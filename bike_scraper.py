"""
Bike Scraper – eBay Kleinanzeigen, eBay, OLX.pl, Allegro.pl, Otomoto.pl
Searches for a specific bike and emails only new, verified listings.
"""

import hashlib
import json
import os
import random
import re
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup



from curl_cffi import requests as cffi_requests

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
CONFIG_FILE    = Path(__file__).parent / "config.json"
SEEN_FILE      = Path(__file__).parent / "seen_listings.json"

# ──────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────
HEADERS_DE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}
HEADERS_PL = {**HEADERS_DE, "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8"}

# Use a clean, minimal header block. 
# Too many custom headers can actually trigger bot flags.
HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

def _get(url: str, headers: dict = HEADERS, timeout: int = 20):
    # Randomize the browser profile to avoid static fingerprinting
    browsers = ["chrome124", "chrome120", "safari17_0", "edge122"]
    chosen_browser = random.choice(browsers)
    
    try:
        resp = cffi_requests.get(
            url, 
            headers=headers, 
            timeout=timeout, 
            impersonate=chosen_browser
        )
        
        if resp.status_code != 200:
            print(f"    [!] Warning: Got HTTP {resp.status_code} from {url}")
            # Quick check to see if it's a hard block or a Captcha
            if "captcha" in resp.text.lower() or "verify you are human" in resp.text.lower():
                print("    [!] We hit a CAPTCHA wall.")
                
        return resp
    except Exception as e:
        print(f"    [!] Connection error: {e}")
        # Return a dummy response object so the script doesn't crash
        class DummyResp:
            status_code = 500
            text = ""
        return DummyResp()

def _id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _sleep():
    time.sleep(random.uniform(2.0, 4.5))


# ──────────────────────────────────────────────
# Keyword matching
# ──────────────────────────────────────────────

def matches_keywords(text: str, required_keywords: list[str]) -> bool:
    """
    Returns True only if EVERY keyword (or multi-word phrase) is found
    in the text (case-insensitive). This is the main guard against
    unrelated listings on Polish platforms.
    """
    text_lower = text.lower()
    return all(kw.lower() in text_lower for kw in required_keywords)


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


def load_seen() -> set:
    if SEEN_FILE.exists():
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen(seen: set) -> None:
    with open(SEEN_FILE, "w") as f:
        json.dump(sorted(seen), f, indent=2)


# ──────────────────────────────────────────────
# Scrapers
# ──────────────────────────────────────────────

def scrape_kleinanzeigen(query: str, required: list[str]) -> list[dict]:
    """eBay Kleinanzeigen (kleinanzeigen.de)"""
    listings = []
    try:
        # Format the query with hyphens (e.g., "Schmolke Carbon" -> "schmolke-carbon")
        formatted_query = query.strip().lower().replace(" ", "-")
        url = f"https://www.kleinanzeigen.de/s-{formatted_query}/k0"
        
        resp = _get(url)
        
        if resp.status_code != 200:
            print(f"  [Kleinanzeigen] BLOCKED or ERROR: HTTP {resp.status_code}")
            return listings

        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Select all listing cards
        cards = soup.select("article.aditem")
        if not cards:
            print("  [Kleinanzeigen] No cards found in the HTML. (Check browser to see if the query yields results)")

        for card in cards:
            title_el = card.select_one("a.ellipsis")
            price_el = card.select_one("p.aditem-main--middle--price-shipping--price")
            loc_el   = card.select_one("div.aditem-main--top--left")
            
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            href  = title_el.get("href", "")
            link  = "https://www.kleinanzeigen.de" + href if href.startswith("/") else href

            # Strict keyword check
            if not matches_keywords(title, required):
                continue

            listings.append({
                "id":      _id(link),
                "title":   title,
                "price":   price_el.get_text(strip=True) if price_el else "N/A",
                "location": loc_el.get_text(strip=True) if loc_el else "N/A",
                "url":     link,
                "source":  "Kleinanzeigen",
            })

        print(f"  [Kleinanzeigen] {len(listings)} matching listings")
    except Exception as exc:
        print(f"  [Kleinanzeigen] ERROR: {exc}")
    
    _sleep()
    return listings

def scrape_olx(query: str, required: list[str]) -> list[dict]:
    """OLX.pl – Polish classifieds"""
    listings = []
    try:
        # OLX prefers clean hyphens instead of URL encoding for the path
        formatted_query = query.strip().lower().replace(" ", "-")
        url = f"https://www.olx.pl/oferty/q-{formatted_query}/"
        
        resp = _get(url, headers=HEADERS_PL)
        
        if resp.status_code != 200:
            print(f"  [OLX.pl] BLOCKED or ERROR: HTTP {resp.status_code}")
            return listings

        soup = BeautifulSoup(resp.text, "html.parser")

        # Relying heavily on data attributes rather than randomized CSS classes
        for card in soup.select("[data-cy='l-card']"):
            title_el = card.select_one("h6")
            price_el = card.select_one("[data-testid='ad-price']")
            loc_el   = card.select_one("[data-testid='location-date']")
            link_el  = card.select_one("a[href]")

            if not (title_el and link_el):
                continue

            title = title_el.get_text(strip=True)
            href  = link_el.get("href", "")
            link  = href if href.startswith("http") else "https://www.olx.pl" + href

            if not matches_keywords(title, required):
                continue

            listings.append({
                "id":      _id(link),
                "title":   title,
                "price":   price_el.get_text(strip=True) if price_el else "N/A",
                "location": loc_el.get_text(strip=True) if loc_el else "N/A",
                "url":     link,
                "source":  "OLX.pl",
            })

        print(f"  [OLX.pl] {len(listings)} matching listings")
    except Exception as exc:
        print(f"  [OLX.pl] ERROR: {exc}")
    _sleep()
    return listings




# ──────────────────────────────────────────────
# Email
# ──────────────────────────────────────────────

EMAIL_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #f0f4f8;
          color: #1e2a38; margin: 0; padding: 0; }}
  .wrapper {{ max-width: 680px; margin: 30px auto; background: #fff;
              border-radius: 14px; overflow: hidden;
              box-shadow: 0 6px 28px rgba(0,0,0,.10); }}
  .header  {{ background: #0f1923; color: #fff; padding: 32px 40px; }}
  .header h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -.3px; }}
  .header p  {{ margin: 6px 0 0; opacity: .55; font-size: 14px; }}
  .badge {{ display: inline-block; background: #e84545; color: #fff;
            border-radius: 20px; padding: 4px 14px; font-size: 12px;
            font-weight: 700; margin-top: 12px; letter-spacing: .04em; }}
  .section-label {{ font-size: 11px; font-weight: 700; letter-spacing: .1em;
                    text-transform: uppercase; color: #8a97a8;
                    padding: 22px 36px 6px; border-top: 1px solid #eef0f3; }}
  .section-label:first-of-type {{ border-top: none; }}
  .card {{ margin: 0 22px 12px; border: 1px solid #e4e9f0; border-radius: 10px;
           padding: 16px 20px; background: #fafbfd;
           transition: box-shadow .2s; }}
  .card a.title {{ font-size: 16px; font-weight: 600; color: #0f1923;
                   text-decoration: none; }}
  .card a.title:hover {{ text-decoration: underline; color: #e84545; }}
  .price {{ font-size: 17px; font-weight: 700; color: #e84545; margin-top: 6px; }}
  .meta  {{ font-size: 12px; color: #7a8899; margin-top: 4px; }}
  .footer {{ text-align: center; padding: 22px; font-size: 11px; color: #aab4c0; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>🚴 Schmolke Carbon – New Listings</h1>
    <p>{date}</p>
    <div class="badge">🔔 {count} new listing{plural}</div>
  </div>
  {sections}
  <div class="footer">Your Bike Scraper · {date}</div>
</div>
</body>
</html>
"""

SECTION_TMPL = """\
<div class="section-label">{source} &nbsp;·&nbsp; {n} listing{plural}</div>
{cards}
"""

CARD_TMPL = """\
<div class="card">
  <a class="title" href="{url}">{title}</a>
  <div class="price">{price}</div>
  <div class="meta">📍 {location}</div>
</div>
"""


def build_html(new_listings: list[dict]) -> str:
    by_source: dict[str, list] = {}
    for item in new_listings:
        by_source.setdefault(item["source"], []).append(item)
    sections = []
    for source, items in by_source.items():
        cards = "".join(CARD_TMPL.format(**i) for i in items)
        sections.append(SECTION_TMPL.format(
            source=source, n=len(items),
            plural="s" if len(items) != 1 else "",
            cards=cards,
        ))
    return EMAIL_HTML.format(
        date=datetime.now().strftime("%B %d, %Y"),
        count=len(new_listings),
        plural="s" if len(new_listings) != 1 else "",
        sections="\n".join(sections),
    )


def build_plain(new_listings: list[dict]) -> str:
    lines = [f"🚴 Schmolke Carbon – New Listings – {datetime.now().strftime('%Y-%m-%d')}", "=" * 50]
    for item in new_listings:
        lines += [
            f"\n[{item['source']}]  {item['title']}",
            f"  Price:    {item['price']}",
            f"  Location: {item['location']}",
            f"  Link:     {item['url']}",
        ]
    return "\n".join(lines)


def send_email(cfg: dict, new_listings: list[dict]) -> None:
    if not new_listings:
        print("No new listings – skipping email.")
        return

    sender    = cfg["email"]["sender"]
    recipient = cfg["email"]["recipient"]
    password  = os.environ.get("GMAIL_APP_PASSWORD") or cfg["email"].get("app_password", "")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚴 {len(new_listings)} New Schmolke Carbon Listing(s) – {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(build_plain(new_listings), "plain", "utf-8"))
    msg.attach(MIMEText(build_html(new_listings),  "html",  "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
        srv.login(sender, password)
        srv.sendmail(sender, recipient, msg.as_string())
    print(f"✅  Email sent to {recipient} ({len(new_listings)} listings).")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

SCRAPER_MAP = {
    "kleinanzeigen": scrape_kleinanzeigen,
    "olx":           scrape_olx
}


def main() -> None:
    cfg      = load_config()
    query    = cfg["search"]["query"]
    required = cfg["search"]["required_keywords"]   # every word must appear in the title
    platforms = [p.lower() for p in cfg["search"].get("platforms", list(SCRAPER_MAP))]
    seen     = load_seen()

    all_listings: list[dict] = []

    for platform in platforms:
        if platform not in SCRAPER_MAP:
            print(f"Unknown platform '{platform}', skipping.")
            continue
        print(f"\n── Scraping {platform} ──")
        all_listings += SCRAPER_MAP[platform](query, required)

    print(f"\nTotal fetched (after keyword filter) : {len(all_listings)}")

    new_listings = [item for item in all_listings if item["id"] not in seen]
    print(f"New (unseen)                         : {len(new_listings)}")

    seen.update(item["id"] for item in all_listings)
    save_seen(seen)
    print(f"Seen cache: {len(seen)} IDs saved → {SEEN_FILE}")

    send_email(cfg, new_listings)


if __name__ == "__main__":

     main() # Keep your main function commented out until these return 200
