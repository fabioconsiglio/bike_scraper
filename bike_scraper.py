"""
Bike Scraper – eBay Kleinanzeigen, OLX.pl
Searches for a specific bike, runs visual matching via YOLOv8 & DINOv2, 
and emails visually verified new listings.
"""

import hashlib
import json
import os
import random
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Paths & Settings
# ──────────────────────────────────────────────
CONFIG_FILE    = Path(__file__).parent / "config.json"
SEEN_FILE      = Path(__file__).parent / "seen_listings.json"

# Force CPU execution for GitHub Actions
DEVICE = torch.device("cpu")

HEADERS_DE = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8",
}
HEADERS_PL = {**HEADERS_DE, "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8"}
HEADERS = {"Accept-Language": "en-US,en;q=0.9"}

# ──────────────────────────────────────────────
# ML Setup (YOLO & DINOv2)
# ──────────────────────────────────────────────
print("Loading ML Models...")
# YOLO Nano for fast object detection
yolo_model = YOLO('yolov8n.pt') 

# DINOv2 Small for robust image embedding
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
dino_model.eval()

# DINOv2 required transforms
transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def process_image(img: Image.Image) -> torch.Tensor:
    """Crops the bike using YOLO, then embeds it with DINO."""
    # 1. Run YOLO to find bounding boxes
    results = yolo_model(img, verbose=False)
    
    cropped_img = img
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class 1 is 'bicycle' in COCO dataset
            if int(box.cls[0]) == 1: 
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                # Expand box slightly for context
                w, h = img.size
                x1, y1 = max(0, x1-10), max(0, y1-10)
                x2, y2 = min(w, x2+10), min(h, y2+10)
                cropped_img = img.crop((x1, y1, x2, y2))
                break # Just take the first bike found
    
    # 2. Transform and Embed
    img_t = transform(cropped_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = dino_model(img_t)
        
    return F.normalize(embedding, p=2, dim=1)

def get_embedding_from_url(url: str) -> torch.Tensor:
    try:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return process_image(img)
    except Exception as e:
        print(f"    [!] Failed to process image URL {url}: {e}")
        return None

# ──────────────────────────────────────────────
# Core Helpers
# ──────────────────────────────────────────────
def _get(url: str, headers: dict = HEADERS, timeout: int = 20):
    browsers = ["chrome124", "chrome120", "safari17_0", "edge122"]
    chosen_browser = random.choice(browsers)
    try:
        resp = cffi_requests.get(url, headers=headers, timeout=timeout, impersonate=chosen_browser)
        if resp.status_code != 200:
            print(f"    [!] Warning: Got HTTP {resp.status_code} from {url}")
        return resp
    except Exception as e:
        print(f"    [!] Connection error: {e}")
        class DummyResp:
            status_code = 500
            text = ""
        return DummyResp()

def _id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def _sleep():
    time.sleep(random.uniform(2.0, 4.5))

def matches_keywords(text: str, required_keywords: list[str]) -> bool:
    text_lower = text.lower()
    return all(kw.lower() in text_lower for kw in required_keywords)

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
    listings = []
    try:
        formatted_query = query.strip().lower().replace(" ", "-")
        url = f"https://www.kleinanzeigen.de/s-{formatted_query}/k0"
        resp = _get(url)
        
        if resp.status_code != 200:
            return listings

        soup = BeautifulSoup(resp.text, "html.parser")
        for card in soup.select("article.aditem"):
            title_el = card.select_one("a.ellipsis")
            price_el = card.select_one("p.aditem-main--middle--price-shipping--price")
            loc_el   = card.select_one("div.aditem-main--top--left")
            img_el   = card.select_one("div.imagebox img") # NEW: Extract Image
            
            if not title_el: continue

            title = title_el.get_text(strip=True)
            href  = title_el.get("href", "")
            link  = "https://www.kleinanzeigen.de" + href if href.startswith("/") else href
            img_url = img_el.get("src") if img_el else None

            if not matches_keywords(title, required): continue

            listings.append({
                "id": _id(link), "title": title, 
                "price": price_el.get_text(strip=True) if price_el else "N/A",
                "location": loc_el.get_text(strip=True) if loc_el else "N/A",
                "url": link, "image_url": img_url, "source": "Kleinanzeigen"
            })
    except Exception as exc:
        print(f"  [Kleinanzeigen] ERROR: {exc}")
    _sleep()
    return listings

def scrape_olx(query: str, required: list[str]) -> list[dict]:
    listings = []
    try:
        formatted_query = query.strip().lower().replace(" ", "-")
        url = f"https://www.olx.pl/oferty/q-{formatted_query}/"
        resp = _get(url, headers=HEADERS_PL)
        
        if resp.status_code != 200: return listings

        soup = BeautifulSoup(resp.text, "html.parser")
        for card in soup.select("[data-cy='l-card']"):
            title_el = card.select_one("h6")
            price_el = card.select_one("[data-testid='ad-price']")
            loc_el   = card.select_one("[data-testid='location-date']")
            link_el  = card.select_one("a[href]")
            img_el   = card.select_one("img") # NEW: Extract Image

            if not (title_el and link_el): continue

            title = title_el.get_text(strip=True)
            href  = link_el.get("href", "")
            link  = href if href.startswith("http") else "https://www.olx.pl" + href
            img_url = img_el.get("src") if img_el else None

            if not matches_keywords(title, required): continue

            listings.append({
                "id": _id(link), "title": title,
                "price": price_el.get_text(strip=True) if price_el else "N/A",
                "location": loc_el.get_text(strip=True) if loc_el else "N/A",
                "url": link, "image_url": img_url, "source": "OLX.pl"
            })
    except Exception as exc:
        print(f"  [OLX.pl] ERROR: {exc}")
    _sleep()
    return listings

# ──────────────────────────────────────────────
# Email Templates (Updated for Images)
# ──────────────────────────────────────────────
EMAIL_HTML = """\
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: sans-serif; background: #f0f4f8; margin: 0; padding: 0; }}
  .wrapper {{ max-width: 680px; margin: 30px auto; background: #fff; border-radius: 14px; padding-bottom: 20px; box-shadow: 0 6px 28px rgba(0,0,0,.10); }}
  .header {{ background: #0f1923; color: #fff; padding: 30px; }}
  .card {{ display: flex; margin: 15px 20px; border: 1px solid #e4e9f0; border-radius: 10px; overflow: hidden; }}
  .card img {{ width: 150px; height: 150px; object-fit: cover; border-right: 1px solid #e4e9f0; }}
  .card-content {{ padding: 15px; display: flex; flex-direction: column; justify-content: center; }}
  .card a.title {{ font-size: 16px; font-weight: 600; color: #0f1923; text-decoration: none; }}
  .price {{ font-size: 17px; font-weight: 700; color: #e84545; margin-top: 6px; }}
  .meta {{ font-size: 12px; color: #7a8899; margin-top: 4px; }}
  .score {{ margin-top: 8px; font-size: 13px; font-weight: bold; color: #2ecc71; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h2>🚴 Stolen Bike Match Alerts</h2>
    <p>{date} - Found {count} visual match(es)</p>
  </div>
  {cards}
</div>
</body>
</html>
"""

CARD_TMPL = """\
<div class="card">
  <img src="{image_url}" alt="Listing Thumbnail" />
  <div class="card-content">
    <a class="title" href="{url}">{title}</a>
    <div class="price">{price}</div>
    <div class="meta">📍 {location} | Source: {source}</div>
    <div class="score">Visual Match: {similarity_score:.1f}%</div>
  </div>
</div>
"""

def build_html(new_listings: list[dict]) -> str:
    cards = "".join(CARD_TMPL.format(**{
        **i, 
        "image_url": i.get("image_url") or "https://via.placeholder.com/150",
        "similarity_score": i.get("similarity_score", 0) * 100
    }) for i in new_listings)
    return EMAIL_HTML.format(date=datetime.now().strftime("%B %d, %Y"), count=len(new_listings), cards=cards)

def send_email(cfg: dict, new_listings: list[dict]) -> None:
    if not new_listings:
        print("No visually matching listings – skipping email.")
        return

    sender = cfg["email"]["sender"]
    recipient = cfg["email"]["recipient"]
    password = os.environ.get("GMAIL_APP_PASSWORD") or cfg["email"].get("app_password", "")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 {len(new_listings)} Potential Visual Match(es) Found!"
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText("Matches found. View email in HTML mode.", "plain", "utf-8"))
    msg.attach(MIMEText(build_html(new_listings), "html", "utf-8"))

    import smtplib
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
        srv.login(sender, password)
        srv.sendmail(sender, recipient, msg.as_string())
    print(f"✅ Email sent to {recipient}.")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
SCRAPER_MAP = {"kleinanzeigen": scrape_kleinanzeigen, "olx": scrape_olx}

def main() -> None:
    cfg = load_config()
    query = cfg["search"]["query"]
    required = cfg["search"]["required_keywords"]
    platforms = [p.lower() for p in cfg["search"].get("platforms", list(SCRAPER_MAP))]
    seen = load_seen()

    # Load Reference Image
    ref_image_path = cfg["search"].get("reference_image")
    if not ref_image_path or not Path(ref_image_path).exists():
        print(f"❌ Error: Reference image '{ref_image_path}' not found. Cannot do visual matching.")
        return
    
    print("Generating Reference Embedding...")
    ref_img = Image.open(ref_image_path).convert("RGB")
    reference_embedding = process_image(ref_img)

    all_listings: list[dict] = []
    for platform in platforms:
        if platform in SCRAPER_MAP:
            print(f"\n── Scraping {platform} ──")
            all_listings += SCRAPER_MAP[platform](query, required)

    new_listings = [item for item in all_listings if item["id"] not in seen]
    print(f"\nNew (unseen) Text Matches: {len(new_listings)}")

    # ──────────────────────────────────────────────
    # Visual Matching Step
    # ──────────────────────────────────────────────
    threshold = cfg["search"].get("similarity_threshold", 0.85)
    visually_matched = []

    for item in new_listings:
        if not item.get("image_url"):
            continue
            
        print(f"Analyzing Image for: {item['title'][:30]}...")
        listing_emb = get_embedding_from_url(item["image_url"])
        
        if listing_emb is not None:
            similarity = F.cosine_similarity(reference_embedding, listing_emb).item()
            print(f"  -> Similarity: {similarity:.2f}")
            
            if similarity >= threshold:
                item["similarity_score"] = similarity
                visually_matched.append(item)

    # Save IDs to seen cache
    seen.update(item["id"] for item in all_listings)
    save_seen(seen)

    send_email(cfg, visually_matched)

if __name__ == "__main__":
    main()