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

import cv2
import numpy as np 

# ──────────────────────────────────────────────
# Paths & Settings
# ──────────────────────────────────────────────
CONFIG_FILE    = Path(__file__).parent / "config.json"
SEEN_FILE      = Path(__file__).parent / "seen_listings.json"
TEMPLATE_DIR   = Path(__file__).parent / "templates"

# Force CPU execution for GitHub Actions - sufficient for inference
DEVICE = torch.device("cpu")

# headers for the scraping 
HEADERS_DE = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8",
}
HEADERS_PL = {**HEADERS_DE, "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8"}
HEADERS = {"Accept-Language": "en-US,en;q=0.9"}

# ──────────────────────────────────────────────
# ML Setup (YOLO, DINOv2, & OpenCV)
# ──────────────────────────────────────────────
print("Loading ML Models...")

yolo_model = YOLO('yolov8n.pt') 
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
dino_model.eval() 

transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def get_color_histogram(cropped_pil_img: Image.Image) -> np.ndarray:
    """Extracts a normalized HSV color histogram from a cropped image."""
    cv_img = cv2.cvtColor(np.array(cropped_pil_img), cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def color_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compares two histograms. Returns a score from -1.0 to 1.0."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def process_image(img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Crops the bike using YOLO, then returns its DINO embedding AND Color Histogram."""
    results = yolo_model(img, verbose=False)
    
    cropped_img = img
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls[0]) == 1: # Bicycle
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                w, h = img.size
                x1, y1 = max(0, x1-10), max(0, y1-10)
                x2, y2 = min(w, x2+10), min(h, y2+10)
                cropped_img = img.crop((x1, y1, x2, y2))
                break 
    
    # 1. Structural Embedding (DINOv2)
    img_t = transform(cropped_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = dino_model(img_t)
    normalized_embedding = F.normalize(embedding, p=2, dim=1)
    
    # 2. Color Profile (OpenCV)
    color_hist = get_color_histogram(cropped_img)
    
    return normalized_embedding, color_hist

def get_features_from_url(url: str) -> tuple[torch.Tensor, np.ndarray]:
    """Downloads an image and extracts both structural and color features."""
    try:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return process_image(img)
    except Exception as e:
        print(f"    [!] Failed to process image URL {url}: {e}")
        return None, None

# ──────────────────────────────────────────────
# Core Helpers
# ──────────────────────────────────────────────
def _get(url: str, headers: dict = HEADERS, timeout: int = 20):
    browsers = ["chrome124", "chrome120", "safari17_0", "edge101"]
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
# Email Handling
# ──────────────────────────────────────────────
def build_html(new_listings: list[dict]) -> str:
    """Builds the HTML for the email from templates."""
    try:
        with open(TEMPLATE_DIR / "email_layout.html", "r", encoding="utf-8") as f:
            layout_tmpl = f.read()
        with open(TEMPLATE_DIR / "email_card.html", "r", encoding="utf-8") as f:
            card_tmpl = f.read()
    except FileNotFoundError as e:
        print(f"❌ Email template not found: {e}. Cannot build email.")
        return ""

    cards = "".join(card_tmpl.format(**{
        **i, 
        "image_url": i.get("image_url") or "https://via.placeholder.com/150",
        "similarity_score": i.get("similarity_score", 0) * 100
    }) for i in new_listings)
    
    return layout_tmpl.replace("{date}", datetime.now().strftime("%B %d, %Y")) \
                      .replace("{count}", str(len(new_listings))) \
                      .replace("{cards}", cards)

def send_email(cfg: dict, new_listings: list[dict]) -> None:
    if not new_listings:
        print("No visually matching listings – skipping email.")
        return

    html_content = build_html(new_listings)
    if not html_content:
        return  # Error message already printed in build_html

    sender = cfg["email"]["sender"]
    recipient = cfg["email"]["recipient"]
    password =  os.environ.get("GMAIL_APP_PASSWORD") or cfg["email"].get("app_password", "")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 {len(new_listings)} Potential Visual Match(es) Found!"
    msg["From"] = sender
    msg["To"] = ", ".join(recipient) if isinstance(recipient, list) else recipient
    msg.attach(MIMEText("Matches found. View email in HTML mode.", "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    import smtplib
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
        srv.login(sender, password)
        srv.sendmail(sender, recipient, msg.as_string())
    print(f"✅ Email sent to {recipient}.")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
SCRAPER_MAP = {"kleinanzeigen": scrape_kleinanzeigen, "olx": scrape_olx}

def scrape_all_platforms(platforms: list[str], query: str, required: list[str]) -> list[dict]:
    """Iterates through enabled platforms and runs their scrapers."""
    all_listings: list[dict] = []
    for platform in platforms:
        if platform in SCRAPER_MAP:
            print(f"\n── Scraping {platform} ──")
            all_listings.extend(SCRAPER_MAP[platform](query, required))
    return all_listings


def perform_visual_match(
    listings: list[dict], 
    reference_embedding: torch.Tensor, 
    reference_color_hist: np.ndarray,
    dino_threshold: float,
    color_threshold: float
) -> list[dict]:
    """Filters listings by comparing both their image embeddings and color histograms to a reference."""
    visually_matched = []
    for item in listings:
        if not item.get("image_url"):
            continue
            
        print(f"Analyzing Image for: {item['title'][:30]}...")
        listing_emb, listing_color = get_features_from_url(item["image_url"])
        
        if listing_emb is not None and listing_color is not None:
            # Calculate both scores
            dino_sim = F.cosine_similarity(reference_embedding, listing_emb).item()
            color_sim = color_similarity(reference_color_hist, listing_color)
            
            print(f"  -> Structure: {dino_sim:.2f} | Color: {color_sim:.2f}")
            
            # Must pass BOTH thresholds
            if dino_sim >= dino_threshold and color_sim >= color_threshold:
                item["similarity_score"] = dino_sim
                item["color_score"] = color_sim
                visually_matched.append(item)
                
    return visually_matched

def main() -> None:
    cfg = load_config()
    query = cfg["search"]["query"]
    required = cfg["search"]["required_keywords"]
    platforms = [p.lower() for p in cfg["search"].get("platforms", list(SCRAPER_MAP))]
    seen = load_seen()

    # 1. Generate Reference Features
    ref_image_path = cfg["search"].get("reference_image")
    if not ref_image_path or not Path(ref_image_path).exists():
        print(f"❌ Error: Reference image '{ref_image_path}' not found. Cannot do visual matching.")
        return
    
    print("Generating Reference Features...")
    with Image.open(ref_image_path) as ref_img:
        reference_embedding, reference_color_hist = process_image(ref_img.convert("RGB"))

    # 2. Scrape platforms and filter for new items
    all_listings = scrape_all_platforms(platforms, query, required)
    new_listings = [item for item in all_listings if item["id"] not in seen]
    print(f"\nNew (unseen) Text Matches: {len(new_listings)}")

    # 3. Perform visual matching
    dino_threshold = cfg["search"].get("similarity_threshold") # Fallback to 
    color_threshold = cfg["search"].get("color_threshold") # Fallback to 0.50 if not in config
    
    visually_matched = perform_visual_match(
        new_listings, reference_embedding, reference_color_hist, dino_threshold, color_threshold
    )

    # 4. Update seen cache and send email
    seen.update(item["id"] for item in all_listings)
    save_seen(seen)

    send_email(cfg, visually_matched)

if __name__ == "__main__":
    main()
