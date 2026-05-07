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
# ML Setup (YOLO & DINOv2)
# ──────────────────────────────────────────────
print("Loading ML Models...")

# 1. Object Detection (YOLOv8 Nano)
# We use YOLO (You Only Look Once) to find the actual bike in the image and draw a bounding box around it.
# This prevents background elements (like trees, cars, or walls) from confusing the visual similarity check.
# 'yolov8n.pt' is the smallest and fastest version of YOLOv8, which is perfect for this usecase.
yolo_model = YOLO('yolov8n.pt') 

# 2. Image Embedding / Feature Extraction (DINOv2)
# DINOv2 is a state-of-the-art vision model by Meta that understands image features without labels.
# It converts an image into a embedding representing its visual characteristics.
# We use the 'vits14' (Vision Transformer Small) variant, mapping it to the chosen device (CPU/GPU).
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
dino_model.eval() # Set to evaluation mode (disables training-specific layers like dropout)

# 3. Image Preprocessing Pipeline
# DINOv2 expects images in a very specific format and size.
# This pipeline standardizes any input image before feeding it into the DINOv2 model.
transform = T.Compose([
    # Resize the smaller edge to 256 pixels while maintaining aspect ratio (using high-quality BICUBIC interpolation)
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    # Extract the central 224x224 pixel square (standard input size for Vision Transformers)
    T.CenterCrop(224),
    # Convert the PIL image to a PyTorch tensor (pixels become values between 0.0 and 1.0)
    T.ToTensor(),
    # Normalize the pixel values using standard ImageNet mean and standard deviation.
    # This centers the data around 0, helping the neural network process it more effectively.
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def process_image(img: Image.Image) -> torch.Tensor:
    """
    Takes a raw PIL image, crops the bike using YOLO, and computes its visual embedding using DINOv2.
    """
    # 1. Run YOLO object detection to find bounding boxes for all objects in the image
    results = yolo_model(img, verbose=False)
    
    cropped_img = img
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Check if the detected object is a 'bicycle' (Class 1 in the standard COCO dataset)
            if int(box.cls[0]) == 1: 
                # Extract the top-left (x1, y1) and bottom-right (x2, y2) coordinates of the bounding box
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                
                # Expand the bounding box slightly (by 10 pixels) to include some context around the bike
                # This ensures we don't accidentally clip edges of the tires or handlebars
                w, h = img.size
                x1, y1 = max(0, x1-10), max(0, y1-10)
                x2, y2 = min(w, x2+10), min(h, y2+10)
                
                # Crop the original image to just the expanded bounding box
                cropped_img = img.crop((x1, y1, x2, y2))
                break # Just take the first bike found (assuming the main subject is the first/most prominent detection)
    
    # 2. Transform the cropped image (resize, crop to 224x224, normalize) and add a batch dimension
    # unsqueeze(0) changes shape from [Channels, Height, Width] to [Batch=1, Channels, Height, Width]
    img_t = transform(cropped_img).unsqueeze(0).to(DEVICE)
    
    # 3. Generate the visual embedding (vector representation)
    # torch.no_grad() saves memory and speeds up execution by disabling gradient calculation (we aren't training)
    with torch.no_grad():
        embedding = dino_model(img_t)
        
    # 4. L2 Normalize the final embedding vector.
    # This scales the vector to have a length of 1, which makes cosine similarity calculations mathematically sound and stable.
    return F.normalize(embedding, p=2, dim=1)

def get_embedding_from_url(url: str) -> torch.Tensor:
    try:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        # runs the image feature extraction after receiving the image
        return process_image(img)
    except Exception as e:
        print(f"    [!] Failed to process image URL {url}: {e}")
        return None

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
    password = os.environ.get("GMAIL_APP_PASSWORD") or cfg["email"].get("app_password", "")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 {len(new_listings)} Potential Visual Match(es) Found!"
    msg["From"] = sender
    msg["To"] = recipient
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
    threshold: float
) -> list[dict]:
    """Filters listings by comparing their image embeddings to a reference."""
    visually_matched = []
    for item in listings:
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
    return visually_matched

def main() -> None:
    """
    Main execution pipeline for the bike scraper:
    1. Loads configuration, search keywords, and history of seen listings.
    2. Generates a visual embedding for the reference image (the bike to look for).
    3. Scrapes selected platforms for listings that match the text keywords.
    4. Downloads images of new unseen listings and computes their visual embeddings.
    5. Compares listing images to the reference using cosine similarity.
    6. Sends an email alert with listings that meet the visual similarity threshold and updates the seen cache.
    """
    cfg = load_config()
    query = cfg["search"]["query"]
    required = cfg["search"]["required_keywords"]
    platforms = [p.lower() for p in cfg["search"].get("platforms", list(SCRAPER_MAP))]
    seen = load_seen()

    # 1. Generate Reference Embedding
    ref_image_path = cfg["search"].get("reference_image")
    if not ref_image_path or not Path(ref_image_path).exists():
        print(f"❌ Error: Reference image '{ref_image_path}' not found. Cannot do visual matching.")
        return
    
    print("Generating Reference Embedding...")
    with Image.open(ref_image_path) as ref_img:
        reference_embedding = process_image(ref_img.convert("RGB"))

    # 2. Scrape platforms and filter for new items
    all_listings = scrape_all_platforms(platforms, query, required)
    new_listings = [item for item in all_listings if item["id"] not in seen]
    print(f"\nNew (unseen) Text Matches: {len(new_listings)}")

    # 3. Perform visual matching
    threshold = cfg["search"].get("similarity_threshold", 0.85)
    visually_matched = perform_visual_match(
        new_listings, reference_embedding, threshold
    )

    # 4. Update seen cache and send email
    seen.update(item["id"] for item in all_listings)
    save_seen(seen)

    send_email(cfg, visually_matched)

if __name__ == "__main__":
    main()
