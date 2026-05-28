# 🚴 Find my Bike Scraper with ML Visual Matching

Monitors **Kleinanzeigen** and **OLX.pl** for
specific bike listings. Uses **YOLOv8** and **DINOv2** to perform visual matching against a reference image, emailing you only the listings that both match your keywords and look visually similar to your bike!


---



## Project structure

```
bike-scraper/
├── .github/
│   └── workflows/
│       └── daily_scrape.yml   # Runs every day at 07:00 UTC (09:00 CEST)
├── bike_scraper.py            # Main scraper
├── config.json                # Search query, keyword filter, email settings
├── templates/                 # HTML templates for email alerts
│   ├── email_layout.html
│   └── email_card.html
├── seen_listings.json         # Auto-generated; tracks already-sent listings
├── requirements.txt
└── .gitignore
```

---

## Quick-start

### 1. Edit `config.json`

```json
{
  "search": {
    "query": "Your Bike",
    "required_keywords": ["bike", "green"],
    "platforms": ["kleinanzeigen", "olx"],
    "reference_image": "your_bike.png",
    "similarity_threshold": 0.82
  }
}
```

**`required_keywords`** — every word in this list must appear in the listing title.
This is the main defence against unrelated results on Polish platforms.
**`reference_image`** — provide a clear image of the bike you are looking for (e.g., `favorite_bike.png`). The scraper will use this to verify listings visually.
Leave `app_password` blank in public repositories; use a GitHub Secret instead (see step 3).

### 2. Get a Gmail App Password

1. Google Account → **Security** → **2-Step Verification** (must be enabled first).
2. Scroll down → **App passwords**.
3. Generate one (name it "bike-scraper"), copy the 16-char code.

### 3. Add it as a GitHub Secret

`Repo → Settings → Secrets and variables → Actions → New repository secret`

- Name: `GMAIL_APP_PASSWORD`
- Value: the 16-char app password

### 4. Push to `main` and test

```bash
git add .
git commit -m "feat: bike scraper"
git push origin main
```

Go to **Actions tab → Daily Bike Scraper → Run workflow**

I added a cron job in my ``yml``-file to trigger the task every day at 12 p.m. 

---

## Running locally

```bash
pip install -r requirements.txt
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
python bike_scraper.py
```

---

# Bike Finding and Matching

## How filtering & matching works

1. **Keyword Filter**: After scraping, each listing title is checked against `required_keywords` — **all** words must be present (case-insensitive). This filters out spam/unrelated results.
2. **Visual Matching (YOLOv8 + DINOv2)**: For listings that pass the text filter, the image is downloaded. YOLOv8 locates the bike in the image and crops it. DINOv2 then generates a feature embedding of the cropped bike and compares it to your `reference_image`. 
3. **Alert**: If the visual similarity score is above your `similarity_threshold` (e.g., `0.82`), you receive an email alert.

### Editing Keywords

You can make the text filter stricter or looser by editing `required_keywords`:

```json
"required_keywords": ["company"]           // looser – only "schmolke" required
"required_keywords": ["company", "carbon"] // default
"required_keywords": ["company", "carbon", "gravel"]  // stricter
```

---

## ⚠️ Notes on scraping
- Random 2–4.5 s delays between requests keep traffic polite.
