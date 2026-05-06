# 🚴 Schmolke Carbon Bike Scraper

Monitors **Kleinanzeigen** and **OLX.pl** for
Schmolke Carbon bike listings. Only emails you listings that genuinely contain your keywords — crucial for Polish platforms which return many unrelated results.

---

## Project structure

```
bike-scraper/
├── .github/
│   └── workflows/
│       └── daily_scrape.yml   # Runs every day at 07:00 UTC (09:00 CEST)
├── bike_scraper.py            # Main scraper
├── config.json                # Search query, keyword filter, email settings
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
    "query": "Schmolke Carbon",
    "required_keywords": ["schmolke", "carbon"],
    "platforms": ["kleinanzeigen", "ebay", "olx", "allegro"]
  },
  "email": {
    "sender": "your.gmail@gmail.com",
    "recipient": "your.gmail@gmail.com",
    "app_password": ""
  }
}
```

**`required_keywords`** — every word in this list must appear in the listing title.
This is the main defence against unrelated results on Polish platforms.
Leave `app_password` blank; use a GitHub Secret instead (see step 3).

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

Go to **Actions tab → Daily Bike Scraper → Run workflow**.

---

## Running locally

```bash
pip install -r requirements.txt
export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
python bike_scraper.py
```

---

## How the keyword filter works

Polish platforms (OLX) often return listings that don't contain
your search string at all. After scraping, each listing title is checked against
`required_keywords` — **all** words must be present (case-insensitive) for the
listing to pass through. Anything that doesn't match is silently dropped.

You can make the filter stricter or looser by editing `required_keywords`:

```json
"required_keywords": ["schmolke"]           // looser – only "schmolke" required
"required_keywords": ["schmolke", "carbon"] // default
"required_keywords": ["schmolke", "carbon", "rennrad"]  // stricter
```

---

## ⚠️ Notes on scraping


- Random 2–4.5 s delays between requests keep traffic polite.
