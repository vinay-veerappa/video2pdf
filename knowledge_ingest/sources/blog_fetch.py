"""
Blog -> clean text front stage (e.g. MenthorQ, other trading blogs).

Fetches article HTML, strips to clean body text, writes .md into an input dir
with URL + date provenance and source_type: blog. Then the normal pipeline
(prose segmenter) handles it.

Per-source config: each blog source declares which VOCABULARY DOMAIN its content
maps to (MenthorQ = gex/gamma, not ict), plus its credibility. This is why the
vocab registry is domain-pluggable — blogs and transcripts are different bases.

RESPECT THE SOURCE: check robots.txt / terms; prefer an RSS feed or official API
if one exists; rate-limit. This tool fetches only URLs you explicitly provide.

Requires: pip install requests beautifulsoup4 trafilatura
(trafilatura does the best job of main-content extraction; bs4 is the fallback.)

Usage:
    # single url
    python -m knowledge_ingest.sources.blog_fetch \
        --url https://menthorq.com/blog/some-post \
        --out "C:\\path\\to\\blog_input" \
        --source menthorq

    # many urls from a file (one per line)
    python -m knowledge_ingest.sources.blog_fetch \
        --url-file urls.txt --out ... --source menthorq
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests

# image filtering heuristics
_MIN_IMG_BYTES = 8000
_SKIP_NAME_HINTS = ("logo", "icon", "avatar", "sprite", "pixel", "spacer",
                    "banner", "share", "social", "favicon", "emoji")

# --- per-source config: domain + credibility. Extend as you add blogs. ----- #
SOURCES = {
    "menthorq": {
        "vocab_domain": "gex",           # maps to GEX/gamma vocab (build later)
        "credibility": "vendor",
        "rate_limit_s": 3,
    },
    "nqstats": {
        "vocab_domain": "nqstats",           # maps to nqstats vocab (build later)
        "credibility": "vendor",
        "rate_limit_s": 3,
    },
    "generic": {
        "vocab_domain": "ict",
        "credibility": "community",
        "rate_limit_s": 3,
    },
}


def extract_images(html, base_url, out_dir, stem):
    """Download substantive in-content <img> images; skip nav/icon junk.
    Also detect JS-rendered charts (canvas/svg) we CANNOT download and report them.
    Returns (downloaded_paths, js_chart_count)."""
    try:
        from bs4 import BeautifulSoup
    except Exception:
        print("  (bs4 not installed; cannot extract images)")
        return [], 0

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "header", "footer", "aside", "script", "style"]):
        tag.decompose()

    js_charts = len(soup.find_all(["canvas", "svg"]))  # uncapturable as files

    os.makedirs(out_dir, exist_ok=True)
    downloaded, seen = [], set()
    for i, img in enumerate(soup.find_all("img")):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:"):
            continue
        if any(h in src.lower() for h in _SKIP_NAME_HINTS):
            continue
        try:
            w = int(img.get("width", "999")); h = int(img.get("height", "999"))
            if w < 100 or h < 100:
                continue
        except (ValueError, TypeError):
            pass
        full = urljoin(base_url, src)
        if full in seen:
            continue
        seen.add(full)
        try:
            resp = requests.get(full, timeout=30,
                                headers={"User-Agent": "research-ingest/1.0"})
            resp.raise_for_status()
            if len(resp.content) < _MIN_IMG_BYTES:
                continue
            ext = os.path.splitext(urlparse(full).path)[1] or ".png"
            fp = os.path.join(out_dir, f"{stem}_img{i}{ext}")
            with open(fp, "wb") as f:
                f.write(resp.content)
            downloaded.append(fp)
        except Exception:
            continue
    return downloaded, js_charts


def clean_html(html: str) -> str:
    """Extract main article text. trafilatura preferred; bs4 fallback."""
    try:
        import trafilatura
        txt = trafilatura.extract(html, include_comments=False,
                                   include_tables=True)
        if txt and txt.strip():
            return txt
    except Exception:
        pass
    # fallback: strip tags crudely, drop nav/script/style
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        # prefer <article> or <main> if present
        main = soup.find("article") or soup.find("main") or soup.body
        return main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    except Exception:
        return ""


def slugify(url: str) -> str:
    import re
    s = re.sub(r'^https?://', '', url)
    s = re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    return s[:80]


def fetch_one(url: str, out_dir: str, source_cfg: dict, grab_images: bool = True) -> bool:
    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "research-ingest/1.0"})
        r.raise_for_status()
        # honor the real charset: requests guesses latin-1 for text/* without a
        # declared charset, which mangles UTF-8 (· -> Â·, σ -> Ï, − -> â).
        r.encoding = r.apparent_encoding or "utf-8"
    except Exception as e:
        print(f"  ERROR fetching {url}: {str(e)[:80]}")
        return False
    text = clean_html(r.text)
    if not text.strip():
        print(f"  WARN: no content extracted from {url}")
        return False
    stem = slugify(url)
    out_fp = os.path.join(out_dir, f"{stem}.md")

    # capture in-content images so image-borne meaning isn't lost (mirrors the
    # mixed-PDF router). They route to chart-extract, like PDF-page images.
    imgs, js_charts = ([], 0)
    if grab_images:
        img_dir = os.path.join(out_dir, "_chart_images")
        imgs, js_charts = extract_images(r.text, url, img_dir, stem)

    with open(out_fp, "w", encoding="utf-8") as f:
        f.write(f"<!-- source_type: blog | url: {url} | "
                f"vocab_domain: {source_cfg['vocab_domain']} | "
                f"images: {len(imgs)} | js_charts_uncaptured: {js_charts} | "
                f"fetched: {datetime.now(timezone.utc).isoformat(timespec='seconds')} -->\n\n")
        f.write(text)
    msg = f"  ok: {url}  -> {os.path.basename(out_fp)}"
    if imgs:
        msg += f"  (+{len(imgs)} images -> _chart_images/)"
    print(msg)
    if js_charts:
        print(f"     ! {js_charts} JS-rendered chart(s) (canvas/svg) present but NOT "
              f"downloadable — screenshot manually into chart-extract if they matter.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="single article URL")
    ap.add_argument("--url-file", help="file with one URL per line")
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--source", default="generic",
                    help=f"source key; known: {list(SOURCES)}")
    ap.add_argument("--no-images", action="store_true",
                    help="skip downloading in-content images")
    args = ap.parse_args()

    if args.source not in SOURCES:
        print(f"Unknown source '{args.source}'. Known: {list(SOURCES)}. "
              f"Add it to SOURCES with its vocab_domain.")
        raise SystemExit(1)
    cfg = SOURCES[args.source]
    os.makedirs(args.out_dir, exist_ok=True)

    urls = []
    if args.url:
        urls.append(args.url)
    if args.url_file:
        urls += [l.strip() for l in open(args.url_file) if l.strip()]
    if not urls:
        print("No URLs given (use --url or --url-file).")
        raise SystemExit(1)

    print(f"Fetching {len(urls)} article(s) from source '{args.source}' "
          f"(vocab domain: {cfg['vocab_domain']})")
    ok = 0
    for i, u in enumerate(urls):
        if fetch_one(u, args.out_dir, cfg, grab_images=not args.no_images):
            ok += 1
        if i < len(urls) - 1:
            time.sleep(cfg["rate_limit_s"])  # be polite

    print(f"\nDone. {ok}/{len(urls)} fetched -> {args.out_dir}")
    print(f"Next: run the pipeline with --source-type blog pointed at {args.out_dir}")
    print(f"Then map with --vocab {cfg['vocab_domain']} in the report/recanonicalize steps.")


if __name__ == "__main__":
    main()
