"""Step 1a: download recent Google Play reviews for the apps in config/apps.yaml.

Usage:
    python scrape.py                                  # every app in the config
    python scrape.py --app imagineart                 # one app
    python scrape.py --app imagineart --max-reviews 400   # quick test run

Output, per app:
    data/raw/<key>_reviews.csv   one row per review (newest first)
    data/raw/<key>_app.json      snapshot of the store listing at scrape time (rating, installs, version)

Only public review fields are kept. Reviewer names and profile pictures are not saved.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd
from google_play_scraper import Sort, app as fetch_app_info, reviews as fetch_review_page

from common import RAW_DIR, select_apps, write_csv

PAGE_SIZE = 200       # Google Play returns at most about 200 reviews per request
PAUSE_SECONDS = 1.5   # wait between requests so we don't hammer Google's servers
MAX_RETRIES = 3       # attempts per request before giving up


def with_retries(fn, *args, **kwargs):
    """Call fn(*args, **kwargs). If it fails (network blip, rate limit), wait longer each time and retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as error:
            if attempt == MAX_RETRIES:
                raise
            wait = PAUSE_SECONDS * 2 ** attempt  # 3s, then 6s
            print(f"  request failed ({error}); retrying in {wait:.0f}s")
            time.sleep(wait)


def scrape_reviews(package: str, lang: str, country: str, max_reviews: int) -> list[dict]:
    """Page through an app's reviews, newest first, until we have max_reviews or there are no more.

    Google Play serves reviews in pages. Each response includes a continuation token that
    points to the next page; we pass it back in to get the following page.
    """
    collected, token = [], None
    while len(collected) < max_reviews:
        page, token = with_retries(
            fetch_review_page, package, lang=lang, country=country,
            sort=Sort.NEWEST, count=PAGE_SIZE, continuation_token=token,
        )
        collected.extend(page)
        print(f"  {len(collected):>5} reviews fetched")
        if not page or token is None or token.token is None:
            break  # reached the oldest review
        time.sleep(PAUSE_SECONDS)
    return collected[:max_reviews]


def to_row(review: dict) -> dict:
    """Keep only the fields we analyze, under clear column names."""
    return {
        "review_id": review["reviewId"],
        "date": review["at"],
        "rating": review["score"],  # 1-5 stars
        # The app version the reviewer had installed. Missing when Google doesn't know it.
        "app_version": review.get("reviewCreatedVersion") or review.get("appVersion"),
        "thumbs_up": review["thumbsUpCount"],  # other users who marked the review helpful
        "has_dev_reply": bool(review.get("replyContent")),  # did the developer reply publicly?
        "text": review["content"],
    }


def store_snapshot(app: dict, info: dict, df: pd.DataFrame) -> dict:
    """Store-listing numbers at scrape time, kept for context in the dashboard."""
    return {
        "key": app["key"],
        "name": app["name"],
        "package": app["package"],
        "title": info.get("title"),
        "developer": info.get("developer"),
        "store_rating": info.get("score"),
        "rating_count": info.get("ratings"),
        "installs": info.get("installs"),
        "current_version": info.get("version"),
        "lang": app["lang"],
        "country": app["country"],
        "reviews_saved": len(df),
        "oldest_review": str(df["date"].min()),
        "newest_review": str(df["date"].max()),
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Google Play reviews for the configured apps.")
    parser.add_argument("--app", help="key of one app in config/apps.yaml (default: all apps)")
    parser.add_argument("--max-reviews", type=int, help="override max_reviews from the config")
    args = parser.parse_args()

    for app in select_apps(args.app):
        max_reviews = args.max_reviews or app["max_reviews"]
        print(f"{app['name']} ({app['package']}): fetching up to {max_reviews} reviews")

        info = with_retries(fetch_app_info, app["package"], lang=app["lang"], country=app["country"])
        raw = scrape_reviews(app["package"], app["lang"], app["country"], max_reviews)
        df = pd.DataFrame([to_row(r) for r in raw])

        reviews_path = RAW_DIR / f"{app['key']}_reviews.csv"
        write_csv(df, reviews_path)
        snapshot = store_snapshot(app, info, df)
        (RAW_DIR / f"{app['key']}_app.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

        print(f"  saved {len(df)} reviews ({snapshot['oldest_review'][:10]} to "
              f"{snapshot['newest_review'][:10]}) -> {reviews_path.relative_to(RAW_DIR.parent.parent)}\n")


if __name__ == "__main__":
    main()
