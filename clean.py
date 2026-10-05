"""Step 1b: clean the raw reviews so they're ready to classify.

Each step prints how many reviews it removed, so every dropped row is accounted for:
  1. Drop reviews with no text.
  2. Drop repeated review IDs (the same review returned twice by the scraper).
     Identical texts from different reviewers are kept: in this data they are short praise
     ("very good app") written by different people, not copy-pasted spam.
  3. Drop very short reviews (fewer than MIN_WORDS words). "good app" names no theme.
  4. Detect each review's language (rules in detect_language below). Non-English reviews are
     kept, but flagged, so the classifier still sees them.

Then it writes a 100-review reading sample (20 per star rating) for hand-reading the data
before the theme list is finalized.

Usage:
    python clean.py                 # every app in config/apps.yaml
    python clean.py --app imagineart

Output, per app:
    data/processed/<key>_clean.csv
    data/processed/<key>_reading_sample.csv
"""
import argparse
import re
import unicodedata

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

from common import PROCESSED_DIR, RAW_DIR, read_reviews, select_apps, write_csv

MIN_WORDS = 3            # reviews with fewer words are dropped
SAMPLE_PER_RATING = 20   # reading sample: this many English reviews per star rating (x5 = 100)
SAMPLE_SEED = 42         # fixed seed, so re-running gives the same sample

# --- Language detection -------------------------------------------------------------------------
# lingua is a language detector that works offline. Given all 75 of its languages, it often
# labels short, informal English as a rare language ("I love it" came back as Slovenian), so we
# only let it choose between languages we actually expect in this store.
LIKELY_LANGUAGES = [
    Language.ENGLISH, Language.SPANISH, Language.PORTUGUESE, Language.FRENCH, Language.GERMAN,
    Language.ITALIAN, Language.POLISH, Language.TURKISH, Language.INDONESIAN, Language.MALAY,
    Language.TAGALOG, Language.VIETNAMESE, Language.RUSSIAN, Language.UKRAINIAN, Language.ARABIC,
    Language.PERSIAN, Language.URDU, Language.HINDI, Language.BENGALI, Language.CHINESE,
    Language.JAPANESE, Language.KOREAN, Language.THAI,
]
# Minimum relative distance: when the top two guesses are this close, lingua answers "unsure"
# (None) instead of guessing. 0.2 was picked by testing on real reviews from this app.
DETECTOR = LanguageDetectorBuilder.from_languages(*LIKELY_LANGUAGES).with_minimum_relative_distance(0.2).build()

# lingua has no model for Hindi/Urdu written in Latin letters ("bahut acha app hai"), which many
# South Asian users write in. Reviews where enough words come from this list are tagged "hi-latn".
ROMAN_HINDI_URDU = set("""
    hai hain hy nahi nahin nhi kya kyu kyun ka ki ke ko se mein mai mujhe mera mere meri hum
    tum ye yeh wo woh bahut bohot boht bhot accha acha achha achcha acche kar karo karna
    karke karta raha rahe rahi tha thi bhi kuch kuchh sab sabse bekar bakwas paisa paise ek
    fir phir kaisa kesa jaldi thik theek koi dekhe itna iske bare sakta liye wala wali
""".split())

# Very common English words, including words used in app reviews. A review where at least half
# the words are on this list is English, even if lingua is unsure (short reviews confuse it).
COMMON_ENGLISH = set("""
    a about after all also am an and any are as at be because but by can can't cant could do
    does doesn't dont don't even ever every for from get give got has have how i i'm im i'll if
    in is it it's its just like make many me more most much my need never new no not now of ok
    okay on one only or other out please really should so some still than thank thanks that the
    their them then there they this to too try use used very want was way we well what when why
    will with without would you your good bad best better worst nice love great amazing awesome
    excellent fantastic perfect beautiful cool fun wow waste money pay paid free trial app apps
    ai art image images picture pictures photo photos video videos generate generator create
    quality prompt update ads subscription premium credits account version work works working
""".split())


def share_of_words_in(words: list[str], vocabulary: set[str]) -> float:
    return sum(word in vocabulary for word in words) / len(words)


def detect_language(text: str) -> str:
    """Return a language code: "en", "es", "ar", ... , "hi-latn" (Hindi/Urdu in Latin letters), or "unknown"."""
    text = unicodedata.normalize("NFKC", text)            # fancy letters like "𝒂𝒑𝒑" become plain "app"
    letters = re.findall(r"[^\W\d_]", text)               # letters in any script
    latin_words = re.findall(r"[a-z']+", text.lower())
    mostly_latin = bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.5

    if mostly_latin and latin_words:
        if share_of_words_in(latin_words, ROMAN_HINDI_URDU) >= 0.3:
            return "hi-latn"
        if share_of_words_in(latin_words, COMMON_ENGLISH) >= 0.5:
            return "en"

    language = DETECTOR.detect_language_of(text)          # None when lingua is unsure
    if language is None:
        # Nearly every Latin-letter review in a US English store is English, so assume that.
        return "en" if mostly_latin else "unknown"
    return language.iso_code_639_1.name.lower()


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Run the cleaning steps. Returns the cleaned reviews and (step, rows removed) for the log."""
    log = []
    df = raw.copy()

    def record(step: str, before: int) -> None:
        log.append((step, before - len(df)))

    n = len(df)
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"] != ""]
    record("no text", n)

    n = len(df)
    df = df.drop_duplicates(subset="review_id", keep="first")
    record("repeated review ID", n)

    n = len(df)
    # Count runs of letters/digits (apostrophes allowed, so "don't" is one word). Emoji don't count.
    df["n_words"] = df["text"].str.count(r"[\w']+")
    df = df[df["n_words"] >= MIN_WORDS]
    record(f"shorter than {MIN_WORDS} words", n)

    df["language"] = df["text"].map(detect_language)
    df["is_english"] = df["language"] == "en"

    return df.reset_index(drop=True), log


def reading_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Pick SAMPLE_PER_RATING random English reviews from each star rating, for reading by hand.

    Equal numbers per rating, so the rarer 2-4 star reviews (often the most specific) aren't
    crowded out by 1 and 5 star ones. Empty columns are there for notes while reading.
    """
    english = df[df["is_english"]]
    picks = [
        group.sample(min(len(group), SAMPLE_PER_RATING), random_state=SAMPLE_SEED)
        for _, group in english.groupby("rating")
    ]
    sample = pd.concat(picks).sort_values(["rating", "date"])
    return sample[["review_id", "date", "rating", "app_version", "text"]].assign(my_theme="", notes="")


def print_summary(name: str, raw: pd.DataFrame, df: pd.DataFrame, log: list[tuple[str, int]]) -> None:
    """Print what the cleaned data looks like, to sanity-check it before classifying."""
    print(f"\n=== {name} ===")
    print(f"raw reviews: {len(raw)}")
    for step, removed in log:
        print(f"  - removed {removed:>4} ({step})")
    print(f"clean reviews: {len(df)}  ({df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y})")

    weekly = df.set_index("date").resample("W")["review_id"].count()
    print(f"reviews per week: median {weekly.median():.0f}, min {weekly.min()}, max {weekly.max()}")

    shares = df["rating"].value_counts(normalize=True).sort_index()
    print("ratings: " + "  ".join(f"{star}★ {share:.0%}" for star, share in shares.items()))

    langs = df["language"].value_counts()
    print(f"English: {df['is_english'].mean():.1%}. Other languages: "
          + ", ".join(f"{code} {count}" for code, count in langs.drop("en", errors="ignore").head(8).items()))

    has_version = df["app_version"].notna()
    versions = df.loc[has_version].groupby("app_version").agg(
        reviews=("review_id", "count"), first_seen=("date", "min"), last_seen=("date", "max"))
    print(f"app version known for {has_version.mean():.1%} of reviews; {len(versions)} versions, "
          f"{(versions['reviews'] >= 30).sum()} with 30+ reviews")
    top = versions.sort_values("first_seen", ascending=False).head(12)
    for version, row in top.iterrows():
        print(f"    {version:<10} {row['reviews']:>4} reviews   first seen {row['first_seen']:%d %b %Y}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw reviews and write a reading sample.")
    parser.add_argument("--app", help="key of one app in config/apps.yaml (default: all apps)")
    args = parser.parse_args()

    for app in select_apps(args.app):
        raw_path = RAW_DIR / f"{app['key']}_reviews.csv"
        if not raw_path.exists():
            print(f"Skipping {app['name']}: {raw_path.name} not found. Run scrape.py first.")
            continue
        raw = read_reviews(raw_path)
        df, log = clean(raw)
        write_csv(df, PROCESSED_DIR / f"{app['key']}_clean.csv")
        write_csv(reading_sample(df), PROCESSED_DIR / f"{app['key']}_reading_sample.csv")
        print_summary(app["name"], raw, df, log)


if __name__ == "__main__":
    main()
