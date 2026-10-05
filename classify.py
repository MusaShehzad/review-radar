"""Step 3: label each clean review with a theme, a severity and a one-line summary, using an LLM.

How it works:
  1. Build instructions for the model from config/themes.yaml (theme ids and descriptions).
  2. Send the reviews in batches. The model has to answer with JSON matching a schema, so it can
     only choose theme ids from our list.
  3. Check every answer: right review ids, known themes, severity 1-3, a summary. Reviews whose
     answer is missing or invalid are sent again, up to MAX_ROUNDS times.
  4. Save each good label to a cache file straight away. A review is only sent again if its text,
     the instructions or the model change, so re-runs are free, and a run that hits the daily
     quota picks up tomorrow where it stopped.

Usage:
    python classify.py                                 # every app: reviews_per_app reviews each (classifier.yaml)
    python classify.py --app imagineart                # one app
    python classify.py --app imagineart --sample 200   # quick test run on 200 reviews
    python classify.py --app imagineart --dry-run      # print the prompt, call nothing

Output, per app:
    data/processed/<key>_classified.csv   (<key>_sample_classified.csv with --sample)
    data/cache/<key>_labels.jsonl         the label cache
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from common import CACHE_DIR, PROCESSED_DIR, ROOT, WORKSPACE, load_config, read_reviews, select_apps, write_csv
from llm import BadResponse, FatalError, RateLimited, TemporaryError, make_client

MAX_ROUNDS = 3       # times a review is sent before we give up on it for this run
LABEL_FIELDS = ["theme", "secondary_theme", "severity", "summary", "model"]

SEVERITY_GUIDE = """\
  1 = minor, or no problem at all (praise, a suggestion, not a review yet)
  2 = a real problem that makes the app worse to use
  3 = blocks the user or costs them money (can't generate anything, charged unexpectedly, wants a refund, uninstalling)"""


class DailyQuotaReached(Exception):
    """The provider's daily quota is used up. Stop, keep what's done, resume later."""


# --- Which reviews ----------------------------------------------------------------------------------

def pick_reviews(reviews: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """A random sample of n reviews that stays the same from run to run.

    Each review id is turned into a random-looking number (its SHA-256 hash) and we keep the n
    lowest. The hash doesn't depend on the review's content, date or rating, so this is as good as
    a random sample. Unlike a shuffle, it picks the same reviews every run: a bigger n keeps all
    the earlier picks, and newly scraped reviews only add to the pool, so cached labels stay in use.
    """
    if not n or n >= len(reviews):
        return reviews
    hashes = reviews["review_id"].map(lambda review_id: hashlib.sha256(review_id.encode("utf-8")).hexdigest())
    return reviews.loc[hashes.sort_values().index[:n]]


def shared_window_start() -> pd.Timestamp | None:
    """First day that every app in config/apps.yaml has clean reviews for.

    Busy apps get hundreds of reviews a week, so their data only reaches back a few weeks.
    Comparing apps is only fair over the same weeks, so besides its overall sample, each app also
    gets a sample of up to reviews_per_app reviews from inside this window. For an app with few
    reviews per week (ImagineArt), that means every review in the window.
    """
    starts = []
    for app in select_apps():
        path = PROCESSED_DIR / f"{app['key']}_clean.csv"
        if path.exists():
            starts.append(read_reviews(path)["date"].min())
    return max(starts) if len(starts) > 1 else None


# --- What we send ---------------------------------------------------------------------------------

def build_instructions(app_name: str, description: str, themes: list[dict]) -> str:
    """The system prompt: the task, the severity scale and the theme list."""
    theme_lines = "\n".join(f"- {t['id']}: {t['description']}" for t in themes)
    return f"""You label Google Play reviews of {app_name}, {description}, for its product team.

For each review, decide:
- theme: the one theme below that best matches the main point of the review.
- secondary_theme: another theme from the list only if the review clearly raises a second issue, otherwise "none".
- severity: how much the issue hurts the user.
{SEVERITY_GUIDE}
- summary: one short line in English (at most 15 words) naming the specific issue or point. Translate reviews that aren't in English.

Themes:
{theme_lines}

Judge by the review text. Use the star rating only to break ties.
Answer with one entry per review, using the review's id."""


def build_prompt(batch: list[dict]) -> str:
    """The user message: one JSON object per review. Short ids (1, 2, 3...) keep answers easy to match."""
    lines = [
        json.dumps({"id": i, "stars": int(review["rating"]), "review": review["text"]}, ensure_ascii=False)
        for i, review in enumerate(batch, start=1)
    ]
    return "Label these reviews (one JSON object per line):\n" + "\n".join(lines)


def build_schema(theme_ids: list[str]) -> dict:
    """JSON schema the answer must follow. The enums stop the model from inventing themes."""
    label = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "theme": {"type": "string", "enum": theme_ids},
            "secondary_theme": {"type": "string", "enum": theme_ids + ["none"]},
            "severity": {"type": "integer"},
            "summary": {"type": "string"},
        },
        "required": ["id", "theme", "secondary_theme", "severity", "summary"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"labels": {"type": "array", "items": label}},
        "required": ["labels"],
        "additionalProperties": False,
    }


# --- Checking what comes back ---------------------------------------------------------------------

def validate(raw: str, batch_size: int, theme_ids: list[str]) -> tuple[dict[int, dict], list[str]]:
    """Parse the model's answer. Returns {review id: label} for valid entries, plus a list of problems.

    The schema should already guarantee most of this, but we never trust it blindly: a review only
    gets a label if every field checks out.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}, ["answer was not valid JSON"]
    items = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}, ["answer had no 'labels' list"]

    valid, problems = {}, []
    for item in items:
        if not isinstance(item, dict):
            problems.append("entry is not an object")
            continue
        review_id, theme = item.get("id"), item.get("theme")
        secondary, severity, summary = item.get("secondary_theme"), item.get("severity"), item.get("summary")
        if type(review_id) is not int or not 1 <= review_id <= batch_size or review_id in valid:
            problems.append(f"bad or repeated id {review_id!r}")
        elif theme not in theme_ids:
            problems.append(f"unknown theme {theme!r}")
        elif secondary not in theme_ids + ["none", None, ""]:
            problems.append(f"unknown secondary theme {secondary!r}")
        elif type(severity) is not int or severity not in (1, 2, 3):
            problems.append(f"severity {severity!r} is not 1, 2 or 3")
        elif not isinstance(summary, str) or not summary.strip():
            problems.append("empty summary")
        else:
            valid[review_id] = {
                "theme": theme,
                # "none", or the same theme twice, both mean there's no second theme
                "secondary_theme": secondary if secondary in theme_ids and secondary != theme else None,
                "severity": severity,
                "summary": summary.strip()[:200],
            }
    missing = batch_size - len(valid)
    if missing:
        problems.append(f"{missing} of {batch_size} reviews without a valid label")
    return valid, problems


# --- Remembering what's done ----------------------------------------------------------------------

class LabelCache:
    """Labels saved on disk, one JSON object per line.

    Each label is stored under a key built from the review (id, rating, text) and a fingerprint of
    the exact instructions and model. Change a theme description or switch model and the
    fingerprint changes, so every review gets labeled again under the new setup.
    """

    def __init__(self, path: Path, fingerprint: str):
        self.path, self.fingerprint, self.labels = path, fingerprint, {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written line from an interrupted run
                self.labels[entry["key"]] = entry

    def key(self, review: dict) -> str:
        raw = f"{self.fingerprint}|{review['review_id']}|{review['rating']}|{review['text']}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def get(self, review: dict) -> dict | None:
        return self.labels.get(self.key(review))

    def save(self, labeled: list[tuple[dict, dict]], model: str) -> None:
        """Append (review, label) pairs to the cache file right away."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.path.open("a", encoding="utf-8") as f:
            for review, label in labeled:
                entry = {"key": self.key(review), "review_id": review["review_id"], **label,
                         "model": model, "labeled_at": now}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self.labels[entry["key"]] = entry


def fingerprint(provider: str, model: str, instructions: str) -> str:
    return hashlib.sha256(f"{provider}|{model}|{instructions}".encode("utf-8")).hexdigest()[:16]


# --- Calling the model politely -------------------------------------------------------------------

class Pacer:
    """Keeps requests at least 60 / requests_per_minute seconds apart."""

    def __init__(self, requests_per_minute: float):
        self.gap, self.last = 60 / requests_per_minute, 0.0

    def wait(self) -> None:
        time.sleep(max(0.0, self.last + self.gap - time.monotonic()))
        self.last = time.monotonic()


def call_with_retries(client, instructions: str, prompt: str, schema: dict, pacer: Pacer, max_attempts: int) -> str:
    """One request, retried with growing waits when the API is busy or rate-limited."""
    for attempt in range(1, max_attempts + 1):
        pacer.wait()
        try:
            return client.generate(instructions, prompt, schema)
        except RateLimited as error:
            if error.daily:
                raise DailyQuotaReached() from error
            wait, reason = error.wait_seconds or 10 * 2 ** (attempt - 1), "rate limited"
        except TemporaryError as error:
            wait, reason = 5 * 2 ** (attempt - 1), str(error)
        if attempt < max_attempts:
            print(f"    {reason}; waiting {wait:.0f}s (attempt {attempt} of {max_attempts})")
            time.sleep(min(wait, 120))
    raise TemporaryError(f"gave up after {max_attempts} attempts")


def label_reviews(todo: list[dict], client, instructions: str, schema: dict, theme_ids: list[str],
                  cache: LabelCache, model: str, batch_size: int, pacer: Pacer, max_attempts: int) -> list[dict]:
    """Label every review in `todo`. Returns the reviews that still have no label at the end."""
    pending = todo
    for round_number in range(1, MAX_ROUNDS + 1):
        if round_number > 1:
            print(f"  round {round_number}: retrying {len(pending)} reviews that came back missing or invalid")
        retry = []
        batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
        for number, batch in enumerate(batches, start=1):
            try:
                raw = call_with_retries(client, instructions, build_prompt(batch), schema, pacer, max_attempts)
                valid, problems = validate(raw, len(batch), theme_ids)
            except (BadResponse, TemporaryError) as error:
                valid, problems = {}, [str(error)]
            cache.save([(review, valid[i]) for i, review in enumerate(batch, start=1) if i in valid], model)
            retry += [review for i, review in enumerate(batch, start=1) if i not in valid]
            note = f"  ({problems[0]})" if problems else ""
            print(f"  batch {number}/{len(batches)}: {len(valid)}/{len(batch)} labeled{note}")
        pending = retry
        if not pending:
            break
    return pending


# --- Output ---------------------------------------------------------------------------------------

def attach_labels(reviews: pd.DataFrame, cache: LabelCache) -> pd.DataFrame:
    """Add the label columns to the reviews (left empty for reviews without a label)."""
    rows = [cache.get(review) or {} for review in reviews.to_dict("records")]
    labels = pd.DataFrame([{field: row.get(field) for field in LABEL_FIELDS} for row in rows], index=reviews.index)
    labels["severity"] = labels["severity"].astype("Int64")
    return reviews.join(labels)


def print_summary(df: pd.DataFrame) -> None:
    done = df[df["theme"].notna()]
    print(f"\n  labeled {len(done)} of {len(df)} reviews")
    if len(done) < len(df):
        print(f"  {len(df) - len(done)} have no label yet. Run the same command again to retry just those.")
    if done.empty:
        return
    table = done.groupby("theme").agg(
        reviews=("review_id", "count"), avg_stars=("rating", "mean"), avg_severity=("severity", "mean"))
    table.insert(1, "share", table["reviews"] / len(done))
    table = table.sort_values("reviews", ascending=False)
    print(table.to_string(formatters={"share": "{:.0%}".format, "avg_stars": "{:.1f}".format,
                                      "avg_severity": "{:.1f}".format}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Label clean reviews with themes using an LLM.")
    parser.add_argument("--app", help="key of one app in config/apps.yaml (default: all apps)")
    parser.add_argument("--sample", type=int, help="label only this many random reviews (a test run)")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt for the first batch, call nothing")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")  # puts the API keys from .env into the environment
    config = load_config("classifier")
    provider = config["provider"]
    settings = config[provider]
    themes = load_config("themes")["themes"]
    theme_ids = [theme["id"] for theme in themes]
    schema = build_schema(theme_ids)

    try:
        client = None if args.dry_run else make_client(provider, settings)
    except FatalError as error:
        raise SystemExit(str(error))
    pacer = Pacer(settings["requests_per_minute"])
    window_start = None if args.sample else shared_window_start()
    if window_start is not None:
        print(f"Shared comparison window starts {window_start:%d %b %Y}: each app also gets a sample from it.")

    for app in select_apps(args.app):
        clean_path = PROCESSED_DIR / f"{app['key']}_clean.csv"
        if not clean_path.exists():
            print(f"Skipping {app['name']}: {clean_path.name} not found. Run clean.py first.")
            continue
        all_reviews = read_reviews(clean_path)
        reviews = pick_reviews(all_reviews, args.sample or config.get("reviews_per_app"))
        if window_start is not None:  # plus a sample from inside the shared comparison window
            window = pick_reviews(all_reviews[all_reviews["date"] >= window_start], config.get("reviews_per_app"))
            reviews = pd.concat([reviews, window[~window.index.isin(reviews.index)]]).sort_index()

        instructions = build_instructions(app["name"], app.get("description", "an app on Google Play"), themes)
        cache = LabelCache(CACHE_DIR / f"{app['key']}_labels.jsonl",
                           fingerprint(provider, settings["model"], instructions))
        records = reviews.to_dict("records")
        todo = [review for review in records if cache.get(review) is None]
        print(f"\n{app['name']}: {len(records)} of {len(all_reviews)} clean reviews picked, "
              f"{len(records) - len(todo)} already labeled, {len(todo)} to label with {settings['model']}")

        if args.dry_run:
            print("\n--- instructions (system prompt) ---\n" + instructions)
            print("\n--- first batch (user message) ---\n" + build_prompt((todo or records)[:config["batch_size"]]))
            continue

        quota_reached = False
        try:
            if todo:
                label_reviews(todo, client, instructions, schema, theme_ids, cache, settings["model"],
                              config["batch_size"], pacer, config["max_attempts"])
        except DailyQuotaReached:
            quota_reached = True
        except FatalError as error:
            raise SystemExit(f"{error}\nLabels finished so far are saved in the cache.")

        labeled = attach_labels(reviews, cache)
        suffix = "sample_classified" if args.sample else "classified"
        out_path = PROCESSED_DIR / f"{app['key']}_{suffix}.csv"
        write_csv(labeled, out_path)
        print_summary(labeled)
        print(f"  saved -> {out_path.relative_to(WORKSPACE)}")

        if quota_reached:
            raise SystemExit("\nThe daily free-tier quota is used up. Everything labeled so far is saved. "
                             "Run the same command after midnight Pacific time to continue.")


if __name__ == "__main__":
    main()
