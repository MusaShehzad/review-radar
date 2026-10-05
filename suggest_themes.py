"""Draft a theme list and funnel map for any app, by letting the LLM read a sample of its reviews.

This automates the step we did by hand for ImagineArt (read ~100 reviews, then write the themes).
The result is a starting point, not a final answer: read it, edit it, then run classify.py.

Usage:
    python suggest_themes.py            # for the target app in apps.yaml (run.py calls this for you)
    python suggest_themes.py --force    # overwrite existing themes.yaml / funnel_map.yaml

Writes, in the workspace's config folder:
    themes.yaml       the themes the classifier will use
    funnel_map.yaml   funnel stage and KPI per theme
If those files already exist (and --force isn't given), it writes *.suggested.yaml next to them instead.
"""
import argparse
import json
import re

import pandas as pd
import yaml
from dotenv import load_dotenv

from classify import Pacer, call_with_retries
from common import CONFIG_DIR, PROCESSED_DIR, ROOT, load_config, read_reviews, select_apps
from llm import FatalError, make_client

PER_RATING = 30       # reviews shown to the model per star rating (5 x 30 = up to 150)
STAGES = ["activation", "trial_to_paid", "retention", "refunds"]

# The dashboard relies on these three themes, so every theme list gets them.
FIXED_THEMES = [
    {"id": "praise", "label": "Praise / positive", "stage": "none", "kpi": "Share of 4-5 star reviews",
     "description": "Positive feedback with no specific complaint."},
    {"id": "not_a_review", "label": "Not a review yet", "stage": "activation",
     "kpi": "Share of rating prompts shown before the user's first success",
     "description": "The user hasn't really used the app yet (\"I'll try it\", \"don't know yet\"), or typed "
                    "something into the review box that isn't a review."},
    {"id": "other", "label": "Other", "stage": "none", "kpi": "-",
     "description": "Fits none of the themes above, or gives no clear reason."},
]


def sample_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Up to PER_RATING English reviews of 8+ words per star rating, most helpful first.

    Equal numbers per rating, so rarer 2-4 star reviews (often the most specific) are well covered.
    """
    pool = reviews[reviews["is_english"].astype(bool) & (reviews["n_words"] >= 8)]
    picks = [g.sort_values(["thumbs_up", "date"], ascending=False).head(PER_RATING) for _, g in pool.groupby("rating")]
    return pd.concat(picks) if picks else pool.head(0)


def build_prompt(app: dict, sample: pd.DataFrame) -> tuple[str, str]:
    instructions = f"""You help a product team set up review analysis for {app['name']}, {app.get('description', 'an app on Google Play')}.

Read the Google Play reviews below and propose 8 to 12 themes that together cover the problems and
requests users write about. A good theme is specific enough to act on ("Login fails after update"
is too narrow, "Problems" is too broad) and different from the other themes.

For each theme give:
- id: short snake_case key, e.g. "billing" or "sync_errors"
- label: a short name for a dashboard, e.g. "Billing & refunds"
- description: one or two sentences telling a classifier exactly what belongs in the theme
- stage: the part of the funnel it hurts most: activation (a new user getting to a first good
  result), trial_to_paid (a free user deciding to pay), retention (users coming back), or refunds
  (paying users asking for their money back)
- kpi: one product metric that would show whether fixing this theme worked

Do not include praise, "not a review yet" or "other": those are added automatically."""
    lines = [json.dumps({"stars": int(r.rating), "review": r.text}, ensure_ascii=False) for r in sample.itertuples()]
    return instructions, "Reviews (one JSON object per line):\n" + "\n".join(lines)


SCHEMA = {
    "type": "object",
    "properties": {"themes": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"}, "label": {"type": "string"}, "description": {"type": "string"},
            "stage": {"type": "string", "enum": STAGES}, "kpi": {"type": "string"},
        },
        "required": ["id", "label", "description", "stage", "kpi"],
        "additionalProperties": False,
    }}},
    "required": ["themes"],
    "additionalProperties": False,
}


def clean_themes(raw: str) -> list[dict]:
    """Check the model's answer and tidy it: snake_case unique ids, valid stages, no reserved ids."""
    themes, seen = [], {t["id"] for t in FIXED_THEMES}
    for item in json.loads(raw).get("themes", []):
        theme_id = re.sub(r"[^a-z0-9]+", "_", str(item.get("id", "")).lower()).strip("_")
        if not theme_id or theme_id in seen or item.get("stage") not in STAGES:
            continue
        seen.add(theme_id)
        themes.append({"id": theme_id, "label": str(item["label"]).strip(), "stage": item["stage"],
                       "kpi": str(item["kpi"]).strip(), "description": str(item["description"]).strip()})
    if len(themes) < 4:
        raise SystemExit("The model suggested too few usable themes. Run again, or write themes.yaml by hand.")
    return themes + FIXED_THEMES


def write_yaml(path, header: str, content: dict) -> None:
    body = yaml.safe_dump(content, sort_keys=False, allow_unicode=True, width=110)
    path.write_text(header + "\n" + body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft themes.yaml and funnel_map.yaml from an app's reviews.")
    parser.add_argument("--force", action="store_true", help="overwrite existing theme and funnel files")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    apps = select_apps()
    app = next((a for a in apps if a.get("role") == "target"), apps[0])
    clean_path = PROCESSED_DIR / f"{app['key']}_clean.csv"
    if not clean_path.exists():
        raise SystemExit(f"{clean_path.name} not found. Run scrape.py and clean.py first.")
    sample = sample_reviews(read_reviews(clean_path))
    if len(sample) < 20:
        raise SystemExit(f"Only {len(sample)} usable reviews for {app['name']}: too few to suggest themes.")

    config = load_config("classifier")
    settings = config[config["provider"]]
    instructions, prompt = build_prompt(app, sample)
    print(f"Asking {settings['model']} to suggest themes from {len(sample)} {app['name']} reviews...")
    try:
        client = make_client(config["provider"], settings)
        raw = call_with_retries(client, instructions, prompt, SCHEMA, Pacer(settings["requests_per_minute"]),
                                config["max_attempts"])
    except FatalError as error:
        raise SystemExit(str(error))
    themes = clean_themes(raw)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    exists = (CONFIG_DIR / "themes.yaml").exists() or (CONFIG_DIR / "funnel_map.yaml").exists()
    suffix = ".suggested.yaml" if exists and not args.force else ".yaml"
    note = (f"# DRAFT suggested by suggest_themes.py from {len(sample)} {app['name']} reviews.\n"
            "# Read it against real reviews and edit freely before running classify.py.\n")
    write_yaml(CONFIG_DIR / f"themes{suffix}", note + "# id: stable key  label: dashboard name  "
               "description: tells the classifier what belongs in the theme\n",
               {"themes": [{k: t[k] for k in ("id", "label", "description")} for t in themes]})
    write_yaml(CONFIG_DIR / f"funnel_map{suffix}", note + "# stage: activation | trial_to_paid | retention | "
               "refunds | none\n", {"themes": {t["id"]: {"stage": t["stage"], "kpi": t["kpi"]} for t in themes}})

    print(f"\n{len(themes)} themes written to {CONFIG_DIR / ('themes' + suffix)}:")
    for t in themes:
        print(f"  {t['id']:<24} {t['stage']:<14} {t['label']}")
    if suffix != ".yaml":
        print("\nExisting theme files were kept. Compare them with the .suggested.yaml files and merge by hand.")


if __name__ == "__main__":
    main()
