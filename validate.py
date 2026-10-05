"""Step 6: check the LLM's labels against your own.

1. Export a random sample of labeled reviews WITHOUT the model's labels, so you label them blind:
       python validate.py export                 # 100 reviews of the target app
2. Open data/processed/validation_sample.csv and fill in my_label for each review with one theme id
   (listed when you export). Leave a row empty to skip it.
3. Score the model against your labels:
       python validate.py score

The score compares your label with the model's main theme. It reports:
  - agreement: share of reviews where you and the model picked the same theme
  - lenient agreement: your label matches the model's main OR second theme
  - Cohen's kappa: agreement after removing the part you'd expect by chance (1 = perfect, 0 = chance)
  - per-theme numbers and the disagreements, so you can see which theme descriptions need work.
Results are also saved to data/processed/validation_results.json, which the dashboard's Method page shows.
"""
import argparse
import json

import pandas as pd

from common import PROCESSED_DIR, load_config, read_reviews, select_apps, write_csv

SAMPLE_PATH = PROCESSED_DIR / "validation_sample.csv"
RESULTS_PATH = PROCESSED_DIR / "validation_results.json"
SEED = 7


def target_app() -> dict:
    apps = select_apps()
    return next((a for a in apps if a.get("role") == "target"), apps[0])


def export(n: int) -> None:
    app = target_app()
    labeled = read_reviews(PROCESSED_DIR / f"{app['key']}_classified.csv").dropna(subset=["theme"])
    if SAMPLE_PATH.exists():
        raise SystemExit(f"{SAMPLE_PATH.name} already exists. Delete it first if you want a new sample "
                         "(your labels in it would be lost).")
    sample = labeled.sample(min(n, len(labeled)), random_state=SEED)
    write_csv(sample[["review_id", "rating", "text"]].assign(my_label=""), SAMPLE_PATH)
    print(f"Wrote {len(sample)} {app['name']} reviews to {SAMPLE_PATH}")
    print("Fill in my_label with one of these theme ids:")
    for theme in load_config("themes")["themes"]:
        print(f"  {theme['id']:<22} {theme['label']}")


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Agreement corrected for chance: (observed - expected) / (1 - expected)."""
    observed = (a == b).mean()
    expected = sum(a.value_counts(normalize=True).get(t, 0) * b.value_counts(normalize=True).get(t, 0)
                   for t in set(a) | set(b))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def score() -> None:
    if not SAMPLE_PATH.exists():
        raise SystemExit("No validation sample yet. Run: python validate.py export")
    themes = load_config("themes")["themes"]
    ids = {t["id"] for t in themes}
    by_label = {t["label"].lower(): t["id"] for t in themes}  # also accept the readable label

    mine = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig", dtype={"review_id": "string", "my_label": "string"})
    mine["my_label"] = mine["my_label"].fillna("").str.strip().str.lower()
    mine["my_label"] = mine["my_label"].map(lambda v: by_label.get(v, v))
    mine = mine[mine["my_label"] != ""]
    unknown = mine[~mine["my_label"].isin(ids)]
    if len(unknown):
        print(f"Skipping {len(unknown)} rows with an unknown theme id: {sorted(unknown['my_label'].unique())}")
        mine = mine[mine["my_label"].isin(ids)]
    if mine.empty:
        raise SystemExit(f"No labels found. Fill in the my_label column in {SAMPLE_PATH.name} first.")

    app = target_app()
    model = read_reviews(PROCESSED_DIR / f"{app['key']}_classified.csv")[["review_id", "theme", "secondary_theme"]]
    both = mine.merge(model, on="review_id", how="inner")
    exact = (both["my_label"] == both["theme"]).mean()
    lenient = ((both["my_label"] == both["theme"]) | (both["my_label"] == both["secondary_theme"])).mean()
    kappa = cohens_kappa(both["my_label"], both["theme"])

    print(f"\n{len(both)} reviews compared ({app['name']})")
    print(f"  agreement (main theme):          {exact:.0%}")
    print(f"  lenient (main or second theme):  {lenient:.0%}")
    print(f"  Cohen's kappa:                   {kappa:.2f}")

    rows = []
    for theme in sorted(set(both["my_label"]) | set(both["theme"])):
        yours, models = both["my_label"] == theme, both["theme"] == theme
        hit = (yours & models).sum()
        rows.append({"theme": theme, "you": int(yours.sum()), "model": int(models.sum()),
                     "precision": hit / models.sum() if models.sum() else float("nan"),
                     "recall": hit / yours.sum() if yours.sum() else float("nan")})
    table = pd.DataFrame(rows).set_index("theme").sort_values("you", ascending=False)
    print("\nPer theme (precision: when the model says X, how often you agree; recall: of your X, how many it found)")
    print(table.to_string(formatters={"precision": "{:.0%}".format, "recall": "{:.0%}".format}, na_rep="-"))

    misses = both[both["my_label"] != both["theme"]]
    write_csv(misses[["review_id", "rating", "my_label", "theme", "secondary_theme", "text"]],
              PROCESSED_DIR / "validation_disagreements.csv")
    print(f"\n{len(misses)} disagreements saved to validation_disagreements.csv")

    RESULTS_PATH.write_text(json.dumps({"app": app["name"], "reviews": len(both), "agreement": exact,
                                        "lenient_agreement": lenient, "kappa": kappa}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the LLM's labels with your own.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="write a blind sample to label").add_argument("-n", type=int, default=100)
    sub.add_parser("score", help="score the model against your labels")
    args = parser.parse_args()
    export(args.n) if args.command == "export" else score()


if __name__ == "__main__":
    main()
