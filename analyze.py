"""Step 5a: turn the labeled reviews into the numbers and findings the dashboard shows.

Every function takes the combined table from load_labeled() and returns a small table (or a list of
findings), so the same numbers can be printed (run this file), checked in a notebook, or drawn by
app.py. Nothing in here knows about colors or layout.

Usage:
    python analyze.py    # print the headline findings for every app
"""
import json
import math

import pandas as pd

from common import PROCESSED_DIR, RAW_DIR, load_config, read_reviews, select_apps

NEGATIVE_STARS = (1, 2)
NOT_COMPLAINTS = {"praise", "other", "not_a_review"}  # themes left out of "top complaint"
PERIOD_DAYS = 90       # "recent" = the last 90 days of data, compared with the 90 days before
MIN_PERIOD_REVIEWS = 40  # don't compare two periods if either has fewer reviews than this
Z_95 = 1.96            # z-score for roughly 95% confidence


# --- Loading --------------------------------------------------------------------------------------

def load_labeled() -> pd.DataFrame:
    """All labeled reviews from every app in one table, with app name and time/version helpers."""
    frames = []
    for app in select_apps():
        path = PROCESSED_DIR / f"{app['key']}_classified.csv"
        if path.exists():
            frames.append(read_reviews(path).assign(app=app["name"], role=app["role"]))
    if not frames:
        raise SystemExit("No labeled reviews found. Run classify.py first.")
    data = pd.concat(frames, ignore_index=True)
    data = data[data["theme"].notna()].copy()
    data["severity"] = data["severity"].astype(int)
    # 7.11.1 -> 7.11. Single patch versions have too few reviews to analyze on their own.
    data["minor_version"] = data["app_version"].str.extract(r"^(\d+\.\d+)", expand=False)
    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp()
    data["week"] = data["date"].dt.to_period("W-SUN").dt.start_time  # weeks start on Monday
    return data


def theme_info() -> pd.DataFrame:
    """One row per theme: its label, funnel stage and KPI, in the order of config/themes.yaml."""
    funnel = load_config("funnel_map")["themes"]
    rows = [{"theme": t["id"], "label": t["label"],
             "stage": funnel.get(t["id"], {}).get("stage", "none"),
             "kpi": funnel.get(t["id"], {}).get("kpi", "")}
            for t in load_config("themes")["themes"]]
    return pd.DataFrame(rows).set_index("theme")


def store_snapshot(app_key: str) -> dict:
    """Play Store listing numbers saved by scrape.py (rating, installs), or {} if not available."""
    path = RAW_DIR / f"{app_key}_app.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# --- Small statistics helper ------------------------------------------------------------------------

def z_score(p0: float, n0: int, p1: float, n1: int) -> float:
    """Two-proportion z-test: how surprising is the change from p0 (out of n0) to p1 (out of n1)?

    Around 1.96 or more (in either direction) means the change is unlikely to be chance
    (about 95% confidence). Returns 0 when there isn't enough data to tell.
    """
    if n0 == 0 or n1 == 0:
        return 0.0
    pooled = (p0 * n0 + p1 * n1) / (n0 + n1)
    if not 0 < pooled < 1:
        return 0.0
    return (p1 - p0) / math.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))


# --- Summaries --------------------------------------------------------------------------------------

def app_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Headline numbers per app."""
    rows = []
    for app, d in data.groupby("app", sort=False):
        complaints = d[~d["theme"].isin(NOT_COMPLAINTS)]
        top = complaints["theme"].value_counts()
        rows.append({
            "app": app,
            "reviews": len(d),
            "avg_stars": d["rating"].mean(),
            "negative_share": d["rating"].isin(NEGATIVE_STARS).mean(),
            "severe_share": (d["severity"] == 3).mean(),  # blocks the user or costs them money
            "top_complaint": top.index[0] if len(top) else None,
            "top_complaint_share": top.iloc[0] / len(d) if len(top) else 0.0,
            "first_review": d["date"].min(),
            "last_review": d["date"].max(),
        })
    return pd.DataFrame(rows).set_index("app")


def theme_table(data: pd.DataFrame) -> pd.DataFrame:
    """Per theme: share of reviews, number of reviews, average stars and severity."""
    table = data.groupby("theme").agg(reviews=("review_id", "size"), avg_stars=("rating", "mean"),
                                      avg_severity=("severity", "mean"))
    table["share"] = table["reviews"] / len(data)
    return table.sort_values("share", ascending=False)


def funnel_table(data: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Share of reviews per funnel stage, with its biggest theme and that theme's KPI."""
    themes = theme_table(data).join(info)
    themes = themes[themes["stage"] != "none"]
    rows = []
    for stage, group in themes.groupby("stage"):
        group = group.sort_values("share", ascending=False)
        rows.append({"stage": stage, "share": group["share"].sum(), "top_theme": group.index[0],
                     "top_share": group["share"].iloc[0], "kpi": group["kpi"].iloc[0],
                     "themes": list(group.index)})
    return pd.DataFrame(rows).set_index("stage").sort_values("share", ascending=False)


# --- Recent change ----------------------------------------------------------------------------------

def split_periods(d: pd.DataFrame, days: int = PERIOD_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The last `days` days of an app's reviews, and the `days` days before that."""
    end = d["date"].max()
    recent = d[d["date"] > end - pd.Timedelta(days=days)]
    prior = d[(d["date"] <= end - pd.Timedelta(days=days)) & (d["date"] > end - pd.Timedelta(days=2 * days))]
    return recent, prior


def metric_change(d: pd.DataFrame) -> dict:
    """Headline numbers for the recent period and the one before (None when there's too little data)."""
    recent, prior = split_periods(d)

    def numbers(part: pd.DataFrame) -> dict | None:
        if len(part) < MIN_PERIOD_REVIEWS:
            return None
        return {"reviews": len(part), "avg_stars": part["rating"].mean(),
                "negative_share": part["rating"].isin(NEGATIVE_STARS).mean(),
                "severe_share": (part["severity"] == 3).mean()}

    return {"recent": numbers(recent), "prior": numbers(prior)}


def theme_changes(d: pd.DataFrame) -> pd.DataFrame:
    """Each theme's share in the recent period vs the period before, with a significance test.

    Empty when either period has fewer than MIN_PERIOD_REVIEWS reviews.
    """
    recent, prior = split_periods(d)
    if len(recent) < MIN_PERIOD_REVIEWS or len(prior) < MIN_PERIOD_REVIEWS:
        return pd.DataFrame(columns=["before", "after", "change", "z"])
    before = prior["theme"].value_counts(normalize=True)
    after = recent["theme"].value_counts(normalize=True)
    table = pd.DataFrame({"before": before, "after": after}).fillna(0.0)
    table["change"] = table["after"] - table["before"]
    table["z"] = [z_score(b, len(prior), a, len(recent)) for b, a in zip(table["before"], table["after"])]
    return table.sort_values("change", ascending=False)


# --- Over time and by version -----------------------------------------------------------------------

def shares_by(data: pd.DataFrame, column: str, min_reviews: int) -> tuple[pd.DataFrame, pd.Series]:
    """Theme shares within each value of `column` (a month, week or version).

    Returns (shares, counts): shares has one row per period/version and one column per theme
    (each row adds up to 1). Periods/versions with fewer than min_reviews reviews are left out,
    because a share computed from a handful of reviews is mostly noise.
    """
    d = data[data[column].notna()]
    counts = d.groupby(column).size()
    keep = counts[counts >= min_reviews].index
    shares = pd.crosstab(d[column], d["theme"], normalize="index").loc[keep]
    return shares, counts.loc[keep]


def monthly_metrics(d: pd.DataFrame, min_reviews: int = 15) -> pd.DataFrame:
    """Per month: reviews, average stars, negative share and severe share (thin months dropped)."""
    table = d.groupby("month").agg(
        reviews=("review_id", "size"), avg_stars=("rating", "mean"),
        negative_share=("rating", lambda s: s.isin(NEGATIVE_STARS).mean()),
        severe_share=("severity", lambda s: (s == 3).mean()))
    return table[table["reviews"] >= min_reviews]


def version_order(data: pd.DataFrame) -> pd.Series:
    """First date each minor version shows up in reviews: our stand-in for its release date."""
    return data.dropna(subset=["minor_version"]).groupby("minor_version")["date"].min().sort_values()


def find_spikes(shares: pd.DataFrame, counts: pd.Series, min_jump: float = 0.10) -> pd.DataFrame:
    """Themes whose share jumped compared with the previous version.

    A jump is flagged when it is at least `min_jump` (0.10 = 10 percentage points) AND unlikely to
    be chance (z >= 1.96). `shares` must be in release order. Each version is compared with the
    previous version that has enough reviews.
    """
    rows = []
    versions = list(shares.index)
    for previous, current in zip(versions, versions[1:]):
        n0, n1 = counts[previous], counts[current]
        for theme in shares.columns:
            p0, p1 = shares.at[previous, theme], shares.at[current, theme]
            z = z_score(p0, n0, p1, n1)
            if p1 - p0 >= min_jump and z >= Z_95:
                rows.append({"version": current, "previous": previous, "theme": theme,
                             "before": p0, "after": p1, "jump": p1 - p0, "z": z,
                             "reviews_before": n0, "reviews_after": n1})
    return pd.DataFrame(rows, columns=["version", "previous", "theme", "before", "after", "jump", "z",
                                       "reviews_before", "reviews_after"])


def release_spikes(d: pd.DataFrame, min_reviews: int = 25, min_jump: float = 0.10) -> tuple:
    """Version heatmap data for one app: (shares in release order, counts, release dates, spikes)."""
    order = version_order(d)
    shares, counts = shares_by(d, "minor_version", min_reviews)
    versions = [v for v in order.index if v in shares.index]
    shares, counts = shares.loc[versions], counts.loc[versions]
    return shares, counts, order, find_spikes(shares, counts, min_jump)


# --- Quotes -----------------------------------------------------------------------------------------

def quotes(d: pd.DataFrame, theme: str | None = None, version: str | None = None, n: int = 3) -> pd.DataFrame:
    """Representative English reviews: the ones other users marked most helpful, then the newest.

    Only reviews of 8 to 80 words, so a quote is specific but still quick to read.
    """
    pool = d[d["is_english"].astype(bool) & d["n_words"].between(8, 80)]
    if theme:
        pool = pool[pool["theme"] == theme]
    if version:
        pool = pool[pool["minor_version"] == version]
    return pool.sort_values(["thumbs_up", "date"], ascending=False).head(n)


# --- Comparing apps -------------------------------------------------------------------------------

def comparison_window(data: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The weeks every app has reviews for. Busy apps' data only reaches back a few weeks."""
    return data.groupby("app")["date"].min().max(), data["date"].max()


def compare_apps(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, tuple]:
    """Theme shares per app over the shared window. Returns (shares, review counts, window)."""
    start, end = comparison_window(data)
    window = data[data["date"] >= start]
    shares = pd.crosstab(window["theme"], window["app"], normalize="columns")
    return shares, window["app"].value_counts(), (start, end)


def competitor_gaps(data: pd.DataFrame, app: str) -> pd.DataFrame:
    """Per theme, over the shared window: this app's share vs all other apps' reviews pooled.

    `significant` marks gaps unlikely to be chance (z beyond +-1.96).
    """
    start, _ = comparison_window(data)
    window = data[data["date"] >= start]
    mine, others = window[window["app"] == app], window[window["app"] != app]
    if mine.empty or others.empty:
        return pd.DataFrame(columns=["share", "others", "gap", "z", "significant"])
    table = pd.DataFrame({"share": mine["theme"].value_counts(normalize=True),
                          "others": others["theme"].value_counts(normalize=True)}).fillna(0.0)
    table["gap"] = table["share"] - table["others"]
    table["z"] = [z_score(o, len(others), s, len(mine)) for o, s in zip(table["others"], table["share"])]
    table["significant"] = table["z"].abs() >= Z_95
    return table.sort_values("gap", ascending=False)


# --- Findings in plain words ----------------------------------------------------------------------

def key_takeaways(data: pd.DataFrame, app: str, info: pd.DataFrame) -> list[dict]:
    """A short list of findings for one app, each {"kind", "text"}.

    kind is one of: "top" (biggest complaint), "emerging" / "declining" (significant change in the
    last 90 days), "release" (a flagged jump after a release), "gap" (vs competitors), "note".
    Only changes that pass the significance test are reported as emerging, declining or gaps.
    """
    labels = info["label"]
    d = data[data["app"] == app]
    found = []

    summary = app_summary(d).iloc[0]
    if summary.top_complaint:
        stars = d.loc[d["theme"] == summary.top_complaint, "rating"].mean()
        found.append({"kind": "top", "text": f"<b>{labels[summary.top_complaint]}</b> is the most common "
                      f"complaint: {summary.top_complaint_share:.0%} of reviews, averaging {stars:.1f} stars."})

    changes = theme_changes(d)
    complaints = changes[~changes.index.isin(NOT_COMPLAINTS)]
    rising = complaints[(complaints["change"] >= 0.03) & (complaints["z"] >= Z_95)]
    falling = complaints[(complaints["change"] <= -0.03) & (complaints["z"] <= -Z_95)]
    for theme, row in rising.head(2).iterrows():
        found.append({"kind": "emerging", "text": f"<b>{labels[theme]}</b> rose from {row.before:.0%} to "
                      f"{row.after:.0%} of reviews in the last {PERIOD_DAYS} days."})
    for theme, row in falling.sort_values("change").head(1).iterrows():
        found.append({"kind": "declining", "text": f"<b>{labels[theme]}</b> fell from {row.before:.0%} to "
                      f"{row.after:.0%} of reviews in the last {PERIOD_DAYS} days."})
    if changes.empty:
        found.append({"kind": "note", "text": f"Not enough history to compare the last {PERIOD_DAYS} days "
                      "with the period before."})
    elif rising.empty and falling.empty:
        found.append({"kind": "note", "text": f"No complaint theme changed significantly in the last "
                      f"{PERIOD_DAYS} days compared with the {PERIOD_DAYS} days before."})

    _, _, order, spikes = release_spikes(d)
    complaint_spikes = spikes[~spikes["theme"].isin(NOT_COMPLAINTS)]
    if len(complaint_spikes):
        biggest = complaint_spikes.sort_values("jump", ascending=False).iloc[0]
        found.append({"kind": "release", "text": f"Version <b>{biggest.version}</b> ({order[biggest.version]:%b %Y}): "
                      f"{labels[biggest.theme]} jumped from {biggest.before:.0%} to {biggest.after:.0%} of reviews."})

    gaps = competitor_gaps(data, app)
    worse = gaps[gaps["significant"] & (gaps["gap"] >= 0.04) & ~gaps.index.isin(NOT_COMPLAINTS)]
    if len(worse):
        theme, row = next(worse.iterrows())
        found.append({"kind": "gap", "text": f"<b>{labels[theme]}</b>: {row.share:.0%} of {app}'s reviews vs "
                      f"{row.others:.0%} for the other apps over the same weeks."})
    if "not_a_review" in gaps.index:
        row = gaps.loc["not_a_review"]
        if row.significant and row.gap >= 0.05:
            found.append({"kind": "gap", "text": f"<b>{row.share:.0%}</b> of {app}'s reviews come from people who "
                          f"haven't used the app yet, vs {row.others:.0%} for the other apps. The rating prompt may "
                          "appear too early."})
    return found


# --- Printed report ----------------------------------------------------------------------------------

def main() -> None:
    data, info = load_labeled(), theme_info()
    labels = info["label"]

    print("=== Headline numbers per app (all labeled reviews) ===")
    for app, row in app_summary(data).iterrows():
        print(f"{app}: {row.reviews} reviews ({row.first_review:%d %b %Y} to {row.last_review:%d %b %Y}), "
              f"{row.avg_stars:.2f} stars, {row.negative_share:.0%} negative, {row.severe_share:.0%} severe, "
              f"top complaint: {labels[row.top_complaint]} ({row.top_complaint_share:.0%})")

    shares, counts, (start, end) = compare_apps(data)
    print(f"\n=== Same weeks for every app: {start:%d %b} to {end:%d %b %Y} ===")
    print("reviews: " + ", ".join(f"{app} {n}" for app, n in counts.items()))
    table = shares.mul(100).round(1).rename(index=labels).sort_values(shares.columns[0], ascending=False)
    print(table.to_string())

    for app in data["app"].unique():
        print(f"\n=== {app}: key takeaways ===")
        for item in key_takeaways(data, app, info):
            print(f"  [{item['kind']}] " + item["text"].replace("<b>", "").replace("</b>", ""))


if __name__ == "__main__":
    main()
