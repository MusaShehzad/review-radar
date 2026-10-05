"""Step 5b: the Review Radar dashboard.

Run it with:   streamlit run app.py

analyze.py computes every number and ui.py decides how things look. This file wires them together:
the sidebar, the six pages, and which numbers each page shows.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analyze
import ui
from common import CONFIG_DIR, PROCESSED_DIR, load_config, select_apps

st.set_page_config(page_title="Review Radar", page_icon=":material/radar:", layout="wide")
P = ui.palette()
ui.inject_css(P)

STAGE_ORDER = ["activation", "trial_to_paid", "retention", "refunds", "none"]
STAGE_NAMES = {"activation": "Activation", "trial_to_paid": "Trial to paid", "retention": "Retention",
               "refunds": "Refunds", "none": "Not a problem"}
MARKDOWN_COLORS = ["blue", "orange", "green"]  # set to the app colors in .streamlit/config.toml


# --- Data -----------------------------------------------------------------------------------------

def files_stamp() -> tuple:
    """Names and last-modified times of the data and config files. When the pipeline rewrites a
    file, this changes, so the cached data below is reloaded instead of going stale."""
    files = sorted(PROCESSED_DIR.glob("*_c*.csv")) + sorted(CONFIG_DIR.glob("*.yaml"))
    return tuple((f.name, f.stat().st_mtime) for f in files)


@st.cache_data
def load(stamp: tuple) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    clean_counts = {app["key"]: len(pd.read_csv(path, usecols=["review_id"]))
                    for app in select_apps() if (path := PROCESSED_DIR / f"{app['key']}_clean.csv").exists()}
    return analyze.load_labeled(), analyze.theme_info(), clean_counts


DATA, INFO, CLEAN_COUNTS = load(files_stamp())
LABELS = INFO["label"]
ALL_APPS = select_apps()
APPS = [a for a in ALL_APPS if a["name"] in set(DATA["app"])]
APP_NAMES = [a["name"] for a in APPS]
APP_KEY = {a["name"]: a["key"] for a in ALL_APPS}
APP_COLOR = {a["name"]: P.apps[i % len(P.apps)] for i, a in enumerate(ALL_APPS)}  # fixed per app
APP_MD_COLOR = {a["name"]: MARKDOWN_COLORS[i % 3] for i, a in enumerate(ALL_APPS)}
TARGET = next((a["name"] for a in APPS if a.get("role") == "target"), APP_NAMES[0])


def current_app() -> str:
    return st.session_state.get("app", TARGET)


def pct(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%"


def ordered_themes(d: pd.DataFrame, complaints_only: bool) -> list[str]:
    """Themes in funnel order (activation first), biggest first within a stage."""
    share = d["theme"].value_counts(normalize=True)
    themes = INFO.assign(share=share).fillna({"share": 0})
    if complaints_only:
        themes = themes.drop(index=[t for t in analyze.NOT_COMPLAINTS if t in themes.index])
    themes["rank"] = themes["stage"].map(STAGE_ORDER.index)
    return list(themes.sort_values(["rank", "share"], ascending=[True, False]).index)


def change_class(theme: str, change: float, z: float) -> str:
    """CSS class for a change in a theme's share: red if things got worse, green if better, gray if
    the change could be chance. More praise is good news; more of any other theme is bad news."""
    if abs(z) < analyze.Z_95 or change == 0:
        return "rr-flat"
    worse = change < 0 if theme == "praise" else change > 0
    return "rr-bad" if worse else "rr-good"


def change_tone(diff: float, higher_is_better: bool, threshold: float) -> str:
    if abs(diff) < threshold:
        return "flat"
    return "good" if (diff > 0) == higher_is_better else "bad"


# --- Overview -------------------------------------------------------------------------------------

def overview() -> None:
    app = current_app()
    d = DATA[DATA["app"] == app]
    summary = analyze.app_summary(d).iloc[0]
    change = analyze.metric_change(d)
    recent, prior = change["recent"], change["prior"]
    months = analyze.monthly_metrics(d).tail(12)
    store = analyze.store_snapshot(APP_KEY[app])
    now = recent or {"reviews": len(d), "avg_stars": summary.avg_stars,
                     "negative_share": summary.negative_share, "severe_share": summary.severe_share}
    window = f"last {analyze.PERIOD_DAYS} days" if recent else "all labeled reviews"

    ui.page_header(f"Overview · {app}", f"What {app} reviewers are telling you",
                   f"{len(d):,} labeled Google Play reviews, {summary.first_review:%b %Y} to "
                   f"{summary.last_review:%d %b %Y}. Headline numbers cover the {window}; changes compare "
                   f"them with the {analyze.PERIOD_DAYS} days before.")

    def delta(key: str, higher_is_better: bool, stars: bool = False) -> str:
        if not (recent and prior):
            return ""
        diff = recent[key] - prior[key]
        text = f"{diff:+.2f}" if stars else f"{diff * 100:+.0f} pts"
        return ui.delta_chip(text, change_tone(diff, higher_is_better, 0.05 if stars else 0.01))

    compare_note = f"vs prior {analyze.PERIOD_DAYS} days" if (recent and prior) else "too little history to compare"
    store_note = f"store listing shows {store['store_rating']:.1f}" if store.get("store_rating") else ""
    ui.kpi_grid([
        ui.kpi("Average rating", f"{now['avg_stars']:.2f}", delta("avg_stars", True, stars=True),
               ui.sparkline(months["avg_stars"], P.accent), compare_note if recent and prior else store_note),
        ui.kpi("Negative reviews", pct(now["negative_share"]), delta("negative_share", False),
               ui.sparkline(months["negative_share"], P.accent), "1 or 2 stars"),
        ui.kpi("Severe issues", pct(now["severe_share"]), delta("severe_share", False),
               ui.sparkline(months["severe_share"], P.accent), "blocks the user or costs money"),
        ui.kpi("Reviews analyzed", f"{len(d):,}", "", "",
               f"of {CLEAN_COUNTS.get(APP_KEY[app], len(d)):,} clean reviews · {now['reviews']:,} in {window}"),
    ])

    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        with st.container(key="card-takeaways"):
            ui.html('<div class="rr-card-title">Key takeaways</div>' +
                    ui.takeaways(analyze.key_takeaways(DATA, app, INFO)))
    with right:
        with st.container(key="card-funnel"):
            funnel = analyze.funnel_table(d, INFO)
            top = max(funnel["share"].max(), 0.01)
            rows = "".join(
                f'<div><div class="rr-stage-head"><span>{STAGE_NAMES[stage]}</span><span>{pct(r.share)}</span></div>'
                f'<div class="rr-track"><div class="rr-fill" style="width:{r.share / top * 100:.1f}%"></div></div>'
                f'<div class="rr-stage-foot">Mostly <b>{LABELS[r.top_theme]}</b> ({pct(r.top_share)}). '
                f'Watch: {r.kpi}</div></div>'
                for stage, r in funnel.iterrows())
            ui.html('<div class="rr-card-title">Where it hurts the funnel</div>'
                    f'<div class="rr-funnel">{rows}</div>')

    ui.section("Themes", f"Share of reviews · trend: monthly share, last 12 months · change: last "
                         f"{analyze.PERIOD_DAYS} days vs the {analyze.PERIOD_DAYS} before")
    monthly, _ = analyze.shares_by(d, "month", 15)
    monthly = monthly.tail(12)
    with st.container(key="card-themes"):
        show_all = st.toggle("Include praise and other non-complaints", key="themes_all")
        table = analyze.theme_table(d).join(INFO)
        if not show_all:
            table = table[~table.index.isin(analyze.NOT_COMPLAINTS)]
        changes = analyze.theme_changes(d)
        widest = table["share"].max()
        rows = []
        for theme, r in table.iterrows():
            trend = monthly[theme] if theme in monthly else pd.Series(dtype=float)
            if theme in changes.index:
                c = changes.loc[theme]
                tone = change_class(theme, c.change, c.z)
                arrow = "▲" if c.change > 0 else ("▼" if c.change < 0 else "")
                change_html = f'<span class="{tone}">{arrow} {abs(c.change) * 100:.0f} pts</span>'
            else:
                change_html = '<span class="rr-flat">–</span>'
            rows.append(
                f'<tr><td><div class="rr-theme-name">{LABELS[theme]}</div><div class="rr-stage">{STAGE_NAMES[r.stage]}</div></td>'
                f'<td style="width:28%"><div class="rr-sharebar"><div style="width:{r.share / widest * 100:.1f}%"></div>'
                f'<span>{pct(r.share)}</span></div></td>'
                f'<td class="num">{int(r.reviews):,}</td>'
                f'<td class="rr-hide-sm">{ui.sparkline(trend.values, P.accent, width=96, height=26)}</td>'
                f'<td class="num">{change_html}</td>'
                f'<td class="num rr-hide-sm">{r.avg_stars:.1f} ★</td></tr>')
        ui.html('<div class="rr-tablewrap"><table class="rr-table"><thead><tr><th>Theme</th><th>Share</th>'
                '<th class="num">Reviews</th><th class="rr-hide-sm">Trend</th><th class="num">Change</th>'
                '<th class="num rr-hide-sm">Avg rating</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
                '<div class="rr-muted" style="margin-top:10px">Changes are colored only when they are unlikely '
                'to be chance (95% confidence): red means worse, green means better.</div>')

    ui.section("Theme spotlight", "The reviews other users found most helpful")
    complaints = [t for t in analyze.theme_table(d).index if t not in analyze.NOT_COMPLAINTS]
    if complaints:
        theme = st.selectbox("Theme", complaints, format_func=lambda t: LABELS[t], key=f"spotlight_{app}")
        with st.container(key="card-spotlight"):
            t = d[d["theme"] == theme]
            left, right = st.columns([1, 1.2], gap="large")
            with left:
                ui.html(f'<div class="rr-eyebrow">{STAGE_NAMES[INFO.at[theme, "stage"]]}</div>'
                        f'<div class="rr-tile-title" style="font-size:18px">{LABELS[theme]}</div>'
                        f'<div class="rr-muted">{pct(len(t) / len(d))} of reviews · {len(t):,} reviews · '
                        f'{t["rating"].mean():.1f} ★ average · {(t["severity"] == 3).mean():.0%} severe</div>'
                        f'<div class="rr-muted" style="margin-top:8px"><b>KPI to watch:</b> {INFO.at[theme, "kpi"]}</div>')
                share = monthly[theme] if theme in monthly else pd.Series(dtype=float)
                if len(share) >= 2:
                    fig = go.Figure(go.Scatter(x=share.index, y=share.values * 100, mode="lines",
                                               line=dict(color=P.accent, width=2), fill="tozeroy",
                                               fillcolor=ui.mix((P.surface, P.accent), 0.12),
                                               hovertemplate="%{x|%b %Y}: %{y:.0f}% of reviews<extra></extra>"))
                    fig.update_yaxes(ticksuffix="%", rangemode="tozero", nticks=4)
                    fig.update_xaxes(showgrid=False, tickformat="%b %y")
                    ui.show(ui.style_fig(fig, P, 190))
            with right:
                quotes = analyze.quotes(d, theme=theme, n=3)
                ui.html('<div class="rr-quotes">' + "".join(ui.quote_card(q, LABELS) for _, q in quotes.iterrows())
                        + "</div>" if len(quotes) else '<div class="rr-muted">No quotable reviews for this theme.</div>')


# --- Releases -------------------------------------------------------------------------------------

def releases() -> None:
    app = current_app()
    d = DATA[DATA["app"] == app]
    ui.page_header(f"Releases · {app}", "What changed with each release",
                   "Each column is a minor version (7.11 covers 7.11.0 and 7.11.1), dated by when it first "
                   "appears in reviews. A ▲ marks a theme that jumped compared with the previous version.")
    c1, c2, c3 = st.columns([1, 1, 1.1], vertical_alignment="bottom")
    min_reviews = c1.slider("Minimum reviews per version", 10, 80, 25, step=5, key="rel_min")
    min_jump = c2.slider("Flag jumps of at least (points)", 5, 30, 10, key="rel_jump") / 100
    complaints_only = c3.toggle("Complaints only", value=True, key="rel_complaints")

    shares, counts, order, spikes = analyze.release_spikes(d, min_reviews, min_jump)
    if complaints_only:
        spikes = spikes[~spikes["theme"].isin(analyze.NOT_COMPLAINTS)]
    if len(shares) < 2:
        st.info("Not enough reviews with a known app version. Lower the minimum reviews per version.")
        return
    with st.container(key="card-release-heatmap"):
        cols = [f"{v}<br>{order[v]:%b %y}" for v in shares.index]
        marked = set(zip(spikes["version"], spikes["theme"]))
        ui.show(ui.share_heatmap(shares, counts, ordered_themes(d, complaints_only), LABELS, cols, "Version", P, marked))
        ui.html('<div class="rr-muted">Each column adds up to 100% across all themes, including the hidden ones. '
                f'Versions with fewer than {min_reviews} labeled reviews are left out.</div>')

    ui.section("Flagged jumps", f"At least {min_jump * 100:.0f} points above the previous version, 95% confidence")
    if spikes.empty:
        ui.html('<div class="rr-muted">No jumps at these settings.</div>')
    else:
        cards = []
        for _, s in spikes.sort_values("jump", ascending=False).iterrows():
            tone = "good" if s.theme == "praise" else "bad"
            quotes = analyze.quotes(d, theme=s.theme, version=s.version, n=2)
            cards.append(
                f'<div class="rr-tile"><div class="rr-eyebrow">v{s.version} · first seen {order[s.version]:%b %Y} · '
                f'{int(s.reviews_after)} reviews</div><div class="rr-tile-title">{LABELS[s.theme]}</div>'
                f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:10px">'
                f'<span class="rr-big">{pct(s.before)} → {pct(s.after)}</span>'
                f'{ui.delta_chip(f"+{s.jump * 100:.0f} pts vs v{s.previous}", tone)}</div>'
                f'<div class="rr-quotes">{"".join(ui.quote_card(q, LABELS) for _, q in quotes.iterrows())}</div></div>')
        ui.html('<div class="rr-grid">' + "".join(cards) + "</div>")

    ui.section("Average rating by version")
    with st.container(key="card-release-stars"):
        stars = d[d["minor_version"].isin(shares.index)].groupby("minor_version")["rating"].mean().loc[shares.index]
        fig = go.Figure(go.Scatter(
            x=[f"{v} · {order[v]:%b %y}" for v in shares.index], y=stars.values, mode="lines+markers",
            line=dict(color=P.accent, width=2), marker=dict(size=8, color=P.accent, line=dict(color=P.surface, width=2)),
            customdata=counts.values, hovertemplate="%{x}: %{y:.2f} ★ (%{customdata} reviews)<extra></extra>"))
        fig.update_yaxes(range=[1, 5], dtick=1, ticksuffix=" ★")
        fig.update_xaxes(showgrid=False)
        ui.show(ui.style_fig(fig, P, 240))


# --- Trends ---------------------------------------------------------------------------------------

def trends() -> None:
    app = current_app()
    d = DATA[DATA["app"] == app]
    ui.page_header(f"Trends · {app}", "How themes move over time",
                   "Each column is one month (or week) and adds up to 100% across all themes. Thin periods are "
                   "hidden, because a share from a handful of reviews is mostly noise.")
    c1, c2, c3 = st.columns([1, 1, 1.1], vertical_alignment="bottom")
    grain = c1.segmented_control("Period", ["Month", "Week"], default="Month", key="trend_grain") or "Month"
    min_period = c2.slider("Minimum reviews per period", 10, 100, 20, step=5, key="trend_min")
    complaints_only = c3.toggle("Complaints only", value=True, key="trend_complaints")

    column = "month" if grain == "Month" else "week"
    shares, counts = analyze.shares_by(d, column, min_period)
    if len(shares) < 2:
        st.info("Not enough reviews per period. Lower the minimum or switch to months.")
        return
    fmt = "%b %y" if grain == "Month" else "%d %b"
    themes = ordered_themes(d, complaints_only)
    with st.container(key="card-trend-heatmap"):
        ui.show(ui.share_heatmap(shares, counts, themes, LABELS, [c.strftime(fmt) for c in shares.index], grain, P))

    changes = analyze.theme_changes(d)
    if not changes.empty:
        ui.section(f"Last {analyze.PERIOD_DAYS} days vs the {analyze.PERIOD_DAYS} before",
                   "● = unlikely to be chance (95% confidence)")
        rising = changes[changes["change"] > 0].head(5)
        falling = changes[changes["change"] < 0].sort_values("change").head(5)

        def change_list(rows: pd.DataFrame) -> str:
            items = "".join(f'<li><span>{LABELS[t]}{" ●" if abs(r.z) >= analyze.Z_95 else ""}</span>'
                            f'<span class="{change_class(t, r.change, r.z)}">{pct(r.before)} → {pct(r.after)}</span></li>'
                            for t, r in rows.iterrows())
            return f'<ul class="rr-list">{items}</ul>' if items else '<div class="rr-muted">Nothing.</div>'

        a, b = st.columns(2, gap="medium")
        with a, st.container(key="card-rising"):
            ui.html('<div class="rr-card-title">Growing</div>' + change_list(rising))
        with b, st.container(key="card-falling"):
            ui.html('<div class="rr-card-title">Shrinking</div>' + change_list(falling))

    ui.section("Follow one theme")
    with st.container(key="card-follow"):
        pick = st.selectbox("Theme", themes, format_func=lambda t: LABELS[t], key="trend_follow",
                            label_visibility="collapsed")
        series = shares[pick] if pick in shares else pd.Series(0.0, index=shares.index)
        fig = go.Figure(go.Scatter(x=series.index, y=series.values * 100, mode="lines+markers",
                                   line=dict(color=P.accent, width=2),
                                   marker=dict(size=7, color=P.accent, line=dict(color=P.surface, width=2)),
                                   customdata=counts.values,
                                   hovertemplate="%{x|" + fmt + "}: %{y:.0f}% of %{customdata} reviews<extra></extra>"))
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")
        fig.update_xaxes(showgrid=False)
        ui.show(ui.style_fig(fig, P, 240))


# --- Competitors ----------------------------------------------------------------------------------

def competitors() -> None:
    if len(APP_NAMES) < 2:
        ui.page_header("Competitors", f"No competitors for {TARGET} yet",
                       "Add up to two competitors with run.py --competitor &lt;package&gt; (or in apps.yaml), "
                       "label their reviews, and this page compares the apps theme by theme.")
        return
    shares, counts, (start, end) = analyze.compare_apps(DATA)
    others = [a for a in APP_NAMES if a != TARGET]
    ui.page_header("Competitors", f"{TARGET} vs {' and '.join(others)}",
                   f"Only the weeks every app has data for: {start:%d %b} to {end:%d %b %Y}. Busy apps get "
                   "hundreds of reviews a week, so their data doesn't reach further back.")

    window = DATA[DATA["date"] >= start]
    summary = analyze.app_summary(window)
    tiles = []
    for name in APP_NAMES:
        r = summary.loc[name]
        tiles.append(
            f'<div class="rr-tile"><div class="rr-label"><span class="rr-dot" style="background:{APP_COLOR[name]}"></span>'
            f'{name}</div><div style="display:flex;align-items:baseline;gap:8px;margin:8px 0 6px">'
            f'<span class="rr-big">{r.avg_stars:.2f}</span><span class="rr-muted">average rating</span></div>'
            f'<div class="rr-muted">{pct(r.negative_share)} negative · {counts[name]:,} reviews<br>'
            f'Top complaint: <b>{LABELS[r.top_complaint]}</b></div></div>')
    ui.html('<div class="rr-apps">' + "".join(tiles) + "</div>")

    ui.section("Theme by theme", "Share of each app's reviews over the same weeks")
    with st.container(key="card-dots"):
        show_all = st.toggle("Include praise and other non-complaints", key="comp_all")
        themes = [t for t in ordered_themes(window, not show_all) if t in shares.index]
        themes = sorted(themes, key=lambda t: shares.loc[t, TARGET], reverse=True)
        names = [ui.wrap(LABELS[t], 26) for t in themes]
        fig = go.Figure()
        for theme, name in zip(themes, names):  # a faint line spanning the apps' values for each theme
            values = shares.loc[theme, APP_NAMES] * 100
            fig.add_trace(go.Scatter(x=[values.min(), values.max()], y=[name, name], mode="lines",
                                     line=dict(color=P.border, width=6), hoverinfo="skip", showlegend=False))
        for app in APP_NAMES:
            fig.add_trace(go.Scatter(
                x=shares.loc[themes, app] * 100, y=names, mode="markers", name=app,
                marker=dict(size=12, color=APP_COLOR[app], line=dict(color=P.surface, width=2)),
                hovertemplate=f"{app} · %{{y}}: %{{x:.1f}}% of {counts[app]:,} reviews<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=P.ink2, size=12))
        fig.update_xaxes(ticksuffix="%", rangemode="tozero")
        ui.show(ui.style_fig(fig, P, 90 + 40 * len(themes), legend=True))

    gaps = analyze.competitor_gaps(DATA, TARGET)
    gaps = gaps[gaps["significant"]]

    def gap_list(rows: pd.DataFrame) -> str:
        items = "".join(f'<li><span>{LABELS[t]}</span><span>{pct(r.share)} vs {pct(r.others)}</span></li>'
                        for t, r in rows.iterrows())
        return f'<ul class="rr-list">{items}</ul>' if items else '<div class="rr-muted">No clear differences.</div>'

    ui.section(f"Where {TARGET} stands out", f"{TARGET} vs the other apps pooled · only differences unlikely to be chance")
    a, b = st.columns(2, gap="medium")
    with a, st.container(key="card-worse"):
        ui.html(f'<div class="rr-card-title">More common in {TARGET} reviews</div>'
                + gap_list(gaps[gaps["gap"] > 0]))
    with b, st.container(key="card-better"):
        ui.html(f'<div class="rr-card-title">Less common in {TARGET} reviews</div>'
                + gap_list(gaps[gaps["gap"] < 0].sort_values("gap")))
    with st.expander("Table view"):
        st.dataframe(shares.rename(index=LABELS)[APP_NAMES].map(lambda v: f"{v:.1%}"), width="stretch")


# --- Reviews --------------------------------------------------------------------------------------

def reviews() -> None:
    app = current_app()
    ui.page_header("Reviews", "Read the reviews behind the numbers",
                   "Every labeled review with the model's theme, severity and one-line summary.")
    with st.container(key="card-filters"):
        c1, c2, c3 = st.columns([1.4, 1, 1])
        pick_apps = c1.pills("Apps", APP_NAMES, selection_mode="multi", default=[app], key="rev_apps")
        pick_stars = c2.pills("Stars", [1, 2, 3, 4, 5], selection_mode="multi", format_func=lambda s: f"{s} ★",
                              key="rev_stars")
        pick_sev = c3.pills("Severity", [3, 2, 1], selection_mode="multi",
                            format_func=lambda s: ui.SEVERITY_NAMES[s], key="rev_sev")
        c4, c5, c6 = st.columns([1.4, 1, 1])
        pick_themes = c4.multiselect("Themes", list(INFO.index), format_func=lambda t: LABELS[t], key="rev_themes")
        search = c5.text_input("Search", placeholder="Words in the review", key="rev_search")
        order = c6.selectbox("Sort", ["Most helpful", "Newest", "Oldest"], key="rev_sort")

    view = DATA[DATA["app"].isin(pick_apps or APP_NAMES)]
    if pick_stars:
        view = view[view["rating"].isin(pick_stars)]
    if pick_sev:
        view = view[view["severity"].isin(pick_sev)]
    if pick_themes:
        view = view[view["theme"].isin(pick_themes)]
    if search:
        view = view[view["text"].str.contains(search, case=False, regex=False)]
    sort = {"Most helpful": (["thumbs_up", "date"], False), "Newest": (["date"], False), "Oldest": (["date"], True)}[order]
    view = view.sort_values(sort[0], ascending=sort[1])

    per_page = 20
    pages = max(1, -(-len(view) // per_page))
    signature = (tuple(pick_apps), tuple(pick_stars), tuple(pick_sev), tuple(pick_themes), search, order)
    if st.session_state.get("rev_signature") != signature:  # new filters: back to page 1
        st.session_state.rev_signature, st.session_state.rev_page = signature, 1
    page = min(st.session_state.get("rev_page", 1), pages)

    ui.section(f"{len(view):,} reviews", f"Page {page} of {pages}")
    chunk = view.iloc[(page - 1) * per_page: page * per_page]
    if chunk.empty:
        ui.html('<div class="rr-muted">No reviews match these filters.</div>')
    else:
        ui.html('<div class="rr-grid">' + "".join(
            ui.quote_card(r, LABELS, show_app=len(pick_apps or APP_NAMES) > 1, show_theme=True, full=True)
            for _, r in chunk.iterrows()) + "</div>")
    if pages > 1:
        b1, b2, _ = st.columns([1, 1, 6])
        if b1.button("Previous", disabled=page <= 1, key="rev_prev", width="stretch"):
            st.session_state.rev_page = page - 1
            st.rerun()
        if b2.button("Next", disabled=page >= pages, key="rev_next", width="stretch"):
            st.session_state.rev_page = page + 1
            st.rerun()


# --- Method ---------------------------------------------------------------------------------------

def method() -> None:
    model = DATA["model"].dropna().iloc[0] if DATA["model"].notna().any() else "an LLM"
    defaults = load_config("apps")["defaults"]
    per_app = load_config("classifier").get("reviews_per_app")
    sample_text = (f"About {per_app:,} random reviews per app" if per_app else "Every clean review")
    margin = 1.96 * (0.15 * 0.85 / (per_app or 2000)) ** 0.5 * 100  # 95% margin for a theme at about 15%
    results_path = PROCESSED_DIR / "validation_results.json"
    if results_path.exists():  # written by validate.py score
        v = json.loads(results_path.read_text(encoding="utf-8"))
        accuracy = (f"<b>Hand-checked.</b> On {v['reviews']} randomly chosen {v['app']} reviews labeled by hand "
                    f"without seeing the model's answer, the model picked the same main theme {v['agreement']:.0%} "
                    f"of the time ({v['lenient_agreement']:.0%} counting its second theme; Cohen's kappa "
                    f"{v['kappa']:.2f}).<br>")
    else:
        accuracy = "<b>Not hand-checked yet.</b> Run validate.py to measure label accuracy.<br>"
    ui.page_header("Method", "How this was made, and what it can't tell you",
                   "Everything here is reproducible from the scripts in this project.")
    with st.container(key="card-method"):
        ui.html(f"""<div class="rr-prose"><ol class="rr-steps">
<li><div><b>Collect.</b> The {defaults['max_reviews']:,} most recent Google Play reviews per app
({defaults['country'].upper()} store, language "{defaults['lang']}"), with star rating,
date and app version. Reviewer names are not stored. <span class="rr-muted">scrape.py</span></div></li>
<li><div><b>Clean.</b> Reviews under 3 words are dropped, because "nice app" names no theme. That removes over half
of all reviews, mostly short praise, so praise is under-counted. Language is detected and non-English reviews are
kept. <span class="rr-muted">clean.py</span></div></li>
<li><div><b>Label.</b> {model} gives each review one main theme from a fixed list, an optional second theme, a
severity from 1 to 3 and a one-line summary. Answers must match a JSON schema and are checked before they are
saved. <span class="rr-muted">classify.py, config/themes.yaml</span></div></li>
<li><div><b>Sample.</b> {sample_text}, plus a sample from the weeks all apps share so the competitor
comparison has enough data. At that size, a theme's share is accurate to roughly ±{margin:.0f} percentage points.
<span class="rr-muted">config/classifier.yaml</span></div></li>
<li><div><b>Analyze.</b> Jumps between versions, changes over the last {analyze.PERIOD_DAYS} days and gaps between
apps are only highlighted when a two-proportion z-test says they are unlikely to be chance (95% confidence).
<span class="rr-muted">analyze.py</span></div></li>
</ol>
<h4>Limits</h4>
{accuracy}<b>Reviews over-represent unhappy users.</b> They show what hurts, not how many users are affected.<br>
<b>Labels are not perfect.</b> In a re-run test on 200 ImagineArt reviews, 4% changed theme. Treat shares as ±2 points on top of
sampling error.<br>
<b>Version dates are estimates:</b> when a version first appears in reviews, not its official release date.<br>
<b>Funnel stages and KPIs are product judgment</b> (config/funnel_map.yaml), not something the data proves.
</div>""")


# --- Navigation and sidebar -----------------------------------------------------------------------

PAGES = [
    st.Page(overview, title="Overview", icon=":material/space_dashboard:", default=True),  # served at /
    st.Page(releases, title="Releases", icon=":material/new_releases:", url_path="releases"),
    st.Page(trends, title="Trends", icon=":material/show_chart:", url_path="trends"),
    st.Page(competitors, title="Competitors", icon=":material/compare_arrows:", url_path="competitors"),
    st.Page(reviews, title="Reviews", icon=":material/reviews:", url_path="reviews"),
    st.Page(method, title="Method", icon=":material/menu_book:", url_path="method"),
]
navigation = st.navigation(PAGES, position="hidden")

with st.sidebar:
    ui.html(ui.brand(P))
    ui.html('<div class="rr-side-label">App</div>')
    st.radio("App", APP_NAMES, index=APP_NAMES.index(TARGET), key="app", label_visibility="collapsed",
             format_func=lambda a: f":{APP_MD_COLOR[a]}[●]  {a}",
             help="Used by Overview, Releases, Trends and Reviews.")
    ui.html('<div class="rr-side-label">Views</div>')
    for page in PAGES:
        st.page_link(page)
    model_name = DATA["model"].dropna().iloc[0] if DATA["model"].notna().any() else "LLM"
    ui.html(f'<div class="rr-side-foot"><b>{len(DATA):,}</b> reviews labeled across {len(APP_NAMES)} apps<br>'
            f'Data through {DATA["date"].max():%d %b %Y}<br>Labels by {model_name}</div>')

navigation.run()
