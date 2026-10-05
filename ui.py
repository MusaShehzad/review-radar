"""How the dashboard looks: color tokens, global CSS and small HTML building blocks.

Streamlit's own widgets are themed in .streamlit/config.toml. Everything custom (headline tiles,
takeaways, the theme table, review cards) is plain HTML styled by the CSS below, so app.py only
decides *what* to show, and this file decides how it looks.
"""
import base64
from dataclasses import dataclass
from html import escape
from string import Template

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FONT = "'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


# --- Color tokens ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Palette:
    dark: bool
    page: str
    surface: str
    surface2: str      # quiet fill: bar tracks, chips
    border: str
    ink: str           # main text
    ink2: str          # secondary text
    ink3: str          # labels, captions
    accent: str        # the one data/UI blue
    accent_soft: str
    good: str          # "getting better" text
    good_soft: str
    bad: str           # "getting worse" text
    bad_soft: str
    warn: str
    warn_soft: str
    star: str
    apps: tuple        # one fixed color per app, in config order (colorblind-checked)
    scale: tuple       # heatmap scale, low -> high (low values fade into the surface)


LIGHT = Palette(
    dark=False, page="#f4f5f7", surface="#ffffff", surface2="#eff1f4", border="#e2e5ea",
    ink="#11161c", ink2="#4f5967", ink3="#7f8995", accent="#2a78d6", accent_soft="#e3edfb",
    good="#0b7a34", good_soft="#e2f3e6", bad="#c02f2f", bad_soft="#fbe8e7", warn="#8a5a00",
    warn_soft="#fcf0d4", star="#e8a100", apps=("#2a78d6", "#eb6834", "#1baf7a"),
    scale=("#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"),
)
DARK = Palette(
    dark=True, page="#0c0f13", surface="#13171d", surface2="#1c2129", border="#262c35",
    ink="#e8ecf1", ink2="#a9b2bd", ink3="#7c8692", accent="#3987e5", accent_soft="#15253b",
    good="#3fbf63", good_soft="#122a1a", bad="#ef6f6f", bad_soft="#361719", warn="#e5b450",
    warn_soft="#33280f", star="#f2b632", apps=("#3987e5", "#d95926", "#199e70"),
    scale=("#161d27", "#17304d", "#1d4675", "#2860a1", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)


def palette() -> Palette:
    """The palette for the viewer's current light/dark setting."""
    return DARK if st.context.theme.type == "dark" else LIGHT


# --- Global CSS -----------------------------------------------------------------------------------

CSS = Template("""
<style>
.stApp { background: $page; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 5rem; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .4rem; }
[data-testid="stPageLink"] a { border-radius: 8px; }
[data-testid="stPageLink"] a[aria-current="page"] { background: $surface2; }
[data-testid="stPageLink"] a[aria-current="page"] p { font-weight: 600; color: $ink; }
[data-testid="stWidgetLabel"] p { font-size: 12.5px; color: $ink2; }

/* Streamlit containers created with key="card-..." become cards */
[class*="st-key-card"] { background: $surface; border: 1px solid $border; border-radius: 12px;
  padding: 18px 20px 12px; }

.rr-brand { display: flex; align-items: center; gap: 10px; padding: 4px 2px 14px; }
.rr-brand-name { font-size: 16px; font-weight: 600; color: $ink; letter-spacing: -.01em; line-height: 1.1; }
.rr-brand-sub { font-family: $mono; font-size: 10.5px; color: $ink3; letter-spacing: .04em; }
.rr-side-label { font-family: $mono; font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase;
  color: $ink3; margin: 14px 2px 4px; }
.rr-side-foot { font-size: 12px; color: $ink3; line-height: 1.55; border-top: 1px solid $border;
  padding-top: 12px; margin-top: 8px; }
.rr-side-foot b { color: $ink2; font-weight: 600; }

.rr-head { margin: 0 0 22px; }
.rr-eyebrow { font-family: $mono; font-size: 11px; font-weight: 500; letter-spacing: .08em;
  text-transform: uppercase; color: $ink3; }
.rr-title { font-size: 28px; line-height: 1.15; font-weight: 600; letter-spacing: -.02em; color: $ink;
  margin: 6px 0 8px; text-wrap: balance; }
.rr-sub { font-size: 14px; line-height: 1.5; color: $ink2; max-width: 72ch; }
.rr-section { display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  margin: 30px 0 10px; flex-wrap: wrap; }
.rr-section h3 { font-size: 15.5px; font-weight: 600; color: $ink; margin: 0; padding: 0; letter-spacing: -.01em; }
.rr-section span { font-size: 12.5px; color: $ink3; }
.rr-card-title { font-size: 14.5px; font-weight: 600; color: $ink; margin: 0 0 12px; }
.rr-muted { font-size: 12.5px; color: $ink3; line-height: 1.5; }

.rr-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 1050px) { .rr-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .rr-kpis { grid-template-columns: 1fr; } }
.rr-kpi { background: $surface; border: 1px solid $border; border-radius: 12px; padding: 14px 16px 12px;
  display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.rr-label { font-family: $mono; font-size: 10.5px; font-weight: 500; letter-spacing: .07em;
  text-transform: uppercase; color: $ink3; }
.rr-kpi-main { display: flex; align-items: flex-end; justify-content: space-between; gap: 10px; }
.rr-kpi-value { font-size: 30px; font-weight: 600; letter-spacing: -.025em; color: $ink; line-height: 1; }
.rr-kpi-foot { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; color: $ink3; }
.rr-delta { font-size: 11.5px; font-weight: 600; padding: 2px 7px; border-radius: 999px; white-space: nowrap;
  font-variant-numeric: tabular-nums; }
.rr-delta.good { color: $good; background: $good_soft; }
.rr-delta.bad { color: $bad; background: $bad_soft; }
.rr-delta.flat { color: $ink2; background: $surface2; }
.rr-spark { flex: none; }

[data-testid="stHtml"] ul.rr-takeaways, [data-testid="stHtml"] ul.rr-list,
[data-testid="stHtml"] ol.rr-steps { padding-left: 0; margin: 0; }  /* undo Streamlit's list indent */
.rr-takeaways { list-style: none; margin: 0; padding: 0; }
.rr-takeaways li { display: grid; grid-template-columns: 104px 1fr; gap: 12px; align-items: baseline;
  padding: 11px 0; border-top: 1px solid $border; font-size: 14px; color: $ink2; line-height: 1.5; }
.rr-takeaways li:first-child { border-top: 0; padding-top: 0; }
.rr-takeaways b { color: $ink; font-weight: 600; }
@media (max-width: 560px) { .rr-takeaways li { grid-template-columns: 1fr; gap: 4px; } }
.rr-pill { display: inline-block; font-family: $mono; font-size: 10px; font-weight: 500; letter-spacing: .06em;
  text-transform: uppercase; padding: 3px 8px; border-radius: 999px; white-space: nowrap; justify-self: start; }
.rr-pill.top, .rr-pill.note { color: $ink2; background: $surface2; }
.rr-pill.emerging { color: $bad; background: $bad_soft; }
.rr-pill.declining { color: $good; background: $good_soft; }
.rr-pill.release { color: $warn; background: $warn_soft; }
.rr-pill.gap { color: $accent; background: $accent_soft; }

.rr-funnel { display: flex; flex-direction: column; gap: 16px; }
.rr-stage-head { display: flex; justify-content: space-between; gap: 8px; font-size: 13.5px; font-weight: 600; color: $ink; }
.rr-stage-head span:last-child { font-variant-numeric: tabular-nums; color: $ink2; }
.rr-track { height: 6px; border-radius: 3px; background: $surface2; overflow: hidden; margin: 7px 0 6px; }
.rr-fill { height: 100%; border-radius: 3px; background: $accent; }
.rr-stage-foot { font-size: 12.5px; color: $ink3; line-height: 1.45; }
.rr-stage-foot b { color: $ink2; font-weight: 500; }

.rr-tablewrap { overflow-x: auto; }
.rr-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.rr-table th { font-family: $mono; font-size: 10.5px; font-weight: 500; letter-spacing: .07em; text-transform: uppercase;
  color: $ink3; text-align: left; padding: 0 10px 9px; border-bottom: 1px solid $border; white-space: nowrap; }
.rr-table td { padding: 8px 10px; border-bottom: 1px solid $border; color: $ink2; vertical-align: middle; }
.rr-table tr:last-child td { border-bottom: 0; }
.rr-table th.num, .rr-table td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.rr-table td:first-child, .rr-table th:first-child { padding-left: 0; }
.rr-table td:last-child, .rr-table th:last-child { padding-right: 0; }
.rr-theme-name { color: $ink; font-weight: 600; white-space: nowrap; }
.rr-stage { font-family: $mono; font-size: 10px; letter-spacing: .05em; text-transform: uppercase; color: $ink3; }
.rr-sharebar { position: relative; height: 24px; border-radius: 6px; background: $surface2; min-width: 96px; }
.rr-sharebar > div { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 6px; background: $accent_soft;
  border-right: 2px solid $accent; }
.rr-sharebar > span { position: relative; padding: 0 9px; line-height: 24px; font-variant-numeric: tabular-nums;
  color: $ink; font-weight: 600; }
.rr-bad { color: $bad; font-weight: 600; } .rr-good { color: $good; font-weight: 600; } .rr-flat { color: $ink3; }
@media (max-width: 760px) { .rr-hide-sm { display: none; } }

.rr-quotes { display: grid; gap: 10px; }
.rr-quote { border: 1px solid $border; border-radius: 10px; padding: 12px 14px; background: $surface; }
.rr-meta { display: flex; gap: 6px 14px; align-items: center; flex-wrap: wrap; font-family: $mono; font-size: 11px;
  color: $ink3; margin-bottom: 6px; }
.rr-stars { color: $star; letter-spacing: 1px; font-family: $font; font-size: 12.5px; }
.rr-stars i { color: $border; font-style: normal; }
.rr-quote-text { font-size: 13.5px; color: $ink; line-height: 1.55; }
.rr-summary { font-size: 13.5px; font-weight: 600; color: $ink; margin-bottom: 4px; }
.rr-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 9px; }
.rr-chip { font-size: 11.5px; padding: 2px 8px; border-radius: 999px; background: $surface2; color: $ink2; white-space: nowrap; }
.rr-chip.sev3 { color: $bad; background: $bad_soft; }
.rr-chip.sev2 { color: $warn; background: $warn_soft; }
.rr-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }

.rr-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.rr-apps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 860px) { .rr-apps { grid-template-columns: 1fr; } }
.rr-tile { background: $surface; border: 1px solid $border; border-radius: 12px; padding: 14px 16px; min-width: 0; }
.rr-tile-title { font-size: 14px; font-weight: 600; color: $ink; margin: 6px 0 4px; }
.rr-big { font-size: 26px; font-weight: 600; letter-spacing: -.02em; color: $ink; }
.rr-list { list-style: none; margin: 0; padding: 0; }
.rr-list li { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-top: 1px solid $border;
  font-size: 13.5px; color: $ink2; }
.rr-list li:first-child { border-top: 0; }
.rr-list li span:last-child { font-variant-numeric: tabular-nums; white-space: nowrap; }
.rr-prose { font-size: 14px; line-height: 1.65; color: $ink2; max-width: 74ch; }
.rr-prose h4 { font-size: 14.5px; color: $ink; margin: 22px 0 6px; font-weight: 600; }
.rr-prose b { color: $ink; font-weight: 600; }
.rr-steps { counter-reset: step; list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }
.rr-steps li { counter-increment: step; display: grid; grid-template-columns: 32px 1fr; gap: 12px; }
.rr-steps li::before { content: counter(step, decimal-leading-zero); font-family: $mono; font-size: 12px; color: $ink3; padding-top: 2px; }
</style>
""")


def inject_css(p: Palette) -> None:
    st.html(CSS.substitute(**{k: v for k, v in p.__dict__.items() if isinstance(v, str)}, font=FONT, mono=MONO))


# --- HTML building blocks --------------------------------------------------------------------------

def html(markup: str) -> None:
    st.html(markup)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    html(f'<div class="rr-head"><div class="rr-eyebrow">{escape(eyebrow)}</div>'
         f'<div class="rr-title">{escape(title)}</div><div class="rr-sub">{subtitle}</div></div>')


def section(title: str, note: str = "") -> None:
    html(f'<div class="rr-section"><h3>{escape(title)}</h3><span>{note}</span></div>')


def brand(p: Palette) -> str:
    """Logo mark (three radar arcs) and product name for the sidebar."""
    mark = svg_image(f'<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28">'
            f'<rect width="28" height="28" rx="7" fill="{p.accent}"/>'
            f'<path d="M8 20a10 10 0 0 1 12-12" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round" opacity=".55"/>'
            f'<path d="M11.5 20a6.5 6.5 0 0 1 8.5-8.5" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round" opacity=".8"/>'
            f'<circle cx="18.5" cy="18.5" r="2.4" fill="#fff"/></svg>', 28, 28)
    return (f'<div class="rr-brand">{mark}<div><div class="rr-brand-name">Review Radar</div>'
            f'<div class="rr-brand-sub">GOOGLE PLAY · US</div></div></div>')


def svg_image(svg: str, width: int, height: int, css_class: str = "") -> str:
    """Wrap an SVG drawing in an <img> tag. Streamlit's HTML sanitizer removes inline <svg>
    elements, but keeps images, so small drawings are embedded as base64 data."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img class="{css_class}" width="{width}" height="{height}" alt="" '
            f'src="data:image/svg+xml;base64,{data}">')


def sparkline(values, color: str, width: int = 112, height: int = 34) -> str:
    """A small inline trend line with a dot on the latest value. Empty if fewer than 2 points."""
    values = [float(v) for v in values if pd.notna(v)]
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    points = [(2 + i * (width - 6) / (len(values) - 1), height - 4 - (v - low) / span * (height - 9))
              for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"2,{height} {line} {points[-1][0]:.1f},{height}"
    x, y = points[-1]
    return svg_image(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polygon points="{area}" fill="{color}" fill-opacity="0.12"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/></svg>', width, height, css_class="rr-spark")


def delta_chip(text: str, tone: str) -> str:
    """tone: good, bad or flat."""
    return f'<span class="rr-delta {tone}">{escape(text)}</span>' if text else ""


def kpi(label: str, value: str, delta: str = "", spark: str = "", foot: str = "") -> str:
    return (f'<div class="rr-kpi"><div class="rr-label">{escape(label)}</div>'
            f'<div class="rr-kpi-main"><div class="rr-kpi-value">{value}</div>{spark}</div>'
            f'<div class="rr-kpi-foot">{delta}<span>{foot}</span></div></div>')


def kpi_grid(cards: list[str]) -> None:
    html('<div class="rr-kpis">' + "".join(cards) + "</div>")


PILL_NAMES = {"top": "Top issue", "emerging": "Rising", "declining": "Falling", "release": "Release",
              "gap": "Vs rivals", "note": "Note"}


def takeaways(items: list[dict]) -> str:
    rows = "".join(f'<li><span class="rr-pill {i["kind"]}">{PILL_NAMES[i["kind"]]}</span><span>{i["text"]}</span></li>'
                   for i in items)
    return f'<ul class="rr-takeaways">{rows}</ul>'


def stars(rating: int) -> str:
    rating = int(rating)
    return f'<span class="rr-stars" aria-label="{rating} stars">{"★" * rating}<i>{"★" * (5 - rating)}</i></span>'


SEVERITY_NAMES = {1: "Minor", 2: "Moderate", 3: "Severe"}


def quote_card(row, labels: pd.Series, show_app: bool = False, show_theme: bool = False,
               full: bool = False) -> str:
    """One review as a card: stars and metadata, the model's one-line summary, the review text, chips."""
    text = str(row["text"])
    if not full and len(text) > 280:
        text = text[:277].rsplit(" ", 1)[0] + "…"
    meta = [stars(row["rating"]), f'<span>{row["date"]:%d %b %Y}</span>']
    if pd.notna(row.get("app_version")):
        meta.append(f'<span>v{escape(str(row["app_version"]))}</span>')
    if row.get("thumbs_up", 0):
        meta.append(f'<span>{int(row["thumbs_up"])} found helpful</span>')
    chips = []
    if show_app:
        chips.append(f'<span class="rr-chip">{escape(row["app"])}</span>')
    if show_theme:
        chips.append(f'<span class="rr-chip">{escape(labels[row["theme"]])}</span>')
        if pd.notna(row.get("secondary_theme")) and row["secondary_theme"] in labels.index:
            chips.append(f'<span class="rr-chip">{escape(labels[row["secondary_theme"]])}</span>')
    severity = int(row["severity"])
    if severity > 1:
        chips.append(f'<span class="rr-chip sev{severity}">{SEVERITY_NAMES[severity]}</span>')
    chips_html = f'<div class="rr-chips">{"".join(chips)}</div>' if chips else ""
    summary = f'<div class="rr-summary">{escape(str(row["summary"]))}</div>' if full else ""
    return (f'<div class="rr-quote"><div class="rr-meta">{"".join(meta)}</div>{summary}'
            f'<div class="rr-quote-text">{escape(text)}</div>{chips_html}</div>')


# --- Charts ---------------------------------------------------------------------------------------

def style_fig(fig: go.Figure, p: Palette, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=4, r=8, t=8, b=4), showlegend=legend,
        font=dict(family=FONT, size=12, color=p.ink2),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=p.surface, bordercolor=p.border, font=dict(family=FONT, size=12, color=p.ink)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, font=dict(size=12, color=p.ink2)),
    )
    fig.update_xaxes(gridcolor=p.border, linecolor=p.border, zeroline=False, automargin=True,
                     tickfont=dict(color=p.ink3, size=11))
    fig.update_yaxes(gridcolor=p.border, linecolor=p.border, zeroline=False, automargin=True,
                     tickfont=dict(color=p.ink3, size=11))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def mix(scale: tuple, t: float) -> str:
    """Color at position t (0-1) along the scale."""
    t = min(max(t, 0.0), 1.0) * (len(scale) - 1)
    i = min(int(t), len(scale) - 2)
    a, b = scale[i], scale[i + 1]
    f = t - i
    rgb = [round(int(a[k:k + 2], 16) * (1 - f) + int(b[k:k + 2], 16) * f) for k in (1, 3, 5)]
    return "#" + "".join(f"{c:02x}" for c in rgb)


def _is_light(color: str) -> bool:
    r, g, b = (int(color[k:k + 2], 16) / 255 for k in (1, 3, 5))
    lum = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lum[0] + 0.7152 * lum[1] + 0.0722 * lum[2] > 0.3


def wrap(text: str, width: int = 20) -> str:
    """Break a long label onto two lines so it takes less room beside a chart."""
    if len(text) <= width:
        return text
    cut = text.rfind(" ", 0, width + 1)
    return text if cut < 0 else text[:cut] + "<br>" + text[cut + 1:]


def share_heatmap(shares: pd.DataFrame, counts: pd.Series, themes: list[str], labels: pd.Series,
                  col_names: list[str], col_title: str, p: Palette, marked: set = frozenset()) -> go.Figure:
    """Rows = themes, columns = versions or periods (left to right in time), color = share of that
    column's reviews. `shares` has one row per version/period, as analyze.shares_by returns it."""
    z = shares.reindex(columns=themes, fill_value=0).mul(100).T
    zmax = max(float(z.values.max()), 1.0)
    names = [wrap(labels[t]) for t in themes]
    fig = go.Figure(go.Heatmap(
        z=z.values, x=col_names, y=names, customdata=[[counts[c] for c in z.columns] for _ in themes],
        zmin=0, zmax=zmax, xgap=3, ygap=3, showscale=False,
        colorscale=[[i / (len(p.scale) - 1), c] for i, c in enumerate(p.scale)],
        hovertemplate=f"{col_title} %{{x}}<br>%{{y}}: %{{z:.0f}}% of %{{customdata}} reviews<extra></extra>",
    ))
    for i, theme in enumerate(themes):
        for j, col in enumerate(z.columns):
            value = z.values[i][j]
            if value < 0.5:
                continue
            cell = mix(p.scale, value / zmax)
            fig.add_annotation(x=col_names[j], y=names[i], showarrow=False,
                               text=f"{value:.0f}%" + (" ▲" if (col, theme) in marked else ""),
                               font=dict(size=11, family=FONT, color="#0b0f14" if _is_light(cell) else "#ffffff"))
    fig.update_xaxes(side="top", showgrid=False, tickmode="array", tickvals=col_names, ticktext=col_names,
                     tickangle=0 if len(col_names) <= 12 else -45, tickfont=dict(color=p.ink2, size=11))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickmode="array", tickvals=names, ticktext=names,
                     tickfont=dict(color=p.ink2, size=12))
    return style_fig(fig, p, 70 + 40 * len(themes))
