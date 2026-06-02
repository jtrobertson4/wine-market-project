"""
build_html_story.py
-------------------
Builds "The Global Wine Market: A Data Portrait" -- a guided, scrollable Plotly HTML
data story.

Run AFTER the notebook has written the data/processed CSVs:
    python build_html_story.py

Saves the SAME self-contained file to two places:
    html/index.html   (local organisation)
    docs/index.html   (GitHub Pages publishing)

Look & feel: light eggshell page, deep-brown headings, Courier typewriter font.
Colour rules used throughout:
  - anything tied to a GRAPE / VARIETY / WINE STYLE is coloured by that thing's colour;
  - everything else (production, consumption, counts, magnitudes) uses BROWN scales;
  - every bar is slightly transparent with a slightly darker outline for a soft look.
Plotly.js is inlined so charts work offline (only the choropleth base map needs internet).
Nothing is invented -- every number shown is computed from the data.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import get_plotlyjs

# --------------------------------------------------------------------------- paths
ROOT = Path.cwd()
TABLEAU = ROOT / "tableau"            # holds only the Tableau map-dashboard master file
PROC = ROOT / "data" / "processed"    # the story's input CSVs live here
HTML = ROOT / "html"
DOCS = ROOT / "docs"
HTML.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------- palette
BG        = "#FAF4E8"   # light eggshell page background
CARD      = "#FFFCF6"   # near-white warm cream for cards
INK       = "#3A2A20"   # dark brown body text
BROWN     = "#4F2E1C"   # deep brown -- headings, accents
RED       = "#A97187"   # red wine (dusty rose accent, replaces the old burgundy)
WHITE_W   = "#E6CC83"   # white wine gold
ROSE      = "#CE8C9A"   # rosé
SPARKLING = "#C3C3BB"   # sparkling
OTHER     = "#7A5436"   # other / mid brown
NEUTRAL   = "#97897A"   # unknown / neutral
GRID      = "#E9DCC4"   # faint gridline
FONT      = "Courier New, Courier, monospace"

# wine-STYLE colours (finished-wine categories used in the review charts)
STYLE_COLORS = {"red": RED, "white": WHITE_W, "rose": ROSE,
                "sparkling": SPARKLING, "other": OTHER, "unknown": NEUTRAL}
# grape-COLOUR colours (the grape's colour, used in the variety/region charts)
GRAPE_COLOR_MAP = {"red": RED, "white": WHITE_W, "grey_or_gris": "#9C8C7A",
                   "rose": ROSE, "unknown": NEUTRAL}
# brown sequential scales for non-grape magnitudes (with the light page at the low end)
BROWN_SCALE = [[0, "#F3E6D0"], [0.5, "#B07C45"], [1, BROWN]]          # generic magnitude
BROWN_SCALE2 = [[0, "#F1E7D6"], [0.5, "#9C8A55"], [1, "#5C4A1E"]]     # consumption (olive-brown)
# olive-green sequential scale -- soft sage/khaki at the low end deepening to muted olive,
# matching the Tableau dashboard look (gentle, not saturated).
GREEN_SCALE = [[0, "#EDEBD3"], [0.5, "#C3BE86"], [1, "#8A8B52"]]      # production (soft olive)

FIG_KW = dict(full_html=False, include_plotlyjs=False,
              config={"responsive": True, "displayModeBar": False})


# ------------------------------------------------------------------ small helpers
def hex_to_rgba(hex_color, alpha):
    """'#6B1F2B' -> 'rgba(107,31,43,alpha)'. Passes rgba()/named colours through."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def darker(hex_color, factor=0.65):
    """Return a darker shade of a hex colour, for bar outlines."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return BROWN
    r, g, b = (int(int(h[i:i + 2], 16) * factor) for i in (0, 2, 4))
    return f"rgb({r},{g},{b})"


def bar_marker(colors, alpha=0.82):
    """Build a Bar marker dict: translucent fill(s) + slightly darker outline(s).
    `colors` may be one hex or a list of hex (one per bar)."""
    if isinstance(colors, (list, tuple, pd.Series)):
        fill = [hex_to_rgba(c, alpha) for c in colors]
        line = [darker(c) for c in colors]
    else:
        fill = hex_to_rgba(colors, alpha)
        line = darker(colors)
    return dict(color=fill, line=dict(color=line, width=1.1))


def tonal_scale(hex_color, amin=0.2):
    """A light->dark colourscale in ONE colour: from alpha=amin (light) to alpha=1 (dark).
    Used so each grape row ramps within its own colour (red rows red, white rows gold)."""
    h = str(hex_color).lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return [[0.0, f"rgba({r},{g},{b},{amin})"], [1.0, f"rgba({r},{g},{b},1.0)"]]


def per_row_tonal_heatmap(pivot, row_colors, hover_unit="", height=540, title=None,
                          subtitle=None, left_margin=150, bottom_margin=100,
                          cbar_title="amount"):
    """Render a variety x category matrix where EACH ROW gets its own light->dark tonal
    ramp in that row's grape colour (red grapes red, white grapes gold), 20% min alpha.

    Implemented as one Heatmap trace per variety. To keep the y-axis labels aligned 1:1
    with the rows, every trace declares the FULL ordered category list on y and writes its
    values only into its own row (others left as NaN), so Plotly renders a single clean
    grid with one tick per variety.

    Each row has its own hue, so a single multi-hue colourbar is impossible. Instead a
    shared NEUTRAL intensity scale is shown on the side: the hue of a cell tells you the
    grape, its depth tells you the magnitude (light = less, dark = more)."""
    fig = go.Figure()
    ys = list(pivot.index)          # variety order (top -> bottom as given)
    xs = list(pivot.columns)
    n = len(ys)
    vmax = np.nanmax(pivot.values) if pivot.size else 1
    for i, variety in enumerate(ys):
        # full-height z, all NaN except this variety's row
        z = [[np.nan] * len(xs) for _ in range(n)]
        z[i] = list(pivot.loc[variety].values)
        col = row_colors.get(variety, NEUTRAL)
        fig.add_trace(go.Heatmap(
            z=z, x=xs, y=ys, colorscale=tonal_scale(col), zmin=0, zmax=vmax,
            showscale=False, xgap=2, ygap=2, hoverongaps=False,
            hovertemplate="%{y} \u00b7 %{x}<br>%{z:,.0f}" + hover_unit + "<extra></extra>"))
    # two reference gradients (red + white) flush side by side, sharing ONE axis with
    # 0/25/50/75/100% ticks. Cell hue tells the grape colour; depth tells magnitude.
    import math as _m
    def _nice(v):
        if v <= 0:
            return 1.0
        mag = 10 ** _m.floor(_m.log10(v))
        return _m.ceil(v / mag) * mag
    top = _nice(float(vmax))
    ticks = [0, top * .25, top * .5, top * .75, top]
    big = top >= 1000
    ticktext = [(f"{t/1000:.0f}k" if big else f"{t:.0f}") for t in ticks]
    legend_grads = [(tonal_scale(STYLE_COLORS["red"]), 1.02, False),
                    (tonal_scale(STYLE_COLORS["white"]), 1.075, True)]
    for scale, xpos, show in legend_grads:
        fig.add_trace(go.Heatmap(
            z=[[0, top]], x=[xs[0], xs[0]], y=[ys[0], ys[0]], opacity=0,
            colorscale=scale, zmin=0, zmax=top, showscale=True, hoverinfo="skip",
            colorbar=dict(title=dict(text=(cbar_title if show else ""),
                                     font=dict(size=10, color=INK), side="top"),
                          x=xpos, thickness=12, len=0.66, outlinewidth=0,
                          tickvals=(ticks if show else []),
                          ticktext=(ticktext if show else []),
                          tickfont=dict(family=FONT, size=9, color=INK))))
    fig.update_yaxes(autorange="reversed", type="category",
                     categoryorder="array", categoryarray=ys)
    fig.update_xaxes(type="category")
    style_fig(fig, height=height, title=title, subtitle=subtitle)
    fig.update_layout(margin=dict(l=left_margin, r=110,
                                  t=104 if (title and subtitle) else 60, b=bottom_margin))
    return fig


def style_fig(fig, height=460, title=None, subtitle=None):
    """Apply the project look + a proper title/subtitle to any figure."""
    title_obj = None
    if title:
        # all graph titles render in uppercase for a consistent editorial look
        title_obj = dict(text=f"<b>{title.upper()}</b>", font=dict(size=17, color=BROWN), x=0.02,
                         xanchor="left", y=0.97, yanchor="top")
        if subtitle:
            title_obj["subtitle"] = dict(text=subtitle,
                                         font=dict(size=12.5, color="#7a6750"))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(l=60, r=30, t=104 if (title and subtitle) else (64 if title else 24), b=50),
        title=title_obj,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def safe_read(name, fallback=None, **kw):
    p = PROC / name
    if p.exists():
        return pd.read_csv(p, **kw)
    if fallback and (PROC / fallback).exists():
        print(f"  note: {name} not found, using {fallback} instead")
        return pd.read_csv(PROC / fallback, **kw)
    print(f"  WARNING: {name} not found -- charts needing it will be skipped")
    return None


# =========================================================================== load
print("Loading story CSVs from data/processed ...")
oiv = safe_read("oiv_country_summary.csv", fallback="oiv_map.csv")
var = safe_read("variety_production.csv")
vreg = safe_read("variety_region.csv", low_memory=False)
rev = safe_read("reviews.csv", low_memory=False)

for label, df in [("oiv", oiv), ("variety", var), ("variety_region", vreg), ("reviews", rev)]:
    if df is not None:
        print(f"  {label}: {len(df):,} rows")

# variety -> grape colour (from the variety data) and variety -> wine style (from reviews)
VARIETY_GRAPE_COLOR = {}
if var is not None and "grape_color" in var.columns:
    VARIETY_GRAPE_COLOR = (var.dropna(subset=["grape_color"])
                              .groupby("variety")["grape_color"]
                              .agg(lambda s: s.mode().iloc[0]).to_dict())
VARIETY_STYLE = {}
if rev is not None and "wine_style" in rev.columns:
    VARIETY_STYLE = (rev.groupby("variety")["wine_style"]
                        .agg(lambda s: s.mode().iloc[0]).to_dict())


def colors_for_varieties(varieties, by="grape"):
    """Map each variety name to its colour. by='grape' uses grape colour; by='style'
    uses the finished-wine style colour. Falls back to neutral brown."""
    out = []
    for v in varieties:
        if by == "grape":
            out.append(GRAPE_COLOR_MAP.get(VARIETY_GRAPE_COLOR.get(v, "unknown"), NEUTRAL))
        else:
            out.append(STYLE_COLORS.get(VARIETY_STYLE.get(v, "other"), OTHER))
    return out


charts = {}


# ===================================================== SECTION 1: global market
def build_global_market(oiv):
    if oiv is None or "avg_production_hl" not in oiv.columns:
        return "<p class='skip'>Production data unavailable -- section skipped.</p>", {}
    prod_col, cons_col = "avg_production_hl", "avg_consumption_hl"
    d = oiv.dropna(subset=[prod_col]).copy()
    # drop non-country aggregate rows (e.g. "Global") -- they aren't real map locations and
    # their huge totals would blow out the colour scale.
    AGG_ROWS = {"Global", "World", "Total", "EU", "European Union"}
    d = d[~d["country"].isin(AGG_ROWS)].copy()

    # --- choropleth, production/consumption toggle. Production = olive-green scale.
    # With aggregates removed we can colour to the TRUE max -- the real top producer
    # (Italy ~49M hl) anchors the dark end, so every country reads on a real scale.
    metrics = [("Avg production", prod_col, GREEN_SCALE),
               ("Avg consumption", cons_col, BROWN_SCALE2)]
    metrics = [m for m in metrics if m[1] in d.columns]

    geo = go.Figure()
    for i, (label, col, scale) in enumerate(metrics):
        cap = float(d[col].max())                # true max now that aggregates are gone
        geo.add_trace(go.Choropleth(
            locations=d["country"], locationmode="country names", z=d[col],
            colorscale=scale, zmin=0, zmax=cap,
            marker_line_color="#EAD9BC", marker_line_width=0.3,
            colorbar=dict(title="hl", thickness=12, len=0.7),
            name=label, visible=(i == 0),
            hovertemplate="%{location}<br>" + label + ": %{z:,.0f} hl<extra></extra>"))
    geo.update_layout(updatemenus=[dict(
        type="dropdown", direction="down", x=1.0, xanchor="right", y=1.26,
        bgcolor=CARD, bordercolor=GRID, font=dict(family=FONT, size=12),
        buttons=[dict(label=label, method="update",
                      args=[{"visible": [j == i for j in range(len(metrics))]}])
                 for i, (label, _, _) in enumerate(metrics)])])
    geo.update_geos(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False,
                    showland=True, landcolor="#EFE6D4", showocean=False,
                    lataxis_range=[-58, 85], projection_type="natural earth")
    style_fig(geo, height=500, title="Wine production around the world",
              subtitle="Mean annual volume across all reported years \u00b7 hectolitres (hl) \u00b7 darker denotes greater volume")
    geo.update_layout(margin=dict(l=10, r=10, t=70, b=10))

    # --- top-10 bar, production vs consumption, brown + olive-brown, translucent ---
    top = d.nlargest(10, prod_col).sort_values(prod_col)
    bar = go.Figure()
    bar.add_trace(go.Bar(y=top["country"], x=top[prod_col], orientation="h",
                         name="Production", marker=bar_marker("#A6702F"),
                         hovertemplate="%{y}: %{x:,.0f} k hl<extra></extra>"))
    if cons_col in top.columns:
        bar.add_trace(go.Bar(y=top["country"], x=top[cons_col], orientation="h",
                             name="Consumption", marker=bar_marker("#9C8A55"),
                             hovertemplate="%{y}: %{x:,.0f} k hl<extra></extra>"))
    bar.update_layout(barmode="group")
    style_fig(bar, height=470, title="Top 10 producing countries: production vs consumption",
              subtitle="Mean annual volume, hectolitres (hl) \u00b7 the 10 largest producers \u00b7 a fairer comparison than the map, which over-weights land area")
    bar.update_xaxes(title_text="Average, hl")

    total = d[prod_col].sum()
    stats = {"top5_share": d.nlargest(5, prod_col)[prod_col].sum() / total * 100,
             "top10_share": d.nlargest(10, prod_col)[prod_col].sum() / total * 100,
             "leaders": ", ".join(d.nlargest(3, prod_col)["country"].tolist())}
    return geo.to_html(**FIG_KW) + bar.to_html(**FIG_KW), stats


# ====================================================== SECTION 2: grape geography
def build_grape_geography(var):
    if var is None or "estimated_variety_production_tons" not in var.columns:
        return "<p class='skip'>Variety estimates unavailable -- section skipped.</p>", {}
    est = var[var["estimated_variety_production_tons"].notna()].copy()
    if est.empty:
        return "<p class='skip'>No variety production estimates -- section skipped.</p>", {}
    years = sorted(est["year"].dropna().unique().tolist())
    yr = int(max(years))
    e = est[est["year"] == yr]

    # --- top varieties, YEAR dropdown, each bar coloured by its GRAPE colour ---
    fig1 = go.Figure()
    for k, y in enumerate(years):
        ev = est[est["year"] == y]
        tv = (ev.groupby("variety")["estimated_variety_production_tons"].sum()
                .sort_values(ascending=False).head(10).sort_values())
        fig1.add_trace(go.Bar(y=tv.index, x=tv.values, orientation="h",
                              marker=bar_marker(colors_for_varieties(tv.index, "grape")),
                              visible=(y == yr), name=str(y),
                              hovertemplate="%{y}: %{x:,.0f} t<extra></extra>"))
    fig1.update_layout(updatemenus=[dict(
        active=years.index(yr), x=1.0, xanchor="right", y=1.26, bgcolor=CARD,
        bordercolor=GRID, font=dict(family=FONT, size=11),
        buttons=[dict(label=f"Year {y}", method="update",
                      args=[{"visible": [j == k for j in range(len(years))]}])
                 for k, y in enumerate(years)])])
    style_fig(fig1, height=470, title="Top 10 grape varieties by estimated production",
              subtitle="Estimated tons (planted area \u00d7 national yield) \u00b7 the 10 largest varieties \u00b7 select a year")
    fig1.update_xaxes(title_text="Estimated production, tons")

    # --- variety dropdown -> its top countries (country bars = brown, translucent) ---
    top12 = (e.groupby("variety")["estimated_variety_production_tons"].sum()
               .sort_values(ascending=False).head(12).index.tolist())
    fig2 = go.Figure()
    for i, v in enumerate(top12):
        cc = (e[e["variety"] == v].groupby("country")["estimated_variety_production_tons"]
                .sum().sort_values(ascending=False).head(8).sort_values())
        gc = GRAPE_COLOR_MAP.get(VARIETY_GRAPE_COLOR.get(v, "unknown"), OTHER)
        fig2.add_trace(go.Bar(y=cc.index, x=cc.values, orientation="h",
                              marker=bar_marker(gc), visible=(i == 0), name=v,
                              hovertemplate="%{y}: %{x:,.0f} t<extra></extra>"))
    fig2.update_layout(updatemenus=[dict(
        active=0, x=1.0, xanchor="right", y=1.26, bgcolor=CARD, bordercolor=GRID,
        font=dict(family=FONT, size=11),
        buttons=[dict(label=v, method="update",
                      args=[{"visible": [j == i for j in range(len(top12))]}])
                 for i, v in enumerate(top12)])])
    style_fig(fig2, height=470, title="Where a grape grows: leading countries by variety",
              subtitle=f"Estimated production, tons ({yr}) \u00b7 top 8 countries for the selected variety")
    fig2.update_xaxes(title_text=f"Estimated production, tons ({yr})")

    # --- HEATMAP: top varieties x top countries, EACH ROW tonal in its grape's colour ---
    topv = (e.groupby("variety")["estimated_variety_production_tons"].sum()
              .sort_values(ascending=False).head(12).index.tolist())
    topc = (e.groupby("country")["estimated_variety_production_tons"].sum()
              .sort_values(ascending=False).head(12).index.tolist())
    piv = (e[e["variety"].isin(topv) & e["country"].isin(topc)]
           .pivot_table(index="variety", columns="country",
                        values="estimated_variety_production_tons", aggfunc="sum")
           .reindex(index=topv, columns=topc))
    # drop any variety with no production in the 12 shown countries (no empty rows)
    piv = piv.loc[piv.fillna(0).sum(axis=1) > 0]
    topv = list(piv.index)
    row_colors = {v: GRAPE_COLOR_MAP.get(VARIETY_GRAPE_COLOR.get(v, "unknown"), NEUTRAL)
                  for v in topv}
    heat = per_row_tonal_heatmap(
        piv, row_colors, hover_unit=" t", height=540, left_margin=150, bottom_margin=100,
        cbar_title="tons",
        title="Grape \u00d7 country production heatmap",
        subtitle=f"Estimated production, tons ({yr}) \u00b7 top 12 varieties \u00d7 top 12 producing countries \u00b7 red grapes shaded red, white grapes gold \u00b7 darker denotes more")

    topv_all = (e.groupby("variety")["estimated_variety_production_tons"].sum()
                  .sort_values(ascending=False))
    stats = {"year": yr, "top_variety": topv_all.index[0],
             "top3": ", ".join(topv_all.head(3).index.tolist())}
    return (fig1.to_html(**FIG_KW) + fig2.to_html(**FIG_KW) + heat.to_html(**FIG_KW)), stats


# =================================== REGIONAL SPOTLIGHT: region x variety heatmap
def build_regional_spotlight(vreg):
    """Reworked to be informative: for a chosen country, a heatmap of its top regions
    x top grape varieties by planted area. Country dropdown drives it."""
    if vreg is None or "variety_area_ha" not in vreg.columns:
        return "<p class='skip'>Region detail unavailable -- spotlight skipped.</p>", {}

    d = vreg.dropna(subset=["variety_area_ha", "region", "variety"]).copy()
    # tidy any stray whitespace in region names for display
    d["region"] = d["region"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # countries with the richest region+variety detail
    rank = (d.groupby("country")
             .agg(n=("variety_area_ha", "size"),
                  nreg=("region", "nunique")).query("nreg >= 4")
             .sort_values("n", ascending=False))
    countries = rank.head(8).index.tolist()
    if not countries:
        return "<p class='skip'>Not enough region detail -- spotlight skipped.</p>", {}

    fig = go.Figure()
    trace_country = []          # which country each trace belongs to (for the dropdown)
    grape_color_of = {v: GRAPE_COLOR_MAP.get(VARIETY_GRAPE_COLOR.get(v, "unknown"), NEUTRAL)
                      for v in d["variety"].unique()}
    country_vmax = {}           # remember each country's true max for its legend bars
    for c in countries:
        dc = d[d["country"] == c]
        topr = (dc.groupby("region")["variety_area_ha"].sum()
                  .sort_values(ascending=False).head(12).index.tolist())
        topv = (dc.groupby("variety")["variety_area_ha"].sum()
                  .sort_values(ascending=False).head(12).index.tolist())
        piv = (dc[dc["region"].isin(topr) & dc["variety"].isin(topv)]
               .pivot_table(index="variety", columns="region",
                            values="variety_area_ha", aggfunc="sum")
               .reindex(index=topv, columns=topr))
        # a variety can rank top-12 overall yet have NO area in the 12 shown regions
        # (its plantings live elsewhere) -> that leaves an empty row. Drop those.
        piv = piv.loc[piv.fillna(0).sum(axis=1) > 0]
        topv = list(piv.index)
        nrows = len(topv)
        vmax = float(np.nanmax(piv.values)) if piv.size else 1.0
        country_vmax[c] = vmax
        # one trace per variety; each declares the full y-category list and writes only
        # its own row (others NaN) so the y tick labels line up 1:1 with the rows.
        for ri, variety in enumerate(topv):
            z = [[np.nan] * len(topr) for _ in range(nrows)]
            z[ri] = list(piv.loc[variety].values)
            fig.add_trace(go.Heatmap(
                z=z, x=topr, y=topv, colorscale=tonal_scale(grape_color_of.get(variety, NEUTRAL)),
                zmin=0, zmax=vmax, showscale=False, xgap=2, ygap=2, hoverongaps=False,
                visible=(c == countries[0]),
                hovertemplate="%{y} \u00b7 %{x}<br>%{z:,.0f} ha<extra></extra>"))
            trace_country.append(c)

    # four grape-colour gradients (red / white / rose / sparkling) flush side by side,
    # sharing ONE axis (0 -> active country's max, with 25/50/75% ticks). No per-bar style
    # labels -- just a single "hectares" title on the right-most bar. Per-country scaled.
    def nice_top(v):
        """Round a max up to a clean number for the legend (e.g. 47,213 -> 50k)."""
        if v <= 0:
            return 1.0
        import math
        mag = 10 ** math.floor(math.log10(v))
        return math.ceil(v / mag) * mag
    cbar_specs = [
        (tonal_scale(STYLE_COLORS["red"]),       1.015),
        (tonal_scale(STYLE_COLORS["white"]),     1.045),
        (tonal_scale(STYLE_COLORS["rose"]),      1.075),
        (tonal_scale(STYLE_COLORS["sparkling"]), 1.105),
    ]
    for c in countries:
        top = nice_top(country_vmax[c])
        ticks = [0, top * 0.25, top * 0.5, top * 0.75, top]
        def fmt(v):
            return f"{v/1000:.0f}k" if top >= 1000 else f"{v:.0f}"
        ticktext = [fmt(t) for t in ticks]
        for idx, (scale, xpos) in enumerate(cbar_specs):
            show_ticks = (idx == len(cbar_specs) - 1)   # labels + title only on the last bar
            fig.add_trace(go.Heatmap(
                z=[[0, top]], x=[None, None], y=[None, None], opacity=0,
                colorscale=scale, zmin=0, zmax=top, showscale=True, hoverinfo="skip",
                visible=(c == countries[0]),
                colorbar=dict(
                    title=dict(text=("hectares" if show_ticks else ""),
                               font=dict(size=10, color=INK), side="top"),
                    x=xpos, thickness=11, len=0.6, outlinewidth=0,
                    tickvals=(ticks if show_ticks else []),
                    ticktext=(ticktext if show_ticks else []),
                    tickfont=dict(family=FONT, size=9, color=INK))))
            trace_country.append(c)

    buttons = []
    for c in countries:
        vis = [(tc == c) for tc in trace_country]   # cells + that country's 3 legend bars
        buttons.append(dict(label=c, method="update", args=[{"visible": vis}]))
    fig.update_layout(updatemenus=[dict(
        active=0, x=1.0, xanchor="right", y=1.26, bgcolor=CARD, bordercolor=GRID,
        font=dict(family=FONT, size=11), buttons=buttons)])
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_xaxes(type="category")
    style_fig(fig, height=560, title="Inside a country: regions \u00d7 grape varieties",
              subtitle="Planted area (ha) by region and variety, latest year \u00b7 top 12 regions \u00d7 top 12 "
                       "varieties \u00b7 pick a country \u00b7 red grapes shaded red, white grapes gold")
    fig.update_layout(margin=dict(l=150, r=130, t=104, b=110))

    stats = {"n_countries": int(d["country"].nunique()),
             "default_country": countries[0]}
    return fig.to_html(**FIG_KW), stats


# ====================================================== SECTION 3: what critics review
def build_what_critics_review(rev):
    if rev is None:
        return "<p class='skip'>Review data unavailable -- section skipped.</p>", {}
    stats = {"n_reviews": len(rev), "n_countries": rev["country"].nunique(),
             "n_varieties": rev["variety"].nunique(),
             "avg_points": rev["points"].mean(), "median_price": rev["price"].median()}
    html = ""

    # --- stacked bar: reviews by country (top 15) x wine_style (style colours) ---
    if "wine_style" in rev.columns:
        top_c = rev["country"].value_counts().head(15).index.tolist()
        sub = rev[rev["country"].isin(top_c)]
        ct = sub.groupby(["country", "wine_style"]).size().reset_index(name="n")
        order = sub["country"].value_counts().index.tolist()[::-1]
        fig = go.Figure()
        for st in ["red", "white", "rose", "sparkling", "other"]:
            s = ct[ct["wine_style"] == st].set_index("country").reindex(order).reset_index()
            fig.add_trace(go.Bar(y=s["country"], x=s["n"], orientation="h", name=st,
                                 marker=bar_marker(STYLE_COLORS.get(st, NEUTRAL)),
                                 hovertemplate="%{y} -- " + st + ": %{x:,}<extra></extra>"))
        fig.update_layout(barmode="stack")
        style_fig(fig, height=540, title="Critic coverage by country and wine style",
                  subtitle="Review count \u00b7 top 15 countries by coverage \u00b7 reflects critical attention, not market size")
        fig.update_xaxes(title_text="Number of reviews")
        html += fig.to_html(**FIG_KW)

    # --- top reviewed varieties, STYLE filter dropdown, bars by variety's STYLE colour ---
    style_opts = ["All styles"] + (["red", "white", "rose", "sparkling", "other"]
                                   if "wine_style" in rev.columns else [])
    fig2 = go.Figure()
    for i, opt in enumerate(style_opts):
        subset = rev if opt == "All styles" else rev[rev["wine_style"] == opt]
        tv = subset["variety"].value_counts().head(10).sort_values()
        fig2.add_trace(go.Bar(y=tv.index, x=tv.values, orientation="h", visible=(i == 0),
                              marker=bar_marker(colors_for_varieties(tv.index, "style")),
                              name=opt, hovertemplate="%{y}: %{x:,} reviews<extra></extra>"))
    fig2.update_layout(updatemenus=[dict(
        active=0, x=1.0, xanchor="right", y=1.26, bgcolor=CARD, bordercolor=GRID,
        font=dict(family=FONT, size=11),
        buttons=[dict(label=opt, method="update",
                      args=[{"visible": [j == i for j in range(len(style_opts))]}])
                 for i, opt in enumerate(style_opts)])])
    style_fig(fig2, height=450, title="Most-reviewed grape varieties",
              subtitle="Review count \u00b7 top 10 varieties by coverage \u00b7 filter by style")
    fig2.update_xaxes(title_text="Number of reviews")
    html += fig2.to_html(**FIG_KW)

    rc = rev["country"].value_counts()
    stats["top_review_country"] = rc.index[0]
    stats["top_review_share"] = rc.iloc[0] / len(rev) * 100
    return html, stats


# ====================================================== SECTION 4: quality and value
def build_quality_value(rev):
    if rev is None:
        return "<p class='skip'>Review data unavailable.</p>", {}, ""
    html = ""

    # --- box plot of points by style (exact quartiles, no sampling), style colours ---
    if "wine_style" in rev.columns:
        fig = go.Figure()
        for st in ["red", "white", "rose", "sparkling", "other"]:
            pts = rev.loc[rev["wine_style"] == st, "points"].dropna()
            if pts.empty:
                continue
            q1, med, q3 = pts.quantile([.25, .5, .75]); iqr = q3 - q1
            fig.add_trace(go.Box(
                name=st, x=[st], q1=[q1], median=[med], q3=[q3],
                lowerfence=[max(pts.min(), q1 - 1.5 * iqr)],
                upperfence=[min(pts.max(), q3 + 1.5 * iqr)],
                fillcolor=hex_to_rgba(STYLE_COLORS.get(st, NEUTRAL), 0.7),
                line=dict(color=darker(STYLE_COLORS.get(st, NEUTRAL)), width=1.4)))
        fig.update_layout(showlegend=False)
        style_fig(fig, height=440, title="Critic scores by wine style",
                  subtitle="Distribution of points (80\u2013100 scale), all reviews \u00b7 box shows median and interquartile range")
        fig.update_yaxes(title_text="Critic points")
        fig.update_xaxes(title_text="Wine style")
        html += fig.to_html(**FIG_KW)

    # --- scatter price vs points, style colours, + OLS regression line ---
    sc = rev.dropna(subset=["price", "points"]); sc = sc[sc["price"] > 0]
    n_full = len(sc)
    if "wine_style" in sc.columns:
        parts = [g.sample(min(len(g), 900), random_state=1) for _, g in sc.groupby("wine_style")]
        sample = pd.concat(parts)
    else:
        sample = sc.sample(min(len(sc), 4000), random_state=1)
    fig2 = px.scatter(sample, x="price", y="points",
                      color="wine_style" if "wine_style" in sample else None,
                      color_discrete_map=STYLE_COLORS,
                      hover_data=["variety", "country"], log_x=True)
    fig2.update_traces(marker=dict(size=5, opacity=0.45,
                                   line=dict(width=0.3, color="rgba(60,40,30,0.4)")))
    # regression on log10(price) vs points over the FULL priced set (numpy, no deps)
    lp = np.log10(sc["price"].values)
    m, b = np.polyfit(lp, sc["points"].values, 1)
    r = float(np.corrcoef(lp, sc["points"].values)[0, 1])
    xs = np.linspace(sc["price"].min(), sc["price"].max(), 60)
    ys = m * np.log10(xs) + b
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="trend (OLS)",
                              line=dict(color=BROWN, width=3, dash="solid"),
                              hovertemplate="trend<extra></extra>"))
    fig2.add_annotation(xref="paper", yref="paper", x=0.03, y=0.97, showarrow=False,
                        align="left", bgcolor=hex_to_rgba(CARD, 0.85),
                        bordercolor=GRID, borderwidth=1,
                        font=dict(family=FONT, size=12, color=BROWN),
                        text=f"<b>r = {r:.2f}</b>  (log price vs points)<br>"
                             f"+{m:.1f} pts per 10\u00d7 price")
    if "wine_style" in sample.columns:
        names = [t.name for t in fig2.data if t.name in STYLE_COLORS]
        buttons = [dict(label="All styles", method="update",
                        args=[{"visible": [True] * len(fig2.data)}])]
        for st in names:
            vis = [(t.name == st or t.name == "trend (OLS)") for t in fig2.data]
            buttons.append(dict(label=st, method="update", args=[{"visible": vis}]))
        fig2.update_layout(updatemenus=[dict(active=0, x=1.0, xanchor="right", y=1.26,
                            bgcolor=CARD, bordercolor=GRID,
                            font=dict(family=FONT, size=11), buttons=buttons)])
    style_fig(fig2, height=500, title="Does a higher price secure a higher score?",
              subtitle="Critic points vs price in USD (log scale) \u00b7 random sample of priced reviews \u00b7 brown line = ordinary-least-squares trend \u00b7 filter by style")
    fig2.update_xaxes(title_text="Price (USD, log scale)")
    fig2.update_yaxes(title_text="Critic points")
    html += fig2.to_html(**FIG_KW)

    # --- correlation heatmap of the numeric review features (the "pop-out") ---
    feats = sc.rename(columns={"points": "points", "price": "price",
                               "value_score": "value", "vintage_year": "vintage"})
    cols = [c for c in ["points", "price", "value", "vintage"] if c in feats.columns]
    corr = feats[cols].corr()
    hm = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        # soft translucent olive-green (the score-box "acceptable" green): pale at the
        # negative end, deeper olive-green at the positive end
        colorscale=[[0.0, "rgba(187,174,120,0.14)"], [0.5, "rgba(169,154,82,0.35)"],
                    [1.0, "rgba(90,82,38,0.66)"]],
        colorbar=dict(title="r", thickness=12, len=0.8), xgap=2, ygap=2,
        text=corr.round(2).values, texttemplate="%{text}",
        textfont=dict(family=FONT, size=13, color=INK),
        hovertemplate="%{y} vs %{x}: r = %{z:.2f}<extra></extra>"))
    hm.update_yaxes(autorange="reversed")
    style_fig(hm, height=420, title="How the review numbers move together",
              subtitle="Pearson correlation coefficients across all priced reviews \u00b7 deeper green = stronger positive \u00b7 points, price (USD), value_score, vintage")
    hm.update_layout(margin=dict(l=90, r=30, t=104, b=60))
    html += hm.to_html(**FIG_KW)

    # --- top value table (value_score, price>0, points>=88); style as colour dot ---
    table_html = ""
    if "value_score" in rev.columns:
        cols2 = [c for c in ["title", "country", "variety", "wine_style",
                             "points", "price", "value_score"] if c in rev.columns]
        val = rev[(rev["price"] > 0) & (rev["points"] >= 88) & rev["value_score"].notna()]
        val = val.nlargest(12, "value_score")[cols2].copy()
        headers = ""
        for c in cols2:
            label = "Style" if c == "wine_style" else c.replace("_", " ").title()
            headers += f"<th>{label}</th>"
        rows = ""
        for _, rrow in val.iterrows():
            cells = ""
            for c in cols2:
                if c == "wine_style":
                    col = STYLE_COLORS.get(rrow[c], NEUTRAL)
                    cells += (f"<td><span class='dot' title='{rrow[c]}' "
                              f"style='background:{col}'></span></td>")
                elif c == "price":
                    cells += f"<td>${rrow[c]:,.0f}</td>"
                elif c == "value_score":
                    cells += f"<td>{rrow[c]:.1f}</td>"
                else:
                    cells += f"<td>{rrow[c]}</td>"
            rows += f"<tr>{cells}</tr>"
        table_html = (f"<table class='vtable'><thead><tr>{headers}</tr></thead>"
                      f"<tbody>{rows}</tbody></table>")

    # clean correlation: value_score vs log10(price) over rows that have both
    vp_rows = sc.dropna(subset=["value_score", "price"])
    corr_vp = float(np.corrcoef(np.log10(vp_rows["price"]), vp_rows["value_score"])[0, 1])

    stats = {"sample_n": len(sample), "full_n": n_full,
             "median_price": sc["price"].median(), "r_pp": r, "slope": m,
             "pts_band": (rev["points"].between(86, 92)).mean() * 100,
             "corr_value_price": corr_vp}
    return html, stats, table_html


# =========================================================================== build
print("\nBuilding charts ...")
charts["market"], s_market = build_global_market(oiv)
charts["grape"], s_grape = build_grape_geography(var)
charts["region"], s_region = build_regional_spotlight(vreg)
charts["review"], s_review = build_what_critics_review(rev)
charts["value"], s_value, value_table = build_quality_value(rev)


def fmt(x, d=0):
    return f"{x:,.{d}f}" if isinstance(x, (int, float, np.floating)) else x

# ---------------------------------------------------- data-driven takeaway bullets
takeaways = []
if s_market:
    takeaways.append(f"Wine production is concentrated: the top five countries "
                     f"({s_market['leaders']} and two more) make up about "
                     f"{fmt(s_market['top5_share'])}% of average global output, and the top ten about "
                     f"{fmt(s_market['top10_share'])}%.")
if s_review and s_market:
    takeaways.append(f"Critic coverage is not the same as production volume: "
                     f"{s_review['top_review_country']} accounts for roughly "
                     f"{fmt(s_review['top_review_share'])}% of all reviews, even though the largest "
                     f"producer by volume is {s_market['leaders'].split(',')[0]}.")
if s_value:
    takeaways.append(f"Critic points cluster in a narrow band: about {fmt(s_value['pts_band'])}% of "
                     f"reviewed wines score between 86 and 92, so price and value differences carry a "
                     f"lot of the signal.")
    takeaways.append(f"Price and score move together only moderately (r = {s_value['r_pp']:.2f}): a "
                     f"ten-fold price increase buys about {s_value['slope']:.1f} extra points on average, "
                     f"so paying more does not guarantee a proportionally better score.")
if s_grape:
    takeaways.append(f"By estimated production in {s_grape['year']}, the leading grapes are "
                     f"{s_grape['top3']} -- a useful geography of where grapes grow, though the figures "
                     f"are estimates (area \u00d7 average yield), not measured tonnage.")

takeaway_html = "".join(f"<li>{t}</li>" for t in takeaways)

# stat cards for the review section
cards_html = ""
if s_review:
    cards = [("Reviews", fmt(s_review['n_reviews'])), ("Countries", fmt(s_review['n_countries'])),
             ("Varieties", fmt(s_review['n_varieties'])),
             ("Avg points", fmt(s_review['avg_points'], 1)),
             ("Median price", f"${fmt(s_review['median_price'])}")]
    cards_html = "".join(f"<div class='stat'><div class='num'>{v}</div>"
                         f"<div class='lbl'>{k}</div></div>" for k, v in cards)

# value-score explanation + scatter/heatmap notes
value_explain = ""
scatter_note = ""
if s_value:
    value_explain = (
        f"<strong>How value_score works.</strong> Critic points run 80&ndash;100 and price is in "
        f"US dollars, so <code>value_score = points &divide; price</code> is a simple "
        f"<em>points-per-dollar</em> index. A $10 wine scoring 90 earns 9.0; a $90 wine scoring 90 "
        f"earns just 1.0. It is only computed where a positive price exists, and because points barely "
        f"vary while price varies enormously, the index is driven mostly by price &mdash; the "
        f"correlation between value_score and (log) price is {s_value['corr_value_price']:.2f}. Treat it "
        f"as a rough comparison tool that rewards cheap-but-decent bottles, not a verdict on quality.")
    scatter_note = (f"The scatter shows a random sample of {fmt(s_value['sample_n'])} reviewed wines "
                    f"(from {fmt(s_value['full_n'])} with a listed price) so it stays readable; the box "
                    f"plot and correlations use every review. The brown line is the overall trend.")

# "About The Scores" tier box -- chips in brown / olive tones.
# (chip text colour flips to dark brown on the two light olive chips so it stays legible)
SCORE_TIERS = [
    ("98-100", "Classic", "The pinnacle of quality", "#4F2E1C", "#FFFFFF"),
    ("94-97", "Superb", "A great achievement", "#5E3A22", "#FFFFFF"),
    ("90-93", "Excellent", "Highly recommended", "#6E4A28", "#FFFFFF"),
    ("87-89", "Very Good", "Often good value; well recommended", "#7E5A2E", "#FFFFFF"),
    ("83-86", "Good", "Everyday drinking, often good value", "#A99A52", "#3A2A20"),
    ("80-82", "Acceptable", "For casual, less-critical moments", "#BBAE78", "#3A2A20"),
]
score_rows = "".join(
    f"<div class='tier'><span class='chip' style='background:{chip};color:{txt}'>{rng}</span>"
    f"<span class='tname'>{name}</span><span class='tdesc'>{desc}</span></div>"
    for rng, name, desc, chip, txt in SCORE_TIERS)
score_box = (f"<div class='scorebox'><div class='scorehead'>ABOUT THE SCORES</div>"
             f"<p class='scoreintro'>Wine Enthusiast doesn't publish reviews for wines scoring "
             f"below 80 (deemed unacceptable), so the visible range runs 80&ndash;100 &mdash; which is "
             f"exactly why the scores above sit in such a tight band.</p>{score_rows}"
             f"<p class='scoresrc'>Scale &amp; descriptions: Wine Enthusiast.</p></div>")

# --- units primer: a cute "what the numbers mean" card, score-box styling ---
# Reads the project's three working units (hl, hectares, tons) and distils the
# Dr. Vinny / Napa grape-to-bottle conversions into a friendly, INTERACTIVE scale:
# pick a unit, set a quantity, and the conversions update live. Each converted value is
# bolded in the same translucent-green shade as its source pill.
# Base conversions (per 1 unit), sourced from Dr. Vinny (Wine Spectator) + Napa chart:
#   1 ton grapes  -> ~720 bottles, ~60 cases, ~2.5 barrels, ~150 gallons, ~3175 glasses
#   1 hectare     -> ~105 hl (midpoint of 35-175) -> ~14000 bottles (illustrative midpoint)
#   1 hl          -> 100 L -> ~133 bottles (750 ml) -> ~11 cases
UNIT_DEFS = [
    ("ton", "ton(s) of grapes", "rgba(94,58,34,0.50)",
     [("bottles (750 ml)", 720), ("cases (12 btl)", 60), ("barrels", 2.5),
      ("gallons", 150), ("six-oz glasses", 3175)]),
    ("hectare", "hectare(s) of vines", "rgba(110,74,40,0.46)",
     [("hectolitres (midpoint)", 105), ("bottles (~midpoint)", 14000),
      ("acres", 2.47), ("cases (~midpoint)", 1167)]),
    ("hl", "hectolitres (the OIV unit)", "rgba(169,154,82,0.50)",
     [("litres", 100), ("bottles (750 ml)", 133.33), ("cases (12 btl)", 11.11)]),
]
import json as _json
units_payload = _json.dumps([
    {"id": uid, "label": lab, "color": col,
     "conv": [{"name": n, "factor": f} for n, f in convs]}
    for uid, lab, col, convs in UNIT_DEFS])

units_box = (
    f"<div class='scorebox unitsbox'><div class='scorehead'>A SENSE OF SCALE</div>"
    f"<p class='scoreintro'>Three units carry this story &mdash; <strong>tons</strong> of grapes at "
    f"harvest, <strong>hectares</strong> of vines, and <strong>hectolitres (hl)</strong> "
    f"of national volume. Choose a unit and a quantity to see what it becomes once the grapes are "
    f"wine.</p>"
    f"<div class='uconv'>"
    f"  <div class='upills' id='upills'></div>"
    f"  <div class='uinput'><label>Quantity:&nbsp;</label>"
    f"      <input id='uqty' type='number' value='1' min='0' step='1'> "
    f"      <span id='uunit' class='uunitlabel'></span></div>"
    f"  <div class='uout' id='uout'></div>"
    f"</div>"
    f"<p class='scoresrc'>Conversions: Wine Spectator, &ldquo;Ask Dr. Vinny&rdquo; (Jul 27, 2016), "
    f"<a href='https://www.winespectator.com/articles/how-many-bottles-wine-per-hectare-53465'>"
    f"winespectator.com</a>; and Napa Valley Private Label Wine, &ldquo;Grape Growing to Bottled "
    f"Wine Conversion Chart.&rdquo; Figures are approximate (yields vary widely).</p></div>"
    f"<script>(function(){{"
    f"  var U={units_payload}; var cur=0;"
    f"  var pills=document.getElementById('upills');"
    f"  var qty=document.getElementById('uqty');"
    f"  var out=document.getElementById('uout');"
    f"  var unit=document.getElementById('uunit');"
    f"  function render(){{"
    f"    pills.innerHTML='';"
    f"    U.forEach(function(u,i){{"
    f"      var b=document.createElement('button');"
    f"      b.className='upill'+(i===cur?' on':''); b.textContent=u.label;"
    f"      b.style.background=u.color;"
    f"      b.onclick=function(){{cur=i; render();}}; pills.appendChild(b);"
    f"    }});"
    f"    var u=U[cur]; var q=parseFloat(qty.value)||0; unit.textContent=u.label;"
    f"    out.innerHTML='';"
    f"    u.conv.forEach(function(c){{"
    f"      var v=q*c.factor;"
    f"      var row=document.createElement('div'); row.className='urow';"
    f"      var val=v>=1000?Math.round(v).toLocaleString():"
    f"        (Math.round(v*100)/100).toLocaleString();"
    f"      row.innerHTML='<span class=\\'ulabel\\'>'+c.name+'</span>'+"
    f"        '<span class=\\'uval\\' style=\\'color:'+u.color.replace(/0\\.\\d+\\)/,'1)')+"
    f"        '\\'>'+val+'</span>';"
    f"      out.appendChild(row);"
    f"    }});"
    f"  }}"
    f"  qty.addEventListener('input',render); render();"
    f"}})();</script>")


# =========================================================================== HTML
print("Assembling HTML ...")
PLOTLY_JS = get_plotlyjs()

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wine Market Data Portrait</title>
<script>{PLOTLY_JS}</script>
<style>
  :root {{ --bg:{BG}; --card:{CARD}; --ink:{INK}; --brown:{BROWN};
           --serif: Georgia, 'Times New Roman', 'Iowan Old Style', serif; }}
  * {{ box-sizing: border-box; }}
  /* body stays typewriter monospace; headings use a clean, more formal serif */
  body {{ margin:0; background:var(--bg); color:var(--brown); font-family:{FONT}; line-height:1.7; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 0 22px; }}
  header.hero {{ text-align:center; padding: 64px 22px 30px; }}
  /* intro area: no white frame -- transparent, borderless, just centered text */
  .herobox {{ background: transparent; border: none; border-radius: 0;
              padding: 8px 0 0; margin-top: 18px; box-shadow: none; }}
  header.hero h1 {{ font-family: {FONT}; font-weight: 700;
                    font-size: clamp(26px, 3.9vw, 42px); letter-spacing: 1px; margin: 0 0 14px;
                    color: {BROWN}; text-transform: uppercase; }}
  /* constrain width so the subtitle breaks into two balanced lines, not an orphan tail */
  header.hero .sub {{ font-family: {FONT}; font-weight: 700; font-size: clamp(15px, 2vw, 20px);
                      color: var(--brown); margin: 0 auto; max-width: 620px; letter-spacing: 0.5px;
                      text-transform: uppercase; line-height: 1.5; text-wrap: balance; }}
  header.hero .intro {{ margin: 22px auto 0; max-width: 760px; font-size: 15px; color: var(--ink);
                        text-align: justify; text-justify: inter-word; }}
  .outputs {{ font-size: 13px; color:#6f5c46; margin: 16px auto 0; max-width: 760px;
              letter-spacing: 0.3px; text-align: justify; text-justify: inter-word; }}
  section {{ padding: 46px 0; border-top: 1px solid {GRID}; }}
  section h2 {{ font-family: {FONT}; font-weight: 700; font-size: clamp(23px, 4vw, 33px);
                color: var(--brown); margin: 0 0 6px; letter-spacing: 0.5px; text-transform: uppercase; }}
  section h3 {{ font-family: {FONT}; font-weight: 700; color: var(--brown);
                letter-spacing: 0.3px; text-transform: uppercase; }}
  .kicker {{ text-transform: uppercase; letter-spacing: 3px; font-size: 12px; color:#8a7659;
             margin-bottom: 4px; }}
  section p {{ font-size: 15px; max-width: 760px; color: var(--ink); }}
  .subhead {{ font-family: {FONT}; font-weight: 700; font-size: clamp(19px,3vw,25px);
              color: var(--brown); margin: 30px 0 2px; letter-spacing: 0.3px;
              text-transform: uppercase; }}
  .card {{ background: var(--card); border: 1px solid {GRID}; border-radius: 12px; padding: 18px;
           margin: 22px 0; box-shadow: 0 2px 10px rgba(79,46,28,0.05); }}
  .note {{ font-size: 13px; color:#6f5c46; font-style: italic; }}
  .skip {{ font-size: 14px; color:#8a7659; font-style: italic; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:14px; margin: 22px 0; }}
  .stat {{ flex:1 1 120px; background:var(--card); border:1px solid {GRID}; border-radius:12px;
           padding:16px; text-align:center; box-shadow: 0 2px 10px rgba(79,46,28,0.05); }}
  .stat .num {{ font-family: {FONT}; font-size: 27px; color: var(--brown); font-weight: bold; }}
  .stat .lbl {{ font-size: 12px; color:#6f5c46; text-transform: uppercase; letter-spacing:1px; }}
  ul.takeaways {{ max-width: 820px; }}
  ul.takeaways li {{ margin: 12px 0; font-size: 15px; color: var(--ink); }}
  .vexplain {{ background: var(--card); border-left: 4px solid {BROWN}; border-radius: 8px;
               padding: 14px 18px; margin: 18px 0; font-size: 14px; max-width: 820px; color: var(--ink); }}
  /* data sources (top) + references (bottom) */
  .sources {{ display:flex; flex-wrap:wrap; gap:16px; margin: 22px 0; }}
  .source {{ flex:1 1 240px; background:var(--card); border:1px solid {GRID}; border-radius:12px;
             padding:16px 18px; box-shadow: 0 2px 10px rgba(79,46,28,0.05); }}
  .source .stitle {{ font-family: {FONT}; font-weight:700; color:var(--brown); font-size:15px;
                     margin-bottom:4px; }}
  .source .sbody {{ font-size:13px; color:var(--ink); }}
  ol.refs {{ max-width: 860px; padding-left: 20px; }}
  ol.refs li {{ margin: 12px 0; font-size: 13.5px; color: var(--ink); line-height: 1.6; }}
  ol.refs a {{ color: var(--brown); word-break: break-all; }}
  table.vtable {{ width:100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px;
                  border: 2px solid rgba(110,74,40,0.55); border-radius: 6px; overflow: hidden; }}
  table.vtable th {{ background: rgba(110,74,40,0.85); color: #fff; padding: 8px; text-align:left;
                     text-transform: uppercase; letter-spacing: 0.5px; }}
  table.vtable td {{ padding: 7px 8px; border-bottom: 1px solid {GRID}; color: var(--ink); }}
  table.vtable tr:nth-child(even) td {{ background: rgba(110,74,40,0.05); }}
  .dot {{ display:inline-block; width:13px; height:13px; border-radius:50%;
          border:1px solid rgba(60,40,30,0.45); }}
  .scorebox {{ background: var(--card); border: 1px solid {GRID}; border-radius: 12px; padding: 20px;
               margin: 22px 0; box-shadow: 0 2px 10px rgba(79,46,28,0.05); }}
  .scorehead {{ font-family: {FONT}; font-weight:700; color: var(--brown); font-size: 17px;
                letter-spacing: 0.5px; margin-bottom: 8px; }}
  .scoreintro {{ font-size: 13.5px; color:#4a3829; margin: 0 0 14px; max-width: 760px; }}
  .scoresrc {{ font-size: 9px; color:#8a7659; font-style: italic; margin-top: 14px;
               text-align: right; }}
  .scoresrc a {{ color:#8a7659; }}
  .unitsbox .chip {{ min-width: 150px; }}
  .uconv {{ margin: 6px 0 4px; }}
  .upills {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom: 14px; }}
  .upill {{ font-family:{FONT}; font-size: 12.5px; color: var(--brown); cursor:pointer;
            border:1px solid rgba(62,74,42,0.4); border-radius: 20px; padding: 6px 14px;
            transition: all .15s; }}
  .upill.on {{ outline: 2px solid rgba(62,74,42,0.7); font-weight: bold; }}
  .upill:hover {{ filter: brightness(0.97); }}
  .uinput {{ font-size: 13.5px; color: var(--ink); margin-bottom: 14px; }}
  .uinput input {{ font-family:{FONT}; font-size: 14px; width: 90px; padding: 5px 8px;
                   border:1px solid {GRID}; border-radius: 6px; background: var(--bg);
                   color: var(--brown); }}
  .uunitlabel {{ font-size: 12.5px; color:#6f5c46; }}
  .uout {{ display:flex; flex-wrap:wrap; gap:10px 26px; }}
  .urow {{ display:flex; align-items:baseline; gap:8px; min-width: 200px;
           border-bottom: 1px dashed {GRID}; padding: 5px 0; flex: 1 1 200px; }}
  .urow .ulabel {{ font-size: 13px; color:#5a4636; flex:1; }}
  .urow .uval {{ font-weight: bold; font-size: 15px; }}
  .tier {{ display:flex; align-items:center; gap:14px; padding: 7px 0; border-top: 1px dashed {GRID}; }}
  .chip {{ font-size: 12.5px; font-weight: bold; padding: 4px 10px; border-radius: 20px;
           min-width: 78px; text-align:center; }}
  .tname {{ color: var(--brown); font-weight: bold; min-width: 110px; }}
  .tdesc {{ color:#4a3829; font-size: 13.5px; }}
  footer {{ padding: 46px 0 72px; border-top: 1px solid {GRID}; font-size: 13px; color:#6f5c46; }}
  footer a {{ color: var(--brown); }}
  .sig {{ margin-top: 18px; font-size: 13px; color: var(--brown); text-transform: uppercase;
          letter-spacing: 1px; }}
  .limits li {{ font-size: 13.5px; margin: 7px 0; max-width: 820px; }}
</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <h1>Wine Market Data Portrait</h1>
    <div class="sub">Production, grape geography, critic reviews, and value across the global wine market.</div>
   <div class="herobox">
    <div class="intro">These data investigations are sourced from three independent datasets: global
      <strong>production and consumption</strong> from the OIV (in hectolitres), the
      <strong>geography of grape varieties</strong> derived from University of Adelaide / Wine
      Economics bearing-area and yield figures, and roughly 130,000 <strong>critic reviews</strong>
      from Wine Enthusiast. Read together, they trace where the world's wine originates, which grapes
      occupy which ground, and how professional scores, prices, and value align across the market.</div>
    <div class="outputs">Audience: an educated, curious reader interested in wine and markets, who
      wants the data to reveal where wine comes from, what grows where, and how critics value it &mdash;
      without assuming prior expertise in viticulture or statistics.</div>
   </div>
  </div>
</header>

<div class="wrap">

  <section style="border-top:none;">
    <div class="kicker">00 &middot; Before we begin</div>
    <h2>The data behind this story</h2>
    <p>Three independent public datasets are combined here. They measure different things in
      different units, so each chart notes which source it draws on, and full citations appear in
      the References at the end.</p>
    <div class="sources">
      <div class="source">
        <div class="stitle">OIV &mdash; production &amp; consumption</div>
        <div class="sbody">The International Organisation of Vine and Wine's country-level wine
          production and consumption series, reported in hectolitres (hl).
          Used for the global market section.</div>
      </div>
      <div class="source">
        <div class="stitle">Adelaide / Wine Economics &mdash; grape geography</div>
        <div class="sbody">Anderson, Nelgen &amp; Puga's database of regional, national and global
          winegrape bearing areas by variety (2000&ndash;2023), plus country grape yields. Used to
          estimate variety production by area &times; yield.</div>
      </div>
      <div class="source">
        <div class="stitle">Wine Enthusiast / Kaggle &mdash; critic reviews</div>
        <div class="sbody">~130k Wine Enthusiast reviews (variety, country, winery, price, points)
          scraped in 2017 and published on Kaggle by user zynicide. Used for ratings, prices,
          value and review coverage.</div>
      </div>
    </div>
    {units_box}
  </section>

  <section>
    <div class="kicker">01 &middot; The global picture</div>
    <h2>Where wine is made &mdash; and drunk</h2>
    <p>The OIV reports production and consumption in hectolitres (hl), the unit
      retained throughout this analysis. The choropleth renders each country's <strong>mean</strong>
      over all reported years; the toggle alternates between average production and average
      consumption. Because area distorts a map's visual weight, the accompanying bar chart of the
      ten largest producers offers the more rigorous comparison.</p>
    <div class="card">{charts['market']}</div>
    <p class="note">A map can visually overstate large countries, so the bar chart of the top ten
      is the fairer read. Production is strikingly concentrated in a handful of countries.</p>
  </section>

  <section>
    <div class="kicker">02 &middot; Grape geography</div>
    <h2>Which grapes grow where</h2>
    <p>Variety production is <strong>estimated</strong> as planted area (hectares) multiplied by each
      country's average grape yield (tons per hectare). This deliberately assumes every variety
      yields at the national mean, so the figures are best read as an approximate <em>geography of
      grapes</em> rather than measured tonnage. The year
      selector compares the 2000, 2010 and 2016 vintages, and selecting a grape reveals its leading
      producing countries.</p>
    <div class="card">{charts['grape']}</div>

    <div class="subhead">Regional spotlight</div>
    <p>Zooming from countries into <strong>regions</strong>: choose a country to see a heatmap of its
      top regions against its top grape varieties, by planted area. It reveals each region's
      signature grapes &mdash; and how concentrated (or varied) a region's plantings are. The variety
      data carries region detail for {s_region.get('n_countries', 'several')} countries.</p>
    <div class="card">{charts['region']}</div>
  </section>

  <section>
    <div class="kicker">03 &middot; What critics review</div>
    <h2>The shape of the review data</h2>
    <p>These figures describe <strong>critical coverage</strong> &mdash; the wines that appear in Wine
      Enthusiast's reviews &mdash; rather than the scale of any country's market. A higher review count
      signals greater <em>critical attention</em>, not necessarily greater production or consumption.</p>
    <div class="stats">{cards_html}</div>
    <div class="card">{charts['review']}</div>
    <p class="note">Wine style is inferred from grape variety and the wine's title
      (red / white / rosé / sparkling / other). It's a label for the finished wine, which isn't the
      same as the grape's colour.</p>
  </section>

  <section>
    <div class="kicker">04 &middot; Quality &amp; value</div>
    <h2>Scores, prices, and points per dollar</h2>
    <p>Critic scores occupy a narrow band, so price carries much of the discriminating signal. The
      charts below examine how scores distribute across styles, whether higher prices command higher
      points, and how the review measures correlate with one another.</p>
    {score_box}
    <div class="card">{charts['value']}</div>
    <p class="note">{scatter_note}</p>
    <div class="vexplain">{value_explain}</div>
    <h3 style="margin-top:30px;">Best points-per-dollar (score &ge; 88)</h3>
    <p class="note">The 12 highest value-score wines meeting the criteria (score \u2265 88, positive price).
      Wine style is encoded as a coloured dot (hover for the label); winery is omitted, as it already
      appears within each wine's title.</p>
    {value_table}
  </section>

  <section>
    <div class="kicker">05 &middot; The short version</div>
    <h2>Key takeaways</h2>
    <ul class="takeaways">{takeaway_html}</ul>
  </section>

  <section>
    <div class="kicker">06 &middot; Fine print</div>
    <h2>Data &amp; limitations</h2>
    <ul class="limits">
      <li>OIV data supplies country-level wine production and consumption (OIV reports in 1000 hl; shown here in hl).</li>
      <li>University of Adelaide / Wine Economics data supplies grape variety area and country yields.</li>
      <li>Variety production is <strong>estimated</strong> from area &times; country yield, only for
        years where yield data exists.</li>
      <li>Wine Enthusiast reviews cover <strong>reviewed wines</strong>, not the full market; review
        counts are coverage, not market size.</li>
      <li>Price data is incomplete &mdash; value figures use only rows with a positive price.</li>
      <li>Wine-style classification is inferred from variety and title; fortified and genuinely mixed
        wines are left as "other".</li>
    </ul>
  </section>

  <section>
    <div class="kicker">07 &middot; References</div>
    <h2>Data sources &amp; citations</h2>
    <p>Datasets are cited in Chicago style. Accessed May 2026.</p>
    <ol class="refs">
      <li>Anderson, Kym, Signe Nelgen, and Germ&aacute;n Puga. <em>Database of Regional, National and
        Global Winegrape Bearing Areas by Variety, 2000 to 2023</em>. Wine Economics Research Centre,
        University of Adelaide, December 2025.</li>
      <li>International Organisation of Vine and Wine. <em>Data Discovery Report</em> (wine production
        and consumption by country). Accessed May 2026.
        <a href="https://www.oiv.int/what-we-do/data-discovery-report">https://www.oiv.int/what-we-do/data-discovery-report</a>.</li>
      <li>Thoutt, Zachary (zynicide). <em>Wine Reviews</em> (winemag-data-130k-v2). Kaggle, 2017.
        Data scraped from Wine Enthusiast (winemag.com). Accessed May 2026.
        <a href="https://www.kaggle.com/datasets/zynicide/wine-reviews">https://www.kaggle.com/datasets/zynicide/wine-reviews</a>.</li>
    </ol>

    <h3 style="margin-top:28px;">Further reading</h3>
    <p>For readers who want to go deeper into the production figures and the geography of grape
      varieties:</p>
    <ol class="refs">
      <li>OIV. <em>World Wine Production Outlook 2025</em>. International Organisation of Vine and Wine,
        November 2025. ISBN 978-2-85038-130-0.
        <a href="https://www.oiv.int/what-we-do/statistics">https://www.oiv.int/what-we-do/statistics</a>.</li>
      <li>Anderson, Kym, and Signe Nelgen. <em>Which Winegrape Varieties Are Grown Where? A Global
        Empirical Picture</em>, rev. ed. Adelaide: University of Adelaide Press, 2020.
        <a href="https://www.adelaide.edu.au/press/titles/winegrapes">adelaide.edu.au/press</a>.</li>
      <li>Anderson, Kym. &ldquo;China's Wine Market: What Next?&rdquo; WERC Wine Brief No. 52, December
        2024 (also in <em>Australian and New Zealand Grapegrower and Winemaker</em> 733, February 2025).</li>
      <li>Anderson, Kym. &ldquo;Economics and Wine, and Globalization.&rdquo; WERC Wine Brief No. 31,
        June 2021.</li>
    </ol>
  </section>

  <footer>
    <strong>Wine Market Data Portrait</strong><br>
    Tableau dashboard: <a href="https://public.tableau.com/app/profile/jillian.robertson3467/viz/globalandregionalwineproductionmap/GlobalWineDashboard">view on Tableau Public</a> &middot;
    GitHub repository: <a href="https://github.com/jtrobertson4/wine-market-project">github.com/jtrobertson4/wine-market-project</a> &middot;
    SVG poster: <a href="https://github.com/jtrobertson4/wine-market-project/blob/main/svg/wine_consumer_poster.svg">svg/wine_consumer_poster.svg</a>
    <div class="sig">Jillian Robertson &middot; Wine World Data Project &middot; 05/29/26</div>
  </footer>

</div>
</body>
</html>
"""

for out in (HTML / "index.html", DOCS / "index.html"):
    out.write_text(page, encoding="utf-8")
    print(f"saved: {out.relative_to(ROOT)}  ({len(page):,} bytes)")

print("\nDone. Open html/index.html in any browser (offline OK; map base layer needs internet).")
