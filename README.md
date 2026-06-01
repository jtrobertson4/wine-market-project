# The Global Wine Market: A Data Portrait

An exploration of the global wine market — where wine is produced and consumed, where grape
varieties grow, and how critics rate and value wine — told through a **Tableau map dashboard**,
an interactive **Plotly HTML data story**, and a consumer **SVG poster**.

## Links

- **Tableau dashboard (Tableau Public):** https://public.tableau.com/app/profile/jillian.robertson3467/viz/globalandregionalwineproductionmap/GlobalWineDashboard
- **HTML data story (GitHub Pages):** https://jtrobertson4.github.io/wine-market-project/
- **SVG poster:** in this repository --> `svg/wine_consumer_poster.svg`

## Audience & questions

**Audience:** a curious wine drinker or a wanting-to-be-more-informed wine investor — and, for the poster, a casual shopper (i.e., the everyday consumer). No
viticulture or statistics background assumed.
**Questions:** Where does the world's wine come from and where is it drunk? Which grapes grow
where? Do higher scores or prices actually signal better wine, and where does value hide?

## Tools (Visualization Tool Usage)

- **Tableau Public** — the interactive global wine map dashboard (primary tool).
- **Python (Plotly)** — the HTML data story, published to the web via GitHub Pages.
- **Inkscape-compatible SVG** — the editable consumer poster.

## Data sets (Data Set Use)

- **OIV** — country wine production & consumption (hectolitres), 1995–2025.
- **University of Adelaide / Wine Economics** — grape variety bearing area & country yield. Old World / New World country classification.
- **Kaggle WineMag / Wine Enthusiast** — ~130,000 critic reviews (well beyond the 100-row /
  5-column "heft" bar: 13 columns).

## Process & methods (Process Writeup)

Data was cleaned and validated in `notebook.ipynb`: standardized country/year/variety fields,
expressed OIV volumes in hectolitres, combined varietal area across years, parsed country yield
tables, and dropped rows that couldn't be classified. Derived measures: variety production ≈
variety area × country grape yield; value_score = points / price; wine_style inferred from review
title and variety. Wine-region coordinates were geocoded for the Tableau map.

**Use of LLMs:** ChatGPT / Claude were used as a coding and design assistant — debugging pandas
cleaning steps, structuring the long-format Tableau file, troubleshooting Tableau filters and map
layers, and refining color/typography choices. All analytical decisions, data interpretation, and
final design were author-directed.

## Design (Principles of Good Design)

A consistent warm, editorial identity across all three artifacts: cream background (#FAF4E8),
brown ink (#3A2A20), wine-tone accents, Courier body type with serif headings. Color encodes
meaning (sequential brown ramp for volume; wine-style hues for grape categories); Gestalt grouping
via cards and proximity; on-graph titles, captions, and tooltips let each visual stand alone.

## Limitations

Variety-level and region-level production is **estimated**, not directly observed. Reviews reflect critic-reviewed
wines, not the full market. Price data is incomplete. Wine-style mapping simplifies ambiguous
blends and fortified wines. The Tableau map uses three snapshot years — **2000, 2010, 2016** —
where country production and region-level estimates align.

## Reproduce

`pip install -r requirements.txt` -> run `notebook.ipynb` top to bottom -> open `docs/index.html`
(or the GitHub Pages link) -> load `tableau/wine_map_tableau.csv` into Tableau -> open the poster
in Inkscape.
