# B2B Segmentation Methodology

A public demonstration of the statistical/analytical methods used in a real dissertation/consultancy customer-segmentation project for an enterprise technology services client. The client's identity, the original survey instrument's wording, the raw respondent data, and every real result (loadings, segment sizes, accuracy figures) have been removed or genericised throughout this repository. **These scripts will not reproduce any real finding if run as-is** — see [`data/README.md`](data/README.md).

## Project guide

| Project | Question and approach | Main tools |
|---|---|---|
| [B2B needs-based segmentation](projects/b2b-needs-based-segmentation/README.md) | Reduce a large survey item pool to reliable "need" factors, then build and validate a customer segmentation with a held-out assignment rule. | R, PCA/EFA (polychoric/mixed correlations), hierarchical & k-means clustering, decision-tree assignment rule |
| [Segment-prediction model](projects/segment-prediction-model/README.md) | Predict segment membership from CRM-plausible attributes outside the survey, with an explicit leakage-eligibility audit and a bounded conclusion about deployability. | Python, scikit-learn, nested cross-validation |

## How to read this repository

1. Start with a project's README to understand its question, scope, and what it does/doesn't prove.
2. Follow the code map to the R Markdown or Python files.
3. Read [`data/README.md`](data/README.md) to see what inputs are intentionally absent and why.

## Team and authorship

The needs-based segmentation project was a two-person dissertation/consultancy workstream; the R code there was primarily authored by my teammate, **Ayan Nazm**, with my direct involvement in methodology and review — see that project's README for the full breakdown. The segment-prediction model is my own work, built with AI-assisted coding during the dissertation.

## Confidentiality

This repository demonstrates methodology only. It contains no real respondent data, no original survey question text, no client name, and no real statistical findings from the underlying client engagement.
