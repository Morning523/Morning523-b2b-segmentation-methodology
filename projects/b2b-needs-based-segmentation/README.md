# B2B Needs-Based Segmentation

## Question and approach

How can a large B2B customer survey (hundreds of items covering technology attitudes, buying criteria, and firmographics) be reduced to a small set of reliable "need" factors, and then used to build a defensible customer segmentation?

This was a real consultancy-style segmentation project for an enterprise technology services client, carried out as part of a dissertation/consultancy engagement. The code here is adapted for public demonstration: the client's identity, the original survey instrument's question wording, the raw respondent data, and every real result (loadings, segment sizes, percentages) have been removed or genericised. Running the scripts as-is will not reproduce any real finding — see [`../../data/README.md`](../../data/README.md).

The method, in order:

1. **Reduction stage** ([`r/02c_Reduction_Stage_WS1Needs.Rmd`](r/02c_Reduction_Stage_WS1Needs.Rmd), [`r/02d_Reduction_Stage_Hierarchical.Rmd`](r/02d_Reduction_Stage_Hierarchical.Rmd)) — mixed-type (ordinal/binary) correlation matrices, PCA/exploratory factor analysis, reliability checks, and per-item missingness diagnostics to collapse the item pool into a smaller set of need factors.
2. **Clustering stage** ([`r/02d_Clustering_Hierarchical.Rmd`](r/02d_Clustering_Hierarchical.Rmd)) — hierarchical and k-means candidate segmentations compared on size adequacy, separation, stability and interpretability, then a held-out decision-tree assignment rule so new customers can be routed to a segment without re-running the full clustering.

## Code map

| File | Covers |
|---|---|
| `r/02c_Reduction_Stage_WS1Needs.Rmd` | Factor reduction on the primary need-item pool |
| `r/02d_Reduction_Stage_Hierarchical.Rmd` | Factor reduction, block-by-block, feeding the hierarchical clustering stage |
| `r/02d_Clustering_Hierarchical.Rmd` | Candidate segmentation comparison, profiling, held-out assignment-rule validation |

## Team and authorship

This was a two-person dissertation/consultancy workstream. The R Markdown files were primarily authored by my project teammate, **Ayan Nazm**, with my direct involvement in the underlying methodology design, data preparation, and review. I am including this code here as a demonstration of the segmentation methodology our team used, not as a claim of sole or primary authorship of every line.

## Data

No real survey data, the original survey instrument, or any real result is included. See [`../../data/README.md`](../../data/README.md).
