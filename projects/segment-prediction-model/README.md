# Segment-Prediction Model

## Question and approach

Once a needs-based customer segmentation exists, can segment membership be predicted from attributes that are *not* part of the survey items used to build the segments — specifically, firmographic and role attributes that could plausibly be available in an operational CRM? And if so, how well, and with what caveats about real-world deployability?

This is adapted from a real dissertation/consultancy analysis for an enterprise technology services client, for public demonstration of methodology only. The client's identity, the original sample size, segment sizes, and every real accuracy/kappa figure have been removed or genericised. Running the script as-is will not reproduce any real finding — see [`../../data/README.md`](../../data/README.md). This part of the analysis is my own work (built with AI-assisted coding during the dissertation), not the teammate-authored segmentation code in the sibling project.

## Method

1. **Predictor eligibility audit, built before any model is fit** ([`python/segment_prediction_model.py`](python/segment_prediction_model.py)) — every candidate predictor or block is individually reviewed and excluded/included with a written reason, covering: whether it was used to construct the segmentation itself (definitional leakage), whether it's a documented supporting/near-duplicate facet of a construction item (indirect leakage), whether it's a legally protected characteristic (age, gender — excluded on lawfulness grounds regardless of predictive value), and whether its availability outside the survey (e.g. in an operational CRM) is actually established.
2. **Leakage-controlled modelling** — majority-class baseline, multinomial logistic regression, and a tuned random forest, compared via nested cross-validation (outer 5-fold for evaluation, inner 3-fold for hyperparameter search), with all imputation/encoding fitted inside the training folds only.
3. **Bounded reporting** — per-segment precision/recall/F1, a confusion matrix, probability calibration, and an explicit "what this does and doesn't prove" conclusion: statistical association within the sample is not evidence of operational deployability.

## Code map

| File | Covers |
|---|---|
| `python/segment_prediction_model.py` | Full pipeline: eligibility audit → nested-CV model comparison → confusion matrix/calibration → bounded written report |

## Data

No real survey data or needs-mapping table is included. See [`../../data/README.md`](../../data/README.md).
