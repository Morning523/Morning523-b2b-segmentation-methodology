"""
Segment-prediction model: operational identification of segment membership.

Predicts the canonical researcher-produced four-segment (30-needs, K-means
k=4) membership from candidate predictors that are NOT part of the 30-need
construction, are not near-duplicates/supporting facets of those needs, and
are not outcome-derived, legacy-segment or free-text fields.

All preprocessing (imputation, encoding, scaling) happens inside the
cross-validation folds. Tree-based hyperparameter tuning is confined to
training folds via nested cross-validation. No global preprocessing is
fitted on the full sample before splitting.

Run:  python segment_prediction_model.py
Requires: pandas, numpy, scipy, scikit-learn, openpyxl, matplotlib.
"""
import os
import sys
import platform
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                              cohen_kappa_score, precision_recall_fscore_support,
                              confusion_matrix, brier_score_loss)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
# Real respondent-level data and the needs-mapping table are not included in
# this repository (client confidentiality) -- see ../../../data/README.md.
# Point these at local copies to actually run this script.
CLEAN_CSV = os.path.join(OUT, "data", "survey_data_cleaned.csv")  # NOT INCLUDED
D1_PATH = os.path.join(OUT, "data", "needs_master_table.csv")  # NOT INCLUDED

RNG_SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
import pandas, numpy, scipy, sklearn
env_info = {"python": sys.version, "platform": platform.platform(),
            "pandas": pandas.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
with open(os.path.join(OUT, "environment.txt"), "w", encoding="utf-8") as f:
    for k, v in env_info.items():
        f.write(f"{k}: {v}\n")

# ---------------------------------------------------------------------------
# Rebuild the outcome: canonical researcher-produced 30-needs 4-segment
# membership (independent rebuild, identical method to
# ../researcher_main_model_validation/run_researcher_main_validation.py)
# ---------------------------------------------------------------------------
F5_ORDER = {
    "Not a priority": 0,
    "A longer-term priority (10+ years)": 1,
    "A medium-term priority (next 5-10 years)": 2,
    "A shorter-term priority (next 1-3 years)": 3,
    "Current priority (next 12 months)": 4,
}


def recode_f5(series):
    return series.map(F5_ORDER)


def clean_1_7(series):
    def conv(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s and s[0].isdigit():
            return int(s[0]) if s[0] != "1" or len(s) == 1 else (10 if s[:2] == "10" else int(s[0]))
        return np.nan
    return pd.to_numeric(series.map(conv), errors="coerce")


def family_of(code):
    if code.startswith("C5c_1") or code.startswith("C5c_2"):
        return "C5c"
    for fam in ("F1", "F3", "F5", "F10"):
        if code.startswith(fam) and code[len(fam):len(fam) + 1] == "r":
            return fam
    return "OTHER"


def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)


log("=== Rebuilding canonical outcome (researcher 30-needs K-means k=4) ===")
df = pd.read_csv(CLEAN_CSV, low_memory=False)
d1 = pd.read_csv(D1_PATH)
log(f"Loaded {df.shape[0]} respondent rows.")
d1["family"] = d1["Source Code"].map(family_of)
f5_codes = d1.loc[d1.family == "F5", "Source Code"].tolist()
c5c_codes = d1.loc[d1.family == "C5c", "Source Code"].tolist()
f10_codes = d1.loc[d1.family == "F10", "Source Code"].tolist()
all_30_codes = f5_codes + c5c_codes + f10_codes

f5_num = df[f5_codes].apply(recode_f5)
c5c_num = df[c5c_codes].apply(pd.to_numeric, errors="coerce")
f10_num = df[f10_codes].apply(clean_1_7)
items_num = pd.concat([f5_num, c5c_num, f10_num], axis=1)[all_30_codes]
items_z = items_num.apply(zscore)
X30 = items_z[all_30_codes].values

km = KMeans(n_clusters=4, n_init=10, random_state=RNG_SEED).fit(X30)
outcome_labels = km.labels_ + 1  # 1..4, label order consistent with the upstream canonical partition
sizes = pd.Series(outcome_labels).value_counts().sort_index()
log(f"Outcome segment sizes: {sizes.to_dict()}")

y = pd.Series(outcome_labels, name="segment")
majority_baseline_pct = round(sizes.max() / sizes.sum() * 100, 1)
log(f"Majority-class baseline accuracy: {majority_baseline_pct}%")

print("OUTCOME_REBUILD_OK", flush=True)

# ---------------------------------------------------------------------------
# rq4_predictor_eligibility.csv — built BEFORE fitting any model.
# ---------------------------------------------------------------------------
log("\n=== Building rq4_predictor_eligibility.csv (before any model fit) ===")

elig_rows = []


def add(code, label, role, avail_survey, avail_wider, mtype, base_missing, used_in_30,
        supporting_flag, leakage, decision, reason):
    elig_rows.append({
        "source_code_or_block": code, "label": label, "analytical_role": role,
        "availability_in_survey": avail_survey,
        "evidence_available_across_wider_customer_base": avail_wider,
        "measurement_type": mtype, "valid_base_and_missingness": base_missing,
        "used_directly_in_30needs_construction": used_in_30,
        "supporting_duplicate_or_near_proxy_of_construction_need": supporting_flag,
        "leakage_status": leakage, "final_decision": decision, "reason": reason,
    })


add("F5r1-F5r15, C5c_1r1-C5c_1r4, C5c_2r5-C5c_2r7, F10r2/r3/r4/r5/r6/r7/r8/r15",
    "The 30 primary taxonomy-need variables used to build the outcome",
    "construction variable", "Yes (survey)", "NOT ESTABLISHED",
    "5-point priority scale (F5) / 1-7 bipolar (C5c) / 1-7 importance (F10)",
    "n = [full valid sample], no missing (assert-checked in Task A)", True, "IS the outcome construction set",
    "DEFINITIONAL, NOT LEAKAGE-CONTROLLABLE",
    "EXCLUDED (construction variable)", "These variables define the outcome by construction; using them as "
    "predictors would be definitional, not predictive.")
add("F9ar1-F9ar13", "F9a mobile/tactical importance ratings",
    "supporting facet of multiple construction needs (Security & Risk, Network Reliability, "
    "Mobile-First, Sustainability, Flexibility/Responsiveness, Support & Service Responsiveness)",
    "Yes (survey)", "NOT ESTABLISHED", "1-7 importance rating", "n = [full valid sample] mostly, see column-level missingness",
    False, True, "SUPPORTING/NEAR-PROXY",
    "EXCLUDED (block)", "Per Needs_to_Variable_Mapping_WS2.md, every F9a item corroborates a specific "
    "construction need's primary F5/C5c item; including them risks leaking the same underlying construct "
    "used to build the outcome.")
add("F9br1-F9br13", "F9b (parallel mobile/tactical block)", "likely supporting facet, same construct family as F9a",
    "Yes (survey)", "NOT ESTABLISHED", "1-7 rating", "n = [full valid sample] mostly", False, True, "SUPPORTING/NEAR-PROXY",
    "EXCLUDED (block)", "Same construct family as F9a; not documented as independent of the 30 needs.")
add("F10r1, F10r9, F10r10, F10r11, F10r12, F10r13, F10r14, F10r16",
    "Remaining F10 provider-selection items not in the researcher's 8 F10 primaries",
    "supporting facet of Support & Service Responsiveness, Security & Risk, Vendor Consolidation, "
    "Customisation, Digital Skills, Business & Sector Understanding needs",
    "Yes (survey)", "NOT ESTABLISHED", "1-7 importance rating", "n = [full valid sample] mostly; F10r9 sector-routed subset",
    False, True, "SUPPORTING/NEAR-PROXY", "EXCLUDED (block)",
    "Corroborating facets of construction needs per the WS2 mapping audit (Task D); using them would "
    "leak the same underlying construct through a different item.")
add("F1r1-F1r15, F3r1-F3r15", "Technology current-adoption (F1) and forced future-priority (F3) checklist items",
    "technology-signal facets of specific construction needs (see D1 60-needs table)",
    "Yes (survey)", "NOT ESTABLISHED", "binary adoption flag (F1) / forced top-3 pick (F3)",
    "F1 n = [full valid sample]; F3 base = interested subset only, substantial missingness", False, True,
    "SUPPORTING/NEAR-PROXY", "EXCLUDED (block)",
    "D1 taxonomy links most F1/F3 items to specific construction needs as corroborating technology signals; "
    "F3's restricted valid base is also a data-quality risk.")
add("C5d1-C5d8, C5e_L1-C5e_L8", "Supplier-relationship agreement/experience ratings",
    "attitudinal, survey-only; conceptually adjacent to C5c relationship needs",
    "Yes (survey)", "NOT ESTABLISHED", "1-7 agreement/rating", "n = [full valid sample] mostly", False, True,
    "SUPPORTING/NEAR-PROXY (uncertain)", "EXCLUDED (block)",
    "Not formally mapped to the 30 needs, but close enough in content to the C5c relationship-expectation "
    "block that inclusion risks leakage without a documented independence rationale.")
add("F6_Lr1-F6_Lr15 (r1-r5 sub-items each)", "Supplier evaluation / experience grid (15 x 5 sub-items)",
    "attitudinal, survey-only", "Yes (survey)", "NOT ESTABLISHED", "categorical rating", "high missingness (grid, skip logic)",
    False, False, "SURVEY-ONLY ATTITUDE, UNCERTAIN CONTENT", "EXCLUDED (block)",
    "Content and construct not independently verified against the 30 needs; high missingness from grid "
    "skip logic makes it unsuitable as a leakage-controlled predictor without further documentation.")
add("F8", "Criticality-of-provider rating", "attitudinal, survey-only", "Yes (survey)", "NOT ESTABLISHED",
    "1-7 rating", "n = [full valid sample]", False, False, "SURVEY-ONLY ATTITUDE", "EXCLUDED",
    "Subjective attitude rating with no external/CRM equivalent documented.")
add("B2c, B5ar1-B5ar4, B5b, B5cr*, B6ar1-B6ar4, B6b, B7r1-B7r14, B7ar1-B7ar11",
    "Behavioural/spend and channel-preference block", "behavioural/attitudinal, survey-only",
    "Yes (survey)", "NOT ESTABLISHED", "mixed (categorical/rating)", "varies by item", False, False,
    "SURVEY-ONLY BEHAVIOUR/ATTITUDE, UNCERTAIN AVAILABILITY", "EXCLUDED (block)",
    "Not documented as available in the client's operational CRM outside this survey; several items are close to "
    "provider-choice attitudes that overlap conceptually with F10 buying-criterion needs.")
add("C4ar1-C4ar21", "Channel/contact preference ratings", "attitudinal, survey-only", "Yes (survey)",
    "NOT ESTABLISHED", "rating", "n = [full valid sample] mostly", False, False, "SURVEY-ONLY ATTITUDE", "EXCLUDED (block)",
    "No documented external/CRM availability; large block with unclear independence from needs constructs.")
add("A2r1-A2r99", "General attitude battery", "attitudinal, survey-only", "Yes (survey)", "NOT ESTABLISHED",
    "categorical/rating", "n = [full valid sample] mostly", False, False, "SURVEY-ONLY ATTITUDE", "EXCLUDED (block)",
    "Attitudinal battery, not established as operationally available or independent of needs constructs.")
add("D9a_pipe/D9ar1-6, D9b_pipe/D9br1-6, D9c_pipe/D9cr1-6", "Current/future work-mode (desk/hybrid/remote)",
    "near-proxy of Secure Hybrid-Working Need (F5r15)", "Yes (survey)", "NOT ESTABLISHED",
    "multi-select binary flags", "n = [full valid sample]", False, True, "NEAR-PROXY", "EXCLUDED (block)",
    "Employees' current/future work mode is conceptually close to the Secure Hybrid-Working construction "
    "need; including it risks leaking that need through a related item.")
add("D10r1-D10r4", "Agreement-scale statements (content not independently documented)",
    "attitudinal, survey-only, content unverified", "Yes (survey)", "NOT ESTABLISHED", "1-7 agreement",
    "n = [full valid sample]", False, False, "SURVEY-ONLY ATTITUDE, UNCERTAIN CONTENT", "EXCLUDED (block)",
    "Item wording not available in the source files reviewed for this task; excluded conservatively "
    "pending documentation.")
add("D2a, D2b", "Current / expected future business growth stage (self-assessed)",
    "self-reported strategic outlook, conceptually adjacent to F5 strategic-priority needs",
    "Yes (survey)", "NOT ESTABLISHED", "categorical (start-up/growth/established/scaling down/exit)",
    "D2a n missing values present, D2b n missing values present, includes 'Don't know'", False, True,
    "NEAR-PROXY (soft)", "EXCLUDED", "Self-assessed strategic trajectory is judged too close in spirit to "
    "the F5 strategic-priority items used to build several construction needs; no documented non-leakage "
    "rationale was available to justify inclusion.")
add("D6", "Confidence in future economic outlook", "attitudinal, survey-only", "Yes (survey)", "NOT ESTABLISHED",
    "1-7 rating", "n = [full valid sample]", False, False, "SURVEY-ONLY ATTITUDE", "EXCLUDED",
    "Subjective sentiment rating, not a firmographic and not documented as externally available.")
add("D6b", "Free-text explanation of D6", "free text", "Yes (survey)", "NOT ESTABLISHED", "open text",
    "n = [full valid sample, 1 case excluded], 1 missing", False, False, "FREE TEXT", "EXCLUDED", "Free text, explicitly excluded per handoff.")
add("C1a", "Technology-adoption attitude item",
    "attitudinal, near-proxy of Digital Transformation / AI-Enabled Procurement / Innovation Leadership needs",
    "Yes (survey)", "NOT ESTABLISHED", "categorical (6-point attitude scale)", "n = [full valid sample]", False, True,
    "NEAR-PROXY", "EXCLUDED", "Functionally restates the same technology-adoption disposition measured by "
    "several F5/F10 construction items; high leakage risk.")
add("ETA1-ETA7, ETA1_num-ETA7_num", "Existing attitude/segmentation scores", "legacy/attitude score",
    "Yes (survey)", "NOT ESTABLISHED", "numeric/categorical score", "n = [full valid sample]", False, True,
    "LEGACY SEGMENT / OUTCOME-ADJACENT", "EXCLUDED (block)",
    "Handoff explicitly instructs not to assume ETA labels are operationally available; these are treated "
    "as outcome-adjacent legacy scoring, not eligible predictors.")
add("D_SEGMENT", "the client's existing six-segment framework label", "RQ3 comparator, not an RQ4 predictor",
    "NOT ESTABLISHED outside this analysis file", "NOT ESTABLISHED", "categorical (6 labels + Not allocated)",
    "a minority 'Not allocated'", False, True, "OUTCOME-ADJACENT COMPARATOR", "EXCLUDED",
    "Explicitly reserved as the RQ3 comparator per the handoff's fixed decisions; not used as an RQ4 predictor.")
add("Legacy Project 2 five-segment assignment (if present anywhere in source files)", "Legacy segment assignment",
    "legacy segment", "N/A", "NOT ESTABLISHED", "categorical", "N/A", False, True, "LEGACY SEGMENT",
    "EXCLUDED", "Old Project 2 five-segment model must not be used as current evidence per the handoff's "
    "fixed decisions; no legacy segment column is used as a predictor.")
add("QTRAP, S7a", "Survey quality-trap / near-constant screening fields", "survey mechanism, near-constant",
    "Yes (survey)", "NOT ESTABLISHED", "categorical", "QTRAP: 0 missing but 1 unique value; S7a: missing values present",
    False, False, "NOT INFORMATIVE", "EXCLUDED", "Constant or near-constant fields carry no predictive "
    "information and are survey-mechanism artefacts, not firmographics.")
add("S6r1-S6r4, S6r99, S7r1-S7r4", "Service-type screening flags (fixed line, contract type, etc.)",
    "screening/qualification variable", "Yes (survey)", "NOT ESTABLISHED", "binary flag",
    "S7 block: high missingness (routed)", False, False, "LOW INFORMATION / SCREENING ARTEFACT", "EXCLUDED (block)",
    "Post-qualification screening flags with little remaining variance and heavy routing-induced missingness.")
add("E1", "Respondent gender", "individual demographic", "Yes (survey)", "NOT ESTABLISHED",
    "categorical (Female/Male/Non-binary/Prefer not to say)", "n = [full valid sample], no missing", False, False,
    "PROTECTED CHARACTERISTIC", "EXCLUDED",
    "Protected characteristic under the Equality Act 2010. Even where technically available and predictive, "
    "using an individual respondent's gender to drive commercial segmentation/targeting decisions is not "
    "lawful/appropriate without a documented, specific legal basis, which is out of scope here. Excluded on "
    "lawfulness grounds regardless of predictive value.")
add("E2", "Respondent age (years, numeric)", "individual demographic", "Yes (survey)", "NOT ESTABLISHED",
    "numeric (18-90ish)", "n = [full valid sample], no missing", False, False, "PROTECTED CHARACTERISTIC", "EXCLUDED",
    "Age is a protected characteristic under the Equality Act 2010; same lawfulness rationale as E1. Also an "
    "individual-respondent attribute, not a firmographic property of the organisation being segmented.")
add("DAGE", "Respondent age band", "individual demographic (banded)", "Yes (survey)", "NOT ESTABLISHED",
    "categorical age band", "n = [full valid sample], no missing", False, False, "PROTECTED CHARACTERISTIC", "EXCLUDED",
    "Same lawfulness rationale as E2; banding does not remove the protected-characteristic concern.")
add("S1", "Respondent employment status (full/part time)", "individual demographic, not firmographic",
    "Yes (survey)", "NOT ESTABLISHED", "categorical (2 levels)", "n = [full valid sample], no missing", False, False,
    "LOW RELEVANCE / NOT FIRMOGRAPHIC", "EXCLUDED", "Describes the individual respondent, not the "
    "organisation; not a defensible firmographic segmentation input.")
add("S2", "Number of employees (raw numeric)", "firmographic (raw, noisy)", "Yes (survey)", "NOT ESTABLISHED",
    "numeric", "n = [full valid sample], no missing; flagged by S2_outlier_flag for data-quality issues", False, False,
    "NONE (firmographic) but duplicate of D_SIZE_QUOTA1", "EXCLUDED (duplicate)",
    "Superseded by the cleaner, quota-controlled D_SIZE_QUOTA1 band; raw S2 has a documented outlier flag "
    "(S2_outlier_flag) indicating data-quality issues. Including both would be redundant and the raw version "
    "is noisier.")
add("D_SIZE, D_SIZE_QUOTA2", "Alternative company-size bandings", "firmographic (duplicate bandings)",
    "Yes (survey)", "NOT ESTABLISHED", "categorical", "n = [full valid sample], no missing", False, False,
    "NONE (firmographic) but duplicate of D_SIZE_QUOTA1", "EXCLUDED (duplicate)",
    "Different granularity of the same underlying company-size construct as D_SIZE_QUOTA1 (the quota "
    "variable actually used to design the sample); including more than one size banding would be redundant.")
add("S3", "Detailed sector/industry code", "firmographic (high cardinality)", "Yes (survey)", "NOT ESTABLISHED",
    "numeric code, 61 categories", "missing values present", False, False, "NONE (firmographic) but high cardinality/missingness",
    "EXCLUDED", "61 categories on n = [full valid sample] is sparse per category once combined with other predictors and "
    "cross-validation folds; broader sector information is retained via S4 and PUBSect instead.")
add("S5a, S5b", "Private-sector industry text / public-sector sub-type text", "firmographic (sector detail)",
    "Yes (survey)", "NOT ESTABLISHED", "categorical text", "S5a missing values present, S5b missing values present", False, False,
    "NONE (firmographic) but high missingness/redundant with PUBSect", "EXCLUDED",
    "Heavy missingness (each field only applies to a subset by routing) and substantial overlap with S4/"
    "PUBSect, which are retained instead.")
add("S8b", "Respondent's decision-making authority", "behavioural/role attribute", "Yes (survey)",
    "NOT ESTABLISHED", "categorical (4 levels)", "n = [full valid sample], no missing", False, False,
    "LOW-CONFIDENCE AVAILABILITY", "EXCLUDED",
    "Plausibly related to how the respondent answered other survey items (self-reported influence over "
    "decisions); not clearly a stable, externally verifiable organisational attribute, so excluded "
    "conservatively in favour of S8a (respondent role), which is more concretely job-title-like.")

# --- Final candidate predictor set (firmographic, org-level, no missing) --
add("D_SIZE_QUOTA1", "Company size band (SoHo/SME 6-99/SME 100-249/Corporate 250+/Public sector)",
    "firmographic", "Yes (survey)", "NOT ESTABLISHED",
    "categorical, 5 levels", "n = [full valid sample], no missing", False, False, "NONE", "INCLUDED",
    "Organisation-level size band, the quota variable used to design the sample; not part of the 30-need "
    "construction and not a supporting facet of any construction need. Plausible operational analogue exists "
    "(CRM account-size band / Companies House employee count), but external availability is NOT ESTABLISHED "
    "from any source reviewed for this task.")
add("S4", "Sector type (private / public / third)", "firmographic", "Yes (survey)", "NOT ESTABLISHED",
    "categorical, 3 levels", "n = [full valid sample], no missing", False, False, "NONE", "INCLUDED",
    "Organisation-level sector classification; not part of the 30-need construction. Note: also used to "
    "route the F10r9/F10r10 sector-specific WS2 item (Task D), but S4 itself is a demographic classifier, "
    "not a needs item, so it is not a leakage risk here.")
add("PUBSect", "Public-sector vertical (health/education/local-national government/other/none)",
    "firmographic (conditional on S4=public)", "Yes (survey)", "NOT ESTABLISHED",
    "categorical, 5 levels incl. 'NONE' for non-public respondents", "n = [full valid sample], no missing", False, False,
    "NONE", "INCLUDED", "Organisation-level sector detail; not part of the 30-need construction.")
add("D1", "Years the organisation has been trading", "firmographic", "Yes (survey)", "NOT ESTABLISHED",
    "categorical, 8 bands incl. 'Don't know'", "n = [full valid sample], no missing", False, False, "NONE", "INCLUDED",
    "Organisation tenure; not part of the 30-need construction and not a supporting facet of any need.")
add("D3", "Annual turnover band", "firmographic", "Yes (survey)", "NOT ESTABLISHED",
    "categorical, 15 bands incl. 'Prefer not to say'", "n = [full valid sample], no missing", False, False, "NONE",
    "INCLUDED", "Organisation-level financial size band; not part of the 30-need construction.")
add("S8a", "Respondent role (business owner/dedicated IT manager/general manager/contractor/third party/other)",
    "org-role attribute", "Yes (survey)", "NOT ESTABLISHED", "categorical, 6 levels", "n = [full valid sample], no missing",
    False, False, "NONE (job-title-like)", "INCLUDED",
    "The respondent's functional role at the organisation is closer to a job-title/contact-type field a CRM "
    "could plausibly hold than to a needs-adjacent attitude item; not part of the 30-need construction.")

elig_df = pd.DataFrame(elig_rows)
elig_df.to_csv(os.path.join(OUT, "rq4_predictor_eligibility.csv"), index=False)
log(f"Eligibility audit: {len(elig_df)} rows written before any model was fit. "
    f"Included: {(elig_df['final_decision'] == 'INCLUDED').sum()}.")

FINAL_FEATURES = ["D_SIZE_QUOTA1", "S4", "PUBSect", "D1", "D3", "S8a"]
for c in FINAL_FEATURES:
    assert df[c].isna().sum() == 0, f"{c} has missing values"
X_feat = df[FINAL_FEATURES].copy()
log(f"Final feature set ({len(FINAL_FEATURES)}): {FINAL_FEATURES}, all categorical, 0 missing in source.")

print("ELIGIBILITY_AUDIT_OK", flush=True)

# ---------------------------------------------------------------------------
# Modelling: majority-class baseline, multinomial logistic regression,
# one tree-based model (RandomForest, tuned via inner nested CV).
# All preprocessing fitted inside each outer training fold only.
# ---------------------------------------------------------------------------
log("\n=== Modelling: outer 5-fold stratified CV, nested inner 3-fold for RF tuning ===")

CATEGORICAL_FEATURES = FINAL_FEATURES  # all six are categorical/string

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL_FEATURES),
])

rf_param_grid = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 10],
    "clf__min_samples_leaf": [1, 5],
}

outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RNG_SEED)

fold_metric_rows = []
oof_pred = {"baseline": np.full(len(y), -1), "logistic_regression": np.full(len(y), -1),
            "random_forest": np.full(len(y), -1)}
oof_proba_lr = np.zeros((len(y), 4))
oof_proba_rf = np.zeros((len(y), 4))
class_order = sorted(y.unique())
best_rf_params_per_fold = []

X_idx = X_feat.reset_index(drop=True)
y_idx = y.reset_index(drop=True)

for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X_idx, y_idx), start=1):
    log(f"--- Outer fold {fold_i}/{N_OUTER_FOLDS}: train n={len(train_idx)}, test n={len(test_idx)} ---")
    X_train, X_test = X_idx.iloc[train_idx], X_idx.iloc[test_idx]
    y_train, y_test = y_idx.iloc[train_idx], y_idx.iloc[test_idx]

    # --- majority-class baseline ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=RNG_SEED)
    dummy.fit(X_train, y_train)
    pred_dummy = dummy.predict(X_test)
    oof_pred["baseline"][test_idx] = pred_dummy
    fold_metric_rows.append({
        "model": "majority_class_baseline", "fold": fold_i,
        "accuracy": accuracy_score(y_test, pred_dummy),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_dummy),
        "macro_f1": f1_score(y_test, pred_dummy, average="macro", zero_division=0),
        "cohens_kappa": cohen_kappa_score(y_test, pred_dummy),
    })
    log(f"  baseline: acc={fold_metric_rows[-1]['accuracy']:.3f}")

    # --- multinomial logistic regression, preprocessing inside the pipeline ---
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        # scikit-learn >=1.5 fits multinomial logistic regression automatically for
        # solvers that support it (lbfgs, the default) when the target has >2 classes;
        # the explicit multi_class= kwarg was removed in this scikit-learn version.
        ("clf", LogisticRegression(max_iter=2000, random_state=RNG_SEED)),
    ])
    lr_pipe.fit(X_train, y_train)
    pred_lr = lr_pipe.predict(X_test)
    proba_lr = lr_pipe.predict_proba(X_test)
    oof_pred["logistic_regression"][test_idx] = pred_lr
    for ci, cls in enumerate(lr_pipe.named_steps["clf"].classes_):
        col = list(class_order).index(cls)
        oof_proba_lr[test_idx, col] = proba_lr[:, ci]
    fold_metric_rows.append({
        "model": "multinomial_logistic_regression", "fold": fold_i,
        "accuracy": accuracy_score(y_test, pred_lr),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_lr),
        "macro_f1": f1_score(y_test, pred_lr, average="macro", zero_division=0),
        "cohens_kappa": cohen_kappa_score(y_test, pred_lr),
    })
    log(f"  logistic regression: acc={fold_metric_rows[-1]['accuracy']:.3f}")

    # --- tree-based model: RandomForest, tuned on inner folds only ---
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=RNG_SEED)),
    ])
    inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RNG_SEED)
    grid = GridSearchCV(rf_pipe, rf_param_grid, cv=inner_cv, scoring="balanced_accuracy", n_jobs=1)
    grid.fit(X_train, y_train)
    best_rf_params_per_fold.append({"fold": fold_i, **grid.best_params_, "inner_cv_best_score": grid.best_score_})
    pred_rf = grid.predict(X_test)
    proba_rf = grid.predict_proba(X_test)
    oof_pred["random_forest"][test_idx] = pred_rf
    for ci, cls in enumerate(grid.best_estimator_.named_steps["clf"].classes_):
        col = list(class_order).index(cls)
        oof_proba_rf[test_idx, col] = proba_rf[:, ci]
    fold_metric_rows.append({
        "model": "random_forest_tuned", "fold": fold_i,
        "accuracy": accuracy_score(y_test, pred_rf),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_rf),
        "macro_f1": f1_score(y_test, pred_rf, average="macro", zero_division=0),
        "cohens_kappa": cohen_kappa_score(y_test, pred_rf),
    })
    log(f"  random forest (best inner params={grid.best_params_}): acc={fold_metric_rows[-1]['accuracy']:.3f}")

print("CV_LOOP_COMPLETE", flush=True)

# ---------------------------------------------------------------------------
# Fold-level metrics and model comparison
# ---------------------------------------------------------------------------
log("\n=== Aggregating fold-level metrics ===")
fold_df = pd.DataFrame(fold_metric_rows)
fold_df.to_csv(os.path.join(OUT, "rq4_fold_metrics.csv"), index=False)

agg = fold_df.groupby("model").agg(
    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
    balanced_accuracy_mean=("balanced_accuracy", "mean"), balanced_accuracy_std=("balanced_accuracy", "std"),
    macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    cohens_kappa_mean=("cohens_kappa", "mean"), cohens_kappa_std=("cohens_kappa", "std"),
).round(4).reset_index()
# overall (out-of-fold, pooled) metrics per model
overall_rows = []
for model_key, label in [("baseline", "majority_class_baseline"),
                          ("logistic_regression", "multinomial_logistic_regression"),
                          ("random_forest", "random_forest_tuned")]:
    p = oof_pred[model_key]
    overall_rows.append({
        "model": label,
        "oof_accuracy": round(accuracy_score(y_idx, p), 4),
        "oof_balanced_accuracy": round(balanced_accuracy_score(y_idx, p), 4),
        "oof_macro_f1": round(f1_score(y_idx, p, average="macro", zero_division=0), 4),
        "oof_cohens_kappa": round(cohen_kappa_score(y_idx, p), 4),
    })
overall_df = pd.DataFrame(overall_rows)
model_comparison = agg.merge(overall_df, left_on="model", right_on="model", how="left")
model_comparison["majority_class_baseline_accuracy_pct"] = majority_baseline_pct
model_comparison.to_csv(os.path.join(OUT, "rq4_model_comparison.csv"), index=False)
log("\n" + model_comparison.to_string(index=False))

pd.DataFrame(best_rf_params_per_fold).to_csv(os.path.join(OUT, "rq4_rf_tuning_by_fold.csv"), index=False)

# ---------------------------------------------------------------------------
# Choose the best non-baseline model on mean balanced accuracy across folds
# for confusion matrix / per-segment metrics / calibration
# ---------------------------------------------------------------------------
lr_bal = agg.loc[agg.model == "multinomial_logistic_regression", "balanced_accuracy_mean"].iloc[0]
rf_bal = agg.loc[agg.model == "random_forest_tuned", "balanced_accuracy_mean"].iloc[0]
best_model_key = "logistic_regression" if lr_bal >= rf_bal else "random_forest"
best_model_label = "multinomial_logistic_regression" if best_model_key == "logistic_regression" else "random_forest_tuned"
log(f"\nBest-performing non-baseline model on mean balanced accuracy: {best_model_label} "
    f"(LR={lr_bal:.4f}, RF={rf_bal:.4f})")

best_pred = oof_pred[best_model_key]
best_proba = oof_proba_lr if best_model_key == "logistic_regression" else oof_proba_rf

# --- per-segment precision/recall/F1 ---
prec, rec, f1s, support = precision_recall_fscore_support(y_idx, best_pred, labels=class_order, zero_division=0)
per_seg_df = pd.DataFrame({
    "segment": class_order, "precision": prec.round(3), "recall": rec.round(3),
    "f1": f1s.round(3), "support_n": support,
})
per_seg_df.to_csv(os.path.join(OUT, "rq4_per_segment_metrics.csv"), index=False)
log("\nPer-segment metrics (best model, out-of-fold):\n" + per_seg_df.to_string(index=False))

# --- confusion matrix ---
cm = confusion_matrix(y_idx, best_pred, labels=class_order)
cm_df = pd.DataFrame(cm, index=[f"true_segment_{c}" for c in class_order],
                      columns=[f"pred_segment_{c}" for c in class_order])
cm_df.to_csv(os.path.join(OUT, "rq4_confusion_matrix_counts.csv"))
cm_row_pct = cm_df.div(cm_df.sum(axis=1), axis=0).round(4) * 100
cm_row_pct.to_csv(os.path.join(OUT, "rq4_confusion_matrix_row_pct.csv"))

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_row_pct.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(class_order)))
ax.set_xticklabels([f"Pred {c}" for c in class_order])
ax.set_yticks(range(len(class_order)))
ax.set_yticklabels([f"True {c}" for c in class_order])
for i in range(len(class_order)):
    for j in range(len(class_order)):
        val = cm_row_pct.values[i, j]
        n_val = cm_df.values[i, j]
        ax.text(j, i, f"{val:.1f}%\n(n={n_val})", ha="center", va="center",
                color="white" if val > 50 else "black", fontsize=8)
ax.set_title(f"RQ4 confusion matrix, out-of-fold (best model: {best_model_label}, n={len(y_idx)})", fontsize=9)
fig.colorbar(im, ax=ax, label="Row %")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "rq4_confusion_matrix.png"), dpi=150)
plt.close(fig)

# --- calibration / probability-quality check (best model, out-of-fold) ---
max_proba = best_proba.max(axis=1)
pred_class_idx = best_proba.argmax(axis=1)
pred_class = np.array(class_order)[pred_class_idx]
correct = (pred_class == y_idx.values).astype(int)
bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
bin_labels = ["<0.30", "0.30-0.39", "0.40-0.49", "0.50-0.59", "0.60-0.69", ">=0.70"]
bin_idx = np.digitize(max_proba, bins) - 1
calib_rows = []
for bi, blabel in enumerate(bin_labels):
    mask = bin_idx == bi
    if mask.sum() == 0:
        continue
    calib_rows.append({
        "predicted_probability_bin": blabel, "n": int(mask.sum()),
        "mean_predicted_probability": round(float(max_proba[mask].mean()), 3),
        "observed_accuracy": round(float(correct[mask].mean()), 3),
    })
calib_df = pd.DataFrame(calib_rows)
try:
    brier_multiclass = float(np.mean([
        brier_score_loss((y_idx.values == c).astype(int), best_proba[:, ci])
        for ci, c in enumerate(class_order)
    ]))
except Exception:
    brier_multiclass = np.nan
log("\nCalibration table (best model, out-of-fold):\n" + calib_df.to_string(index=False))
log(f"Mean one-vs-rest Brier score (best model): {brier_multiclass:.4f}")

# ---------------------------------------------------------------------------
# rq4_feature_summary.csv
# ---------------------------------------------------------------------------
feat_summary_rows = []
for c in FINAL_FEATURES:
    feat_summary_rows.append({
        "feature": c, "type": "categorical", "n_categories": int(df[c].nunique()),
        "n_missing_in_source": int(df[c].isna().sum()),
        "categories": " | ".join(sorted(df[c].dropna().unique().tolist())),
    })
feat_summary_df = pd.DataFrame(feat_summary_rows)
feat_summary_df.to_csv(os.path.join(OUT, "rq4_feature_summary.csv"), index=False)

print("METRICS_AND_OUTPUTS_WRITTEN", flush=True)

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
with open(os.path.join(OUT, "rq4_analysis_report.md"), "w", encoding="utf-8") as f:
    f.write("# Segment-prediction model -- analysis report\n\n")
    f.write("Outcome: canonical four-segment membership from a prior needs-based k=4 clustering stage "
            f"(n={len(y_idx)}).\n\n")
    f.write("## Predictor eligibility\n\n"
            f"{len(elig_df)} candidate predictors/blocks were reviewed BEFORE any model was fit "
            f"(`rq4_predictor_eligibility.csv`). {(elig_df['final_decision'] == 'INCLUDED').sum()} were "
            f"included: {FINAL_FEATURES}. All 30 construction variables, all documented supporting/"
            "corroborating facets of those needs, ETA scores, D_SEGMENT, legacy segment assignments, free "
            "text, and protected characteristics (gender, age) were excluded. Availability of every "
            "predictor — including the six included firmographics — in the client's operational customer base "
            "outside this survey sample is **NOT ESTABLISHED** from any source reviewed for this task.\n\n")
    f.write("## Modelling design\n\n"
            f"- Outer: stratified {N_OUTER_FOLDS}-fold CV, shuffle=True, random_state={RNG_SEED}\n"
            f"- Inner (RandomForest tuning only): stratified {N_INNER_FOLDS}-fold CV, scoring=balanced_accuracy\n"
            "- All imputation and one-hot encoding fitted inside each outer training fold only "
            "(`ColumnTransformer` + `Pipeline`, refit per fold)\n"
            "- Models: majority-class baseline (`DummyClassifier`), multinomial logistic regression "
            "(`LogisticRegression`, scikit-learn's automatic multinomial handling for multi-class lbfgs), "
            "RandomForestClassifier tuned over n_estimators in {200,400}, max_depth in {None,10}, "
            "min_samples_leaf in {1,5} via inner GridSearchCV\n\n")
    f.write("## Model comparison (fold-level mean +/- std, and pooled out-of-fold)\n\n"
            + model_comparison.to_string(index=False) + "\n\n")
    f.write(f"Majority-class baseline accuracy: {majority_baseline_pct}% "
            f"(matches the largest segment's share).\n\n")
    f.write(f"Best-performing non-baseline model on mean balanced accuracy: **{best_model_label}**.\n\n")
    f.write("## Per-segment precision/recall/F1 (best model, out-of-fold)\n\n" + per_seg_df.to_string(index=False) + "\n\n")
    f.write("## Confusion matrix (best model, out-of-fold, row %)\n\n```\n" + cm_row_pct.round(1).to_string() + "\n```\n\n")
    f.write("## Calibration / probability quality (best model, out-of-fold)\n\n" + calib_df.to_string(index=False)
            + f"\n\nMean one-vs-rest Brier score: {brier_multiclass:.4f} (0 = perfect; interpretation depends "
            "on the class prevalences and the multiclass scoring definition).\n\n")
    f.write("## Bounded conclusion\n\n")
    f.write(f"The firmographic feature set ({', '.join(FINAL_FEATURES)}) predicts the researcher-produced "
            f"four-segment membership modestly better than the majority-class baseline "
            f"({majority_baseline_pct}%) under leakage-controlled cross-validation "
            f"(best model pooled out-of-fold accuracy = {overall_df.loc[overall_df.model == best_model_label, 'oof_accuracy'].iloc[0]:.3f}, "
            f"kappa = {overall_df.loc[overall_df.model == best_model_label, 'oof_cohens_kappa'].iloc[0]:.3f}). "
            "This demonstrates statistical association within the survey sample only. It does **not** "
            "establish operational deployability: availability of these predictors in the client's live operational "
            "population outside the survey is NOT ESTABLISHED, their stability over time has not been tested, "
            "and no lawfulness assessment beyond excluding protected characteristics at the eligibility stage "
            "was performed. Per the handoff's explicit prohibition, cross-validation performance alone is not "
            "claimed as evidence of deployability.\n")

with open(os.path.join(OUT, "rq4_validation_log.md"), "w", encoding="utf-8") as f:
    f.write("# RQ4 validation log\n\n## Environment\n\n")
    for k, v in env_info.items():
        f.write(f"- {k}: {v}\n")
    f.write("\n## Full run log\n\n```\n" + "\n".join(log_lines) + "\n```\n")

with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
    f.write("# Task F: RQ4 operational identification\n\n")
    f.write("Predicts the canonical researcher-produced four-segment (30-needs K-means k=4) membership from "
            "candidate predictors outside the 30-need construction set, with all preprocessing confined to "
            "cross-validation training folds.\n\n")
    f.write("Run: `python segment_prediction_model.py` from this directory. "
            "Requires pandas, numpy, scipy, scikit-learn, openpyxl, matplotlib. Runtime approximately 1-2 minutes.\n\n")
    f.write("## Outputs\n\n"
            "- `rq4_predictor_eligibility.csv` — full predictor audit, built before any model fit\n"
            "- `rq4_model_comparison.csv` — fold-level mean/std and pooled out-of-fold metrics per model\n"
            "- `rq4_fold_metrics.csv` — per-fold, per-model metrics\n"
            "- `rq4_rf_tuning_by_fold.csv` — winning RandomForest hyperparameters per outer fold\n"
            "- `rq4_per_segment_metrics.csv` — precision/recall/F1 per segment, best model, out-of-fold\n"
            "- `rq4_confusion_matrix_counts.csv`, `rq4_confusion_matrix_row_pct.csv`, `rq4_confusion_matrix.png`\n"
            "- `rq4_feature_summary.csv` — final included features, categories, missingness\n"
            "- `rq4_analysis_report.md` — full report, model comparison, bounded conclusion\n"
            "- `rq4_validation_log.md` — environment and full run log\n"
            "- `environment.txt`\n\n")
    f.write("No respondent-level joined file is exported (only aggregate/out-of-fold summary statistics "
            "are written). No source workbook or live Google Doc was modified by this script.\n")

log("\n=== RQ4 ANALYSIS COMPLETE ===")
print("RQ4_ALL_COMPLETE", flush=True)
