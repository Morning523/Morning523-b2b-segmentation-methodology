# Input data (not included)

No real respondent data, the original survey instrument (question wording), or any real analysis output from the underlying client engagement is included in this repository, by design. The client did not grant permission to publish its data or survey instrument, and this repository is a methodology demonstration, not a reproduction of the original project.

Expected local inputs, if you have your own comparable dataset and permission to use it:

- `projects/b2b-needs-based-segmentation/r/*.Rmd` expect a respondent-level survey extract (`client_survey_data.xlsx`) and a value-label lookup (`value_label_map.csv`) in the same folder as the script.
- `projects/segment-prediction-model/python/segment_prediction_model.py` expects `data/survey_data_cleaned.csv` (respondent-level) and `data/needs_master_table.csv` (item-to-need mapping) under its own folder.

None of these files are provided. The scripts are included to show the analytical approach (factor reduction, clustering, leakage-controlled predictive modelling), not to be run against the original data.
