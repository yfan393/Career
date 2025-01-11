# Humana Case Comp 2024
- https://mays.tamu.edu/humana-tamu-analytics/
- Submit here: https://mays.tamu.edu/humana-tamu-analytics/leaderboard-submission/

## Installation
### Requirements
Python 3.10
- `SQLAlchemy` (2.0.35)
- `pandas` (2.2.3)
- `scikit-learn` (1.5.2)
- `xgboost` (2.1.1)
- `numpy` (2.1.1)
- `jupyterlab` (4.2.5)
- `tqdm` (4.66.5)
- `matplotlib` (3.9.2)

## Operations
### Preprocessing
Create SQLite db from given CSVs (see `preprocess.ipynb` for data exploration):
```python
# See populate_db_test.sh
export DB_CONNECTION_STRING=sqlite:///out/humana.sqlite
python3 db.py -T "../Training_final/" -H "../Holdout_final/"
```
Table `member_id` contains the member ids and their dataset of origni (Training or Holdout).
Other tables are derived from raw dataset CSVs.

Derive CSV from SQLite DB (TODO handle tables where there exist multiple rows per member_id):
```sh
# See dump_to_csv_test.sh
OUTPATH=out/datadump.csv
python3 -m data_analysis dump_to_csv -o "$OUTPATH" -t 10
```

Run `data_analysis.ipynb` to include onetomany columns.

### Model training
Run initial model (train/test):
```sh
# Path to CSV dump created in preprocessing step, containing both training AND test (holdout) data.
DATAPATH=out/datadump.csv
python3 initial_model.py "$DATAPATH" -o out/results.csv
```

Run `traintest_model` to get cross-validation results as well.