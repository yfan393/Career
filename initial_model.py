import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

from tqdm import tqdm

import argparse
from pathlib import Path
import random


ap = argparse.ArgumentParser()
#ap.add_argument('train_data_csv', default="data.csv")
#ap.add_argument('test_data_csv', default="test_data.csv")
ap.add_argument('csv_path')
ap.add_argument('--output_dir', '-o', required=True, help="Model configuration, output CSV etc. will be saved here")
ap.add_argument('--cross_validation', '-cv', type=int, default=0, help="Number of folds for cross-validation. If 0, no cross-validation is performed")
ap.add_argument('--test_ratio', '-t', type=float, default=0.2, help="Only used if cross validating")
ap.add_argument('--test_threshold', default=0.5, help="Model will classify as 1 (== true) if above this threshold. Only used if cross validating.")
ap.add_argument('--random_seed', '-r', type=int, default=-1)
ap.add_argument('--exclude_columns', '-e', nargs='*', default=[])
ap.add_argument('--exclude_id_columns', action='store_true', help="Exclude columns '*_id' columns")
ap.add_argument('--year_to_categorical', action='store_true', help="Convert '*_year' columns to categorical")
ap.add_argument('--use_cuda', action='store_true')
ap.add_argument('--include_unnamed_0', action='store_true',
                help=("If there is a column named 'Unnamed: 0' in your CSV it is likely because in data_analysis.py the CSV was not saved with index=False."
                     " In the original round 1 submission the 'Unnamed: 0' was included (and surprisingly had the second highest feature importance) "
                     "but it will be excluded from hereon unless you set this flag true for obvious reasons"))

args = ap.parse_args()

if args.random_seed > 0:
    rs = args.random_seed
else:
    rs = random.randint(0, 214748364)
R = random.Random(rs)

print("Random state: {}".format(rs))

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

print("Reading CSV ({})".format(args.csv_path))
df = pd.read_csv(args.csv_path)
print("#cols: {}".format(len(df.columns)))
print("#numeric cols: {}".format(len(df.select_dtypes(include=[np.number]).columns)))
print("#object cols: {}".format(len(df.select_dtypes(include=[object]).columns)))

if not args.include_unnamed_0:
    if 'Unnamed: 0' in df.columns:
        print("Dropping the 'Unnamed: 0' column")
        df = df.drop(columns=['Unnamed: 0'])

#df = df.sample(frac=1, random_state=rs).reset_index(drop=True) # Shuffle rows (mainly for CV)
if args.exclude_columns:
    print("Attempting to exclude {} column(s): {}".format(len(args.exclude_columns), args.exclude_columns))
    old_cols_count = len(df.columns)
    for drop_col in args.exclude_columns:
        if drop_col in df.columns:
            df = df.drop(columns=drop_col)
        else:
            print("\tColumn '{}' not found in the dataset".format(drop_col))
    cols_count = len(df.columns)
    print("Dropped {} column(s)".format(old_cols_count - cols_count))
if args.exclude_id_columns:
    print("Excluding '*_id' columns")
    id_cols = [c for c in df.columns if c.endswith('_id')]
    print("Found {} ID columns: {}".format(len(id_cols), id_cols))
    df = df.drop(columns=id_cols, axis='columns')
if args.year_to_categorical:
    print("Converting *_year columns to categorical")
    year_cols = [c for c in df.columns if c.endswith('_year')]
    print("Found {} year columns: {}".format(len(year_cols), year_cols))
    df[year_cols] = df[year_cols].astype(str)

print("DF: {}".format(df.shape))

print("Preprocessing data")
X = df.drop(columns=['preventive_visit_gap_ind', 'id'], axis=1)

# Handling missing data
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# Filling categorical columns with the most frequent value
categorical_cols = [c for c in X.select_dtypes(include=['object']).columns if c != 'data_type']
print("Found {} categorical columns: {}".format(len(categorical_cols), categorical_cols))
X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

# Encoding categorical features
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

train_msk = X['data_type'].str.endswith("Training")
test_msk = X['data_type'].str.endswith("Holdout")
X_train = X[train_msk].drop('data_type', axis=1)
X_test = X[test_msk].drop('data_type', axis=1)
y_train = df[train_msk]['preventive_visit_gap_ind'].astype(np.int32) # Boolean -> Integer

# Model
model_params = dict(
    objective='binary:logistic', 
    eval_metric='auc',  
    max_depth=10,
    learning_rate=0.05,
    n_estimators=500,
    random_state=rs
)
if args.use_cuda:
    print("Using GPU")
    model_params['device'] = 'cuda'

n_cvs = args.cross_validation
if n_cvs > 0:
    n_test_cv = int(len(X_train) * args.test_ratio)
    n_train_cv = len(X_train) - n_test_cv
    print("Cross-validating ({} folds; #train={}, #test={} ({:.0f}%))".format(n_cvs, n_train_cv, n_test_cv, args.test_ratio * 100))
    cv_scores = []
    for i_cv in range(n_cvs):
        print("CV fold {}/{}".format(i_cv+1, n_cvs))
        cv_train_mask = np.ones(len(X_train), dtype=bool)
        cv_train_mask[i_cv*n_test_cv:min(len(X_train), (i_cv+1)*n_test_cv)] = False
        cv_test_mask = ~cv_train_mask

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train[cv_train_mask].to_numpy(), y_train[cv_train_mask].to_numpy())
        pred = model.predict_proba(X_train[cv_test_mask].to_numpy())[:, 1]

        pred_class = pred > args.test_threshold
        acc_score_cv = accuracy_score(y_train[cv_test_mask], pred_class)

        auc_score_cv = roc_auc_score(y_train[cv_test_mask], pred, average='macro')

        print("\tAccuracy (thresh={:.2f}): {:.3f}; AUC: {:.3f}".format(args.test_threshold, acc_score_cv, auc_score_cv))
        cv_scores.append({'accuracy': acc_score_cv, 'auc': auc_score_cv})
    cv_scores_avg = {}
    for k in cv_scores[0].keys():
        cv_scores_avg[k] = np.mean([cv[k] for cv in cv_scores])
    print("Average scores from CV:")
    for k in cv_scores[0].keys():
        print("\t{}: {:.3f}".format(k, cv_scores_avg[k]))

# Train submission model
print("Start training submission model")
model = xgb.XGBClassifier(**model_params)

model.fit(X_train, y_train)

model_save_path = output_dir / "model.json"
model.save_model(model_save_path)
print("Saved trained model to: {}".format(model_save_path))

print("Getting feature importance...") # https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost
feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
feat_df = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
feat_df.to_csv(output_dir / "feature_importance.csv")
ax = feat_df.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features
ax.get_figure().savefig(output_dir / "feature_importance.png")

print("Start running on holdout")
y_pred_proba2 = model.predict_proba(X_test.to_numpy())
assert len(y_pred_proba2.shape) == 2 and y_pred_proba2.shape[1] == 2
score_1 = y_pred_proba2[:, 1] # We want scores for class 1 (1 == True)

results = pd.DataFrame({'ID': df[test_msk]['id'].to_numpy(), 'SCORE': score_1})
results = results.sort_values(by='SCORE', ascending=False).reset_index(drop=True)
results['RANK'] = results.index + 1

results_path = output_dir / "results.csv"
results.to_csv(results_path, index=False)
print("Saved output to: {}".format(results_path))
