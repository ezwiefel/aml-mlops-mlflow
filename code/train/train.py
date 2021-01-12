from argparse import ArgumentParser

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        required=True, help="The path to read the data files from")
    parser.add_argument("--random-seed", type=int,
                        default=42, help="The random seed to use for data splitting and XGBoost")
    parser.add_argument('--colsample-bytree', type=float, default=1.0,
                        help='subsample ratio of columns when constructing each tree (default: 1.0)')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample ratio of the training instances (default: 1.0)')
    return parser.parse_args()


def main(data_path, random_seed, colsample_bytree, subsample):
    """Main Training Loop"""
    with mlflow.start_run():
        df = pd.read_parquet(data_path)

        X = df.copy()
        y = pd.DataFrame(X.pop('target'))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # train model
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mlogloss',
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'seed': random_seed,
        }

        # Log the model after training
        mlflow.xgboost.autolog()
        model = xgb.train(params, dtrain)

        # evaluate model
        y_pred = model.predict(dtest)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        # Log specific evaluation metrics
        mlflow.log_metrics({'val_r2': r2, 'val_rmse': rmse, 'val_mae': mae})


if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.data_path,
         random_seed=args.random_seed,
         subsample=args.subsample,
         colsample_bytree=args.colsample_bytree
         )
