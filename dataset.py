import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class WorksitesDataset(Dataset):

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    SPLIT_FRACTIONS = {
        TRAIN: 0.7,
        VAL: 0.1,
        TEST: 0.2,
    }

    def __init__(
        self, split, label_columns, csv_path, scale_features=True, feature_scaler=None,
    ):
        """
        Class constructor.
        Args:
            - split: the split that this WorksitesDataset instance will process
            - labels_column: the labels column name
            - csv_path: the path to the dataset file
            - scale_features: whether to scale the features or not
            - feature_scaler: a scikit-learn scaler to scale the features, if None a MinMaxScaler is instantiated
        """
        # Sanity check
        assert split in (self.TRAIN, self.VAL, self.TEST)

        self._csv_path = csv_path
        self._split = split
        self._label_columns = label_columns
        self._feature_scaler = feature_scaler
        self._scale_features = scale_features
        self._df = None
        self._features_df = None
        self._labels_df = None

        self._load_dataset()

    def _load_dataset(self):
        df = pd.read_csv(self._csv_path)

        # Get the dataset split
        df = df.sort_values(
            by=["YEAR", "MONTH", "WORKSITE_ID"], ascending=[True, True, True]
        )
        train_idxs = (0, int(self.SPLIT_FRACTIONS[self.TRAIN] * len(df)))
        val_idxs = (
            train_idxs[1],
            train_idxs[1] + int(self.SPLIT_FRACTIONS[self.VAL] * len(df)),
        )
        test_idxs = (
            val_idxs[1],
            val_idxs[1] + int(self.SPLIT_FRACTIONS[self.TEST] * len(df)),
        )
        splits_idxs = {
            self.TRAIN: train_idxs,
            self.VAL: val_idxs,
            self.TEST: test_idxs,
        }
        split_idxs = splits_idxs[self._split]
        split_df = df.iloc[split_idxs[0] : split_idxs[1]].copy()

        split_df = self._preprocess_df(split_df)
        (
            self._features_df,
            self._labels_df,
            self._feature_scaler,
        ) = self._df_to_features_labels(
            df=split_df,
            label_columns=self._label_columns,
            scale_features=self._scale_features,
            feature_scaler=self._feature_scaler,
        )

        self._df = split_df

    @staticmethod
    def _df_to_features_labels(
        df, label_columns, scale_features, feature_scaler=None,
    ):
        """
        Creates features and labels of a DataFrame.
        Args:
            - df: a DataFrame
            - label_columns: name of the DataFrame columns that are labels
            - scale_features: whether to scale the features or not
            - feature_scaler: a scikit-learn scaler to scale the features, if None a MinMaxScaler is instantiated
        Returns:
            - features: a DataFrame with features
            - labels: a Series with labels
            - feature_scaler: the scikit-learn scaler applied onto the features
        """
        # Get features.
        # Also, get `labels_columns` values which will be used when training the model.
        features = df[
            [
                "NUM_FACILITIES",
                "DAYS_SINCE_LAST_INSPECTION",
                "STOPPED_BY_SANCTION",
                "PENDING_ACTIONS",
                "NO_TIME_LOST_COUNT",
                "TIME_LOST_COUNT",
                "FATAL_COUNT",
                "FATAL_TIME_LOST_ACCIDENTS_COUNT",
                "TOTAL_ACCIDENTS_COUNT",
                "HOURS_WORKED",
                "ACCIDENTS_RATE",
            ]
        ].copy()
        labels = df[label_columns].copy()

        # Build dummy variables for MONTH
        df["MONTH"] = pd.Categorical(df["MONTH"], categories=range(1, 13))
        dummy_month = pd.get_dummies(df["MONTH"], prefix="MONTH")
        features = features.join(dummy_month)

        # Build HAS_NEVER_BEEN_INSPECTED variable
        has_never_been_inspected = ~np.isfinite(features["DAYS_SINCE_LAST_INSPECTION"])
        features["HAS_NEVER_BEEN_INSPECTED"] = has_never_been_inspected.astype(int)
        features.loc[has_never_been_inspected, "DAYS_SINCE_LAST_INSPECTION"] = 0

        # Scale features to the same numeric range, preserving values distribution
        if scale_features:
            if feature_scaler is None:
                feature_scaler = MinMaxScaler()
                features[features.columns] = feature_scaler.fit_transform(
                    features[features.columns]
                )
            else:
                features[features.columns] = feature_scaler.transform(
                    features[features.columns]
                )

        return features, labels, feature_scaler

    @staticmethod
    def _preprocess_df(df):
        # Fill None and NaNs with sensible values
        df["LAST_INSPECTION_DATE"].fillna(-np.inf, inplace=True)
        df["LAST_INSPECTION_YEAR"].fillna(-np.inf, inplace=True)
        df["DAYS_SINCE_LAST_INSPECTION"].fillna(np.inf, inplace=True)
        # df["DAYS_UNTIL_NEXT_ACCIDENT"].fillna(np.inf, inplace=True)
        df["HOURS_WORKED"].fillna(0, inplace=True)

        # Create new columns with accidents sums
        df["TOTAL_ACCIDENTS_COUNT"] = (
            df["NO_TIME_LOST_COUNT"] + df["TIME_LOST_COUNT"] + df["FATAL_COUNT"]
        )
        df["FATAL_TIME_LOST_ACCIDENTS_COUNT"] = (
            df["TIME_LOST_COUNT"] + df["FATAL_COUNT"]
        )

        return df

    def __getitem__(self, idx):
        features = self._features_df.iloc[idx].values
        classifier_target, regressor_target = self._labels_df.iloc[idx]
        return {
            "features": torch.tensor(features),
            "classifier_target": torch.tensor(classifier_target > 0, dtype=torch.int64),
            "regressor_target": torch.tensor(regressor_target),
        }

    def __len__(self):
        return len(self._features_df)