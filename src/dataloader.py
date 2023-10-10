import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple


class DataLoader:
    """
    Load a dataset and prepare it for prediction.

    Args:
    ------
    path_to_data (str): The path to the dataset file.
    time_column (str): The name of the datetime column.
    threshold_cat_feature (int): Threshold for categorical features.
    drop_disbalance (str): Oversampling technique ('SMOTE' or None).

    Methods:
    ------
    get_test_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        Get train and test data.

    """

    def __init__(
        self,
        path_to_data: str,
        time_column: str,
        threshold_cat_feature: int,
        drop_disbalance: str,
    ):
        self.path_to_data = path_to_data
        self.time_column = time_column
        self.threshold_cat_feature = threshold_cat_feature
        self.drop_disbalance = drop_disbalance

    def _load_data(self) -> pd.DataFrame:
        """Load the dataset."""
        df = pd.read_csv(self.path_to_data, delimiter=",")
        return df

    def _get_features_from_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-related features from a datetime column."""
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df["Weekday"] = df[self.time_column].dt.weekday
        df["Quarter"] = df[self.time_column].dt.quarter
        df["DayofYear"] = df[self.time_column].dt.dayofyear
        df["IsWeekend"] = (df[self.time_column].dt.weekday >= 5).astype(int)
        df["IsLeapYear"] = (df[self.time_column].dt.is_leap_year).astype(int)
        df = df.drop(columns=[self.time_column])
        return df

    def _encode_cat_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        columns_ohe: list,
        columns_freq_encoder: list,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply one-hot encoding and frequency encoding to categorical features."""
        X_train, X_test = self._apply_ohe(columns_ohe, X_train, X_test)
        X_train, X_test = self._apply_frequency_encoder(
            columns_freq_encoder, X_train, X_test
        )
        return X_train, X_test

    def _balance_data(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply oversampling (SMOTE) if specified."""
        if self.drop_disbalance == "SMOTE":
            X_train, y_train = self._apply_SMOTE(X_train, y_train)
        return X_train, y_train

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load, clean, and prepare data for prediction."""
        df = self._load_data()
        df = self._get_features_from_time(df)
        columns = df.select_dtypes(include=["object"]).columns
        columns_OHE = [
            col for col in columns if df[col].nunique() <= self.threshold_cat_feature
        ]
        columns_freq_encoder = [
            col for col in columns if df[col].nunique() > self.threshold_cat_feature
        ]

        X, y = df.drop(columns="target"), df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        X_train, X_test = self._encode_cat_features(
            X_train,
            X_test,
            columns_ohe=columns_OHE,
            columns_freq_encoder=columns_freq_encoder,
        )
        X_train, y_train = self._balance_data(X_train, y_train)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _apply_ohe(
        cols: list, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply OneHotEncoder to the dataset."""
        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe.fit(X_train[cols])
        cols_after_ohe = ohe.get_feature_names_out(cols)

        X_train_ohe = pd.DataFrame(
            data=ohe.transform(X_train[cols]).toarray(), columns=cols_after_ohe
        )
        X_test_ohe = pd.DataFrame(
            data=ohe.transform(X_test[cols]).toarray(), columns=cols_after_ohe
        )

        X_train = pd.concat([X_train, X_train_ohe], axis=1)  # Concatenate along columns
        X_test = pd.concat([X_test, X_test_ohe], axis=1)  # Concatenate along columns

        X_train.drop(columns=cols, inplace=True)
        X_test.drop(columns=cols, inplace=True)

        return X_train, X_test

    @staticmethod
    def _apply_frequency_encoder(
        cols: list, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply FrequencyEncoder to the dataset."""
        for col in cols:
            for X in [X_train, X_test]:
                fq = X.groupby(col).size() / len(X)
                X.loc[:, f"{col}_freq_encode"] = X[col].map(fq)
                X.drop([col], inplace=True, axis=1)
        return X_train, X_test

    @staticmethod
    def _apply_SMOTE(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE for oversampling."""
        smote = SMOTE()
        X_train_smote, y_train_smote = smote.fit_resample(X, y)
        return X_train_smote, y_train_smote
