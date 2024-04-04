import logging

import pandas as pd
from MemoryReduction import MemoryReduction
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataProcessing:
    df: pd.DataFrame
    replace_data: list[dict[str, any]]
    rename: dict[str, str]

    def __init__(
        self,
        df: pd.DataFrame,
        remove_rows_where=[],
        remove_columns_where=[],
        replace_data=[],
        rename={},
        encode=[],
    ):
        initial_mem = df.memory_usage().sum() / 1024**2
        logging.info("Optimizing types...")
        memory_reduction = MemoryReduction()
        self.df = memory_reduction.type_optimization(df)
        # self.scale()
        if remove_rows_where:
            logging.info("Removing rows...")
            for setting in tqdm(remove_rows_where):
                self.remove_rows_where(**setting)
        if remove_columns_where:
            logging.info("Removing columns...")
            for setting in tqdm(remove_columns_where):
                self.remove_columns_where(**setting)
        if replace_data:
            logging.info("Replacing data...")
            for setting in tqdm(replace_data):
                self.replace(**setting)
        if encode:
            logging.info("Encoding columns...")
            encode = [
                {"column": column, "encoder": OneHotEncoder(sparse_output=False)}
                for column in self.df.select_dtypes(
                    include=["category", "object"]
                ).columns
            ]
            self.encoders = [self.encode_column(**setting) for setting in tqdm(encode)]
        if rename:
            self.df = self.df.rename(columns=rename)

        logging.info("Reducing dimensionality...")
        self.df = memory_reduction.dimensionality_reduction(self.df)
        logging.info("Scaling...")
        self.scalers = self.scale()
        self.df = self.df.astype("float16")
        final_mem = self.df.memory_usage().sum() / 1024**2
        print(
            f"Memory decreased {round(100 * (initial_mem - final_mem) / initial_mem)}%"
        )

    def fillna(self):
        for column in self.df.select_dtypes(include=["float64"]).columns.tolist():
            min_value = self.df[column].min()
            self.df[column].fillna(min_value - 100000, inplace=True)

    def remove_rows_where(self, columns: list[str], condition: callable):
        for column in columns:
            mask = self.df[column].apply(condition)
            self.df = self.df[~mask].reset_index(drop=True)

    def remove_columns_where(self, condition: callable, ignore_columns: list[str]):
        columns = [x for x in self.df.columns if x not in ignore_columns]
        for col in columns:
            if condition(self.df, col):
                self.df = self.df.drop(columns=[col])

    def replace(self, columns: list[str], condition: callable, new_value: any):
        self.df[columns] = self.df[columns].map(
            lambda x: new_value if condition(x) else x
        )
        return self.df

    def encode_column(self, column, encoder):
        if isinstance(encoder, LabelEncoder):
            self.df[f"encoded_{column}"] = encoder.fit_transform(self.df[column])
        elif isinstance(encoder, OneHotEncoder):
            # breakpoint()
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        encoder.fit_transform(self.df[[column]]).astype(bool),
                        columns=encoder.get_feature_names_out([column]),
                    ),
                ],
                axis=1,
            )

            bool_columns = self.df.select_dtypes(include="bool").columns.tolist()
            for col in bool_columns:
                self.df[col].astype(pd.SparseDtype(dtype=bool, fill_value=False))
        self.df = self.df.drop(columns=[column])
        return {"column": column, "encoder": encoder}

    def scale(self):
        scalers = []
        numeric_columns = self.df.select_dtypes(include="number").columns.tolist()
        for column in numeric_columns:
            scaler = StandardScaler()
            self.df[column] = scaler.fit_transform(
                self.df[column].values.reshape(-1, 1)
            )
            scalers.append(scaler)
        return scalers
