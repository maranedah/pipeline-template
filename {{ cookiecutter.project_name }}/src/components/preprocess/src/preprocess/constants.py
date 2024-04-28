from Aggregator import Aggregator
from Encodings import Encodings
from FeatureReduction import FilterColumns, FilterCorrelatedColumns
from Scaler import Scaler
from TypeHandling import TypeHandling, TypeOptimization

ignore_columns = [
    "case_id",
    "date_decision",
    "MONTH",
    "WEEK_NUM",
    "target",
    "num_group1",
    "num_group2",
]

get_encodings = Encodings(
    weekday=False, day_of_month=False, day_of_year=False, date_as_unix=True
)
type_handling = TypeHandling(logging=True, ignore_columns=ignore_columns)
type_optimization = TypeOptimization(logging=True, ignore_columns=ignore_columns)

aggregate_rows = Aggregator(key_column="case_id", ignore_columns=ignore_columns)

filter_columns = FilterColumns(logging=True, ignore_columns=ignore_columns)
filter_columns_lower_threshold = FilterColumns(
    logging=True, ignore_columns=ignore_columns, threshold=0.8
)


filter_correlated_columns = FilterCorrelatedColumns(
    threshold=0.8, ignore_columns=ignore_columns
)

scaler = Scaler(ignore_columns=ignore_columns)
