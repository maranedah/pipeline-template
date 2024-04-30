from Aggregator import Aggregator
from Encodings import Encodings
from Scaler import Scaler
from DataTypeOptimizer import PolarsDataTypeOptimizer
from Compose import Compose
from FeatureReduction import FilterColumnsTooCorrelated, FilterColumnsWithManyNulls, FilterColumnsWithOnlyOneValue, FilterColumnsWithTooManyCategories

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

aggregate_rows = Aggregator(key_column="case_id", ignore_columns=ignore_columns)


scaler = Scaler(ignore_columns=ignore_columns)

step_1_processing = Compose(
    transforms=[
        PolarsDataTypeOptimizer(),
        FilterColumnsWithOnlyOneValue(),
        FilterColumnsWithTooManyCategories(threshold=11),
        Encodings(weekday=False, day_of_month=False, day_of_year=False, date_as_unix=True),
        Aggregator(key_column="case_id", ignore_columns=ignore_columns)

    ]
)

step_2_processing = Compose(
    transforms=[
        FilterColumnsWithManyNulls(threshold=0.95),
        FilterColumnsTooCorrelated(threshold=0.9)

    ]
)