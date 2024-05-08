from .Aggregator import Aggregator
from .Compose import Compose
from .DataTypeOptimizer import PolarsDataTypeOptimizer
from .Encodings import Encodings
from .FeatureReduction import (
    FilterColumnsTooCorrelated,
    FilterColumnsWithManyNulls,
    FilterColumnsWithOnlyOneValue,
    FilterColumnsWithTooManyCategories,
)
from .Scaler import Scaler

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
        # FilterColumnsWithOnlyOneValue(ignore_columns=ignore_columns),
        FilterColumnsWithTooManyCategories(threshold=11, ignore_columns=ignore_columns),
        Encodings(
            weekday=False, day_of_month=False, day_of_year=False, date_as_unix=True
        ),
        Aggregator(
            key_column="case_id",
            ignore_columns=ignore_columns,
            mean=True,
            std=True,
            min=False,
            max=False,
        ),
    ]
)

step_2_processing = Compose(
    transforms=[
        FilterColumnsWithOnlyOneValue(ignore_columns=ignore_columns),
        FilterColumnsWithManyNulls(threshold=0.8, ignore_columns=ignore_columns),
        FilterColumnsTooCorrelated(threshold=0.9, ignore_columns=ignore_columns),
    ]
)

step_3_processing = Compose(transforms=[Scaler(ignore_columns=ignore_columns)])
