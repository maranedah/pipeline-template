import numpy as np
import pandas as pd


def get_pandera_defaults(schema):
    checks_dict = {}
    annotation_default = schema._collect_fields()
    for arg_name, arg_value in annotation_default.items():
        annotation_info, field_info = arg_value
        checks_dict.update({arg_name: {"type": annotation_info.arg, "checks": {}}})
        for check in field_info.checks:
            checks_dict[arg_name]["checks"].update(check._check_kwargs)
    return checks_dict


def get_schema_sample(schema, size):
    defaults = get_pandera_defaults(schema)
    rng = np.random.default_rng()
    data = {}
    for arg, value in defaults.items():
        type = value["type"]
        checks = value["checks"]
        if issubclass(type, np.integer) or issubclass(type, int):
            value = rng.integers(
                low=checks["min_value"],
                high=checks["max_value"],
                size=size,
                dtype=type,
            )
        elif issubclass(type, np.floating) or issubclass(type, float):
            value = rng.uniform(
                low=checks["min_value"],
                high=checks["max_value"],
                size=size,
            ).astype(type)
        elif issubclass(type, str):
            value = rng.choice(list(checks["allowed_values"]), size=size, replace=True)
        data.update({arg: value})
    return pd.DataFrame(data)
