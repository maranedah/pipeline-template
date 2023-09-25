import logging

import lightgbm as lgb 
import pandas as pd

def run_model(gcs_bucket: str) -> tuple[lgb.LGBMRegressor, dict[str, int | float | str], pd.DataFrame, str]:
    logging.info("Loading Data...")