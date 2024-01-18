import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_postprocess(
    model: any, data_uri: str, scaler_uri: StandardScaler
) -> pd.DataFrame:
    data = pd.read_parquet(data_uri)
    scaler = joblib.load(scaler_uri)
    X = scaler.transform(data)
    prediction = model.predict(X)
    data["target"] = scaler.inverse_transform(prediction)
    return data
