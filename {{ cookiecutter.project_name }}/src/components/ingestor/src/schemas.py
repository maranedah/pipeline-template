import pandera as pa
from pandas import Series

class MyDataFrameSchema(pa.SchemaModel):
    A: Series[int] = pa.Field(
        ge=1,
        le=100
    )
    B: Series[float] = pa.Field(
        ge=0,
        le=10
    )
    C: Series[str] = pa.Field(
        isin=["X", "Y", "Z"]
    )