import numpy as np
import pandera as pa
from pandera.typing import Series


class MyDataFrameSchema(pa.SchemaModel):
    A: Series[np.int32] = pa.Field(ge=1, le=100)
    B: Series[np.float16] = pa.Field(ge=0, le=17)
    C: Series[str] = pa.Field(isin=["X", "Y", "Z"])
