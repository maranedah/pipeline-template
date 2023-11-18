import pandera as pa
from pandera.typing import Series


class PalmerPenguinsSchema(pa.SchemaModel):
    species: Series[str] = pa.Field(isin=["Adelie", "Gentoo", "Chinstrap"])
    island: Series[str] = pa.Field(isin=["Torgersen", "Biscoe", "Dream"])
    bill_length_mm: Series[float] = pa.Field(ge=32.1, le=59.6, nullable=True)
    bill_depth_mm: Series[float] = pa.Field(ge=13.1, le=21.5, nullable=True)
    flipper_length_mm: Series[float] = pa.Field(ge=172, le=231, nullable=True)
    body_mass_g: Series[float] = pa.Field(ge=2700, le=6300, nullable=True)
    sex: Series[str] = pa.Field(isin=["male", "female"], nullable=True)
    year: Series[int] = pa.Field(ge=2007, le=2009)
