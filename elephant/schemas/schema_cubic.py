from typing import (
    Any,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)
import neo


import elephant.schemas.field_validator as fv

class PydanticCubic(BaseModel):
    """
    PyDantic Class to wrap the elephant.cubic.cubic function
    with additional type checking and json_schema by PyDantic.
    """
    histogram: Any = Field(..., description="Population histogram of entire population")
    max_interations: int = Field(100, gt=0, description="Maximum number of iterations")
    alpha: float = Field(0.05, ge=0, le=1, description="Significance level")

    @field_validator("histogram")
    @classmethod
    def validate_neo_AnalogSignal(cls, v, info):
        return fv.validate_type(v, info, allowed_types=(neo.AnalogSignal,), allow_none=False)