import quantities as pq
from typing import (
    Any,
    Union,
    Optional
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    field_serializer
)
import neo
from enum import Enum
import elephant
import scipy.sparse as sp

import elephant.schemas.field_validator as fv
import elephant.schemas.field_serializer as fs

from elephant.spike_train_surrogates import SURR_METHODS

class PydanticSurrogates(BaseModel):
    """
    PyDantic Class to wrap the `elephant.spike_train_surrogates.surrogates` function
    with additional type checking and JSON schema generation.
    """
    pass