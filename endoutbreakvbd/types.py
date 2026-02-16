from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

IntArray: TypeAlias = NDArray[np.signedinteger[Any]]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
GenTimeInput: TypeAlias = list[float] | FloatArray
IncidenceSeriesInput: TypeAlias = list[int] | IntArray
IncidenceInitInput: TypeAlias = int | list[int] | IntArray | None
PercRiskThresholdInput: TypeAlias = int | float | IntArray | FloatArray
RepNoOutput: TypeAlias = int | float | np.number[Any] | IntArray | FloatArray
RepNoFunc: TypeAlias = Callable[[int | IntArray], RepNoOutput]
