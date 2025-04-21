import numpy as np
import numpy.typing as npt

def dtw(distances: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...
def dtw_batch(
    distances: npt.NDArray[np.float32],
    sx: npt.NDArray[np.int64],
    sy: npt.NDArray[np.int64],
    symmetric: bool,
) -> npt.NDArray[np.float32]: ...
