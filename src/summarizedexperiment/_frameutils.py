from biocframe import BiocFrame, from_pandas

from .type_checks import is_pandas

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _sanitize_frame(frame, num_rows: int):
    frame = frame if frame is not None else BiocFrame({}, number_of_rows=num_rows)

    if is_pandas(frame):
        frame = from_pandas(frame)

    return frame
