from importlib.metadata import version

from bcs_rs._core import BcsConnection, decode_z85, decode_z85_parallel
from bcs_rs.server import BCSServer

__version__ = version("bcs-rs")
__all__ = [
    "BCSServer",
    "BcsConnection",
    "decode_z85",
    "decode_z85_parallel",
    "__version__",
]
