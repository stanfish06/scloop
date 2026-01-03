import warnings
from types import ModuleType

from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="colorspacious")

pl: ModuleType | None = None
try:
    from . import plotting as pl  # noqa: F401
except ImportError as e:
    logger.warning(f"Could not import plotting modules: {e}")

pp: ModuleType | None = None
try:
    from . import preprocessing as pp  # noqa: F401
except ImportError as e:
    logger.warning(f"Could not import preprocessing modules: {e}")

tl: ModuleType | None = None
try:
    from . import tools as tl  # noqa: F401
except ImportError as e:
    logger.warning(f"Could not import tools modules: {e}")

_io_module: ModuleType | None = None
try:
    from . import io as _io_module  # noqa: F401
    from .io import load_scloop, save_scloop  # noqa: F401
except Exception as e:
    logger.warning(f"Could not import io modules: {type(e).__name__}: {e}")
io = _io_module

__all__ = [pkg for pkg in ["pl", "pp", "tl", "io"] if globals().get(pkg) is not None]
