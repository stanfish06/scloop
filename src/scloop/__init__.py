import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

from . import data
from . import utils
from . import preprocessing as pp
