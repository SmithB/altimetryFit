import os as _os
import sys as _sys
from .set_threads import set_threads
# OpenBLAS and OpenMP read their thread-count environment variables only when
# numpy first loads them, so scripts that take --THREADS set them here, before
# the imports below pull in numpy.
if _sys.argv and _os.path.basename(_sys.argv[0]) in \
        ['fit_altimetry.py', 'fit_DEM_bias.py']:
    set_threads()
#from .check_obj_memory import check_obj_memory
from .read_optical import read_optical_data, laser_key
#from .reread_data_from_fits import reread_data_from_fits
from .check_ATL06_hold_list import *
from .im_subset import *
from .read_ICESat2 import *
from .get_pgc import get_pgc
from .est_DEM_jitter_AT import *
from .SMB_corr_from_grid import SMB_corr_from_grid
from .remove_overlapping_DEM_data import remove_overlapping_DEM_data
from .fit_altimetry import fit_altimetry
from .apply_tides import apply_tides
from .pgc_2m_dem import pgc_2m_dem
from .surfaceModel import surfaceModel
from  .tile_names import tile_centers_from_files
from  .tile_names import tile_centers_from_scripts
from  .get_pgc import *
from .fit_DEM_bias import fit_DEM_bias
from . import eval_model
from . import register_DEMs
from . import bias_functions 
