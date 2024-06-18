import logging

__version__ = '0.7.4'
changelog = '''
* 0.1.0: functioning
* 0.2.0: checks for invalid aVNA_AN and aVNA_PA and updates weights accordingly
* 0.3.0: Identical results for PM from previous run, but then updated to only
         use sites with valid model results in eVNA and aVNA.
* 0.4.0: added functionality with GEOS-CF and PurpleAir only fusion.
         moved ozone.py, pmpaonly.py, and pmanonly.py to drivers.py
         pa.py is retained because it applied pm from AirNow and PurpleAir
* 0.4.1: switched default AirNow data source to be AirNow API (<2d old) or
         AirNow Files (>=2d old)
* 0.4.2: updated logging to capture all logged messages during pm.py execution.
* 0.5.0: Adding GOES capability as an observation dataset.
* 0.5.1: * Updated obs/epa readers to require valid observation as well as mod.
         * Updated pm driver to set a minimum distance of half a grid cell for
           PurpleAir for the purpose of fusing data. This prevents PurpleAir
           from every being closer to the prediction cell centroid than an
           AirNow monitor when they are in the same cell.
         * Updated get_model to apply bbox to NAQFC; allows spatial subsetting
           of the target domain.
         * Update to pull NAQFC based on end hour.
* 0.5.2: * Added NetCDF formatted output option.
* 0.5.3: * Updated to_geopandas for backward matplotlib compatibility
         * Updated to_geojson for convenience
* 0.6.0: * Updated so that the default NAQFC is the KFAN product for both PM
           and ozone. For archived dates, the KFAN is not available. The system
           will switch to non-KFAN (raw forecast).
         * Improved backward compatibility for matplotlib versions (only
           affects) to_geopandas and to_geojson
* 0.6.1: * Updates for windows compatibility with NOAA forecast temp files.
* 0.6.2: * New ANT color palettes and added metadata in nc and geojson files.
* 0.7.0: * Added capability for AirNow API to use "mobile monitors" (default).
         * Fixed pair_airnowaqobsfile and pair_airnowhourlydatafile (unused)
           and added testing for them.
* 0.7.1: * Updated some netcdf metadata options.
         * Added NDGD_HISTORICAL environmental variable. If set to True, true,
           Yes, or Y, then the NAQFC modeling from the THREDDS catalog will
           be loaded from the historical subfolder. To see the historical
           data holdings, go to the link below and choose historical
           www.ncei.noaa.gov/thredds/catalog/model-ndgd-file/catalog.html
* 0.7.2: * Changed default NOAA server from ncep to nomads.
* 0.7.3: * Added initialization hour to ncep and nomads file structure.
* 0.7.4: * Fixed install_requires and requirements.txt
'''

__doc__ = '''
Overview
========

The package contains a proposed spatial fusion method for AirNow.

* Realtime AirNow data expects a ~/.airnowkey file that contains the user's
  AirNow API key. The file should have read permissions for only the user.
* PurpleAir expects a ~/.purpleairkey file that contains the user's PurpleAir
  API key. The file should have read permissions for only the user.

Usage
=====

Usage as a script
-----------------

```bash
python -m airfuse.pm 2023-06-14T14
```

or

```bash
python -m airfuse.drivers -s ozone 2023-06-14T14
```


Usage as a module
-----------------

```python
from airfuse.drivers import fuse
import pandas as pd

date = pd.to_datetime('2023-08-24T18Z')
outpaths = fuse('airnow', 'o3', date, 'naqfc', (-97, 25, -67, 50))
print(outpaths)
```

or

```python
from airfuse.pm import pmfuse
import pandas as pd

date = pd.to_datetime('2023-08-24T18Z')
outpaths = pmfuse(date, 'naqfc', (-97, 25, -67, 50))
print(outpaths)
```


Installation
============

Prerequisites
-------------

 * xarray, pandas, numpy, scipy
 * for NCEI archvied files (past): netCDF4
 * for NWS or NCEP last two day files: cfgrib, eccodes, ecmwflibs
 * ~/.airnowkey and ~/.purpleairkey with keys for data access

Example bash Installation
-------------------------

```bash
cat << EOF > requirements.txt
dask[array]
dask[dataframe]
xarray>=2023.11.0
pandas>=1.1.5
numpy>=1.19.5
scipy>=1.5.4
netCDF4>=1.5.8
pyproj>=2.6.1
cfgrib
eccodes==1.2.0
ecmwflibs
pyrsig
git+https://github.com/barronh/nna_methods.git
git+https://github.com/barronh/airfuse.git
EOF
pip install --user -r requirements.txt
```

Example for JupyterNotebook
---------------------------

# In[1]
%writefile requirements.txt
dask[array]
dask[dataframe]
xarray>=2023.11.0
pandas>=1.1.5
numpy>=1.19.5
scipy>=1.5.4
netCDF4>=1.5.8
pyproj>=2.6.1
cfgrib
eccodes==1.2.0
ecmwflibs
pyrsig
git+https://github.com/barronh/nna_methods.git
git+https://github.com/barronh/airfuse.git

# In[2]
%pip install --user -r requirements.txt
'''

# Set up logging object for library root
logging.getLogger(__name__).addHandler(logging.NullHandler())
