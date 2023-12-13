import logging

__version__ = '0.5.2'
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
'''

__doc__ = '''
Contains proposed spatial fusion method for AirNow.

Requires:
 * xarray, pandas, numpy, scipy
 * for NCEI archvied files (past): netCDF4
 * for NWS or NCEP last two day files: cfgrib, eccodes, ecmwflibs
Usage as a script:
$ python -m airnow_fusion.pm 2023-06-14T14

or

$ python -m airnow_fusion.drivers -s ozone 2023-06-14T14

Installation:

Example for bash:

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
git+https://github.com/barronh/nna_methods.git
EOF
pip install --user -r requirements.txt

Example for JupyterNotebook using two cells.

%writefile requirements.txt
xarray>=0.16.2
pandas>=1.1.5
numpy>=1.19.5
scipy>=1.5.4
netCDF4>=1.5.8
pyproj>=2.6.1
cfgrib
eccodes==1.2.0
ecmwflibs
git+https://gist.github.com/barronh/nna_methods.git

%pip install --user -r requirements.txt
'''

# Set up logging object for library root
logging.getLogger(__name__).addHandler(logging.NullHandler())
