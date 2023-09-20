__version__ = '0.4.1'
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

$ python -m airnow_fusion.ozone 2023-06-14T14

Installation:

Example for bash:

cat << EOF > requirements.txt
xarray>=0.16.2
pandas>=1.1.5
numpy>=1.19.5
scipy>=1.5.4
netCDF4>=1.5.8
pyproj>=2.6.1
cfgrib
eccodes==1.2.0
ecmwflibs
git+https://gist.github.com/barronh/08b6fc259e47badd70b9fdcf2b7039f1.git
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
git+https://gist.github.com/barronh/08b6fc259e47badd70b9fdcf2b7039f1.git

%pip install --user -r requirements.txt
'''
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
