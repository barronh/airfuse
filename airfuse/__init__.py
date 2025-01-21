import logging

__version__ = '0.8.1'
changelog = '''see CHANGELOG.md on github'''

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
