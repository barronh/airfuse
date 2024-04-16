# airfuse

AirFuse provides several data fusion techniques designed to work with AirNow,
PurpleAir, NOAA's Air Quality Forecast and NASA's Composition Forecast.

## Overview

The standard driver applies Nearest Neighbor Averaging, Voronoi Neighbor
Averaging (VNA), extended VNA (eVNA), and additive VNA (aVNA). eVNA corrects the
model surface multiplying the ratio of obs:mod. aVNA is like eVNA, except it
corrects teh model surface by subtracting the bias. For both eVNA and aVNA, the
ratio or bias is interpolated from Voronoi neighbors using inverse distance
weights.

This is currently a research product and is provided as-is with no warranty
expressed or implied. Users should be cautious.


## AirFuse Examples

```python
from airfuse.drivers import fuse

date = '2023-08-24T18Z'
pmpaths = fuse(
    obssource='airnow', species='pm25', startdate=date, model='naqfc'
)
o3paths = fuse(
    obssource='airnow', species='o3', startdate=date, model='naqfc'
)
```


## Install

### Using pip

airfuse currently requires the nna_methods package, which is another github
repository. So, installing requires two calls to pip.

```bash
pip install git+https://github.com/barronh/nna_methods.git
pip install git+https://github.com/barronh/airfuse.git
```

### From Downloaded Source

airfuse can also be installed by downloading the source code.

```
wget https://github.com/barronh/airfuse/archive/refs/heads/main.zip
unzip main
cd airfuse-main
pip install -r requirements.txt
pip install .
```

## Feedback

If you have feedback about airfuse, please
[open an issue](https://github.com/barronh/airfuse/issues).