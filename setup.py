import setuptools

short_description = (
    "airfuse provides several data fusion techniques designed to work with"
    + " AirNow, PurpleAir, NOAA's Air Quality Forecast and NASA's Composition "
    + " Forecast."
)
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("airfuse/__init__.py", "r") as fh:
    for l in fh:
        if l.startswith('__version__'):
            exec(l)
            break
    else:
        __version__ = 'x.y.z'

setuptools.setup(
    name="airfuse",
    version=__version__,
    author="Barron H. Henderson",
    author_email="barronh@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/barronh/airfuse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "xarray>=0.16.2", "pandas>=1.1.5", "numpy>=1.19.5", "scipy>=1.5.4",
        "netCDF4>=1.5.8", "pyproj>=2.6.1",
        # optionally required for reading real-time grib files.
        # "cfgrib", "eccodes==1.2.0", "ecmwflibs"
    ],
)
