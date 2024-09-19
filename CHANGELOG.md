Notable updates in reverse chronological order:

* 0.8.0: * Allow for an optional "mask" variable in NAQFC coordinate files.
         * Add -j --njobs option for prediction (requires nna_methods 0.6.0)
* 0.7.6: * Allow failback when NCEI is missing both file for a particular
           pollutant day (e.g. LZQZ99_KWBP 2024-06-24)
* 0.7.5: * Updating requirements.txt and install_requires to prevent numpy 2
           and fixing new matplotlib registry issue.
* 0.7.4: * Fixed install_requires
* 0.7.3: * Added initialization hour to ncep and nomads file structure.
* 0.7.2: * Changed default NOAA server from ncep to nomads.
* 0.7.1: * Updated some netcdf metadata options.
         * Added NDGD_HISTORICAL environmental variable. If set to True, true,
           Yes, or Y, then the NAQFC modeling from the THREDDS catalog will
           be loaded from the historical subfolder. To see the historical
           data holdings, go to the link below and choose historical
           www.ncei.noaa.gov/thredds/catalog/model-ndgd-file/catalog.html
* 0.7.0: * Added capability for AirNow API to use "mobile monitors" (default).
         * Fixed pair_airnowaqobsfile and pair_airnowhourlydatafile (unused)
           and added testing for them.
* 0.6.2: * New ANT color palettes and added metadata in nc and geojson files.
* 0.6.1: * Updates for windows compatibility with NOAA forecast temp files.
* 0.6.0: * Updated so that the default NAQFC is the KFAN product for both PM
           and ozone. For archived dates, the KFAN is not available. The system
           will switch to non-KFAN (raw forecast).
         * Improved backward compatibility for matplotlib versions (only
           affects) to_geopandas and to_geojson
* 0.5.3: * Updated to_geopandas for backward matplotlib compatibility
         * Updated to_geojson for convenience
* 0.5.2: * Added NetCDF formatted output option.
* 0.5.1: * Updated obs/epa readers to require valid observation as well as mod.
         * Updated pm driver to set a minimum distance of half a grid cell for
           PurpleAir for the purpose of fusing data. This prevents PurpleAir
           from every being closer to the prediction cell centroid than an
           AirNow monitor when they are in the same cell.
         * Updated get_model to apply bbox to NAQFC; allows spatial subsetting
           of the target domain.
         * Update to pull NAQFC based on end hour.
* 0.5.0: Adding GOES capability as an observation dataset.
* 0.4.2: updated logging to capture all logged messages during pm.py execution.
* 0.4.1: switched default AirNow data source to be AirNow API (<2d old) or
         AirNow Files (>=2d old)
* 0.4.0: added functionality with GEOS-CF and PurpleAir only fusion.
         moved ozone.py, pmpaonly.py, and pmanonly.py to drivers.py
         pa.py is retained because it applied pm from AirNow and PurpleAir
* 0.3.0: Identical results for PM from previous run, but then updated to only
         use sites with valid model results in eVNA and aVNA.
* 0.2.0: checks for invalid aVNA_AN and aVNA_PA and updates weights accordingly
* 0.1.0: functioning
