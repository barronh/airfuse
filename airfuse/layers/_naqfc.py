import logging
logger = logging.getLogger('airfuse.layers.naqfc')


class naqfc(object):
    def __init__(
        self, spc, nowcast=False, maxval=2e3, inroot='inputs', **kwds
    ):
        """
        Object to acquire and load NOAA's NAQFC with optional application
        of the Nowcast formulat to the results.

        Arguments
        ---------
        spc : str
            pm25 or ozone
        nowcast : bool
            If True, apply nowcast (avail for pm25)
        maxval : float
            NAQFC can be capped to a maximum value to remove unrealistic values
            similar to those described by Kristen Foley on the CMAQ forum.
            Based on that comment, pm25 has never been observed greater than
            greater than 1401 micrograms/m**3. Thus, values in the 10k+ are
            likely artifacts of fire emissision procesing.
            See https://forum.cmascenter.org/t/screening-out-extreme-pm2-5-\
               model-estimates-in-equates-cmaq-output-files/3010
        kwds : mappable
            Passed to xr.open_dataset

        Returns
        -------
        naqfc:
            Object capable of acquire, open, and get
        """
        import pyproj
        self.name = 'naqfc'
        self.spc = spc
        self.nowcast = nowcast
        self.kwds = kwds
        self.srs = (
            '+proj=lcc +lat_1=25 +lat_0=25 +lon_0=265 +k_0=1 +x_0=0 +y_0=0'
            ' +R=6371229 +no_defs'
        )
        self.proj = pyproj.Proj(self.srs)
        self.maxval = maxval
        self.inroot = inroot

    def acquire(self, date, fdates=None):
        """
        Find and download NAQFC simulation from AWS, NOMADS, or TGFTP.

        Arguments
        ---------
        date : pandas.to_datetime
            Datetime of data
        fdates : list
            Datetime for files to check default (date, date - 24h, date - 48h)

        Returns
        -------
        path : str
            Path to downloaded file
        """
        from urllib.request import urlretrieve
        import os
        import pandas as pd
        import xarray as xr

        spc = self.spc
        fspc = {'ozone': 'o3'}.get(spc, spc)
        date = pd.to_datetime(date)
        dest = date.strftime('%Y/%m/%d/aqm.t12z.ave_1hr_pm25.227.grib2')
        nomadroot = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/aqm/prod'
        adatestr = date.strftime('%Y-%m-%d')
        if adatestr > '2024-05-13':
            av = '7'
        elif adatestr > '2021-07-19':
            av = '6'
        elif adatestr >= '2020-01-01':
            av = '5'
        else:
            raise ValueError('Dates before 2020 are not available')
        awsroot = f'https://noaa-nws-naqfc-pds.s3.amazonaws.com/AQMv{av}/CS'
        tgtroot = 'https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/'
        tgtroot += 'DC.ndgd/GT.aq/AR.conus'
        spc = self.spc
        nmdtmpl12 = f'{nomadroot}/aqm.%Y%m%d/12/aqm.t12z.ave_1hr_{fspc}'
        nmdtmpl12 += '_bc.227.grib2'
        nmdtmpl06 = f'{nomadroot}/aqm.%Y%m%d/06/aqm.t06z.ave_1hr_{fspc}'
        nmdtmpl06 += '_bc.227.grib2'
        awstmpl12 = f'{awsroot}/%Y%m%d/12/aqm.t12z.ave_1hr_{fspc}_bc.%Y%m%d'
        awstmpl12 += '.227.grib2'
        awstmpl06 = f'{awsroot}/%Y%m%d/06/aqm.t06z.ave_1hr_{fspc}_bc.%Y%m%d'
        awstmpl06 += '.227.grib2'
        h06 = pd.to_timedelta('+6h')
        h12 = pd.to_timedelta('+12h')
        h24 = pd.to_timedelta('+24h')
        h72 = pd.to_timedelta('+72h')
        fdates = fdates or [date, date - h24, date - h24 * 2]
        now = pd.to_datetime('now', utc=True).tz_convert(None)
        if now.hour < 11:
            tgtstart = now.floor('1d') - h12
        elif now.hour < 17:
            tgtstart = now.floor('1d') + h06
        else:
            tgtstart = now.floor('1d') + h12
        for fdate in fdates:
            d00Z = fdate.floor('1d')
            urlopts = [
                (d00Z + h12, nmdtmpl12),
                (d00Z + h12, awstmpl12),
                (d00Z + h06, nmdtmpl06),
                (d00Z + h06, awstmpl06),
                (tgtstart, f'{tgtroot}/ds.a{fspc}h01_bc.bin')
            ]
            for sdate, url in urlopts:
                url = fdate.strftime(url)
                edate = sdate + h72
                dest = url.split('://')[1]
                dest = f'{self.inroot}/{dest}'
                if date > sdate and date < edate:
                    if os.path.exists(dest):
                        path = dest
                    else:
                        try:
                            os.makedirs(os.path.dirname(dest), exist_ok=True)
                            path, msg = urlretrieve(url, dest)
                        except Exception as e:
                            logger.warn(str(e))
                            continue
                    with xr.open_dataset(path, **self.kwds) as testf:
                        reftime = pd.to_datetime(testf.time.values)
                        logger.info(f'{path} time={reftime} target={date}')
                        if reftime < date:
                            return path
                        logger.warn(f'{path} time={reftime} after {date}')
        else:
            raise IOError('not found')

    def open(self, date, fdates=None):
        """
        Acquire and open results from an NAQFC simulation that covers date.

        Arguments
        ---------
        date : pandas.to_datetime
            Datetime of data
        fdates : list
            Datetime for files to check default (date, date - 24h, date - 48h)


        Returns
        -------
        f : xarray.Dataset
            file with data for date in it
        """
        import numpy as np
        import pandas as pd
        import xarray as xr
        path = self.acquire(date, fdates=fdates)
        f = xr.open_dataset(path, **self.kwds)
        reftime = f.time
        time = reftime + f.step - pd.to_timedelta('1800s')
        f.coords['step'] = time.data
        f = f.drop_vars(['time']).rename_dims(step='time')
        f = f.rename_vars(step='time')
        rnms = dict(pmtf='pm25', ozcon='ozone')
        rnms = {k: v for k, v in rnms.items() if k in f.data_vars}
        f = f.rename(**rnms)
        tmpvar = f[self.spc]
        swlon = tmpvar.GRIB_longitudeOfFirstGridPointInDegrees
        if swlon > 180:
            swlon = swlon - 360
        swlat = tmpvar.GRIB_latitudeOfFirstGridPointInDegrees
        dy = tmpvar.GRIB_DyInMetres
        dx = tmpvar.GRIB_DxInMetres
        # https://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID227
        sw = self.proj(swlon, swlat)
        nx = 1473  # tmpvar.shape[1]
        ny = 1025  # tmpvar.shape[0]
        x = sw[0] + np.arange(nx) * dx
        x = xr.DataArray(x, dims=('x',), attrs=dict(units='m'))
        y = sw[1] + np.arange(ny) * dy
        y = xr.DataArray(y, dims=('y',), attrs=dict(units='m'))
        f.coords['x'] = x
        f.coords['y'] = y
        now = pd.to_datetime('now', utc=True)
        f.attrs['description'] = f'{path} ({now:%Y-%m-%dT%H:%M:%SZ})'
        f.attrs['crs_proj4'] = self.srs
        return f.drop_vars(['longitude', 'latitude', 'sigma'])

    def get(self, date, fdates=None, nowcast=None):
        """
        Return the appropriate hour value (optionally with nowcast)
        Arguments
        ---------
        date : pandas.to_datetime
            Datetime of data
        fdates : list
            Datetime for files to check default (date, date - 24h, date - 48h)


        Returns
        -------
        outvar : xarray.DataArray
            Must have crs_proj4 and description attributes
            If nowcast, then return nowcasted result.
            Otherwise, return hourly value.
        """
        import pandas as pd
        if nowcast is None:
            nowcast = self.nowcast

        if nowcast:
            import xarray as xr
            from ..utils import xpmnowcast
            date = pd.to_datetime(date)
            sdate = date + pd.to_timedelta('-11.5h')
            edate = date + pd.to_timedelta('0.5h')
            f = self.open(sdate, fdates=fdates)
            invar = f[self.spc].sel(time=slice(sdate, edate))
            invar[:] = invar.where(invar.fillna(0) < self.maxval, self.maxval)
            # note: apply always moves core dimensions to the end
            outvar = xr.apply_ufunc(
                xpmnowcast, invar,
                input_core_dims=[['time']], kwargs={"axis": -1}
            ).expand_dims(time=[date]).transpose('time', 'y', 'x')
            outvar.attrs.update(invar.attrs)
        else:
            f = self.open(date, fdates=fdates)
            outvar = f[self.spc].sel(time=[date], method='nearest')
            outvar[:] = outvar.where(
                outvar.fillna(0) < self.maxval, self.maxval
            )
            if 'valid_time' in outvar.coords:
                outvar = outvar.drop_vars('valid_time')

        outvar.name = self.name
        outvar.attrs['crs_proj4'] = f.attrs['crs_proj4']
        outvar.attrs['description'] = f.attrs['description']
        return outvar
