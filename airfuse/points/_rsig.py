from ._obs import obs
from ..utils._err import log_class_errors


@log_class_errors
class rsig_obs(obs):
    def __init__(
        self, spc, bbox=None, nowcast=False, src='airnow',
        sitekey='site_name', inroot='inputs'
    ):
        """Initialize rsig_obs object

        Arguments
        ---------
        spc : str
            pm25, ozone, co, no2, or any other RSIG AirNow species
        bbox : list
            Bounding box in decimal degrees [swlon, swlat, nelon, nelat]
        nowcast : bool
            If True, species will be nowcasted. If False, return hourly result
        src : str
            airnow, aqs, or other RSIG source of point observations
        sitekey : str
            Lowercase name of field in RSIG ascii output (ignore unit) that
            identifies the site.
        inroot : str
            Path to store cached inputs.

        Returns
        -------
        None
        """
        super().__init__(
            spc=spc, bbox=bbox, nowcast=nowcast,
            sitekey=sitekey, inroot=inroot
        )
        self.src = src
        self._rsigopts = dict(bbox=self.bbox)

    def load(self, date, key=None):
        """load raw data from server.

        Arguments
        ---------
        date : date-like
            Starting hour to load HH:00:00Z to HH:59:59Z
        key : str
            Override the default key (default: src.spc)

        Returns
        -------
        df : pandas.DataFrame.DataArray
            Must have time, longitude, latitude, obs, and sitekey
            If nowcast, load 12 hours of data.
            Otherwise, load 1h.
        """
        import pandas as pd
        import pyrsig
        src = self.src
        spc = self.spc
        date = pd.to_datetime(date)

        if key is None:
            key = f'{src}.{spc}'
        wdir = date.strftime(f'{self.inroot}/rsig/%Y/%m/%d')
        api = pyrsig.RsigApi(workdir=wdir, **self._rsigopts)

        sdate = date
        edate = sdate + pd.to_timedelta('3599s')
        df = api.to_dataframe(
            key, bdate=sdate, edate=edate,
            unit_keys=False, parse_dates=True
        )
        df.columns = [k.lower() for k in df.columns]
        renamer = {
            self.spc: 'obs', 'pm25_hourly': 'obs', 'pm25_corrected': 'obs',
            'pm25_corrected_hourly': 'obs'
        }
        df.rename(columns=renamer, inplace=True)
        df = df.drop(['timestamp'], axis=1)
        return df


@log_class_errors
class airnowrsig(rsig_obs):
    def __init__(
        self, spc, bbox=None, nowcast=False, inroot='inputs'
    ):
        """Initialize airnowrsig object

        Arguments
        ---------
        spc : str
            pm25, ozone, co, no2, or any other RSIG AirNow species
        bbox : list
            Bounding box in decimal degrees [swlon, swlat, nelon, nelat]
        nowcast : bool
            If True, species will be nowcasted. If False, return hourly result
        inroot : str
            Path to store cached inputs.

        Returns
        -------
        None
        """
        super().__init__(
            spc, src='airnow', bbox=bbox, nowcast=nowcast, inroot=inroot
        )


@log_class_errors
class purpleairrsig(rsig_obs):
    def __init__(
        self, spc, bbox=None, nowcast=False, inroot='inputs', exclude=None,
        dust='ignore', drop_outliers=True, min_valid=0.0, max_valid=1000.0,
        api_key=None
    ):
        """Initialize airnowrsig object

        Arguments
        ---------
        spc : str
            pm25, ozone, co, no2, or any other RSIG AirNow species
        bbox : list
            Bounding box in decimal degrees [swlon, swlat, nelon, nelat]
        nowcast : bool
            If True, species will be nowcasted. If False, return hourly result
        inroot : str
            Path to store cached inputs.
        exclude : list
            List-like set of PurpleAir IDs that are known to have bad values
        dust : str
            Choice on how to treat dusty measurements: ignore, exclude, correct
        drop_outliers : bool
            If True, drop outliers using maxdist=100km (see utils.buddycheck)
        min_valid : float
            Values less than this are removed as invalid
        max_valid : float
            Values greater than or equal to this are removed as invalid
        api_key : str
            PurpleAir API key

        Returns
        -------
        None

        Notes
        -----
        The dust option affects rows where the count of small particles (0.3um)
        are less than 190 times the large particles (5um). The reporting of
        small and large particles is not complete, so some records have nan for
        the ratio. The nans are currently treated as not greater than 190 and,
        therefore, as dusty.
        """
        import os
        super().__init__(
            spc, src='purpleair', bbox=bbox, nowcast=nowcast,
            sitekey='station', inroot=inroot
        )
        if api_key is None:
            keypath = os.path.expanduser('~/.purpleairkey')
            if not os.path.exists(keypath):
                emsg = 'If api_key is not provided, the purpleair api key'
                emsg += ' must exist in a file at ~/.purpleairkey'
                raise IOError(emsg)
            with open(keypath, 'r') as kf:
                api_key = kf.read().strip()
        self._rsigopts['purpleair_kw'] = dict(api_key=api_key)
        assert dust in ('exclude', 'correct', 'ignore')
        self.dust = dust
        self.exclude = exclude
        self.min_valid = min_valid
        self.max_valid = max_valid
        self.drop_outliers = drop_outliers

    def load(self, date, key='purpleair.pm25_corrected'):
        """load raw data from server.

        Arguments
        ---------
        date : date-like
            Starting hour to load HH:00:00Z to HH:59:59Z
        key : str
            Override the default key (default: src.spc)

        Returns
        -------
        df : pandas.DataFrame
            Must have time, longitude, latitude, and obs, and sitekey
        """
        import pandas as pd
        import numpy as np
        from ..utils import buddycheck
        import logging
        classname = type(self).__name__
        logger = logging.getLogger(f'airfuse.{classname}.load')
        df = super().load(date, key)
        sitekey = self.sitekey
        if self.exclude is not None:
            exclude = self.exclude
            nbefore = df.shape[0]
            remids = df.query(f'{sitekey}.isin({exclude}) == True')[sitekey]
            remids = list(remids.values)
            df.query(f'{sitekey}.isin({exclude}) == False', inplace=True)
            nafter = df.shape[0]
            if nbefore != nafter:
                ndrop = nbefore - nafter
                nexclude = len(exclude)
                wmsg = f'{ndrop} records removed from {nexclude} to exclude'
                logger.warning(wmsg)
                wmsg = f'Removed ({remids}) of exclude ids ({exclude})'
                logger.debug(wmsg)

        date = pd.to_datetime(date)
        df['time'] = df['time'].dt.floor('1h')
        min_valid = self.min_valid
        max_valid = self.max_valid
        df = df.query(f'obs >= {min_valid} and obs < {max_valid}')
        if self.dust in ('correct', 'exclude'):
            # Code adapted from Sara Farrell; See eq 4 and discussion in
            # Jaffe et al. https://amt.copernicus.org/articles/16/1311/2023/
            sup = super()
            pm03df = sup.load(date, 'purpleair.0_3_um_count')
            pm5df = sup.load(date, 'purpleair.5_um_count')
            dustdf = pd.merge(pm03df, pm5df, on=[self.sitekey, 'time'])
            dustdf['time'] = dustdf['time'].dt.floor('1h')

            # Calculate PM 0.3um counts/PM 5um counts (dust criteria) ratio
            small = dustdf['0_3_um_count_hourly']
            large = dustdf['5_um_count_hourly']
            dustdf['small_to_large'] = small / large
            dustdf.replace([np.inf, -np.inf], np.nan, inplace=True)
            dustgb = dustdf.groupby([self.sitekey, 'time'])
            dustdf = dustgb[['small_to_large']].mean()
            norig = df.shape[0]
            df = pd.merge(df, dustdf.reset_index(), how='inner')
            nmerg = df.shape[0]
            if norig != nmerg:
                wmsg = f'Records count changed during merge {norig} to {nmerg}'
                logger.warning(wmsg)
            if self.dust == 'exclude':
                # Dropping nans by default
                # nan > 190 is False, and not False is true.
                didx = df.query('~(small_to_large > 190)').index
                nrem = didx.shape[0]
                df.drop(didx, axis=0, inplace=True)
                msg = f'{nrem} ({nrem / norig:.1%}) sensors removed'
                msg += ' due to possible dust (0.3um / 5um less than 190).'
            elif self.dust == 'correct':
                # nan defaults to false, so not corrected
                qstr = 'small_to_large <= 190'
                didx = df.query(qstr).index
                nrem = didx.shape[0]
                df.loc[didx, 'obs'] = df.loc[didx, 'obs'] * 5.6
                msg = f'{nrem} ({nrem / norig:.1%}) sensors multiplied by 5.6'
            df.drop('small_to_large', axis=1, inplace=True)
            logger.info(msg)

        if self.drop_outliers:
            # Apply Buddy Check using max distance
            X = pd.DataFrame(dict(
                latr=np.radians(df['latitude']),
                lonr=np.radians(df['longitude'])
            ), index=df.index)[['latr', 'lonr']]
            y = df['obs']
            # haversine returns distance in radians, so using spherical earth
            # to approximate 100km.
            maxdist = 100. / 6371
            keep = buddycheck(X, y, metric='haversine', maxdist=maxdist)
            nkeep = keep.sum()
            norig = keep.shape[0]
            nrem = norig - nkeep
            remids = list(df.loc[~keep, sitekey].values)
            msg = f'{nrem} ({nrem / norig:.1%}) sensors removed by buddy'
            logger.info(msg)
            msg = f'Removed ids: {remids}'
            logger.debug(msg)
            df = df.loc[keep]  # only keep the non-outliers

        maxv = self.max_valid
        minv = self.min_valid
        df = df.query(f'obs >= {minv} and obs < {maxv}')  # add constraint
        return df

    def pair(self, date, modvar, proj=None, qstr=None):
        """
        Applies standard obs.pair method and then adds spatial aggregation
        within a subgrid. The subgrid is defined as 1/2 the grid with of the
        original grid.

        Arguments
        ---------
        date : datetime
            Target date for data
        modvar : xarray.DataArray
            Uncorrected model result for date
        proj : pyproj.Proj
            Projection object that defines the gridded space in of modvar
        qstr : str
            Query string (optional) defaults to requiring both model and obs
            to be valid.

        Returns
        -------
        df : pandas.DataFrame
            Has obs, modvar.name, x, and y variables.
            If nowcast, then obs is nowcasted
        """
        import numpy as np
        import pandas as pd
        df = super().pair(date, modvar, proj=proj, qstr=qstr)
        # group paired data within a cell
        dx = float(modvar.x.diff('x').mean())
        dy = float(modvar.y.diff('y').mean())
        nx = modvar.x.shape[0]
        ny = modvar.y.shape[0]
        sx = float(modvar.x.min())
        sy = float(modvar.y.min())
        xe = sx + np.arange(-.5, nx, 0.5) * dx
        xc = sx + np.arange(0, nx, 0.5) * dx
        ye = sy + np.arange(-.5, ny, 0.5) * dy
        yc = sy + np.arange(0, ny, 0.5) * dy
        x = pd.cut(df['x'], xe, labels=xc).astype('d')
        y = pd.cut(df['y'], ye, labels=yc).astype('d')
        t = df['time'].dt.floor('1h')
        df = df.groupby([t, x, y], observed=True).agg(**{
            'elevation': ('elevation', 'mean'),
            'obs': ('obs', 'mean'),
            'mod': ('mod', 'mean'),
            'x': ('x', 'mean'),
            'y': ('y', 'mean'),
            'count': ('obs', 'count'),
        })
        t = df.index.get_level_values(0)
        x = df.index.get_level_values(1)
        y = df.index.get_level_values(2)
        ismulti = df['count'] > 1
        df['time'] = t
        df.loc[ismulti, 'x'] = x[ismulti]
        df.loc[ismulti, 'y'] = y[ismulti]
        df.reset_index(drop=True, inplace=True)
        return df
