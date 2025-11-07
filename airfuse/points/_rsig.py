from ._obs import obs


class rsig_obs(obs):
    def __init__(
        self, spc, bbox=None, nowcast=False, src='airnow',
        sitekey='site_name', inroot='inputs'
    ):
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
        import numpy as np
        import pandas as pd
        import pyrsig
        src = self.src
        spc = self.spc
        date = pd.to_datetime(date)

        if key is None:
            key = f'{src}.{spc}'
        wdir = date.strftime(f'{self.inroot}/rsig/%Y/%m/%d')
        api = pyrsig.RsigApi(workdir=wdir, **self._rsigopts)
        if self.nowcast:
            dhrs = np.arange(0, -12, -1)
        else:
            dhrs = [0]
        dfs = []
        for dh in dhrs:
            sdate = date + pd.to_timedelta(dh, unit='h')
            edate = sdate + pd.to_timedelta('3599s')
            df = api.to_dataframe(
                key, bdate=sdate, edate=edate,
                unit_keys=False, parse_dates=True
            )
            dfs.append(df)
        if len(dhrs) > 1:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = dfs[0]

        df.columns = [k.lower() for k in df.columns]
        renamer = {
            self.spc: 'obs', 'pm25_hourly': 'obs', 'pm25_corrected': 'obs',
            'pm25_corrected_hourly': 'obs'
        }
        df = df.rename(columns=renamer).drop(['timestamp'], axis=1)
        return df


class airnowrsig(rsig_obs):
    def __init__(
        self, spc, bbox=None, nowcast=False, inroot='inputs'
    ):
        super().__init__(
            spc, src='airnow', bbox=bbox, nowcast=nowcast, inroot=inroot
        )


class purpleairrsig(rsig_obs):
    def __init__(
        self, spc, bbox=None, nowcast=False, inroot='inputs',
        dust='ignore', api_key=None
    ):
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

    def load(self, date):
        import pandas as pd
        import numpy as np
        df = super().load(date, 'purpleair.pm25_corrected')
        date = pd.to_datetime(date)
        df['time'] = df['time'].dt.floor('1h')
        df = df.query('obs > 0.0 and obs < 1000.')
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
                print('WARN:: Records count changed during merge')
            if self.dust == 'exclude':
                # Dropping nans by default
                # nan > 190 is False, and not False is true.
                didx = df.query('~(small_to_large > 190)').index
                nrem = didx.shape[0]
                df.drop(didx, axis=0, inplace=True)
                msg = f'{nrem} ({nrem / norig:.1%}) monitors removed'
                msg += ' due to possible dust (0.3um / 5um less than 190).'
            elif self.dust == 'correct':
                # nan defaults to false, so not corrected
                qstr = 'small_to_large <= 190'
                didx = df.query(qstr).index
                nrem = didx.shape[0]
                df.loc[didx, 'obs'] = df.loc[didx, 'obs'] * 5.6
                msg = f'{nrem} ({nrem / norig:.1%}) monitors multiplied by 5.6'
            df.drop('small_to_large', axis=1, inplace=True)
            print(msg)

        return df.query('obs > 0.0 and obs < 1000.')  # add constraint

    def pair(self, date, modvar, proj=None, qstr=None):
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
