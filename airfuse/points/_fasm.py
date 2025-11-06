from ._obs import obs


class _fasm(obs):
    def __init__(
        self, spc, bbox=None, nowcast=False,
        sitekey='site_name', inroot='inputs'
    ):
        assert spc == 'pm25'
        super().__init__(
            spc=spc, bbox=bbox, nowcast=nowcast, sitekey=sitekey, inroot=inroot
        )

    def get(self, date):
        """Get observational data for date

        Arguments
        ---------
        date : date-like

        Returns
        -------
        df : pandas.DataFrame.DataArray
            Must have time, longitude, latitude, obs.
            If nowcast, then obs will be nowcasted
            Otherwise, obs will be a raw 1-hour value.
        """
        df = self.load(date)
        if self.nowcast:
            df['obs'] = df['nowcast']
        else:
            df['obs'] = df['raw']
        outdf = df[['time', 'longitude', 'latitude', self.sitekey, 'obs']]
        return outdf


class purpleairfasm(_fasm):
    def load(self, date, key=None):
        import io
        import requests
        import pandas as pd
        url = 'https://s3-us-west-2.amazonaws.com/airfire-data-exports/maps'
        url += '/purple_air/v4/pas.csv'
        eurl = 'https://airfire-data-exports.s3.us-west-2.amazonaws.com/elwood'
        eurl += '/exclusion_lists/elwood_exclusion.json'
        with requests.get(eurl) as r:
            r.raise_for_status()
            edf = pd.DataFrame.from_records(r.json())
        with requests.get(url) as r:
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
        df['time'] = pd.to_datetime(df['utc_ts'])
        df['raw'] = df['epa_pm25']
        df['nowcast'] = df['epa_nowcast']
        df['site_name'] = df['sensor_index']
        now = pd.to_datetime('now', utc=True).floor('1h')
        dt = pd.to_timedelta('2h')
        mint = (now - dt)
        keepidx = ~(
            df['sensor_index'].isin(edf['unit_id'].astype('l'))
            | (df['time'] < mint)
        )
        keepcols = ['time', 'longitude', 'latitude', 'site_name']
        keepcols += ['raw', 'nowcast']
        print(keepidx.mean())
        return df.loc[keepidx, keepcols].copy()


class airnowfasm(_fasm):
    def load(self, date, key=None):
        import requests
        import pandas as pd
        url = 'https://s3-us-west-2.amazonaws.com/airfire-data-exports'
        url += '/monitoring/v2/latest/geojson/fasm_airnow_PM2.5_latest.geojson'
        now = pd.to_datetime('now', utc=True).floor('1h')
        dt = pd.to_timedelta('2h')
        minstr = (now - dt).strftime('%Y-%m-%d %H:%M:%S')
        with requests.get(url) as r:
            r.raise_for_status()
            j = r.json()
            rows = []
            for feat in j['features']:
                props = feat['properties']
                if props['lastValidUTCTime'] >= minstr:
                    row = {}
                    row['site_name'] = props['fullAQSID']
                    lon, lat = feat['geometry']['coordinates']
                    row['longitude'] = lon
                    row['latitude'] = lat
                    row['time'] = date
                    row['nowcast'] = props["PM2.5_nowcast"]
                    row['raw'] = props["PM2.5_1hr"]
                    rows.append(row)
            df = pd.DataFrame.from_records(rows)
        return df
