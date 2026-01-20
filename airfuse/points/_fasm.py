from ._obs import obs


class _fasm(obs):
    def __init__(
        self, spc, bbox=None, nowcast=False,
        sitekey='site_name', inroot='inputs', fasmcfgpath=None
    ):
        """Initialize _fasm object

        Arguments
        ---------
        spc : str
            pm25, ozone, co, no2, or any other RSIG AirNow species
        bbox : list
            Bounding box in decimal degrees [swlon, swlat, nelon, nelat]
        nowcast : bool
            If True, species will be nowcasted. If False, return hourly result
        sitekey : str
            Lowercase name of field in RSIG ascii output (ignore unit) that
            identifies the site.
        inroot : str
            Path to store cached inputs.
        fasmcfgpath : str
            Path to fasm configuration path (see Notes). Defaults to
            ./fasm.json or ~/fasm.json

        Returns
        -------
        None

        Notes
        -----
        The fasmcfgpath must point to a json file where the keys are urls for
        three files: "purpleaircsv" is the pa.csv file, "excludejson" is the
        path to the actively excluded sites, "airnowjson" is the a geojson
        that has airnow features.
        """
        import os
        import json
        assert spc == 'pm25'
        super().__init__(
            spc=spc, bbox=bbox, nowcast=nowcast, sitekey=sitekey, inroot=inroot
        )
        if fasmcfgpath is None:
            for fasmcfgpath in ['fasm.json', '~/fasm.json']:
                fasmcfgpath = os.path.expanduser(fasmcfgpath)
                if os.path.exists(fasmcfgpath):
                    break
            else:
                emsg = 'fasm.json nor ~/fasm.json exists; must supply'
                emsg += ' fasmcfgpath'
                raise IOError(emsg)
        else:
            fasmcfgpath = os.path.expanduser(fasmcfgpath)
        fasmcfg = open(fasmcfgpath)
        self.urls = json.load(fasmcfg)

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
        lon = outdf['longitude']
        lat = outdf['latitude']
        bbox = self.bbox
        inlon = (lon >= bbox[0]) & (lon <= bbox[2])
        inlat = (lat >= bbox[1]) & (lat <= bbox[3])
        inbbox = inlon & inlat
        return outdf.loc[inbbox]


class purpleairfasm(_fasm):
    def load(self, date, key=None):
        import io
        import requests
        import pandas as pd
        import logging
        logger = logging.getLogger('airfuse.points.purpleairfasm')
        purl = self.urls['purpleaircsv']
        eurl = self.urls['excludejson']
        with requests.get(eurl) as r:
            r.raise_for_status()
            edf = pd.DataFrame.from_records(r.json())
        with requests.get(purl) as r:
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
        logger.info(f'Keeping {keepidx.mean():.2%} of obs')
        return df.loc[keepidx, keepcols].copy()


class airnowfasm(_fasm):
    def load(self, date, key=None):
        import requests
        import pandas as pd
        aurl = self.urls['airnowjson']
        now = pd.to_datetime('now', utc=True).floor('1h')
        dt = pd.to_timedelta('2h')
        minstr = (now - dt).strftime('%Y-%m-%d %H:%M:%S')
        with requests.get(aurl) as r:
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
                    row['time'] = props['lastValidUTCTime']
                    row['nowcast'] = props["PM2.5_nowcast"]
                    row['raw'] = props["PM2.5_1hr"]
                    rows.append(row)
            df = pd.DataFrame.from_records(rows)
            df = df.groupby('site_name').apply(
                lambda tdf: tdf.sort_values('time', ascending=True).tail(1)
            )
            df['time'] = pd.to_datetime(df['time'])
            df['nowcast'] = df['nowcast'].astype('d')
            df['raw'] = df['raw'].astype('d')
        return df
