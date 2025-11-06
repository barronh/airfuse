from ._obs import obs


class airnowapi(obs):
    def __init__(
        self, spc, bbox=None, nowcast=False,
        sitekey='intlaqscode', inroot='inputs', montype=None, api_key=None
    ):
        import os
        super().__init__(
            spc=spc, bbox=bbox, nowcast=nowcast, sitekey=sitekey, inroot=inroot
        )
        if api_key is None:
            emsg = 'api_key must be provided or ~/.airnowapikey must exist'
            keypath = os.path.expanduser('~/.airnowkey')
            if os.path.exists(keypath):
                api_key = open(keypath, 'r').read().strip()
            else:
                raise KeyError(emsg)
        self.api_key = api_key
        self.montype = montype

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
        df : pandas.DataFrame
            Must have time, longitude, latitude, nowcast, raw, and sitekey
        """
        import os
        import requests
        import pandas as pd
        import numpy as np
        date = pd.to_datetime(date)
        datestr = date.strftime('%Y-%m-%dT%H')
        root = 'https://www.airnowapi.org/aq/data/'
        inroot = self.inroot
        outpath = f'{inroot}/airnowapi/{date:%Y-%m-%d}'
        outpath += f'/airnowapi.{self.spc}.{datestr}.csv'
        if os.path.exists(outpath):
            return pd.read_csv(outpath)
        bboxstr = '{},{},{},{}'.format(*self.bbox)
        pcode = {'pm25': 'PM25', 'ozone': 'O3', 'o3': 'O3'}[self.spc]
        montype = self.montype
        if montype is None:
            now = pd.to_datetime('now').floor('1h')
            ds = (now - date).total_seconds() / 3600.
            if ds > 1:
                montype = '0'
            else:
                montype = '2'

        params = {
            'startDate': datestr, 'endDate': datestr,
            'parameters': pcode, 'BBOX': bboxstr, 'dataType': 'C',
            'format': 'application/json', 'verbose': '1',
            'monitorType': montype, 'includerawconcentrations': '1',
            'API_KEY': self.api_key
        }
        with requests.get(root, params=params) as r:
            r.raise_for_status()
            j = r.json()
            df = pd.DataFrame.from_records(j)
            df.columns = [k.lower() for k in df.columns]
        df.replace(-999, np.nan)
        rnmrs = {'utc': 'time', 'value': 'nowcast', 'rawconcentration': 'raw'}
        df.columns = [rnmrs.get(k, k) for k in df.columns]
        df['time'] = pd.to_datetime(df['time'])
        okeys = ['time', 'longitude', 'latitude']
        okeys += [self.sitekey, 'nowcast', 'raw']
        outdf = df[okeys]
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        outdf.to_csv(outpath, index=False)
        return outdf

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
