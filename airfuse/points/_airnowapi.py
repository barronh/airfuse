from ._obs import obs


class airnowapi(obs):
    def __init__(
        self, spc, bbox=None, nowcast=False,
        sitekey='intlaqscode', inroot='inputs', api_key=None
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

    def load(self, date, key=None):
        """load raw data from server.

        Arguments
        ---------
        date : date-like
        key : str
            Override the default key (default: src.spc)

        Returns
        -------
        df : pandas.DataFrame
            Must have time, longitude, latitude, nowcast, raw, and sitekey
        """
        import requests
        import pandas as pd
        import numpy as np
        date = pd.to_datetime(date)
        root = 'https://www.airnowapi.org/aq/data/'
        bboxstr = '{},{},{},{}'.format(*self.bbox)
        datestr = date.strftime('%Y-%m-%dT%H')
        pcode = {'pm25': 'PM25', 'ozone': 'O3', 'o3': 'O3'}[self.spc]
        params = {
            'startDate': datestr, 'endDate': datestr,
            'parameters': pcode, 'BBOX': bboxstr, 'dataType': 'B',
            'format': 'application/json', 'verbose': '1',
            'monitorType': '2', 'includerawconcentrations': '1',
            'API_KEY': self.api_key
        }
        with requests.get(root, params=params) as r:
            r.raise_for_status()
            j = r.json()
            df = pd.DataFrame.from_records(j)
            df.columns = [k.lower() for k in df.columns]
        renamers = {'utc': 'time', 'value': 'nowcast', 'rawconcentration': 'raw'}
        df.replace(-999, np.nan)
        df.columns = [renamers.get(k, k) for k in df.columns]
        df['time'] = pd.to_datetime(df['time'])
        okeys = [
            'time', 'longitude', 'latitude', self.sitekey, 'nowcast', 'raw'
        ]
        return df[okeys]

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
