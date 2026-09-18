from ..utils._err import log_class_errors


@log_class_errors
class obs:
    def __init__(
        self, spc, bbox=None, nowcast=False,
        sitekey=None, inroot='inputs'
    ):
        if bbox is None:
            bbox = (-135, 15, -55, 80)
        self.spc = spc
        self.nowcast = nowcast
        self.sitekey = sitekey
        self.bbox = bbox
        self.inroot = inroot

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
            Must have time, longitude, latitude, and obs, and sitekey
        """
        raise NotImplementedError('Must be implemented by subclass')

    def get(self, date):
        """Get observational data for date and, if appropriate, apply nowcast

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
        import logging
        import numpy as np
        import pandas as pd
        logger = logging.getLogger(f'airfuse.{self.__class__}')

        spc = self.spc

        if self.nowcast:
            from ..utils import pmnowcast, o3nowcast
            if spc == 'pm25':
                nowcast = pmnowcast
                dhrs = np.arange(0, -12, -1)
            elif spc == 'ozone':
                nowcast = o3nowcast
                dhrs = np.arange(0, -14 * 24, -1)
            dfs = []
            for dh in dhrs:
                hdate = date + pd.to_timedelta(dh, unit='h')
                try:
                    df = self.load(hdate)
                    dfs.append(df)
                except Exception as e:
                    wmsg = f'{self.__class__} failed to retrieve {hdate}'
                    wmsg += f': {str(e)}'
                    logger.warn(wmsg)

            hdf = pd.concat(dfs)
            df = hdf.drop(['obs'], axis=1).groupby(self.sitekey).first()
            nc = hdf.groupby(self.sitekey).apply(
                lambda df: nowcast(df.set_index('time').asfreq('1h')['obs']),
                include_groups=False
            )
            df['obs'] = nc
        else:
            df = self.load(date)
        return df

    def pair(self, date, modvar, proj=None, qstr=None):
        """pair observational from get with modvar

        Arguments
        ---------
        date : datetime
        modvar : xarray.DataArray
        proj : pyproj.Proj
        qstr : str

        Returns
        -------
        df : pandas.DataFrame
            Has obs, modvar.name, x, and y variables.
            If nowcast, then obs is nowcasted
        """
        df = self.get(date)

        if proj is not None:
            df['x'], df['y'] = proj(df['longitude'], df['latitude'])
        else:
            df['x'], df['y'] = df['longitude'], df['latitude']

        cds = df[['x', 'y']].to_xarray()
        # the only dimensions should be x/y. If time exists, it is a unity
        # dimension and can be squeezed out.
        moddf = modvar.squeeze(drop=True).interp(
            x=cds.x, y=cds.y, method='linear'
        ).to_dataframe(name='mod')
        df['mod'] = moddf['mod']
        qstr = qstr or 'obs == obs and mod == mod'
        return df.query(qstr)
