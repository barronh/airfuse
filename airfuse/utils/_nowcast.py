def o3nowcast(x, mdl=0, debug=False):
    raise Exception('not implemented')


def pmnowcast(x, mdl=0, debug=False):
    """
    Accepts a 12-vector of PM25 in micrograms/m3 and returns a single AQI.

    Arguments
    ---------
    x : array-like
      Should be 12-hours of particulate matter

    Returns
    -------
    nc : array-like
      Nowcasted concentration according to steps 1-7 on pg 16-17 in
      https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf

    Notes
    -----
    * Treatment of missing values is based on code provided by Adam Reff.
    * Verified based on AirNowAPI data for IntlAQSCode site 840510590030

    Example
    -------
    import requests
    import getpass
    import pandas as pd
    lat = 38.77335
    lon = -77.10468
    dx = 0.01
    api_key = getpass.getpass('Enter AirNowAPI Key')
    url = (
        'https://www.airnowapi.org/aq/data/?&parameters=PM25'
        '&dataType=B&format=application/json&verbose=1&monitorType=0'
        '&includerawconcentrations=1'
        '&startDate=2024-01-01T00&endDate=2024-11-21T00'
        f'&BBOX={lon - dx},{lat - dx},{lon + dx},{lat + dx}&API_KEY={api_key}'
    )
    r = requests.get(url)
    df = pd.DataFrame.from_records(r.json())
    df = df.replace(-999, float('nan'))
    df['datetime'] = pd.to_datetime(df['UTC'])
    dfr = df.set_index(['IntlAQSCode', 'datetime']).groupby('IntlAQSCode')
    df = dfr.resample('1h', level='datetime').mean(numeric_only=True)
    df['Nowcast'] = df['RawConcentration'].groupby('IntlAQSCode').transform(
        lambda df: df.rolling(window=12, min_periods=1).apply(pmnowcast)
    ) * 10 // 1 / 10
    df['Diff'] = (df['Nowcast'] - df['Value']).replace(0, float('nan'))
    df = df.iloc[12:-12]
    print(df[['Value', 'Nowcast', 'Diff']].describe().round(1).to_markdown())
    # Outputs:
    # |       |   Value |   Nowcast |   Diff |
    # |:------|--------:|----------:|-------:|
    # | count |  7775   |    7775   |      0 |
    # | mean  |     6.5 |       6.5 |    nan |
    # | std   |     3.5 |       3.5 |    nan |
    # | min   |     0.4 |       0.4 |    nan |
    # | 25%   |     4   |       4   |    nan |
    # | 50%   |     5.7 |       5.7 |    nan |
    # | 75%   |     8.1 |       8.1 |    nan |
    # | max   |    27.1 |      27.1 |    nan |
    """
    import numpy as np
    # if there are less than 2 measurements in the last 3 hours
    if np.isfinite(x[-3:]).sum() < 2:
        return np.nan
    # Replace less than MDL with MDL
    ltmdl = x < mdl
    if ltmdl.any():
        c = x.copy()
        c[ltmdl] = mdl
    else:
        c = x
    # Calculate scale value
    cmax = c.max()
    # Exit if max value is zero
    if cmax == 0:
        return np.nan

    s = (cmax - c.min()) / cmax   # Step 1-3
    wf = np.maximum(np.minimum(1 - s, 1), 0.5)  # Step 4-5
    w = wf**np.arange(c.size)[::-1]
    w[np.isnan(c)] = 0
    out = (w * c).sum() / w.sum()
    if debug:
        print(w)
        print(c)
        print(out)

    return out


def xpmnowcast(arr, mdl=0, axis=0):
    """
    Thin warpper around pmnowcast for application along a dimension of an
    xarray.DataArray.

    Arguments
    ---------
    arr : numpy.array
    mdl : float
    axis : int

    Returns
    -------
    out : numpy.array
        Apply pmnowcast to array along time axis
    """
    import numpy as np
    return np.apply_along_axis(pmnowcast, axis, arr, mdl=mdl)
