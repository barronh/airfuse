__all__ = ['get_file', 'wget_file', 'request_file', 'ftp_file']


def get_file(url, local_path, wget=False):
    """
    Download file from ftp or http via wget, ftp_file, or request_file

    Arguments
    ---------
    url : str
        Path on server
    local_path : str
        Path to save file (usually url without file protocol prefix
    wget : bool
        If True, use wget (default: False)

    Returns
    -------
    local_path : str
        local_path
    """
    if wget:
        return wget_file(url, local_path)
    elif url.startswith('ftp://'):
        return ftp_file(url, local_path)
    else:
        return request_file(url, local_path)


def ftp_file(url, local_path):
    """
    While files are on STAR ftp, use this function.

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without ftp://)

    Returns
    -------
    local_path : str
        local_path
    """
    import ftplib
    import os

    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    server = url.split('//')[1].split('/')[0]
    remotepath = url.split(server)[1]
    ftp = ftplib.FTP(server)
    ftp.login()
    with open(local_path, 'wb') as fp:
        ftp.retrbinary(f'RETR {remotepath}', fp.write)
    ftp.quit()
    return local_path


def wget_file(url, local_path):
    """
    If local has wget, this can be used.

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without ftp://)

    Returns
    -------
    local_path : str
        local_path
    """
    import os
    if not os.path.exists(local_path):
        cmd = f'wget -r -N {url}'
        os.system(cmd)

    return local_path


def request_file(url, local_path):
    """
    Only works with http and https

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without https://)

    Returns
    -------
    local_path : str
        local_path
    """
    import requests
    import shutil
    import os

    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_path


def read_netrc(netrcpath, server):
    import netrc
    nf = netrc.netrc(netrcpath)
    return nf.authenticators(server)


def mpestats(df, refkey='obs'):
    """
    Calculate typical model statistics
    [1] https://epa.gov/sites/production/files/2015-11/
        modelperformancestatisticsdefinitions.docx

    Example assuming you have a csv with 'obs' and 'mod' columns:

    ```
    import pandas as pd

    df = pd.read_csv('obs_mod.csv')
    statdf = getstats(df)
    statdf.to_csv('stats.csv')
    ```

    Arguments
    ---------
    df : pandas.DataFrame
        Each column should be the reference observation or an estimate
        of the reference.
    refkey : str
        Column to be used as a reference observation. All other columns are
        will be compared to refkey

    Returns
    -------
    mpedf : pandas.DataFrame
        DataFrame with statistics by estimate.
        - Descriptive statistics :
          - count, mean, std, min, 5%, 25%, 50%, 75%, 95%, max
          - skew : 50% / mean
          - cov : std / mean
        - Evaluation statistics :
          - r : Pearson Correlation
          - mb : mean bias mean(yhat - yref)
          - me : mean bias mean(|yhat - yref|)
          - rmse : Root Mean Square Error mean((yhat - yref)**2)**0.5
          - nmb : mb / mean(yref) * 100 (as %)
          - nme : me / mean(yref) * 100 (as %)
          - fmb : 200 * mb / (mean(yref) + mean(y)) (as %)
          - fme : 200 * me / (mean(yref) + mean(y)) (as %)
          - ioa : 1 - sum((yhat - yref)**2) / sum(
                        |yhat - mean(yref)| + |yref - mean(yref)|
                  )

    """
    sdf = df.describe().T
    sdf['5%'] = df.quantile(0.05)
    sdf['95%'] = df.quantile(0.95)
    dks = [
        'count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max'
    ]
    sdf = sdf[dks].copy()
    om = sdf.loc[refkey, 'mean']
    bias = df.subtract(df[refkey], axis=0)
    minusom = df.subtract(om, axis=0)
    ioaden = (
        minusom.add(minusom[refkey], axis=0)**2
    ).sum()
    se = bias**2
    sse = se.sum()
    sdf['skew'] = sdf['50%'] / sdf['mean']
    sdf['cov'] = sdf['std'] / sdf['mean']
    sdf['r'] = df.corr()[refkey]
    sdf['mb'] = bias.mean().T
    sdf['me'] = bias.abs().mean().T
    sdf['rmse'] = se.mean()**.5
    sdf['nmb'] = sdf['mb'] / om * 100
    sdf['nme'] = sdf['me'] / om * 100
    sdf['fmb'] = sdf['mb'] / (om + sdf['mean']) * 200
    sdf['fme'] = sdf['me'] / (om + sdf['mean']) * 200
    sdf['ioa'] = 1 - sse / ioaden
    sdf.index.name = 'key'

    return sdf
