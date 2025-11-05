__all__ = ['mpestats']


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
                        (|yhat - mean(yref)| + |yref - mean(yref)|)**2
                  )

    """
    from scipy.stats.mstats import linregress
    sdf = df.describe().T
    sdf['5%'] = df.quantile(0.05)
    sdf['95%'] = df.quantile(0.95)
    dks = [
        'count', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max'
    ]
    sdf = sdf[dks].copy()
    om = sdf.loc[refkey, 'mean']
    bias = df.subtract(df[refkey], axis=0)
    minusom = df.subtract(om, axis=0).abs()
    ioaden = (
        minusom.add(minusom[refkey], axis=0)**2
    ).sum()
    se = bias**2
    sse = se.sum()
    sdf['skew'] = sdf['50%'] / sdf['mean']
    sdf['cov'] = sdf['std'] / sdf['mean']
    sdf['mb'] = bias.mean().T
    sdf['me'] = bias.abs().mean().T
    sdf['rmse'] = se.mean()**.5
    sdf['nmb'] = sdf['mb'] / om
    sdf['nme'] = sdf['me'] / om
    sdf['fmb'] = sdf['mb'] / (om + sdf['mean']) * 2
    sdf['fme'] = sdf['me'] / (om + sdf['mean']) * 2
    sdf['r'] = df.corr()[refkey]
    sdf['ioa'] = 1 - sse / ioaden
    sdf['nmb%'] = sdf['nmb'] * 100
    sdf['nme%'] = sdf['nme'] * 100
    sdf['fmb%'] = sdf['fmb'] * 100
    sdf['fme%'] = sdf['fme'] * 100
    sdf['fme%'] = sdf['fme'] * 100
    sdf['rmse%'] = sdf['rmse'] / om * 100
    sdf.index.name = 'key'
    for key in df.columns:
        lr = linregress(df[refkey], df[key])
        sdf.loc[key, 'lr_slope'] = lr.slope
        sdf.loc[key, 'lr_intercept'] = lr.intercept
        sdf.loc[key, 'lr_pvalue'] = lr.pvalue

    return sdf
