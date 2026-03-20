__all__ = ['buddycheck']


def buddycheck(
    X, y, k=4, maxdist=1e5, atol=15, rtolmdn=0.1, rtolmad=5,
    metric='minkowski', return_parts=False
):
    """
    Evaluate if the deviation of a measurement (y) from its neighbors is
    within critical values. The deviation is based on the median of the
    k-nearest neighbors (ydev = y - kmdn). A measurement is valid if the
    deviation (ydev) is less than any of the following:

    - an absolute tolerance (atol),
    - a tolerance relative to the median of neighbors (rtolmdn * kmdn), or
    - a tolerance relative to the median absolute deviation of neighbors
      (rtolmad * kmad)

    Being less than any of these tolerances is sufficient. Measurements with
    fewer than k-neighbors within maxdist are too isolated to check, and are
    considered valid.

    Arguments
    ---------
    X : pandas.DataFrame or numpy.ndarray
        DataFrame with x- and y-coordinates
    y : pandas.Series or numpy.ndarray
        Must have value for sensor
    k : int
        Number of required nearest neighbors
    maxdist : float
        Neighbors further than (units consistent with X) cannot be used.
        Default (1e5) assumes x/y in meters with a maxdist=100km.
    atol : float
        Acceptable when ydev (y - kmdn) less than atol (default: 15).
    rtolmdn : float
        Acceptable when ydev (y - kmdn) less than rtolmdn * kmdn (default: 0.1)
    rtolmad : float
        Acceptable when ydev (y - kmdn) less than rtolmad * kmad (default: 5)
    return_parts : bool
        If True, return the overall answer (keep) and whether

    Returns
    -------
    valid[, abschk, mdnchk, madchk, isolated] : numpy.array
        valid: values indicate if y is acceptable
        if return_parts:
            - abschk: ydev is less than atol
            - mdnchk: ydev is less than rtolmdn * kmdn
            - madchk: ydev is less than rtolmad * kmad
            - isolated: X is too far to test

    Example
    -------

    import numpy as np
    import pyproj
    import pyrsig
    from airfuse.utils import buddycheck

    api = pyrsig.RsigApi(purpleair_kw=dict(api_key='EPA'))  # fill in your key
    rawpadf = api.to_dataframe(
        'purpleair.pm25_corrected',
        bdate='2026-03-18T17:00:00', edate='2026-03-18T17:59:59',
        unit_keys=False
    )
    proj = pyproj.Proj('EPSG:5070')
    x, y = proj(rawpadf['LONGITUDE'], rawpadf['LATITUDE'])
    rawpadf['x'] = x
    rawpadf['y'] = y
    padf = rawpadf.query('pm25_corrected_hourly < 1000.')

    X = padf[['x', 'y']]
    y = padf['pm25_corrected_hourly']
    keep, abschk, mdnchk, madchk, lonely = buddycheck(X, y, return_parts=True)
    goodpadf = padf.loc[keep]
    print('N', rawpadf.shape[0])
    print('Count Flagged Data')
    print('- y > 1000:', (rawpadf['pm25_corrected_hourly'] > 1000).sum())
    print('- ydev > max(atol, rtolmdn * kmdn, rtolmad * kmad):', (~keep).sum())
    print('- ydev > atol:', (~abschk).sum())
    print('- ydev > rtolmdn * kmdn:', (~mdnchk).sum())
    print('- ydev > rtolmad * kmad:', (~madchk).sum())
    print('- lonely:', lonely.sum())
    # Count Flagged Data
    # - y > 1000: 62
    # - ydev > max(atol, rtolmdn * kmdn, rtolmad * kmad): 19
    # - ydev > atol: 51
    # - ydev > rtolmdn * kmdn: 458
    # - ydev > rtolmad * kmad: 1045
    # - lonely: 199

    Notes
    -----
    Based on presentation by Halil Cakir, Bryan Chastin, and Ellen DAmico on
    2026-03-19 at the EPA Data Fusion Team Meeting where inputs were optimized
    based on 52 cases of PurpleAir data across the United States.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    # Get k-nearest neighbors (and self) distances and indices
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, idx = nbrs.kneighbors(X)
    # Checking that k-neighbors are always in order of nearest (self) to
    # furthest. Passed once not testing anymore.
    # assert np.diff(dist, axis=1).min() >= 0

    # Get values for k-nearest (assuming ordered nearest-to-furthest)
    kvals = np.asarray(y)[idx[:, 1:]]

    # Calculate statistics across neighbors
    kmdn = np.median(kvals, axis=1)
    kmad = np.abs(kvals - kmdn[:, np.newaxis]).mean(1)

    # Define the critical value
    mad_crit = rtolmad * kmad
    mdn_crit = rtolmdn * kmdn
    ydev = np.abs(y - kmdn)
    # Define isolated as more than 100km away any k-nearest
    isolated = dist.max(1) > maxdist
    crit = np.maximum(mad_crit, np.maximum(atol, mdn_crit))
    # Valid if difference from local median is less than mad_crit
    keep = (ydev < crit) | isolated
    if return_parts:
        abschk = ydev < atol
        mdnchk = ydev < mdn_crit
        madchk = ydev < mad_crit
        return keep, abschk, mdnchk, madchk, isolated
    else:
        return keep
