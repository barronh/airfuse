def biascorrect(
    df, suffix='', obskey='obs', modkey='mod',
    mbckey='mbc', abckey='abc', bckey='bc'
):
    mod = df[modkey]
    modhat = df[modkey + suffix]
    obshat = df[obskey + suffix]
    mbc = df[mbckey + suffix] = mod / (modhat / obshat)
    abc = df[abckey + suffix] = mod - (modhat - obshat)
    meanbc = (mbc + abc) / 2
    meanbc = meanbc.where(abc > 0, mbc)
    df[bckey + suffix] = meanbc


def addgridded(ds, df, keys, suffix='_GRID'):
    """
    """
    cds = df.to_xarray()
    for key in keys:
        df[key + suffix] = ds[key].sel(x=cds.x, y=cds.y, method='nearest')


def fuse(
    tgtdf, fitdf, obdnr=None, fitkwds=None, kfoldkwds=None,
    obskey='obs', modkey='mod',
    yhatsfx='_dnr', cvsfx='_dnr_cv', xkeys=None
):
    """
    Apply AirFuse to tgtdf after fitting the model with fitdf

    Arguments
    ---------
    tgtdf : pandas.DataFrame
      Target prediction data frame, must have xykeys and modkey
    fitdf : pandas.DataFrame
      Observation datafram used to fit predictions. Must have xykeys, obskey,
      and modkey.
      If None, obdnr is required and will not be refit.
    obdnr : DelaunayNeighborsRegressor
      User supplied DelaunayNeighborsRegressor regressor instance.
      If None, selects from available options as follows:
      - If groups in fitdf, obdnr will be a GroupedDelaunayNeighborsRegressor
      - Otherwise, DelaunayNeighborsRegressor
    fitkwds : mappable
      User supplied params for fitting.
      If None, defaults as follows:
      - if fitdf has column weight, adds sample_weight.
      - if fitdf has column groups, adds groups.
    obskey : str
      Name of observation in fitdf
    modkey : str
      Name of model in fitdf and tgtdf
    yhatsfx : str
      Suffix to use for model applicaion results added to tgtdf
    cvsfx : str
      Suffix to use for cross-validation results added to fitdf
      If None, do not perform cross-validation
    kfoldkwds : mappable
      Default {"random_state": 42, "n_splits": 10, "shuffle": True}

    Returns
    -------
    None
    """
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_predict
    from .. import dnr

    if xkeys is None:
        xkeys = ['x', 'y']
        print(f'INFO:: Using xkeys={xkeys}')

    ykeys = [obskey, modkey]
    print(f'INFO:: Using ykeys={ykeys}')
    if kfoldkwds is None:
        kfoldkwds = {"random_state": 42, "n_splits": 10, "shuffle": True}

    if fitkwds is None:
        fitkwds = {}
        if fitdf is not None:
            if 'sample_weight' not in fitkwds:
                if 'sample_weight' in fitdf.columns:
                    fitkwds['sample_weight'] = fitdf['sample_weight']
            if 'groups' not in fitkwds:
                if 'groups' in fitdf.columns:
                    fitkwds['groups'] = fitdf['groups']

    if obdnr is None:
        if fitdf is None:
            emsg = 'Either fitdf or obdnr is required; you supplied neither'
            raise ValueError(emsg)
        dnrkwds = {}
        print('INFO:: Using weights=lambda d: np.maximum(d, 1e-10)**-2')
        dnrkwds.setdefault('weights', lambda d: np.maximum(d, 1e-10)**-2)
        print('INFO:: Using delaunay_weights="only"')
        dnrkwds.setdefault('delaunay_weights', 'only')
        dnrkwds.setdefault('n_neighbors', 30)
        print('INFO:: Using n_neighbors=30')
        if 'groups' in fitkwds:
            obdnr = dnr.GroupedDelaunayNeighborsRegressor(**dnrkwds)
        else:
            obdnr = dnr.DelaunayNeighborsRegressor(**dnrkwds)

    if cvsfx is not None and fitdf is not None:
        okeys = [k + cvsfx for k in ykeys]
        yhat_cv = cross_val_predict(
            obdnr, fitdf[xkeys], fitdf[ykeys],
            params=fitkwds,
            cv=KFold(**kfoldkwds)
        )
        fitdf[okeys] = yhat_cv
        biascorrect(fitdf, suffix=cvsfx, modkey=modkey, obskey=obskey)
    if fitdf is not None:
        obdnr.fit(fitdf[xkeys], fitdf[ykeys], **fitkwds)
        okeys = [k + yhatsfx for k in ykeys]
        fitdf[okeys] = obdnr.predict(fitdf[xkeys])
        biascorrect(fitdf, suffix=yhatsfx, modkey=modkey, obskey=obskey)
    if tgtdf is not None:
        if all([k in tgtdf.columns for k in xkeys]):
            tgtX = tgtdf[xkeys]
        else:
            tgtX = tgtdf.index.to_frame()[xkeys]
        okeys = [k + yhatsfx for k in ykeys]
        tgtdf[okeys] = obdnr.predict(tgtX)
        biascorrect(tgtdf, suffix=yhatsfx, modkey=modkey, obskey=obskey)
