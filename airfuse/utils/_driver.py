import logging
logger = logging.getLogger('airfuse.utils.driver')


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
    tgtdf, fitdf, obdnr=None, dnrkwds=None, fitkwds=None, kfoldkwds=None,
    obskey='obs', modkey='mod',
    yhatsfx='_dnr', cvsfx='_dnr_cv', xkeys=None,
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
    dnrkwds : mappable
      Default {
        "delaunay_weights": "only",
        "n_neighbors": 30,
        "weights": lambda d: np.maximum(d, 1e-10)**-2)
      }
    fitkwds : mappable
      User supplied params for fitting.
      If None, defaults as follows:
      - if fitdf has column weight, adds sample_weight.
      - if fitdf has column groups, adds groups.
    kfoldkwds : mappable
      Default {"random_state": 42, "n_splits": 10, "shuffle": True}
    obskey : str
      Name of observation in fitdf
    modkey : str
      Name of model in fitdf and tgtdf
    yhatsfx : str
      Suffix to use for model applicaion results added to tgtdf
    cvsfx : str
      Suffix to use for cross-validation results added to fitdf
      If None, do not perform cross-validation

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
        logger.info(f'Using xkeys={xkeys}')

    ykeys = [obskey, modkey]
    if kfoldkwds is None:
        kfoldkwds = {"random_state": 42, "n_splits": 10, "shuffle": True}

    if fitkwds is None:
        fitkwds = {}
        if fitdf is not None:
            if 'sample_weight' not in fitkwds:
                if 'sample_weight' in fitdf.columns:
                    imsg = 'Using sample_weight column to weight samples.'
                    logger.info(imsg)
                    fitkwds['sample_weight'] = fitdf['sample_weight']
            if 'groups' not in fitkwds:
                if 'groups' in fitdf.columns:
                    logger.info('Using groups to stratify Delaunay diagrams')
                    fitkwds['groups'] = fitdf['groups']

    if obdnr is None:
        if fitdf is None:
            emsg = 'Either fitdf or obdnr is required; you supplied neither'
            raise ValueError(emsg)
        if dnrkwds is None:
            dnrkwds = {}
        if 'weights' not in dnrkwds:
            logger.info('Using weights=lambda d: np.maximum(d, 1e-10)**-2')
            dnrkwds.setdefault('weights', lambda d: np.maximum(d, 1e-10)**-2)
        if 'delaunay_weights' not in dnrkwds:
            logger.info('Using delaunay_weights="only"')
            dnrkwds.setdefault('delaunay_weights', 'only')
        if 'n_neighbors' not in dnrkwds:
            logger.info('Using n_neighbors=30')
            dnrkwds.setdefault('n_neighbors', 30)
        if 'groups' in fitkwds:
            obdnr = dnr.BCGroupedDelaunayNeighborsRegressor(**dnrkwds)
        else:
            obdnr = dnr.BCDelaunayNeighborsRegressor(**dnrkwds)

    if cvsfx is not None and fitdf is not None:
        okeys = [f'{k}{cvsfx}' for k in ykeys]
        fitdf[okeys] = cross_val_predict(
            obdnr, fitdf[xkeys], fitdf[ykeys],
            params=fitkwds,
            cv=KFold(**kfoldkwds)
        )
        biascorrect(fitdf, suffix=cvsfx, obskey=obskey, modkey=modkey)
    if fitdf is not None:
        okeys = [f'{k}{yhatsfx}' for k in ykeys]
        obdnr.fit(fitdf[xkeys], fitdf[ykeys], **fitkwds)
        fitdf[okeys] = obdnr.predict(fitdf[xkeys])
        biascorrect(fitdf, suffix=yhatsfx, obskey=obskey, modkey=modkey)
    if tgtdf is not None:
        idx = ~tgtdf[modkey].isna()
        if all([k in tgtdf.columns for k in xkeys]):
            tgtX = tgtdf.loc[idx, xkeys]
        else:
            tgtX = tgtdf.loc[idx].index.to_frame()[xkeys[:-1]]
            tgtX['mod'] = tgtdf['mod']
        okeys = [f'{k}{yhatsfx}' for k in ykeys]
        tgtdf.loc[idx, okeys] = obdnr.predict(tgtX)
        biascorrect(tgtdf, suffix=yhatsfx, obskey=obskey, modkey=modkey)
