__all__ = ['applyfusion', 'get_fusions']


def get_fusions(v=30, i=10):
    """
    Get instances of default models

    Arguments
    ---------
    v : int
        minimum number of neighbors for voronoi
    i : int
        minimum number of neighbors for IDW
    """
    import nna_methods
    models = dict(
        # Consistent with AirNow implementation
        IDW=nna_methods.NNA(method='nearest', k=i, power=-5),
        # Power consistent with eVNA; k set to optimize speed.
        VNA=nna_methods.NNA(method='voronoi', k=v, power=-2),
    )
    return models


def applyfusion(
    mod, prefix, fitdf, tgtdf=None, loodf=None, xkey='x',
    ykey='y', obskey='obs_value', modkey='NAQFC', biaskey='BIAS',
    ratiokey='RATIO', loo=True, cv=True, verbose=0, random_state=None,
    njobs=None
):
    """
    This is a convenience function. This assumes you are interpolating the
    observation, the model, the bias (model - obs), and the ratio
    (model / obs). Its job is to perform 10-fold Cross Validation,
    leave-one-out validation, and apply the model to a full target
    domain. In addition, it has the ability to apply a leave-one-out
    validation to another dataset (i.e., leave out the nearest obs even
    though it is not in the target dataset)

    With add=True, this will add interpolated valeus to the DataFrames. Each
    interpolated value (ykey) is added to fitdf as:
      - CV_{prefix}_{ykey} for Cross Validation, and
      - LOO_{prefix}_{ykey} for leave-one-out
      - LOO_{prefix}_{ykey} for leave-one-out in the loodf

    In addition, it calculates the extended (e{prefix}) and additive
    (a{prefix}) neighbor averaging data fusions:
      - a{prefix} = mod - sum(w_n * bias_n)
      - e{prefix} = mod / sum(w_n * ratio_n)

    fitdf, loodf, and tgtdf must contain xkey and ykey
    In addition, loodf and fitdf must contain obskey and modkey
    if biaskey or ratiokey are not in loodf and/fitdf, they will be added.
    """
    import logging

    ykeys = [obskey, modkey, biaskey, ratiokey]
    xkeys = [xkey, ykey]
    # Add bias and ratio keys if they do not exist.
    if biaskey not in fitdf.columns:
        fitdf[biaskey] = fitdf[modkey] - fitdf[obskey]
        if verbose > 1:
            logging.info(f'Added fitdf {biaskey} = {modkey} - {obskey}')
    if ratiokey not in fitdf.columns:
        fitdf[ratiokey] = fitdf[modkey] / fitdf[obskey]
        if verbose > 1:
            logging.info(f'Added fitdf {ratiokey} = {modkey} / {obskey}')
    if loodf is not None:
        if biaskey not in loodf.columns:
            loodf[biaskey] = loodf[modkey] - loodf[obskey]
            if verbose > 1:
                logging.info(f'Added loodf {biaskey} = {modkey} - {obskey}')
        if ratiokey not in loodf.columns:
            loodf[ratiokey] = loodf[modkey] / loodf[obskey]
            if verbose > 1:
                logging.info(f'Added loodf {ratiokey} = {modkey} / {obskey}')

    # Perform a CV validation
    if cv:
        for ykey in ykeys:
            if verbose > 0:
                logging.info(f'Starting cross validation: {ykey}')
            mod.cross_validate(
                fitdf[xkeys], fitdf[ykey], df=fitdf, ykey=f'{prefix}_{ykey}'
            )
        avna = fitdf[modkey] - fitdf[f'CV_{prefix}_{biaskey}']
        fitdf[f'CV_a{prefix}'] = avna
        evna = fitdf[modkey] / fitdf[f'CV_{prefix}_{ratiokey}']
        fitdf[f'CV_e{prefix}'] = evna
        # Add the distance to nearest during cross validation
        for k, testdf in fitdf.groupby(f'CV_{prefix}_{ykey}_fold'):
            traindf = fitdf.query(f'CV_{prefix}_{ykey}_fold != {k}')
            mod.fit(traindf[xkeys], traindf[obskey])
            fitdf.loc[testdf.index, 'CV_DIST'] = mod.nn(
                testdf[xkeys].values, k=1
            )[0][:, 0]

    # Fit the model
    mod.fit(fitdf[xkeys].values, fitdf[ykeys].values)
    # Perform a leave one out validation.
    if loo:
        if verbose > 0:
            logging.info('Starting LOO')
        looz = mod.predict(fitdf[xkeys].values, loo=True)
        for ykey, y in zip(ykeys, looz.T):
            fitdf[f'LOO_{prefix}_{ykey}'] = y
        avna = fitdf[modkey] - fitdf[f'LOO_{prefix}_{biaskey}']
        fitdf[f'LOO_a{prefix}'] = avna
        evna = fitdf[modkey] / fitdf[f'LOO_{prefix}_{ratiokey}']
        fitdf[f'LOO_e{prefix}'] = evna
        fitdf[f'LOO_{prefix}_DIST'] = mod.nn(
            fitdf[xkeys].values, k=2
        )[0].max(1)
    if loodf is not None and fitdf.shape[0] > 1:
        if verbose > 0:
            logging.info('Starting secondary LOO')
        looz = mod.predict(loodf[xkeys].values, loo=True)
        for ykey, y in zip(ykeys, looz.T):
            loodf[f'LOO_{prefix}_{ykey}'] = y
        avna = loodf[modkey] - loodf[f'LOO_{prefix}_{biaskey}']
        loodf[f'LOO_a{prefix}'] = avna
        evna = loodf[modkey] / loodf[f'LOO_{prefix}_{ratiokey}']
        loodf[f'LOO_e{prefix}'] = evna
        loodf[f'LOO_{prefix}_DIST'] = mod.nn(
            loodf[xkeys].values, k=2
        )[0].max(1)

    if tgtdf is not None:
        tgtx = tgtdf[xkeys].values
        if verbose > 0:
            logging.info('Starting target prediction')
        tgtz = mod.predict(tgtx, njobs=njobs)
        for ykey, y in zip(ykeys, tgtz.T):
            tgtdf[f'{prefix}_{ykey}'] = y
        tgtdf[f'a{prefix}'] = tgtdf[modkey] - tgtdf[f'{prefix}_{biaskey}']
        tgtdf[f'e{prefix}'] = tgtdf[modkey] / tgtdf[f'{prefix}_{ratiokey}']
        tgtdf[f'{prefix}_DIST'] = mod.nn(tgtx, k=1)[0]
