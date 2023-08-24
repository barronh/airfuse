__all__ = ['WeightedEnsemble', 'distweight']


def distweight(
    df, distkeys, valkeys, modkey='NAQFC', ykey='FUSED', power=-2, add=True,
    L=1, k=0.3, x0=125, **scale_kw
):
    """
    Arguments
    ---------
    df : pandas.DataFrame
    distkeys : iterable
        List of distance keys
    valkeys : iterable
        List of  value keys (same order as distkeys)
    modkey : str
        Model name ('NAQFC' or 'GEOSCF')
    ykey : str
        Name for weighted output (e.g., 'FUSED')
    power : int
        Power to use in distance weights (default: -2)
    add : bool
        If True, add the weights and final results to the inptu dataframe
    L : int
        Logistic function numerator (default: 1)
    k : float
        Logistic distance scaling function (default: 0.3)
    x0 : float
        Logistic reference distance (default: 125) must be in teh same distance
        units as the distkeys.
    scale_kw : mappable
        If provided, must be in valkeys and is used to scale the nominal
        weights of values from that valkey

    Results
    -------
    outdf : pandas.DataFrame
        Dataframe with weights and final fused product
    """
    import numpy as np
    rename = dict(zip(distkeys, valkeys))
    dists = df[distkeys].rename(columns=rename)
    vals = df[valkeys]
    wgts = (dists**power).where(~vals.isna()).fillna(0)
    # If a distance was zero, set weight to huge
    wgts = wgts.where(wgts != np.inf).fillna(1e20)
    for scalekey, scaleval in scale_kw.items():
        wgts[scalekey] = wgts[scalekey] * scaleval

    mindist = dists.min(axis=1)
    totwgt = wgts.sum(axis=1)
    nwgts = wgts.divide(totwgt, axis=0).fillna(0)
    bc_wgt = L / (1 + np.exp(k * (mindist - x0)))
    bc_wgt = bc_wgt.where(totwgt > 0).fillna(0)
    outdf = nwgts.multiply(bc_wgt, axis=0)
    outdf[modkey] = (1 - bc_wgt)
    y = (outdf[valkeys + [modkey]] * df[valkeys + [modkey]]).sum(axis=1).where(
        ~df[modkey].isna()
    )
    outdf = outdf.rename(columns=lambda x: x + '_WGT')
    outdf[ykey] = y
    if add:
        for key in outdf.columns:
            df[key] = outdf[key]

    return outdf


class WeightedEnsemble():
    def __init__(self, coordkeys, ekeys, ykey):
        r"""
        Develop an additive model to fuse ensemble members of the form:

        yhat_fused = \sum_e{\alpha_e * yhat_e}

        where:
        * yhat_fused : final results that is fused from ensemble estimates,
        * yhat_e : an estimate by an ensemble member (e), and
        * alpha_e : multiplicative coefficient for the ensemble member (e).

        alpha_e = a_e + \sum_c{b_{e,c} * c}

        * a_e is an intercept specific to the ensemble member
        * b_{e,c} is a slope specific to coordinate (c) and ensemble member (e)
        * c is the coordinate value

        All a_e and b_{e,c} are optimized to minimize the least squares
        optimization with respect to a reference truth (ykey)

        Arguments
        ---------
        coordkeys : iterable
          Coordinate keys to be found within a dataframe during fitting and
          prediction
        ekeys : iterable
          Ensemble member keys to be found within a dataframe during fitting
          and prediction
        ykey : str
          Reference y value to be found within a dataframe during fitting
        """
        self.coordkeys = tuple(coordkeys)
        self.ekeys = tuple(ekeys)
        self.ykey = ykey

    def fit(self, X, y=None, m0=None, **kwds):
        """
        Adjust all coefficients (b_{e,c}) and intercepts (a_e) such as to
        optimize the least squares difference between y and yhat_fused ().

        Arguments
        ---------
        X : pandas.DataFrame or array
          Dataframe that contains coordkeys, ekeys and (of y is None) ykey.
          or array where columns match coordkeys + ekeys
        y : array-like
          Values to use as a reference for optimization. Defaults to X[ykey]
          if X is a dataframe. Otherwise, it is required.
        m0 : array-like
          Initial coefficients sape (len(coordkeys) + 1) * len(ekeys). Defaults
          to all zeros.
        kwds : mappable
          passed to scipy.optimize.least_squares

        Returns
        -------
        None
        """
        import numpy as np
        import pandas as pd
        import scipy.optimize
        import warnings

        if isinstance(X, pd.DataFrame):
            allkeys = list(self.coordkeys + self.ekeys + (self.ykey,))
            records = X[allkeys].copy()
            records['ONE'] = 1
            cpi = records[list(self.coordkeys + ('ONE',))].values
            self._coords_pl_intercept = cpi
            self._models = records[list(self.ekeys)].values
            if y is None:
                y = records[self.ykey].values
            self._yref = records[self.ykey].values
        else:
            self._coords_pl_intercept = X[:, :-list(self.ekeys)].values
            self._models = X[:, -list(self.ekeys):].values
            if y is None:
                raise ValueError('When X is not a DataFrame, y is required')
            self._yref = y

        if m0 is None:
            ncoords = len(self.coordkeys) + 1
            self.m0 = m0 = np.zeros(ncoords * len(self.ekeys), dtype='f')

        self.mopt = scipy.optimize.least_squares(self._multinres, m0, **kwds)
        if self.mopt.status < 1:
            warnings.warn('Convergence error')

    def predict(self, X):
        """
        Fuse multiple ensemble predictions into one based estimate.

        Arguments
        ---------
        df : pandas.DataFrame
          must have coordkeys and ekeys specified at initialization.

        Returns
        -------
        yhat : array
          Best estimate by fusing results
          (get_alphas(df=df) * df[ekeys]).sum(1)
        """
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=self.coordkeys + self.ekeys)

        alphas = self.get_alphas(df=df)
        models = df[list(self.ekeys)].values
        yhat = (alphas * models).sum(1)
        return yhat

    def get_alphas(self, x=None, df=None):
        """
        Calculate weighting factors for all ensemble members.

        Arguments
        ---------
        x : array
          Length (n+1) * m where n is the number of coordiantes and m is the
          number of models being fused. Defaults to x from fitting.
        df : pandas.DataFrame
          This must have coordkeys specified at initialization. If not
          provided, use data fram fitting

        Returns
        -------
        alphas : array
          Weighting factor array of shape (r, m) where r is the number of
          records in df and m is the number of models.
          yhat = (alphas * models).sum(1)
        """
        import numpy as np

        if x is None:
            x = self.mopt.x
        if df is not None:
            allkeys = list(self.coordkeys)
            records = df[allkeys].copy()
            records['ONE'] = 1
            cpi = records[list(self.coordkeys + ('ONE',))].values
            coords_pl_intercept = cpi
        else:
            coords_pl_intercept = self._coords_pl_intercept

        nmodels = len(self.ekeys)
        nrecords = coords_pl_intercept.shape[0]
        ncoords = len(self.coordkeys) + 1  # plus one for ONE
        alphas = np.zeros((nrecords, len(self.ekeys)), dtype='f')
        for mi in range(nmodels):
            start = ncoords * mi
            end = start + ncoords
            alphas[:, mi] = (x[start:end] * coords_pl_intercept).sum(1)

        return alphas

    def _multinres(self, x):
        """
        Calculate residual (yhat - yref) for fitting and not mean to be used
        outside
        """
        # import numpy as np
        # Get alpha_e
        alphas = self.get_alphas(x)
        models = self._models
        yhat = (alphas * models).sum(1)
        res = yhat - self._yref
        # Consider adding a penalty as the deviation from 1. This would force
        # the optimization toward a set of alphas
        #
        # penalty = np.abs(alphas.sum(1) - 1)
        # res = res * penalty
        return res
