__all__ = [
    'BCDelaunayNeighborsRegressor',
    'BCGroupedDelaunayNeighborsRegressor',
    'BCFusedDelaunayNeighborsRegressor'
]

from ._core import DelaunayNeighborsRegressor, \
    GroupedDelaunayNeighborsRegressor
from ._core import _fdnr_parameter_constraints


class _BCRegressor:
    # identifies intermediates to be calculated internally for the regressor
    _needs = {
        'obs': ('obs_dnr', ),
        'abc': ('obs_dnr', 'mod_dnr', 'mod_abc',),
        'mbc': ('obs_dnr', 'mod_dnr', 'mod_mbc',),
        'ambc': ('obs_dnr', 'mod_dnr', 'mod_mbc', 'mod_abc', 'mod_ambc',),
        'individual': (
            'obs_dnr', 'mod_dnr', 'mod_mbc', 'mod_abc', 'mod_ambc',
        ),
        'best': (
            'mod_dnr',
            'obs_dnr', 'mod_abc', 'mod_mbc', 'mod_ambc',
            'obs_dnr_w', 'mod_abc_w', 'mod_mbc_w', 'mod_ambc_w',
            'mod_bbc'
        ),
        'all': (
            'mod_dnr',
            'obs_dnr', 'mod_abc', 'mod_mbc', 'mod_ambc',
            'obs_dnr_w', 'mod_abc_w', 'mod_mbc_w', 'mod_ambc_w',
            'mod_bbc'
        ),
        'debug': (
            'mod_dnr',
            'obs_dnr', 'mod_abc', 'mod_mbc', 'mod_ambc',
            'obs_dnr_w', 'mod_abc_w', 'mod_mbc_w', 'mod_ambc_w',
            'mod_bbc'
        ),
    }
    # identifies results to be returned to the user
    _returns = {
        'obs': ('obs_dnr', ),
        'abc': ('mod_abc',),
        'mbc': ('mod_mbc',),
        'ambc': ('mod_ambc',),
        'individual': ('obs_dnr', 'mod_abc', 'mod_mbc', 'mod_ambc',),
        'best': ('mod_bbc',),
        'debug': (
            'obs_dnr', 'mod_abc', 'mod_mbc', 'mod_ambc',
            'obs_dnr_w', 'mod_mbc_w', 'mod_abc_w', 'mod_ambc_w',
            'mod_bbc'
        ),
        'all': (
            'obs_dnr', 'mod_abc', 'mod_mbc', 'mod_ambc',
            'mod_bbc'
        ),
    }

    @property
    def feature_names_out_(self):
        return list(self._returns[self.how])

    def fit(self, X, y, sample_weight=None, groups=None):
        """
        Arguments
        ---------
        X : n x 3 array
            X should have n records and 3 columns x, y, initial-estimate
        y : n x m
            y should have n records and m features
        sample_weight : array
            sample_weight (if provided) should have n records
        groups : array
            groups (if provided) should have n records

        Returns
        -------
        None
        """
        import numpy as np
        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_predict

        if hasattr(X, 'columns'):
            if all([isinstance(k, str) for k in X.columns]):
                self.feature_names_in_ = tuple(X.columns)

        X = np.asarray(X)
        y = np.asarray(y)
        nr = y.shape[0]
        _X = X[:, :-1]
        if y.ndim == 1:
            y = y[:, None]

        if self.how in ('best', 'all', 'debug'):
            # Create a cross-validation set of all predictions
            _how = self.how
            self.set_how('individual')
            tmpp = self.get_params()
            tmpp.pop('how', None)
            # long-term, random_state, n_splits, and shuffle and n_neighbors
            # should be additional arguments
            # tmpparams['how'] = 'all'
            # _cvregr = BCGroupedDelaunayNeighborsRegressor(**tmpparams)
            _cvregr = self
            fitkwds = dict(sample_weight=sample_weight, groups=groups)
            kf = KFold(random_state=42, n_splits=5, shuffle=True)
            _ycv = cross_val_predict(_cvregr, X, y, params=fitkwds, cv=kf)
            # Get the squared error per site
            serr = (_ycv - y)**2
            # Create a neighbor-based interpolator of err -- each prediction is
            # the mean squared error, which is used to optimize the blending
            cvkn = GroupedDelaunayNeighborsRegressor(**tmpp)
            cvkn.fit(_X, serr, **fitkwds)
            self._serrkn = cvkn
            self.set_how(_how)

        ny = y.shape[1]
        _y = np.empty((nr, ny + 1), dtype=y.dtype)
        _y[:, 1:] = y
        _y[:, 0] = X[:, -1]
        fitkwds = dict(sample_weight=sample_weight, groups=groups)
        return super().fit(_X, _y, **fitkwds)

    def set_how(self, how):
        """
        Change how after fitting and diagnostics.
        """
        self.how = how

    def predict_dataframe(self, X, floor=0):
        """
        Returns predict result with column names. See predict for more details.
        """
        import pandas as pd
        v = self.predict(X, floor=floor)
        return pd.DataFrame(v, columns=self.feature_names_out_)

    def predict(self, X, floor=0):
        """
        Arguments
        ---------
        X : n x 3 array
            X should have n records and 3 columns x, y, initial-estimate
        floor : float
            Minimum value to return
        return_type : str
            array or pandas

        Returns
        -------
        y : n x m
            y should have n records and m features
        """
        import numpy as np
        how = self.how
        assert how in self._needs
        needs = self._needs[how]
        returns = self._returns[how]
        X = np.asarray(X)
        _X = X[:, :-1]
        raw_mod = X[:, -1:]
        _y = super().predict(_X)
        store = {}
        obs_dnr = store['obs_dnr'] = _y[:, 1:]
        mod_dnr = store['mod_dnr'] = _y[:, :1]
        if 'mod_abc' in needs:
            abc = np.maximum(floor, raw_mod + obs_dnr - mod_dnr)
            store['mod_abc'] = abc
        if 'mod_mbc' in needs:
            mbc = np.maximum(floor, raw_mod * obs_dnr / mod_dnr)
            store['mod_mbc'] = mbc

        if 'mod_ambc' in needs:
            store['mod_ambc'] = np.mean([abc, mbc], axis=0)

        if 'mod_bbc' in needs:
            keys = self._returns['individual']
            indiv = np.concatenate([store[k] for k in keys], axis=-1)
            serr = self._serrkn.predict(_X)
            w = serr**-1
            store['mod_bbc'] = (
                (indiv * w).sum(-1, keepdims=True)
                / w.sum(-1, keepdims=True)
            )

        if 'obs_dnr_w' in returns:
            store['obs_dnr_w'] = w[:, :1]
        if 'mod_abc_w' in returns:
            store['mod_abc_w'] = w[:, 1:2]
        if 'mod_mbc_w' in returns:
            store['mod_mbc_w'] = w[:, 2:3]
        if 'mod_ambc_w' in returns:
            store['mod_ambc_w'] = w[:, 3:]

        if len(returns) == 1:
            out = store[returns[0]]
        else:
            out = np.concatenate([store[k] for k in returns], axis=-1)

        return np.squeeze(out)


class BCDelaunayNeighborsRegressor(_BCRegressor, DelaunayNeighborsRegressor):
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        delaunay_weights="only",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        how='best'
    ):
        """
        Create multiple DNRs so that each group has its own Delaunay neighbors,
        but weights are normalized simultaneously.

        Essentially, the delaunay_weights function is applied once for each
        group -- ignoring points from other groups.

        X must have spatial (x, y) and model initial estimate (mod).

        **kwds:
            See DelaunayNeighborRegressor
        how : str
            obs: returns interpolated observation
            abc: returns additive bias correction
            mbc: returns multiplicative bias correction
            ambc: returns where abc is negative, mbc, else mean(abc, mbc)
            individual: returns obs, abc, mbc, ambc - primarily for tuning best
            best: returns weighted obs, abc, mbc, and ambc by CV MSE**-2
            all: returns idividual + best

        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            delaunay_weights=delaunay_weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.how = how


class BCGroupedDelaunayNeighborsRegressor(
    _BCRegressor, GroupedDelaunayNeighborsRegressor
):
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        delaunay_weights="only",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        how='best'
    ):
        """
        Create multiple DNRs so that each group has its own Delaunay neighbors,
        but weights are normalized simultaneously.

        Essentially, the delaunay_weights function is applied once for each
        group -- ignoring points from other groups.

        X must have spatial (x, y) and model initial estimate (mod).

        **kwds:
            See DelaunayNeighborRegressor
        how : str
            obs: returns interpolated observation
            abc: returns additive bias correction
            mbc: returns multiplicative bias correction
            ambc: returns where abc is negative, mbc, else mean(abc, mbc)
            individual: returns obs, abc, mbc, ambc - primarily for tuning best
            best: returns weighted obs, abc, mbc, and ambc by CV MSE**-2
            all: returns idividual + besst

        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            delaunay_weights=delaunay_weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.how = how


class BCFusedDelaunayNeighborsRegressor(
    _BCRegressor, GroupedDelaunayNeighborsRegressor
):
    _parameter_constraints: dict = _fdnr_parameter_constraints

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        delaunay_weights="only",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        fusion_weight=None
    ):
        """
        Create multiple DNRs so that each group has its own Delaunay neighbors,
        and fuse the predictions using fusion weights based on dnr and X.

        fusion_weight : dict
            Dictionary with functions to weight each grouped DNR. Keys must
            match unique groups passed during fit. Values must take two
            arguments dnr and X where dnr is the DelaunayNeighborRegressor
            for that group and X is the target for predictions.
        **kwds:
            See DelaunayNeighborRegressor
        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            delaunay_weights=delaunay_weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        if fusion_weight is None:
            fusion_weight = {}
        self.fusion_weight = fusion_weight
