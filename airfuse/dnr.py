__all__ = [
    'DelaunayNeighborsRegressor', 'GroupedDelaunayNeighborsRegressor',
    'FusedDelaunayNeighborsRegressor'
]
__doc__ = """
DelaunayNeighborsRegressor : KNeighborsRegressor
    Similar to KNeighborsRegressor, but adds spatial diagram capability.
    Adds the capability to increase weights of delaunay/voronoi neighbors
    from a the KNeighbors typically used. Allows for individual sample
    weights during fitting.

GroupedDelaunayNeighborsRegressor : DelaunayNeighborsRegressor
    Similar to DelaunayNeighborsRegressor, but allows for distinct sets
    of neighbors. For example, separately calculate delaunay for airnow
    and for purpleair by adding a "src" column. Optionally, supply a
    dictionary of delaunay_weights keyed by group. Initial wieghts are
    w = weights(d) * delaunay_weights(d). All weights from all groups
    are simultaneously normalized.


FusedDelaunayNeighborsRegressor : DelaunayNeighborsRegressor
    Create separate DelaunayNeighborsRegressor Surfaces and then fuse them
    together using weights calculated from DelaunayNeighborsRegressor
    objects and X
"""
import numpy as np
import copy
from sklearn.neighbors import KNeighborsRegressor


def _isdelaunay(tgtx, X, n_jobs=None):
    """
    Find out which coordinates in X are Delaunay neighbors of tgtx. The
    Delaunay and Voronoi diagrams are dual graphs, where the Delaunay
    creates triangles where any Voronoi centroids are connected by a
    line.

    Arguments
    ---------
    tgtx : array-like
        Array-like of shape (n_samples, n_features) or (n_features,) where
        * n_samples represents the number of coordinates to test,
        * n_features typically 2 (x, y) representing the cartesian location.
    X : array-like
        Array-like of shape (n_samples, k, n_features) or (k, n_features)
        * k is the number of neighbors to check (e.g, k=10 or k=30),

    Returns
    -------
    isdn : array-like
        Array of shape (n_samples, k) where True if the point at the kth
        element is a Delaunay neighbor for the nth sammple
    """
    from scipy.spatial import Delaunay
    tgtx = np.array(tgtx, ndmin=2)
    X = np.array(X, ndmin=3)
    if n_jobs is not None:
        from joblib import Parallel, delayed
        n = X.shape[0]
        ns = [n // n_jobs] * n_jobs
        ns[-1] += (n - sum(ns))
        print('Cells per job', ns)
        se = np.cumsum([0] + ns)
        with Parallel(n_jobs=n_jobs, verbose=10) as par:
            processed_list = par(
                delayed(_isdelaunay)(tgtx[s:e], X[s:e], n_jobs=None)
                for s, e in zip(se[:-1], se[1:])
            )
        yout = np.ma.concatenate(processed_list, axis=0)
        return yout

    outshape = tgtx.shape[-2], X.shape[-2]
    k = X.shape[1]
    n = tgtx.shape[0]
    isdn = np.zeros((n, k), dtype='bool')
    didx = np.arange(k)

    # Get locations with target as last point
    vnxy = np.concatenate([X, np.asarray(tgtx)[:, None, :]], axis=1)

    # For each target with near points, calc Delaunay and find neighbors
    # Tag neighbors as is Delaunay Neighbor isdn
    for i in range(n):
        newxy = vnxy[i]
        tric = Delaunay(newxy)
        tri_indicies, tri_neighbors = tric.vertex_neighbor_vertices
        cidx = tri_neighbors[tri_indicies[k]:tri_indicies[k + 1]]
        isdn[i] = np.in1d(didx, cidx)

    # if there are not neighbors, then you are in the same cell
    # In that case, closest three should be used have 100%
    isdn[np.where(~isdn.any(1))[0], :3] = True

    return isdn.reshape(outshape)


def _get_weights(dist, weights):
    """COPIED FROM sklearn

    Get the weights from an array of distances and a parameter ``weights``.

    Assume weights have already been validated.

    Parameters
    ----------
    dist : ndarray
        The input distances.

    weights : {'uniform', 'distance'}, callable or None
        The kind of weighting used.

    Returns
    -------
    weights_arr : array of the same shape as ``dist``
        If ``weights == 'uniform'``, then returns None.
    """
    import numpy as np
    if weights in (None, "uniform"):
        return None

    if weights == "distance":
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        if dist.dtype is np.dtype(object):
            for point_dist_i, point_dist in enumerate(dist):
                # check if point_dist is iterable
                # (ex: RadiusNeighborClassifier.predict may set an element of
                # dist to 1e-6 to represent an 'outlier')
                if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                    dist[point_dist_i] = point_dist == 0.0
                else:
                    dist[point_dist_i] = 1.0 / point_dist
        else:
            with np.errstate(divide="ignore"):
                dist = 1.0 / dist
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        return dist

    if callable(weights):
        return weights(dist)


_dnr_parameter_constraints: dict = copy.deepcopy(
    KNeighborsRegressor._parameter_constraints
)
_dnr_parameter_constraints["weights"].append(dict)


class DelaunayNeighborsRegressor(KNeighborsRegressor):
    """
    Replaces nna_methods.NNA with a sklearn implementation. Builds on sklearn's
    KNeighborsRegressor by adding a customizable weight that depends on whether
    a neighbor is a Delaunay
    Arguments
    ---------
    delaunay_weights : {str, int, float, callable}, default='only'
        Function to modify weights used in the prediction.  Possible values:

        - 'only' : only use Delaunay Neighbors. Delaunay neighbors use weight
          calculated by weights. Weights for all other points in each
          neighborhood are set to zero.
        - 'equal' : only use Delaunay Neighbors. Delaunay neighbors are equally
          weighted (i.e., 1). Weights for all other points in each neighborhood
          are set to zero.
        - 'none' or None : no special treatment for Delaunay Neighbors. This
          is essentially KNeighborsRegressor, but with the additional
          sample_weight option.
        - int or float : multiply weights of Delaunay Neighbors by scaling
          factor. All other points in each neighborhood are unchanged.
        - [callable] : a user-defined function which accepts an array
          identifying delaunay neighbors (True/False) and and array of
          pre-calculated weights (see weights), and returns an array of the
          same shape containing the updated weights.

        Only Delaunay Neighbors are used by default.

        See the following example for a demonstration of the impact of
        different weighting schemes on predictions:
        :ref:`sphx_glr_auto_examples_neighbors_plot_regression.py`.

    For all other arguments and keywords, see KneighborsRegressor
    """
    _parameter_constraints: dict = _dnr_parameter_constraints

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
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            weights=weights
        )
        if isinstance(delaunay_weights, str):
            if delaunay_weights == "none" or delaunay_weights is None:
                delaunay_weights = None
            elif delaunay_weights == "equal":
                def delaunay_weights(isdn, weights):
                    return np.where(isdn, 1., 0.)
            elif delaunay_weights == "only":
                def delaunay_weights(isdn, weights):
                    dwgt = np.where(isdn, 1., 0.)
                    if weights is not None:
                        dwgt = weights * dwgt
                    return dwgt
        elif isinstance(delaunay_weights, (int, float)):
            vnscale = delaunay_weights

            def delaunay_weights(isdn, weights):
                return np.where(isdn, vnscale, 1.) * weights

        self.delaunay_weights = delaunay_weights

    def fit(self, X, y, sample_weight=None):
        """Fit the k-nearest neighbors regressor from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features + 1)
        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        sample_weight : {array-like, sparse matrix} of shape (n_samples,)
            Row-specific weights (e.g, inverse uncertainty) to be multiplied \
            by the distance based weight from the weight function.

        Returns
        -------
        self : KNeighborsRegressor
            The fitted k-nearest neighbors regressor.
        """
        out = super().fit(X, y)
        if sample_weight is None:
            self._wgt = sample_weight
        else:
            self._wgt = np.asarray(sample_weight)
        return out

    def _get_weights(self, X, weightf=None):
        if self.weights == "uniform":
            # In that case, we do not need the distances to perform
            # the weighting so we do not compute them.
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        srcX = self._fit_X[neigh_ind]
        if weightf is None:
            weightf = self.weights
        weights = _get_weights(neigh_dist, weightf)
        if self.delaunay_weights is not None:
            # self should be treated as Delaunay Neighbor
            isdn = _isdelaunay(X, srcX, n_jobs=self.n_jobs) | (neigh_dist == 0)
            weights = self.delaunay_weights(isdn, weights)

        if self._wgt is not None:
            weights *= self._wgt[neigh_ind]

        return neigh_ind, weights

    def predict(self, X):
        """Predict the target for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed', or None
            Test samples. If `None`, predictions for all indexed points are
            returned; in this case, points are not considered their own
            neighbors.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        """
        neigh_ind, weights = self._get_weights(X)
        _y = self._y

        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))
        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty(
                (neigh_ind.shape[0], _y.shape[1]), dtype=np.float64
            )
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


class GroupedDelaunayNeighborsRegressor(DelaunayNeighborsRegressor):
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
    ):
        """
        Create multiple DNRs so that each group has its own Delaunay neighbors,
        but weights are normalized simultaneously.

        Essentially, the delaunay_weights function is applied once for each
        group -- ignoring points from other groups.

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
        self._dnrs = {}

    def fit(self, X, y, sample_weight=None, groups=None):
        self._dnrs = {}
        self._ug = np.unique(groups)
        for g in self._ug:
            ismine = groups == g
            params = self.get_params()
            params.pop('fusion_weight', None)
            dnr = DelaunayNeighborsRegressor(**params)
            if sample_weight is None:
                sw = None
            else:
                sw = sample_weight[ismine]

            dnr.fit(X[ismine], y[ismine], sample_weight=sw)
            self._dnrs[g] = dnr

    def predict(self, X):
        wgts = []
        inds = []
        dnrs = self._dnrs
        for g in self._ug:
            dnr = dnrs[g]
            if isinstance(self.weights, dict):
                weightf = self.weights[g]
            else:
                weightf = self.weights
            neigh_ind, weights = dnr._get_weights(X, weightf)
            if weights is None:
                weights = np.ones_like(neigh_ind, dtype='d')
            wgts.append(weights)
            inds.append(neigh_ind)

        # Concatenate on the neighbors index
        weights = np.concatenate(wgts, axis=-1)
        denom = np.sum(weights, axis=1)
        nout = dnr._y.shape[1]
        y_pred = np.empty(
            (neigh_ind.shape[0], dnr._y.shape[1]), dtype=np.float64
        )
        for j in range(nout):
            ys = np.concatenate([
                self._dnrs[g]._y[ind, j] for g, ind in zip(self._ug, inds)
            ], axis=1)
            num = np.sum(ys * weights, axis=1)
            y_pred[:, j] = num / denom

        if y_pred.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


_fdnr_parameter_constraints: dict = copy.deepcopy(_dnr_parameter_constraints)
_fdnr_parameter_constraints["fusion_weight"] = [dict]


class FusedDelaunayNeighborsRegressor(GroupedDelaunayNeighborsRegressor):
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

    def predict(self, X):
        for g, exdnr in self._dnrs.items():
            break
        nout = exdnr._y.shape[1]
        ndnrs = len(self._dnrs)
        nx = X.shape[0]
        y_preds = np.empty((nx, ndnrs, nout), dtype=np.float64)
        scales = np.empty((nx, ndnrs), dtype=np.float64)
        # order does not matter becuase predictions and scales are
        # consistent, but could choose to enumerate over self._ug
        for di, (g, dnr) in enumerate(self._dnrs.items()):
            y_preds[:, di, :] = dnr.predict(X)
            scales[:, di] = self.fusion_weight[g](dnr, X)
        denom = scales.sum(1)
        y_pred = np.empty((nx, nout), dtype=np.float64)
        for j in range(nout):
            y_pred[:, j] = (scales * y_preds[..., j]).sum(1) / denom
        return y_pred
