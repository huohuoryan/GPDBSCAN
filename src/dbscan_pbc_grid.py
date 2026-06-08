from sklearn.cluster import DBSCAN
import numpy as np
import itertools
from collections import defaultdict

class GPDBSCAN(DBSCAN):
    """
    DBSCAN with support for Periodic Boundary Conditions (PBC) and optional grid-based optimization.
    Optimized for speed: cached neighbor offsets, squared distance, batch padding, float32 arrays.
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None,
                 algorithm='auto', leaf_size=30, p=None, n_jobs=None, use_grid=False):
        super().__init__(eps=eps, min_samples=min_samples, metric=metric,
                         metric_params=metric_params, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, n_jobs=n_jobs)
        self.use_grid = use_grid
        self.eps_sq = eps ** 2
        self.neighbor_offsets = None
        self.grid_shape = None
        self.periodic_mask = None

    def fit(self, X, pbc_lower=0, pbc_upper=1, return_padded_dbs=False, use_grid=None):
        if len(X) == 0:
            return None

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Input must be of shape (n_samples, n_features), but got {X.ndim} dimensions.")

        D = X.shape[1]

        # 设置 pbc 范围
        pbc_lower, pbc_upper = self._prepare_pbc_bounds(pbc_lower, pbc_upper, D, X)

        # Canonicalize periodic dimensions into [0, L).
        L = np.full(D, -1, dtype=np.float32)
        for d in range(D):
            if pbc_lower[d] is None or pbc_upper[d] is None:
                continue
            L[d] = pbc_upper[d] - pbc_lower[d]
            X[:, d] -= pbc_lower[d]

        if use_grid is None:
            use_grid = self.use_grid

        # 缓存邻居偏移
        if self.neighbor_offsets is None or self.neighbor_offsets.shape[1] != D:
            self.neighbor_offsets = np.array(list(itertools.product([-1, 0, 1], repeat=D)), dtype=int)
        self.periodic_mask = L > 0
        self.grid_shape = np.full(D, -1, dtype=np.int32)
        if np.any(self.periodic_mask):
            self.grid_shape[self.periodic_mask] = np.ceil(L[self.periodic_mask] / self.eps).astype(np.int32)

        if use_grid:
            return self._fit_with_grid(X, L)
        else:
            padded_points, source_idx = self._pad_boundary_points(X, self.eps, L)
            self.padded_points_ = padded_points
            self.source_idx_ = source_idx
            db = super().fit(padded_points)
            labels = db.labels_
            new_labels = self._pbc_cluster_merger(labels, source_idx, X.shape[0], padded_points.shape[0])
            self.labels_ = self._renumber_labels(new_labels)
            return self

    def _prepare_pbc_bounds(self, pbc_lower, pbc_upper, D, X):
        pbc_lower = np.asarray(pbc_lower)
        if pbc_lower.ndim == 0:
            pbc_lower = np.full(D, pbc_lower)
        elif len(pbc_lower) != D:
            raise ValueError("pbc_lower length mismatch")

        pbc_upper = np.asarray(pbc_upper)
        if pbc_upper.ndim == 0:
            pbc_upper = np.full(D, pbc_upper)
        elif len(pbc_upper) != D:
            raise ValueError("pbc_upper length mismatch")

        for d in range(D):
            if pbc_lower[d] is None or pbc_upper[d] is None:
                continue
            if pbc_upper[d] <= pbc_lower[d]:
                raise ValueError("pbc_upper must be larger than pbc_lower")
            if np.min(X[:, d]) < pbc_lower[d] or np.max(X[:, d]) > pbc_upper[d]:
                raise ValueError("Data out of PBC range")
        return pbc_lower, pbc_upper

    def _fit_with_grid(self, X, L):
        grid_dict, grid_size = self._build_grid_index(X, self.eps, L)
        labels = -1 * np.ones(X.shape[0], dtype=int)
        visited = np.zeros(X.shape[0], dtype=bool)
        cluster_id = 0

        for i in range(X.shape[0]):
            if visited[i]:
                continue
            neighbors = self._get_neighbor_points(i, X, grid_dict, grid_size, L)
            if len(neighbors) < self.min_samples:
                visited[i] = True
                continue
            labels[i] = cluster_id
            seed_set = set(neighbors)
            while seed_set:
                j = seed_set.pop()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = self._get_neighbor_points(j, X, grid_dict, grid_size, L)
                    if len(j_neighbors) >= self.min_samples:
                        seed_set.update(j_neighbors)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

        self.labels_ = self._renumber_labels(labels)
        return self

    def _build_grid_index(self, X, eps, L):
        grid_size = eps
        grid_indices = np.floor(X / grid_size).astype(int)
        grid_dict = defaultdict(list)
        for idx, grid in enumerate(grid_indices):
            grid_dict[tuple(grid)].append(idx)
        return grid_dict, grid_size

    def _get_neighbor_points(self, point_idx, X, grid_dict, grid_size, L):
        point = X[point_idx]
        grid_coord = np.floor(point / grid_size).astype(int)
        neighbors = []

        for offset in self.neighbor_offsets:
            neighbor_coord = grid_coord + offset
            if np.any(self.periodic_mask):
                neighbor_coord = neighbor_coord.copy()
                neighbor_coord[self.periodic_mask] %= self.grid_shape[self.periodic_mask]
            if np.any((~self.periodic_mask) & (neighbor_coord < 0)):
                continue
            neighbor_key = tuple(neighbor_coord.tolist())
            if neighbor_key in grid_dict:
                pts_idx = grid_dict[neighbor_key]
                pts = X[pts_idx]
                diff = np.abs(pts - point)
                if np.any(self.periodic_mask):
                    diff[:, self.periodic_mask] = np.minimum(
                        diff[:, self.periodic_mask],
                        L[self.periodic_mask] - diff[:, self.periodic_mask],
                    )
                dist_sq = np.sum(diff ** 2, axis=1)
                neighbors.extend(np.array(pts_idx)[dist_sq <= self.eps_sq])
        return neighbors

    def _pad_boundary_points(self, points, eps, L):
        D = points.shape[1]
        shifts_list = []
        src_idx_list = []

        for idx, point in enumerate(points):
            shift_options = []
            boundary_point = False
            for d in range(D):
                if L[d] < 0:
                    shift_options.append([0])
                    continue
                if point[d] < eps:
                    shift_options.append([-1, 0])
                    boundary_point = True
                elif point[d] > L[d] - eps:
                    shift_options.append([1, 0])
                    boundary_point = True
                else:
                    shift_options.append([0])
            if boundary_point:
                for shift_vec in itertools.product(*shift_options):
                    if all(s == 0 for s in shift_vec):
                        continue
                    shifts_list.append(point - np.array(shift_vec) * L)
                    src_idx_list.append(idx)

        if shifts_list:
            padded_points = np.vstack([points, np.array(shifts_list, dtype=np.float32)])
        else:
            padded_points = points.copy()
        return padded_points, np.array(src_idx_list)

    def _pbc_cluster_merger(self, labels, source_idx, Npoints, Npadded_points):
        merged_cluster = labels[:Npoints].copy()
        matches = []
        for i in range(Npadded_points - Npoints):
            IDX = source_idx[i]
            label_pp = labels[Npoints + i]
            label_src = merged_cluster[IDX]
            if label_pp == label_src:
                continue
            if label_src == -1:
                merged_cluster[IDX] = label_pp
                continue
            if label_pp == -1:
                continue
            found = False
            for match in matches:
                if label_pp in match or label_src in match:
                    match.update([label_pp, label_src])
                    found = True
                    break
            if not found:
                matches.append(set([label_pp, label_src]))
        for match in matches:
            correct_label = min(match)
            for wrong_label in match:
                if wrong_label != correct_label:
                    merged_cluster[merged_cluster == wrong_label] = correct_label
        return merged_cluster

    def _renumber_labels(self, labels):
        new_labels = labels.copy()
        unique_labels = np.unique(new_labels[new_labels >= 0])
        label_map = {old: new for new, old in enumerate(unique_labels)}
        for old, new in label_map.items():
            new_labels[new_labels == old] = new
        return new_labels
