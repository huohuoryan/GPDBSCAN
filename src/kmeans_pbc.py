import numpy as np

class KMeansPBC:
    def __init__(self, n_clusters, box_lengths, max_iter=300, tol=1e-4, random_state=None):
        """
        Parameters
        ----------
        n_clusters : int
            聚类数 k
        box_lengths : array-like, shape (n_features,)
            每个维度的周期长度
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差
        random_state : int or None
            随机数种子
        """
        self.n_clusters = n_clusters
        self.box_lengths = np.asarray(box_lengths, dtype=float)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _pbc_distance(self, X, centers):
        """
        计算周期性边界条件下的距离矩阵
        """
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]

        # 关键修复：按特征维度独立处理周期性
        for d in range(diff.shape[-1]):
            dim_diff = diff[..., d]
            period = self.box_lengths[d]

            # 周期性修正：如果差值超过半周期，用周期长度调整
            mask = np.abs(dim_diff) > period / 2
            dim_diff[mask] = -np.sign(dim_diff[mask]) * (
                    period - np.abs(dim_diff[mask])
            )
        dist_sq = np.sum(diff ** 2, axis=2)
        return np.sqrt(dist_sq)

    def fit(self, X):
        """
        执行 K-Means 聚类（PBC 版本）
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if len(self.box_lengths) != n_features:
            raise ValueError("box_lengths must match number of features")

        rng = np.random.default_rng(self.random_state)
        # 初始化中心
        centers = X[rng.choice(n_samples, self.n_clusters, replace=False)]

        for it in range(self.max_iter):
            # 分配簇
            distances = self._pbc_distance(X, centers)
            labels = np.argmin(distances, axis=1)

            # 保存旧中心
            old_centers = centers.copy()

            # 更新中心（考虑周期边界）
            new_centers = []
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    # 空簇重新随机选一个点
                    new_centers.append(X[rng.integers(0, n_samples)])
                    continue

                # 转换到相对第一个点的坐标系（避免跨边界平均错误）
                ref_point = members[0]
                shifted = members.copy()
                for d in range(n_features):
                    delta = shifted[:, d] - ref_point[d]
                    delta[delta >  self.box_lengths[d] / 2] -= self.box_lengths[d]
                    delta[delta < -self.box_lengths[d] / 2] += self.box_lengths[d]
                    shifted[:, d] = ref_point[d] + delta
                center = np.mean(shifted, axis=0) % self.box_lengths
                new_centers.append(center)

            centers = np.array(new_centers)

            # 检查收敛
            center_shift = np.linalg.norm(centers - old_centers)
            if center_shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        distances = self._pbc_distance(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
