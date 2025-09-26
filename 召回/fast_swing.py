# -*- coding: utf-8 -*-
# @Time: 2025/9/26 20:30
# @Author: Haocheng Xi
# @File: fast_swing.py
# @Software: PyCharm


import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch


class FastRecommender:
    def __init__(self,
                 dataset_path="../ml-100k/ratings.csv",
                 use_rating=False,
                 top_k=50,
                 lastn=30,
                 sim_mode="cosine",
                 alpha=10.0):
        # 路径安全
        p = Path(dataset_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / dataset_path
        self.dataset_path = str(p)

        self.use_rating = use_rating
        self.top_k = top_k
        self.lastn = lastn
        self.sim_mode = sim_mode
        self.alpha = float(alpha)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA"
            print(f"[FastRecommender] 使用设备: cuda ({gpu_name})")
        else:
            self.device = torch.device("cpu")
            print(f"[FastRecommender] 使用设备: cpu")

        # 运行期缓存
        self.user_map = None
        self.item_map = None
        self.user_rev = None
        self.item_rev = None
        self.R = None
        self.item_topk_idx = None
        self.item_topk_val = None
        self.df = None
        self.user_order = []   # 按 CSV 原始行用户首次出现的顺序

    # ---------- 数据加载 ----------
    def load(self):
        df_all = pd.read_csv(self.dataset_path)
        print(f"共{len(df_all):,}条数据")

        df_all["_row"] = np.arange(len(df_all))
        first_seen = (
            df_all.groupby("userId", as_index=False)["_row"]
            .min()
            .sort_values("_row")
        )
        self.user_order = first_seen["userId"].tolist()

        need_cols = [c for c in ["userId", "movieId", "rating", "timestamp"] if c in df_all.columns]
        df = df_all[need_cols].copy()

        if "timestamp" in df.columns:
            df = df.sort_values(["userId", "timestamp"])
            df = df.groupby("userId", group_keys=False).tail(self.lastn)
        else:
            df = df.groupby("userId", group_keys=False).tail(self.lastn)

        self.df = df  # 处理后的数据用来建矩阵

        # 连续化 id
        uids = df["userId"].astype("int64").to_numpy()
        iids = df["movieId"].astype("int64").to_numpy()
        uniq_u, uidx = np.unique(uids, return_inverse=True)
        uniq_i, iidx = np.unique(iids, return_inverse=True)

        self.user_map = {int(u): int(ix) for ix, u in enumerate(uniq_u)}
        self.item_map = {int(i): int(ix) for ix, i in enumerate(uniq_i)}
        self.user_rev = {v: k for k, v in self.user_map.items()}
        self.item_rev = {v: k for k, v in self.item_map.items()}

        rows = torch.tensor(uidx, dtype=torch.long)
        cols = torch.tensor(iidx, dtype=torch.long)
        if self.use_rating and "rating" in df.columns:
            vals = torch.tensor(df["rating"].astype("float32").to_numpy())
        else:
            vals = torch.ones(len(df), dtype=torch.float32)

        m = len(uniq_u)
        n = len(uniq_i)
        indices = torch.stack([rows, cols], dim=0)
        R = torch.sparse_coo_tensor(indices, vals, size=(m, n)).coalesce().to(self.device)
        self.R = R

        print(f"[load] users={m}, items={n}, nnz={self.R._nnz():,}")

    # ---------- 相似度 ----------
    def _topk_from_dense(self, M, k):
        n = M.size(0)
        M = M.clone()
        M.fill_diagonal_(0)
        vals, idx = torch.topk(M, k=min(k, n - 1), dim=1)
        return idx, vals

    def _item_sim_cosine(self):
        Rt = self.R.transpose(0, 1).to(self.device)
        cooc = torch.matmul(Rt.to_dense(), self.R.to_dense())  # (I x I)
        norms = torch.sqrt(torch.clamp(torch.diag(cooc), min=1e-8))
        inv = 1.0 / norms
        S = (cooc * inv.view(-1, 1)) * inv.view(1, -1)
        return S

    def _item_sim_swing_approx(self):
        B = (self.R.to_dense() > 0).float()  # (U x I)
        deg_u = torch.clamp(B.sum(dim=1), min=1.0)
        Rt = B.t()  # (I x U)
        weight_u = 1.0 / (self.alpha + torch.clamp(deg_u - 1.0, min=0.0))  # (U,)
        overlap = torch.matmul(Rt, (B * weight_u.view(-1, 1)))  # (I x I)
        W = 0.5 * (overlap + overlap.t())
        return W

    # ---------- 训练 ----------
    def fit(self):
        assert self.R is not None, "请先调用 load()"
        t0 = time.time()
        if self.sim_mode == "cosine":
            S = self._item_sim_cosine()
        else:
            S = self._item_sim_swing_approx()

        idx, val = self._topk_from_dense(S, self.top_k)
        self.item_topk_idx = idx
        self.item_topk_val = val

        # 按 CSV 原始顺序打印
        for u in self.user_order:
            print("处理用户:", u)

        print("[fit] 完成 item 相似度与 topK 邻居构建 (耗时 {:.2f}s)".format(time.time() - t0))
        return self

    # ---------- 推荐 ----------
    def recommend(self, user_raw_id: int, n_rec: int = 10):
        assert self.item_topk_idx is not None, "请先 fit()"
        assert user_raw_id in self.user_map, "用户不在数据集中"

        u = self.user_map[user_raw_id]
        R_dense = self.R.to_dense()
        user_vec = R_dense[u]

        hist_items = torch.nonzero(user_vec > 0, as_tuple=False).view(-1)
        if hist_items.numel() == 0:
            return []

        scores = torch.zeros(R_dense.size(1), device=self.device)
        for it in hist_items.tolist():
            neighbors = self.item_topk_idx[it]
            sims = self.item_topk_val[it]
            scores.index_add_(0, neighbors, sims)

        scores[hist_items] = -1e9
        vals, idx = torch.topk(scores, k=min(n_rec, scores.size(0) - hist_items.numel()))
        recs = [(int(self.item_rev[i.item()]), float(v.item())) for i, v in zip(idx, vals)]
        return recs


if __name__ == "__main__":
    user_id = 27
    model = FastRecommender(
        dataset_path="../ml-100k/ratings.csv",
        use_rating=False,
        top_k=50,
        lastn=30,
        sim_mode="cosine",  #  "swing_approx" 也行？？
        alpha=10.0,
    )
    model.load()
    model.fit()
    recommendations = model.recommend(user_raw_id=user_id, n_rec=10)
    print(f"为用户 {user_id} 推荐的物品: {recommendations}")
