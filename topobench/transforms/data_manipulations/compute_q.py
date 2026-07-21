import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class QData(Data):
    """
    Data subclass that teaches PyG how to offset the Q_* fields during batching.

    PyG only knows how to auto-increment fields it recognizes by name
    (e.g. 'edge_index'). For custom fields like ours, we override __inc__
    so that node ids (Q_center, Q_v0, Q_v1) get offset by num_nodes and
    edge ids (Q_edge) get offset by the number of edges, exactly the way
    edge_index is offset internally. This makes batch.Q_center / Q_v0 /
    Q_v1 / Q_edge come out already globally-correct after collation,
    for any batch size.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("Q_center", "Q_v0", "Q_v1"):
            return self.num_nodes
        if key == "Q_edge":
            return self.incidence_1.size(1)  # number of edges in this graph
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("Q_center", "Q_v0", "Q_v1", "Q_edge"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class ComputeQDict(BaseTransform):
    """TopoBench-compliant transform step to attach flat Q tensors to graphs."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data):
        B1_tensor = getattr(data, "incidence_1", None)
        B2_tensor = getattr(data, "incidence_2", None)

        if B1_tensor is not None and B2_tensor is not None:
            B1 = safe_to_numpy(B1_tensor)
            B2 = safe_to_numpy(B2_tensor)

            centers, v0s, v1s, edge_idxs = compute_q_tensors(B1, B2, verbose=False)
            data.Q_center = centers
            data.Q_v0 = v0s
            data.Q_v1 = v1s
            data.Q_edge = edge_idxs

            # Swap in the __inc__-aware subclass so PyG's default collate
            # batches Q_* fields with correct offsets, for any batch size.
            # Safe because QData adds no new required state -- it's the
            # same underlying attribute store as Data, just with custom
            # __inc__/__cat_dim__ hooks.
            if data.__class__ is not QData:
                data.__class__ = QData

        return data


def safe_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a PyTorch tensor (sparse or dense), torch_sparse SparseTensor,
    or numpy array into a dense numpy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor

    # Handle torch_sparse SparseTensor wrappers
    if hasattr(tensor, "to_torch_sparse_coo_tensor"):
        tensor = tensor.to_torch_sparse_coo_tensor()

    # Handle sparse tensors with to_dense method
    if hasattr(tensor, "to_dense"):
        try:
            tensor = tensor.to_dense()
        except Exception:
            pass

    # Fallback checking for .is_sparse attribute
    if hasattr(tensor, "is_sparse") and tensor.is_sparse:
        try:
            tensor = tensor.to_dense()
        except Exception:
            pass

    if hasattr(tensor, "numpy"):
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        return tensor.numpy()

    return np.array(tensor)


# ============================================================
# CORE LOGIC
# ============================================================

def triangles_from_B1_B2(
    B1: np.ndarray,
    B2: np.ndarray,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """
    Recover the edge list and triangle list from incidence matrices.
    """
    edges: list[tuple[int, ...]] = []
    for j in range(B1.shape[1]):
        verts = np.where(B1[:, j] != 0)[0]
        edges.append(tuple(int(v) for v in verts))

    triangles: list[tuple[int, ...]] = []
    for j in range(B2.shape[1]):
        edge_ids = np.where(B2[:, j] != 0)[0]
        verts: set[int] = set()
        for e in edge_ids:
            verts.update(edges[e])
        triangles.append(tuple(verts))

    return edges, triangles


def compute_q_tensors(
    B1: np.ndarray,
    B2: np.ndarray,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the flattened Q representation: for every (node, triangle) pair
    where the node is a vertex of the triangle, record the node ('center'),
    the two other triangle vertices ('v0', 'v1'), and the index of the edge
    opposite to the node ('edge'). Equivalent in content to the original
    per-node dict, but as four parallel 1-D tensors so it batches like any
    other PyG field (via QData.__inc__).
    """
    edges, triangles = triangles_from_B1_B2(B1, B2)

    if verbose:
        print(
            f"[compute_q_tensors] {B1.shape[0]} nodes, "
            f"{len(edges)} edges, {len(triangles)} triangles..."
        )

    centers: list[int] = []
    v0s: list[int] = []
    v1s: list[int] = []
    edge_idxs: list[int] = []

    for tri in triangles:
        if len(tri) < 3:
            # degenerate 2-cell -- skip
            continue

        v0, v1, v2 = tri[0], tri[1], tri[2]

        for center, a, b in ((v0, v1, v2), (v1, v0, v2), (v2, v0, v1)):
            for k, (e0, e1) in enumerate(edges):
                if (e0 == a and e1 == b) or (e1 == a and e0 == b):
                    centers.append(center)
                    v0s.append(a)
                    v1s.append(b)
                    edge_idxs.append(k)

    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(v0s, dtype=torch.long),
        torch.tensor(v1s, dtype=torch.long),
        torch.tensor(edge_idxs, dtype=torch.long),
    )