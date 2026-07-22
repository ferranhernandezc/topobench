import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class CCNNLayer(nn.Module):
    """A single layer/block of the Cup and Cap Neural Network with ablation controls."""

    def __init__(
        self,
        dim_hidden: int,
        use_linear_0: bool = True,
        use_linear_1: bool = True,
        use_linear_2: bool = True,
        use_linear_3: bool = True,
        use_linear_4: bool = True,
        use_square: bool = False,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.use_linear_0 = use_linear_0
        self.use_linear_1 = use_linear_1
        self.use_linear_2 = use_linear_2
        self.use_linear_3 = use_linear_3
        self.use_linear_4 = use_linear_4
        self.use_square = use_square
        # Node feature update transformations
        if self.use_linear_0:
            self.linear_0 = nn.Linear(dim_hidden, dim_hidden)
        if self.use_linear_1:
            self.linear_1 = nn.Linear(dim_hidden, dim_hidden)
        if self.use_linear_2:
            self.linear_2 = nn.Linear(dim_hidden, dim_hidden)

        # Edge feature (auxiliary cochain) update transformations
        if self.use_linear_3:
            self.linear_3 = nn.Linear(dim_hidden, dim_hidden)
        if self.use_linear_4:
            self.linear_4 = nn.Linear(dim_hidden, dim_hidden)

        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        if self.use_linear_0:
            self.linear_0.reset_parameters()
        if self.use_linear_1:
            self.linear_1.reset_parameters()
        if self.use_linear_2:
            self.linear_2.reset_parameters()
        if self.use_linear_3:
            self.linear_3.reset_parameters()
        if self.use_linear_4:
            self.linear_4.reset_parameters()

    def compute_triangle_aggregations(
        self,
        Q_center: torch.Tensor,
        Q_v0: torch.Tensor,
        Q_v1: torch.Tensor,
        Q_edge: torch.Tensor,
        chi: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized node-wise triangle boundary feature accumulation.
        Computes for each node i: 1 - sum_{tri layers} (|chi_e| * (x_j + x_k - 2*x_i))

        Q_center, Q_v0, Q_v1, Q_edge are parallel 1-D tensors (one entry per
        (node, opposite-triangle) pair), already batched with correct global
        offsets by QData.__inc__ -- valid for any batch size, including the
        empty case (no triangles in the batch).
        """
        num_nodes = x_0.shape[0]
        tri_features = torch.ones((num_nodes, self.dim_hidden), device=x_0.device, dtype=x_0.dtype)

        if Q_center.numel() == 0:
            # No triangles anywhere in this batch -- nothing to subtract.
            return tri_features

        edge_val = torch.abs(chi[Q_edge])                         # [num_terms, dim_hidden]
        node_diff = x_0[Q_v0] + x_0[Q_v1] - 2 * x_0[Q_center]      # [num_terms, dim_hidden]
        contrib = edge_val * node_diff                            # [num_terms, dim_hidden]

        summed = scatter(contrib, Q_center, dim=0, dim_size=num_nodes, reduce="sum")
        tri_features = tri_features - summed
        return tri_features

    def forward(
        self,
        x_0: torch.Tensor,
        B1: torch.Tensor,
        Q_center: torch.Tensor,
        Q_v0: torch.Tensor,
        Q_v1: torch.Tensor,
        Q_edge: torch.Tensor,
        chi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Update Node Signals (x_0) ---
        node_update = torch.zeros_like(x_0)

        if self.use_linear_0:
            node_update += self.linear_0(x_0)

        if self.use_linear_1
            node_update += self.linear_1(torch.sparse.mm(B1,torch.sparse.mm(B1.T, x_0)))

        if self.use_linear_2:
            tri_accumulations = self.compute_triangle_aggregations(
                Q_center, Q_v0, Q_v1, Q_edge, chi, x_0
            )
            node_update += self.linear_2(tri_accumulations)

        x_0_next = self.act(node_update)

        # --- Update Edge Signals (chi) ---
        edge_update = torch.zeros_like(chi)

        if self.use_linear_3:
            edge_update += self.linear_3(chi)

        if self.use_linear_4:
            # Uses the newly computed x_0 representations for boundary propagation
            if self.use_square:
                edge_update += self.linear_4(torch.square(torch.sparse.mm(B1.T, x_0_next)))
            else:
                edge_update += self.linear_4(torch.abs(torch.sparse.mm(B1.T, x_0_next)))

        chi_next = self.act(edge_update)

        return x_0_next, chi_next


class CCNN(nn.Module):
    """Multi-layer implementation of the Cup and Cap Neural Network."""

    def __init__(
        self,
        dim_hidden: int,
        out_channels: int,
        n_layers: int = 2,
        use_linear_0: bool = True,
        use_linear_1: bool = True,
        use_linear_2: bool = True,
        use_linear_3: bool = True,
        use_linear_4: bool = True,
        use_square: bool = False,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.out_channels = out_channels

        # Stack multiple layers dynamically
        self.layers = nn.ModuleList(
            [
                CCNNLayer(
                    dim_hidden=dim_hidden,
                    use_linear_0=use_linear_0,
                    use_linear_1=use_linear_1,
                    use_linear_2=use_linear_2,
                    use_linear_3=use_linear_3,
                    use_linear_4=use_linear_4,
                    use_square=use_square,
                )
                for _ in range(n_layers)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        x_0: torch.Tensor,
        B1: torch.Tensor,
        Q_center: torch.Tensor,
        Q_v0: torch.Tensor,
        Q_v1: torch.Tensor,
        Q_edge: torch.Tensor,
        chi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x_0, chi = layer(x_0, B1, Q_center, Q_v0, Q_v1, Q_edge, chi)
        return x_0, chi