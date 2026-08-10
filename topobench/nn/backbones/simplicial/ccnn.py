import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm
from torch_geometric.utils import scatter


class CCNNLayer(nn.Module):
    """A single layer/block of the Cup and Cap Neural Network with ablation controls and normalization options."""

    def __init__(
        self,
        dim_hidden: int,
        use_parameter_0: bool = True,
        use_linear_1: bool = True,
        use_linear_2: bool = True,
        use_parameter_3: bool = True,
        use_linear_4: bool = True,
        use_parameter_5: bool = False,
        use_square: bool = False,
        use_product: bool = False,
        dropout: float = 0.0,
        norm_type: str | None = "BatchNorm",
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.use_parameter_0 = use_parameter_0
        self.use_linear_1 = use_linear_1
        self.use_linear_2 = use_linear_2
        self.use_parameter_3 = use_parameter_3
        self.use_linear_4 = use_linear_4
        self.use_parameter_5 = use_parameter_5
        self.use_square = use_square
        self.use_product = use_product
        self.norm_type = norm_type

        # Node feature update transformations
        if self.use_parameter_0:
            self.parameter_0 = nn.Parameter(torch.randn(dim_hidden))
        if self.use_linear_1:
            self.linear_1 = nn.Linear(dim_hidden, dim_hidden)
        if self.use_linear_2:
            self.linear_2 = nn.Linear(dim_hidden, dim_hidden)

        # Edge feature (auxiliary cochain) update transformations
        if self.use_parameter_3:
            self.parameter_3 = nn.Parameter(torch.randn(dim_hidden))
        if self.use_linear_4:
            self.linear_4 = nn.Linear(dim_hidden, dim_hidden)
        if self.use_parameter_5:
            self.parameter_5 = nn.Parameter(torch.randn(dim_hidden))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if self.norm_type in (None, "None", "null", False, "Identity"):
            self.norm_node = nn.Identity()
            self.norm_edge = nn.Identity()
        elif self.norm_type == "BatchNorm":
            self.norm_node = BatchNorm(dim_hidden)
            self.norm_edge = BatchNorm(dim_hidden)
        elif self.norm_type == "LayerNorm":
            self.norm_node = nn.LayerNorm(dim_hidden)
            self.norm_edge = nn.LayerNorm(dim_hidden)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        if self.use_parameter_0:
            self.parameter_0.data = torch.randn_like(self.parameter_0.data)
        if self.use_linear_1:
            self.linear_1.reset_parameters()
        if self.use_linear_2:
            self.linear_2.reset_parameters()
        if self.use_parameter_3:
            self.parameter_3.data = torch.randn_like(self.parameter_3.data)
        if self.use_linear_4:
            self.linear_4.reset_parameters()
        if self.use_parameter_5:
            self.parameter_5.data = torch.randn_like(self.parameter_5.data)

        if hasattr(self.norm_node, "reset_parameters"):
            self.norm_node.reset_parameters()
        if hasattr(self.norm_edge, "reset_parameters"):
            self.norm_edge.reset_parameters()

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
        Computes for each node i:  sum_{tri layers} (|chi_e| * (x_j + x_k - 2*x_i))

        Q_center, Q_v0, Q_v1, Q_edge are parallel 1-D tensors (one entry per
        (node, opposite-triangle) pair), already batched with correct global
        offsets by QData.__inc__ -- valid for any batch size, including the
        empty case (no triangles in the batch).
        """
        num_nodes = x_0.shape[0]
        tri_features = torch.zeros((num_nodes, self.dim_hidden), device=x_0.device, dtype=x_0.dtype)

        if Q_center.numel() == 0:
            # No triangles anywhere in this batch -- nothing to subtract.
            return tri_features

        edge_val = torch.tanh(chi[Q_edge])                         # [num_terms, dim_hidden]
        node_diff = x_0[Q_v0] + x_0[Q_v1] - 2 * x_0[Q_center]      # [num_terms, dim_hidden]
        contrib = edge_val * node_diff                            # [num_terms, dim_hidden]

        summed = scatter(contrib, Q_center, dim=0, dim_size=num_nodes, reduce="sum")
        tri_features = tri_features + summed
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

        if self.use_parameter_0:
            node_update += x_0 * self.parameter_0

        if self.use_linear_1:
            node_update += self.linear_1(torch.sparse.mm(B1, torch.sparse.mm(B1.T, x_0)))

        if self.use_linear_2:
            tri_accumulations = self.compute_triangle_aggregations(
                Q_center, Q_v0, Q_v1, Q_edge, chi, x_0
            )
            node_update += self.linear_2(tri_accumulations)

        node_update = self.dropout(self.act(node_update))
        node_update = self.norm_node(node_update)

        x_0_next = node_update

        # --- Update Edge Signals (chi) ---
        edge_update = torch.zeros_like(chi)

        if self.use_parameter_3:
            edge_update += chi * self.parameter_3

        if self.use_linear_4:
            # Uses the newly computed x_0 representations for boundary propagation
            if self.use_square:
                edge_update += self.linear_4(torch.square(torch.sparse.mm(B1.T, x_0_next)))
            if self.use_product:
                B1_copy = B1.coalesce()
                B1_pos = B1_copy.values().clamp_min_(0)
                B1_neg = (-B1_copy).values().clamp_min_(0)
                edge_update += self.linear_4((B1_pos.T @ x_0_next) * (B1_neg.T @ x_0_next))
            else:
                edge_update += self.linear_4(torch.abs(torch.sparse.mm(B1.T, x_0_next)))

        if self.use_parameter_5:
            edge_update += self.parameter_5

        edge_update = self.dropout(self.act(edge_update))
        edge_update = self.norm_edge(edge_update)

        chi_next = edge_update

        return x_0_next, chi_next


class CCNN(nn.Module):
    """Multi-layer implementation of the Cup and Cap Neural Network."""

    def __init__(
        self,
        dim_hidden: int,
        out_channels: int,
        n_layers: int = 2,
        use_parameter_0: bool = True,
        use_linear_1: bool = True,
        use_linear_2: bool = True,
        use_parameter_3: bool = True,
        use_linear_4: bool = True,
        use_parameter_5: bool = False,
        use_square: bool = False,
        use_product: bool = False,
        dropout: float = 0.0,
        norm_type: str | None = "BatchNorm",
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.out_channels = out_channels

        # Stack multiple layers dynamically
        self.layers = nn.ModuleList(
            [
                CCNNLayer(
                    dim_hidden=dim_hidden,
                    use_parameter_0=use_parameter_0,
                    use_linear_1=use_linear_1,
                    use_linear_2=use_linear_2,
                    use_parameter_3=use_parameter_3,
                    use_linear_4=use_linear_4,
                    use_parameter_5=use_parameter_5,
                    use_square=use_square,
                    use_product=use_product,
                    dropout=dropout,
                    norm_type=norm_type,
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