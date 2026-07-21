import torch
from topobench.nn.wrappers.base import AbstractWrapper


class CCWrapper(AbstractWrapper):
    """Clean, production wrapper aligning the CCNN backbone with TopoBench workflows."""

    _Q_FIELDS = ("Q_center", "Q_v0", "Q_v1", "Q_edge")

    def forward(self, batch) -> dict:
        """
        Intercepts TopoBench batch objects, configures initialization properties,
        and routes features safely to the multi-layer CCNN backbone.
        """
        x_0 = batch.x_0
        B1 = batch.incidence_1

        # Pull the flat Q_* tensors straight off the batch. Because ComputeQDict
        # attaches these via the QData subclass (with a custom __inc__), PyG's
        # default collate already applied the correct global node/edge offsets
        # during batching -- this works identically for batch_size=1 (transductive)
        # and batch_size=N (multiple graphs per batch), with no manual merging.
        missing = [f for f in self._Q_FIELDS if getattr(batch, f, None) is None]
        if missing:
            raise ValueError(
                f"Missing {missing} from the data batch. "
                "Run the 'ComputeQDict' transform step first."
            )
        Q_center, Q_v0, Q_v1, Q_edge = (getattr(batch, f) for f in self._Q_FIELDS)

        # Initialize edge cochains (chi) dynamically if missing from the pipeline
        chi = getattr(batch, "chi", None)
        if chi is None:
            chi = torch.sparse.mm(B1.T, x_0)
            batch.chi = chi

        # Route through the stacked layers of the CCNN model
        x_0_out, chi_out = self.backbone(x_0, B1, Q_center, Q_v0, Q_v1, Q_edge, chi)

        # Cache updated properties back into the batch context
        batch.chi = chi_out

        # Derive a triangle-level (x_2) cochain from the final edge signals.
        # CCNN has no dedicated triangle-update rule, but it does have the
        # boundary map it needs: incidence_2 maps edges -> triangles, so
        # summing each triangle's three boundary edges' embeddings (the same
        # way chi is derived from x_0 via B1.T) gives a well-defined x_2,
        # letting the wrapper output a full node/edge/triangle signature
        # like a standard simplicial network, for use with PropagateSignalDown.
        incidence_2 = getattr(batch, "incidence_2", None)
        if incidence_2 is not None:
            x_2_out = torch.sparse.mm(incidence_2.T, chi_out)
        else:
            x_2_out = torch.zeros((0, chi_out.shape[-1]), device=chi_out.device, dtype=chi_out.dtype)

        # Standard TopoBench packaging expected by subsequent readout modules
        model_out = {
            "x_0": x_0_out,
            "x_1": chi_out,
            "x_2": x_2_out,
            "labels": batch.y,
            "batch_0": batch.batch_0,
        }
        return model_out
