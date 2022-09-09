from torch.nn import LayerNorm, Module, ModuleList
from torch_geometric.nn import MaskLabel, TransformerConv


class UniMP(Module):
    def __init__(self, in_channels, num_classes, hidden_channels, num_layers, heads, dropout=0.3):
        super().__init__()

        if num_layers < 2:
            raise RuntimeError(f"Need at least 2 layers, but got {num_layers}")

        self.label_emb = MaskLabel(num_classes, in_channels)

        self.convs = ModuleList(
            (
                TransformerConv(
                    in_channels,
                    hidden_channels // heads,
                    heads,
                    concat=True,
                    beta=True,
                    dropout=dropout,
                ),
            )
        )
        for i in range(1, num_layers - 1):
            self.convs.append(
                TransformerConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads,
                    concat=True,
                    beta=True,
                    dropout=dropout,
                )
            )
        self.convs.append(
            TransformerConv(hidden_channels, num_classes, heads, concat=False, beta=True, dropout=dropout),
        )
        self.norms = ModuleList([LayerNorm(hidden_channels) for i in range(0, num_layers - 1)])

    def forward(self, x, y, edge_index, label_mask):
        x = self.label_emb(x, y, label_mask)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index)).relu()
        return self.convs[-1](x, edge_index)
