import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from model.S_Mamba import Model as MambaModel  
# from model.Informer import Model as MambaModel  

class GCNSpatial(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


from torch_geometric.nn import MessagePassing

class MeanAggregator(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')  # or 'add', 'max', etc.

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

class Model(nn.Module):
    def __init__(self, gcn_out_channels, edge_index, mamba_args):
        """
        Args:
            gcn_out_channels (int): Number of output channels from GCN
            edge_index (torch.LongTensor): [2, num_edges] community graph edges
            mamba_args (argparse.Namespace): arguments for Mamba model
        """
        super().__init__()
        self.edge_index = edge_index
        self.gcn = GCNSpatial(in_channels=1, out_channels=gcn_out_channels)
        self.mean_agg = MeanAggregator()
        self.mamba_model = MambaModel(mamba_args)

        self.flatten_gcn_output = nn.Linear(mamba_args.enc_in * (1 + gcn_out_channels), mamba_args.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc shape: [B, T, N]
        B, T, N = x_enc.shape

        # Reshape to [B*T, N, 1]
        x = x_enc.reshape(B*T, N).unsqueeze(-1)

        # Apply GCN per time step
        gcn_outputs = []
        for i in range(B*T):
            gcn_out = self.gcn(x[i], self.edge_index)  # [N, gcn_out_channels]
            # gcn_out = self.mean_agg(x[i], self.edge_index)  # [N, gcn_out_channels]
            gcn_outputs.append(gcn_out.unsqueeze(0))

        # Stack: [B*T, N, gcn_out_channels]
        gcn_output = torch.cat(gcn_outputs, dim=0)

        # Reshape: [B, T, N, gcn_out_channels]
        gcn_output = gcn_output.view(B, T, N, -1)

        # Original input reshape: [B, T, N, 1]
        x_enc_exp = x_enc.unsqueeze(-1)


        # Concatenate: [B, T, N, 1 + gcn_out_channels]
        x_combined = torch.cat([x_enc_exp, gcn_output], dim=-1)

        # Flatten node features: [B, T, N*(1 + gcn_out_channels)]
        x_combined = x_combined.view(B, T, -1)
      

        # Project back to [B, T, enc_in] to match Mamba input
        x_projected = self.flatten_gcn_output(x_combined)

        # Call original Mamba
        return self.mamba_model(x_projected, x_mark_enc, x_dec, x_mark_dec)
