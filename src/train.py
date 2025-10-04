# src/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, num_layers=3, dropout=0.2):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels if _ < num_layers - 1 else out_channels))
        
        # Dropout
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, num_layers=3, 
                 heads=4, dropout=0.2):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels if _ < num_layers - 1 else out_channels))
        
        # Dropout
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class TumorSegmentationGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, gnn_out_channels=64, 
                 gnn_type='sage', num_layers=3, dropout=0.2, heads=4, use_consistency=True):
        super(TumorSegmentationGNN, self).__init__()
        self.use_consistency = use_consistency
        
        # GNN backbone
        if gnn_type.lower() == 'sage':
            self.gnn = GraphSAGE(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=gnn_out_channels,
                num_layers=num_layers,
                dropout=dropout
            )
        elif gnn_type.lower() == 'gat':
            self.gnn = GAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=gnn_out_channels,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(gnn_out_channels, gnn_out_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_out_channels // 2, 1)
        )
    
    def forward(self, data):
        # Extract node features and edge index
        x, edge_index = data.x, data.edge_index
        
        # Apply GNN to get node embeddings
        node_embeddings = self.gnn(x, edge_index)
        
        # Apply classifier to get predictions
        logits = self.classifier(node_embeddings).squeeze(-1)
        
        return logits, node_embeddings

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CrossSliceConsistencyLoss(nn.Module):
    def __init__(self, lambda_consistency=0.1):
        super(CrossSliceConsistencyLoss, self).__init__()
        self.lambda_consistency = lambda_consistency
    
    def forward(self, embeddings, slice_mask):
        # Get embeddings for each slice
        slice1_embeddings = embeddings[slice_mask]
        slice2_embeddings = embeddings[~slice_mask]
        
        # If either slice has no embeddings, return 0 loss
        if len(slice1_embeddings) == 0 or len(slice2_embeddings) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Calculate mean embeddings for each slice
        slice1_mean = torch.mean(slice1_embeddings, dim=0)
        slice2_mean = torch.mean(slice2_embeddings, dim=0)
        
        # Calculate consistency loss (cosine similarity)
        cos_sim = F.cosine_similarity(slice1_mean.unsqueeze(0), slice2_mean.unsqueeze(0))
        
        # We want to maximize similarity (minimize negative similarity)
        return self.lambda_consistency * (1 - cos_sim)

class CombinedLoss(nn.Module):
    def __init__(self, lambda_ce=1.0, lambda_dice=1.0, lambda_consistency=0.1, use_consistency=True):
        super(CombinedLoss, self).__init__()
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.lambda_consistency = lambda_consistency
        self.use_consistency = use_consistency
        
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.consistency_loss = CrossSliceConsistencyLoss(lambda_consistency=lambda_consistency)
    
    def forward(self, logits, embeddings, targets, slice_mask):
        # Calculate BCE loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Calculate Dice loss
        dice_loss = self.dice_loss(logits, targets)
        
        # Calculate combined loss
        combined_loss = self.lambda_ce * ce_loss + self.lambda_dice * dice_loss
        
        # Add consistency loss if enabled
        if self.use_consistency:
            consistency_loss = self.consistency_loss(embeddings, slice_mask)
            combined_loss += consistency_loss
            return combined_loss, ce_loss, dice_loss, consistency_loss
        
        return combined_loss, ce_loss, dice_loss, torch.tensor(0.0, device=logits.device)