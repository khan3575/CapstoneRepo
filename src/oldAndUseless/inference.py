# src/inference.py
import os
import numpy as np
import torch
import argparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from pathlib import Path
import glob
import time

from gnn_model import TumorSegmentationGNN

def load_model(model_path, device, in_channels=142):
    """Load a trained model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Initialize model
    model = TumorSegmentationGNN(
        in_channels=in_channels,  # Default, will be overridden if in config
        hidden_channels=model_config.get('hidden_dim', 128),
        gnn_out_channels=model_config.get('out_dim', 64),
        gnn_type=model_config.get('model_type', 'sage'),
        num_layers=model_config.get('num_layers', 3),
        dropout=model_config.get('dropout', 0.2),
        heads=model_config.get('heads', 4),
        use_consistency=not model_config.get('no_consistency', False)
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    return model, checkpoint

def run_inference(model, graphs, device, threshold=0.5, batch_size=1):
    """Run inference on graphs."""
    model.eval()
    all_predictions = []
    all_scores = []
    
    # Create dataloader
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            # Move batch to device
            batch = batch.to(device)
            
            # Run inference
            logits, _ = model(batch)
            scores = torch.sigmoid(logits)
            preds = (scores > threshold).float()
            
            # Store predictions and scores
            all_predictions.append(preds.cpu())
            all_scores.append(scores.cpu())
    
    # Concatenate results
    if all_predictions:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        return all_predictions.numpy(), all_scores.numpy()
    else:
        return np.array([]), np.array([])

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained GNN model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--graph_dir', type=str, required=True, help='Directory with graph files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for predictions')
    parser.add_argument('--n_superpixels', type=int, default=400, choices=[200, 400, 800],
                        help='Number of superpixels per slice')
    parser.add_argument('--use_cnn', action='store_true', help='Use CNN features')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")
    print(f"Model was trained for {checkpoint['epoch']+1} epochs")
    print(f"Validation Dice: {checkpoint['val_metrics']['dice']:.4f}")
    
    # Find all graph files
    if args.use_cnn:
        pattern = f"*_graphs_cnn_{args.n_superpixels}.pt"
    else:
        pattern = f"*_graphs_{args.n_superpixels}.pt"
    
    graph_files = glob.glob(os.path.join(args.graph_dir, "*", pattern))
    
    if not graph_files:
        print(f"No graph files found matching pattern {pattern} in {args.graph_dir}")
        return
    
    print(f"Found {len(graph_files)} graph files")
    
    # Run inference on each patient
    for graph_file in graph_files:
        patient_id = os.path.basename(graph_file).split('_graphs')[0]
        print(f"Processing patient {patient_id}")
        
        # Load graphs
        graphs = torch.load(graph_file)
        
        if not graphs:
            print(f"No graphs found for patient {patient_id}")
            continue
        
        print(f"Loaded {len(graphs)} graphs")
        
        # Run inference
        predictions, scores = run_inference(
            model, graphs, device, threshold=args.threshold, batch_size=args.batch_size
        )
        
        if len(predictions) == 0:
            print(f"No predictions generated for patient {patient_id}")
            continue
        
        # Create patient output directory
        patient_output_dir = os.path.join(args.output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Save predictions and scores
        pred_file = os.path.join(patient_output_dir, f"{patient_id}_predictions.npy")
        score_file = os.path.join(patient_output_dir, f"{patient_id}_scores.npy")
        
        np.save(pred_file, predictions)
        np.save(score_file, scores)
        
        print(f"Saved predictions to {pred_file}")
        
        # Save original graphs with predictions for visualization
        for i, graph in enumerate(graphs):
            # Add predictions to graph
            graph.pred = torch.tensor(predictions[i:i+len(graph.y)], dtype=torch.float)
            graph.score = torch.tensor(scores[i:i+len(graph.y)], dtype=torch.float)
        
        # Save graphs with predictions
        graphs_pred_file = os.path.join(patient_output_dir, f"{patient_id}_graphs_pred.pt")
        torch.save(graphs, graphs_pred_file)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total inference time: {(time.time() - start_time) / 60:.2f} minutes")