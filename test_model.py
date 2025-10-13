#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

print("Starting script...")

# Test basic functionality
try:
    # Import model
    sys.path.append('model')
    from model.model_cnn_attention import ExoplanetCNNAttn
    print("✓ Model imported")
    
    # Load model
    model_path = "cnn_attention_model.pt"
    model = ExoplanetCNNAttn(input_size=3000, embedding_dim=32, period_bins=11, num_dilations=11, pool_size=2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    print("✓ Model loaded")
    
    # Load one sample
    data_dir = Path("data/samples/positive")
    files = list(data_dir.glob("*.npz"))
    if files:
        data = np.load(files[0])
        lightcurve = data['arr_0']
        actual_bins = data['arr_1']
        print(f"✓ Sample loaded: {lightcurve.shape}, {actual_bins.shape}")
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(lightcurve).unsqueeze(0)
            model_output = model(input_tensor).squeeze().numpy()
        print(f"✓ Model prediction: {model_output.shape}")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(actual_bins, 'o-', label='Actual', linewidth=2, markersize=8)
        plt.plot(model_output, 's-', label='Model', linewidth=2, markersize=8)
        plt.title('Model vs Actual Output Bins')
        plt.xlabel('Bin Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('model_vs_actual.png', dpi=300, bbox_inches='tight')
        print("✓ Plot saved as model_vs_actual.png")
        
        # Print values
        print(f"Actual bins: {actual_bins}")
        print(f"Model output: {model_output}")
    
    print("✓ Script completed successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
