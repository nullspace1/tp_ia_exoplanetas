#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random

print("Starting...")

try:
    sys.path.append('model')
    from model_cnn_attention import ExoplanetCNNAttn
    print("Model imported")
    
    # Load model
    model_path = "results/model_CNN_Attn_Kernle3/cnn_attention_2_model.pt"
    model = ExoplanetCNNAttn(input_size=3000, embedding_dim=32, period_bins=11, num_dilations=9, pool_size=2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded")
    
    # Get one sample
    data_dir = Path("data/samples/positive")
    files = list(data_dir.glob("*.npz"))
    if files:
        file_path = files[0]
        data = np.load(file_path)
        lightcurve = data['arr_0']
        actual_bins = data['arr_1']
        
        # Model prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(lightcurve).unsqueeze(0)
            model_output = model(input_tensor).squeeze().numpy()
        
        # Simple plot
        plt.figure(figsize=(8, 4))
        plt.plot(actual_bins, label='Actual', marker='o')
        plt.plot(model_output, label='Model', marker='s')
        plt.title('Model vs Actual')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('test_output.png', dpi=150)
        print("Plot saved as test_output.png")
    
    print("Done!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
