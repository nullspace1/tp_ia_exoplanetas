#!/usr/bin/env python3
"""
Plot training and validation losses
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Loss data
train_losses = [
    0.018446408212184906,
    0.03682271018624306,
    0.019428914412856102,
    0.022593889385461807,
    0.0292439516633749,
    0.01902196928858757,
    0.03274216502904892,
    0.01060363370925188,
    0.022719575092196465,
    0.021781064569950104
]

val_losses = [
    3.21643328666687,
    0.02939845807850361,
    0.04062424972653389,
    0.05488661676645279,
    0.02733035944402218,
    0.029805460944771767,
    0.02736852876842022,
    0.030926475301384926,
    0.03520357981324196,
    0.024232691153883934,
    0.023507853969931602
]

# Create plot
plt.figure(figsize=(10, 6))

# Plot training losses
epochs_train = range(1, len(train_losses) + 1)
plt.plot(epochs_train, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)

# Plot validation losses
epochs_val = range(1, len(val_losses) + 1)
plt.plot(epochs_val, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)

plt.title('Training and Validation Losses', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Use log scale due to the large initial validation loss

# Add some styling
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
print("✓ Loss plot saved as 'loss_plot.png'")

print("Plot completed!")
