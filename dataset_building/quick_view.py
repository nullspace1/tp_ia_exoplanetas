#!/usr/bin/env python3
"""
Quick data viewer script for inspecting generated datasets
Usage: python quick_view.py [options]
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from Positives import Positives
from Negatives import Negatives
from Synthetics import Synthetics

def load_config(config_path="config.json"):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def show_summary():
    """Show dataset summary"""
    config = load_config()
    
    positives = Positives(config)
    negatives = Negatives(config)
    synthetics = Synthetics(config)
    
    print("📊 DATASET SUMMARY")
    print("=" * 50)
    print(f"Positives: {len(positives.df)} total, {positives.sample_count} samples")
    print(f"Negatives: {len(negatives.df)} total, {negatives.sample_count} samples")
    print(f"Synthetics: {len(synthetics.df)} total, {synthetics.sample_count} samples")
    
    if len(positives.df) > 0:
        print(f"\nPositives period: {positives.df['period'].min():.2f} - {positives.df['period'].max():.2f} days")
        print(f"Positives duration: {positives.df['duration'].min():.2f} - {positives.df['duration'].max():.2f} hours")

def show_file_counts():
    """Show count of generated files"""
    print("\n📁 GENERATED FILES")
    print("=" * 30)
    
    folders = ["positive", "negative", "synthetic"]
    for folder in folders:
        path = f"data/samples/{folder}"
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith('.npz')])
            print(f"{folder.capitalize()}: {count} files")
        else:
            print(f"{folder.capitalize()}: 0 files (folder not found)")

def plot_sample(dataset_type, kepler_id=None):
    """Plot a sample lightcurve"""
    config = load_config()
    
    if dataset_type == "positive":
        dataset = Positives(config)
        folder = "positive"
    elif dataset_type == "negative":
        dataset = Negatives(config)
        folder = "negative"
    elif dataset_type == "synthetic":
        dataset = Synthetics(config)
        folder = "synthetic"
    else:
        print(f"❌ Invalid dataset type: {dataset_type}")
        return
    
    # Select random kepler_id if not provided
    if kepler_id is None:
        sample_dir = f"data/samples/{folder}"
        if not os.path.exists(sample_dir):
            print(f"❌ Directory not found: {sample_dir}")
            return
        files = [f.replace('.npz', '') for f in os.listdir(sample_dir) if f.endswith('.npz')]
        if not files:
            print(f"❌ No files found in {sample_dir}")
            return
        kepler_id = np.random.choice(files)
    
    file_path = f"data/samples/{folder}/{kepler_id}.npz"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("💡 Run dataset generation first!")
        return
    
    try:
        data = np.load(file_path)
        lightcurve = data['arr_0']
        
        plt.figure(figsize=(12, 6))
        plt.plot(lightcurve, 'b-', linewidth=0.8)
        plt.title(f'{dataset_type.title()} Lightcurve - KIC {kepler_id}')
        plt.xlabel('Time Points')
        plt.ylabel('Normalized Flux')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.text(0.02, 0.98, f'Points: {len(lightcurve)}\nMin: {lightcurve.min():.4f}\nMax: {lightcurve.max():.4f}\nMean: {lightcurve.mean():.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Plotted {dataset_type} lightcurve for KIC {kepler_id}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")

def plot_distributions():
    """Plot parameter distributions"""
    config = load_config()
    
    positives = Positives(config)
    synthetics = Synthetics(config)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Distributions', fontsize=16)
    
    # Period distributions
    axes[0, 0].hist(positives.df['period'], bins=30, alpha=0.7, color='green', label='Positives')
    axes[0, 0].set_title('Period Distribution - Positives')
    axes[0, 0].set_xlabel('Period (days)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(synthetics.df['period'], bins=30, alpha=0.7, color='blue', label='Synthetics')
    axes[0, 1].set_title('Period Distribution - Synthetics')
    axes[0, 1].set_xlabel('Period (days)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Duration distributions
    axes[1, 0].hist(positives.df['duration'], bins=30, alpha=0.7, color='green', label='Positives')
    axes[1, 0].set_title('Duration Distribution - Positives')
    axes[1, 0].set_xlabel('Duration (hours)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(synthetics.df['duration'], bins=30, alpha=0.7, color='blue', label='Synthetics')
    axes[1, 1].set_title('Duration Distribution - Synthetics')
    axes[1, 1].set_xlabel('Duration (hours)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Plotted parameter distributions")

def main():
    parser = argparse.ArgumentParser(description='Quick data viewer for exoplanet datasets')
    parser.add_argument('--summary', action='store_true', help='Show dataset summary')
    parser.add_argument('--files', action='store_true', help='Show file counts')
    parser.add_argument('--plot', choices=['positive', 'negative', 'synthetic'], 
                       help='Plot a sample lightcurve')
    parser.add_argument('--distributions', action='store_true', help='Plot parameter distributions')
    parser.add_argument('--id', type=str, help='Specific Kepler ID to plot')
    parser.add_argument('--all', action='store_true', help='Show all information')
    
    args = parser.parse_args()
    
    if args.all:
        show_summary()
        show_file_counts()
        plot_distributions()
        return
    
    if args.summary:
        show_summary()
    
    if args.files:
        show_file_counts()
    
    if args.plot:
        plot_sample(args.plot, args.id)
    
    if args.distributions:
        plot_distributions()
    
    # If no arguments provided, show summary and files
    if not any([args.summary, args.files, args.plot, args.distributions]):
        show_summary()
        show_file_counts()
        print("\n💡 Use --help to see all options")

if __name__ == "__main__":
    main()
