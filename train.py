from dataset_building import LightcurveDataset
import pytorch_lightning as pl
import torch
import argparse
import json
import os
from datetime import datetime

from model.model_cnn_attention import ExoplanetCNNAttn
from model.model_cnn_gru import ExoplanetCNNGRU

class LossTracker(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        avg_train_loss = trainer.callback_metrics.get('train_loss', 0)
        if hasattr(avg_train_loss, 'item'):
            avg_train_loss = avg_train_loss.item()
        self.train_losses.append(avg_train_loss)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        avg_val_loss = trainer.callback_metrics.get('val_loss', 0)
        if hasattr(avg_val_loss, 'item'):
            avg_val_loss = avg_val_loss.item()
        self.val_losses.append(avg_val_loss)
    
    def get_losses(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

def get_model(model_name, **kwargs):
    model_templates = {
        'cnn_attention': {
            'class': ExoplanetCNNAttn,
            'params': {
                'input_size': 3000,
                'embedding_dim': 32,
                'period_bins': 11,
                'num_dilations': 11,
                'pool_size': 2
            }
        },
        'cnn_gru': {
            'class': ExoplanetCNNGRU,
            'params': {
                'input_size': 3000,
                'num_classes': 11,
                'hidden_size': 128
            }
        }
    }
    
    if model_name not in model_templates:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(model_templates.keys())}")
    
    if model_name == 'cnn_gru':
        model = ExoplanetCNNGRU(input_size=model_templates['cnn_gru']['params']['input_size'], num_classes=model_templates['cnn_gru']['params']['num_classes'], hidden_size=model_templates['cnn_gru']['params']['hidden_size'])
    
    template = model_templates[model_name]
    params = template['params'].copy()
    params.update(kwargs)
    
    return template['class'](**params)

def main():
    parser = argparse.ArgumentParser(description='Train exoplanet detection models')
    parser.add_argument('--model', type=str, default='cnn_attention', 
                       choices=['cnn_attention', 'cnn_gru'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--limit_batches', type=float, default=0.1, help='Limit training batches (for testing)')
    
    args = parser.parse_args()
    
    dataset = LightcurveDataset.LightCurveDataset(
        "data/samples/positive", "data/samples/negative", "data/samples/synthetic", 0.4, 0.2, 0.4
    )

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model)
    
    loss_tracker = LossTracker()

    trainer = pl.Trainer(
        enable_checkpointing=True,
        limit_train_batches=args.limit_batches, 
        limit_val_batches=args.limit_batches, 
        limit_test_batches=args.limit_batches,
        max_epochs=args.epochs,
        callbacks=[loss_tracker]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    torch.save(model.state_dict(), f"{args.model}_model.pt")
    
    losses_data = loss_tracker.get_losses()
    losses_data['model_name'] = args.model
    losses_data['epochs'] = args.epochs
    losses_data['batch_size'] = args.batch_size
    losses_data['timestamp'] = datetime.now().isoformat()
    
    os.makedirs('results', exist_ok=True)
    
    loss_filename = f"results/{args.model}_losses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(loss_filename, 'w') as f:
        json.dump(losses_data, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Model saved as: {args.model}_model.pt")
    print(f"Loss data saved as: {loss_filename}")
    print(f"Final training loss: {losses_data['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {losses_data['val_losses'][-1]:.6f}")

if __name__ == "__main__":
    main()