from dataset_building import LightcurveDataset
import pytorch_lightning as pl
import torch

from model.model import Model

dataset = LightcurveDataset.LightCurveDataset(
    "data/samples/positive", "data/samples/negative", "data/samples/synthetic", 0.3, 0.5, 0.2
)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = Model(3000, 256, 101)

trainer = pl.Trainer(max_epochs=10)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model, dataloaders=test_loader)

torch.save(model.state_dict(), "model.pt")