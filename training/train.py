import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Add root to python path to allow importing from the training directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.dataset import SongDataset, collate_fn
from training.engine import TrainingEngine

class SongDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the SongDataset.
    Handles the creation of training and validation dataloaders.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.paths = config['paths']
        self.data_params = config['data']
        self.train_params = config['training']
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        """
        Loads the full dataset and splits it into training and validation sets.
        """
        if stage == 'fit' or stage is None:
            full_dataset = SongDataset(manifest_path=os.path.join(self.paths['prepared_data_dir'], 'manifest.jsonl'))

            # Split dataset into training and validation
            dataset_size = len(full_dataset)
            val_size = int(0.1 * dataset_size) # 10% for validation
            train_size = dataset_size - val_size

            # Ensure there's at least one validation sample
            if val_size == 0 and train_size > 0:
                val_size = 1
                train_size -= 1

            if train_size <= 0:
                raise ValueError("Dataset is too small to create a training split. Need at least 2 samples.")

            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.train_params['seed'])
            )
            print(f"Dataset split: {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples.")

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_params['batch_size'],
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.data_params['num_workers'],
            pin_memory=True
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_params['validation_batch_size'],
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.data_params['num_workers'],
            pin_memory=True
        )

def train(config_path: str):
    """
    Main training function.
    - Loads configuration.
    - Sets up the DataModule and TrainingEngine.
    - Initializes and runs the PyTorch Lightning Trainer.
    """
    # --- Load Config ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']
    train_params = config['training']

    # --- Reproducibility ---
    pl.seed_everything(train_params['seed'], workers=True)

    # --- DataModule and Model ---
    datamodule = SongDataModule(config)
    model = TrainingEngine(config_path)

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(paths['output_dir'], 'checkpoints'),
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # --- Logger ---
    logger = TensorBoardLogger(
        save_dir=paths['output_dir'],
        name='logs'
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=train_params['epochs'],
        accelerator='gpu',
        devices=-1, # Use all available GPUs
        strategy='ddp_find_unused_parameters_true', # For DDP
        precision='16-mixed' if train_params['use_fp16'] else 32,
        accumulate_grad_batches=train_params['gradient_accumulation_steps'],
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=train_params['log_interval']
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.fit(model, datamodule)
    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the SongGeneration model.")
    parser.add_argument('--config', type=str, default='training/train_config.yaml',
                        help='Path to the training configuration YAML file.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found at {args.config}")

    train(args.config)
