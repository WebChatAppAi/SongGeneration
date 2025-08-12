import os
import yaml
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler
import lightning as pl
from omegaconf import OmegaConf
from peft import get_peft_model, LoraConfig

# Add root to python path to allow importing from the main project
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from codeclm.models import builders

class CosineLRScheduler(_LRScheduler):
    """
    Cosine LR scheduler with warmup, copied from the original codebase for self-containment.
    """
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int,
                 lr_min_ratio: float = 0.0, cycle_length: float = 1.0):
        self.warmup_steps = warmup_steps
        assert self.warmup_steps >= 0
        self.total_steps = total_steps
        assert self.total_steps >= 0
        self.lr_min_ratio = lr_min_ratio
        self.cycle_length = cycle_length
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            lr_ratio = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
        elif step < self.total_steps:
            s = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_ratio = self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * \
                (1. + math.cos(math.pi * s / self.cycle_length))
        else:
            lr_ratio = self.lr_min_ratio
        return lr_ratio * lr

    def get_lr(self):
        return [self._get_sched_lr(lr, self.last_epoch) for lr in self.base_lrs]


class TrainingEngine(pl.LightningModule):
    """
    PyTorch Lightning module for training the SongGeneration model.
    """
    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.paths = self.config['paths']
        self.train_params = self.config['training']

        # This is a placeholder for total_steps, which will be set properly in setup()
        self.total_steps = 0

        # --- Model Loading ---
        model_config_path = os.path.join(self.paths['base_model_path'], 'config.yaml')
        model_cfg = OmegaConf.load(model_config_path)

        # Override model config with any values from our train_config.yaml
        model_cfg.update(self.config['model'])

        print("Initializing Language Model...")
        self.audiolm = builders.get_lm_model(model_cfg)

        # Load pretrained weights
        ckpt_path = os.path.join(self.paths['base_model_path'], 'model.pt')
        if os.path.exists(ckpt_path):
            print(f"Loading pretrained weights from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            # Filter for audiolm weights and remove the prefix
            audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
            self.audiolm.load_state_dict(audiolm_state_dict, strict=False)
        else:
            print(f"Warning: No pretrained weights found at {ckpt_path}. Training from scratch.")

        # --- LoRA Setup ---
        if self.train_params['type'] == 'lora':
            print("Setting up LoRA...")
            lora_config = self.config['lora']
            peft_config = LoraConfig(
                r=lora_config['r'],
                lora_alpha=lora_config['alpha'],
                target_modules=lora_config['target_modules'],
                lora_dropout=lora_config['dropout'],
                bias="none"
            )
            self.audiolm = get_peft_model(self.audiolm, peft_config)
            self.audiolm.print_trainable_parameters()

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.audiolm.special_token_id)
        self.save_hyperparameters() # Save config to checkpoint

    def forward(self, batch):
        """
        The forward pass of the model.
        """
        tokens = batch['tokens'] # [B, K, T]
        lyrics = batch['lyrics']
        descriptions = batch['descriptions']

        # The model internally expects a batch size of 1 for condition preparation,
        # so we process one by one. This is a limitation of the original codebase's design.
        # A more optimized approach would be to batch this, but we follow the original logic.
        condition_tensors = self.audiolm.prepare_condition_tensors(
            batch_size=len(lyrics),
            text=lyrics,
            descriptions=descriptions,
            # Prompt audio is not used during training from scratch on a dataset.
            # We pass a tensor of "end-of-prompt" tokens. The length 375 corresponds
            # to the model's expected prompt length (15 seconds * 25 fps = 375 tokens).
            audio_qt_emb=torch.full((len(lyrics), 3, 375), 16385, device=self.device).long()
        )

        # compute_predictions handles the codebook pattern and returns logits & mask
        output = self.audiolm.compute_predictions(tokens, condition_tensors)
        return output.logits, tokens, output.mask

    def _calculate_loss(self, logits, labels, mask):
        """
        Calculates the cross-entropy loss, respecting the mask.
        """
        # logits: [B, K, T, Card], labels: [B, K, T], mask: [B, K, T]

        # Flatten the tensors
        logits_flat = logits.view(-1, logits.size(-1)) # [B*K*T, Card]
        labels_flat = labels.view(-1) # [B*K*T]
        mask_flat = mask.view(-1) # [B*K*T]

        # Apply mask by setting labels of masked positions to the ignore_index
        # This is more efficient than filtering the tensors
        masked_labels = torch.where(mask_flat.bool(), labels_flat, self.loss_fn.ignore_index)

        loss = self.loss_fn(logits_flat, masked_labels)
        return loss

    def training_step(self, batch, batch_idx):
        logits, labels, mask = self.forward(batch)
        loss = self._calculate_loss(logits, labels, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels, mask = self.forward(batch)
        loss = self._calculate_loss(logits, labels, mask)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """

        Prepares optimizer and learning rate scheduler.
        """
        # Filter parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = AdamW(
            trainable_params,
            lr=self.train_params['learning_rate'],
            betas=(self.train_params['adam_beta1'], self.train_params['adam_beta2']),
            weight_decay=self.train_params['weight_decay']
        )

        scheduler = CosineLRScheduler(
            optimizer,
            total_steps=self.total_steps,
            warmup_steps=self.train_params['warmup_steps']
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def setup(self, stage: str):
        """
        Called by Lightning to set up data-dependent properties.
        """
        if stage == 'fit':
            # Calculate total steps
            train_loader = self.trainer.datamodule.train_dataloader()
            self.total_steps = int(
                len(train_loader.dataset) /
                (self.train_params['batch_size'] * self.trainer.num_devices) *
                self.train_params['epochs']
            )
            print(f"Calculated total training steps: {self.total_steps}")
