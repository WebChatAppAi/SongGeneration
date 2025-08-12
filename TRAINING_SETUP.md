# Training and Finetuning the SongGeneration Model

This guide provides detailed instructions on how to use the provided modular training pipeline to finetune the SongGeneration model on your own dataset. The pipeline supports both full finetuning and efficient LoRA finetuning.

## Overview of the Training Pipeline

The training pipeline is designed to be modular and is split into several components located in the `training/` directory:

- `train_config.yaml`: The central configuration file. All paths, hyperparameters, and settings are defined here.
- `prepare_data.py`: A script to preprocess your raw audio files and metadata into a tokenized format ready for training.
- `dataset.py`: Contains the PyTorch `Dataset` and `DataLoader` logic for efficiently loading the prepared data.
- `engine.py`: The core training engine, built using PyTorch Lightning. It encapsulates the model loading, training loop, validation loop, and optimization logic.
- `train.py`: The main script used to launch the training process.

---

## Step 1: Initial Setup

### Dependencies
First, ensure you have installed all the necessary dependencies as described in the main `README.md` file.

### Pre-trained Models
You need to download the pre-trained models to use as a starting point for finetuning.
- **Base Model**: Download the `songgeneration_base` model checkpoint from the Hugging Face repository mentioned in the `README.md`.
- **Tokenizers**: The necessary tokenizer models (`audio_tokenizer` and `separate_audio_tokenizer`) are also available from the same Hugging Face repository.

### Folder Structure
Before running the pipeline, organize your files and directories as follows. This structure is crucial for the scripts to locate the necessary files.

```
SongGeneration/
├── ckpt/
│   ├── songgeneration_base/      <-- Place the base model here
│   │   ├── config.yaml
│   │   └── model.pt
│   └── third_party/
│       ├── audio_tokenizer/      <-- Place audio tokenizer here
│       │   ├── config.yaml
│       │   └── model.pt
│       └── separate_audio_tokenizer/ <-- Place separate tokenizer here
│           ├── config.yaml
│           └── model.pt
├── my_dataset/
│   ├── wavs/                     <-- Your raw .wav files
│   │   ├── song1.wav
│   │   └── song2.wav
│   └── metadata.jsonl            <-- Your metadata file
├── training/
│   ├── train_config.yaml         <-- The main config file for training
│   ├── prepare_data.py
│   ├── dataset.py
│   ├── engine.py
│   └── train.py
├── prepared_data/                <-- Output of data preparation (auto-created)
└── training_output/              <-- Training checkpoints and logs (auto-created)
```

---

## Step 2: Configuration

The entire training process is controlled by the `training/train_config.yaml` file. You **must** edit this file to point to your local paths.

### Key Configuration Parameters

- **`paths`**: This is the most important section to configure.
  - `base_model_path`: Path to the base model directory (e.g., `ckpt/songgeneration_base`).
  - `audio_tokenizer_path`: Path to the audio tokenizer checkpoint **directory** (e.g., `ckpt/third_party/audio_tokenizer/`).
  - `separate_audio_tokenizer_path`: Path to the separate audio tokenizer checkpoint **directory** (e.g., `ckpt/third_party/separate_audio_tokenizer/`).
  - `raw_data_dir`: Path to your directory of `.wav` files (e.g., `my_dataset/wavs/`).
  - `metadata_path`: Path to your `.jsonl` metadata file (e.g., `my_dataset/metadata.jsonl`).
  - `prepared_data_dir`: Directory where the processed data will be saved (e.g., `prepared_data/`).
  - `output_dir`: Directory where checkpoints and logs will be saved (e.g., `training_output/`).

- **`training`**:
  - `type`: Set to `'full'` for full finetuning or `'lora'` for LoRA finetuning.
  - `epochs`, `batch_size`, `learning_rate`, etc.: Adjust these based on your dataset size and hardware.

- **`lora`**: These settings are only active if `training.type` is `'lora'`. Adjust `r` and `alpha` as needed.

---

## Step 3: Data Preparation

This step converts your raw `.wav` files into tokenized tensors that the model can consume.

### Metadata Format
Your `metadata.jsonl` file must contain one JSON object per line, corresponding to one `.wav` file. Each object should have the following keys:
- `path`: The relative or absolute path to the audio file.
- `gt_lyric`: The lyrics for the song.
- `descriptions`: Text descriptions of the song's style (e.g., genre, mood).
- `idx`: A unique identifier for the sample.

**Example `metadata.jsonl` line:**
```json
{"path": "my_dataset/wavs/song1.wav", "gt_lyric": "[verse] hello world", "descriptions": "pop, female singer", "idx": "song1"}
```

### Running the Script
Once your `train_config.yaml` is configured, run the following command from the root directory of the project:

```bash
python training/prepare_data.py --config training/train_config.yaml
```

This will process all the audio files listed in your metadata, save the tokenized data into the `prepared_data_dir`, and create a `manifest.jsonl` file inside it.

---

## Step 4: Running the Training

After preparing the data, you can start the training process.

### Full Model Finetuning
1. In `training/train_config.yaml`, ensure `training.type` is set to `'full'`.
2. Run the training script:
   ```bash
   python training/train.py --config training/train_config.yaml
   ```

### LoRA Finetuning
1. In `training/train_config.yaml`, ensure `training.type` is set to `'lora'`.
2. (Optional) Adjust the LoRA parameters (`r`, `alpha`, `target_modules`) in the config file.
3. Run the training script:
   ```bash
   python training/train.py --config training/train_config.yaml
   ```

---

## Step 5: Monitoring and Using Checkpoints

### Monitoring
The training progress, including loss and learning rate, is logged to TensorBoard. You can view the logs by running:
```bash
tensorboard --logdir training_output/logs
```

### Checkpoints
Model checkpoints are saved in the `training_output/checkpoints/` directory.
- For **full finetuning**, these checkpoints are complete model weights and can be used to replace the original `model.pt` for inference.
- For **LoRA finetuning**, the checkpoints only contain the trained adapter weights (e.g., `adapter_model.bin`). These need to be merged with the base model before inference.

### Evaluation / Export
To use your newly finetuned model for generation, you will need to point the inference scripts (like `generate.sh`) to your new checkpoint directory. For LoRA, this requires an extra step of merging the adapter weights with the base model, which can be done using the `peft` library.
