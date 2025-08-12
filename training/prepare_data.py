import os
import argparse
import yaml
import json
from tqdm import tqdm
import torch
import torchaudio
from omegaconf import OmegaConf

# It's better to import from the existing scripts if possible, but to adhere to the "new files only"
# and "no modifications" rule, we can't add an `__init__.py` to the root to make `generate` a module.
# Therefore, we need to add the root to the python path before importing.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generate import Separator
from codeclm.models import builders

def prepare_data(config_path):
    """
    Prepares the training data by converting raw .wav files into tokenized representations.
    Reads configuration from a YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']
    data_params = config['data']

    # --- 1. Load configurations and models ---
    print("Loading models and configurations...")
    # Load the main model config to be used with builders
    model_config_path = os.path.join(paths['base_model_path'], 'config.yaml')
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config not found at {model_config_path}. "
                                f"Please ensure 'base_model_path' in '{config_path}' is correct.")
    model_cfg = OmegaConf.load(model_config_path)

    # Instantiate models
    separator = Separator()
    audio_tokenizer = builders.get_audio_tokenizer_model(paths['audio_tokenizer_path'], model_cfg).cuda().eval()
    separate_tokenizer = builders.get_audio_tokenizer_model(paths['separate_audio_tokenizer_path'], model_cfg).cuda().eval()
    print("Models loaded successfully.")

    # --- 2. Setup directories ---
    output_dir = paths['prepared_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'manifest.jsonl')

    # --- 3. Process data ---
    print(f"Reading metadata from {paths['metadata_path']}...")
    with open(paths['metadata_path'], 'r', encoding='utf-8') as f, \
         open(manifest_path, 'w', encoding='utf-8') as manifest_f:

        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Preparing data")):
            try:
                metadata = json.loads(line)
                audio_path = metadata.get('path') # Assuming the jsonl contains a 'path' key to the audio
                if not audio_path or not os.path.exists(audio_path):
                    print(f"Warning: Audio path not found or does not exist for item: {metadata.get('idx', i)}. Skipping.")
                    continue

                # Separate audio into mixed, vocal, and bgm
                # The separator handles loading and resampling to the model's expected sample rate
                with torch.no_grad():
                    mixed_wav, vocal_wav, bgm_wav = separator.run(audio_path)

                # Truncate or pad to max_duration
                max_len = data_params['sample_rate'] * data_params['max_duration_secs']

                def _process_wav(wav):
                    if wav.shape[-1] > max_len:
                        wav = wav[..., :max_len]
                    else:
                        wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[-1]))
                    return wav.cuda()

                mixed_wav, vocal_wav, bgm_wav = map(_process_wav, [mixed_wav, vocal_wav, bgm_wav])

                # Encode audio to tokens
                with torch.no_grad():
                    # mixed_tokens has shape [1, 1, T], scale is not needed
                    mixed_tokens, _ = audio_tokenizer.encode(mixed_wav.unsqueeze(0))
                    # these have shape [1, 1, T] each
                    vocal_tokens, bgm_tokens = separate_tokenizer.encode(vocal_wav.unsqueeze(0), bgm_wav.unsqueeze(0))

                # Combine tokens into a single tensor for storage [3, T]
                # Order: mixed, vocal, bgm
                all_tokens = torch.cat([
                    mixed_tokens.squeeze(0),
                    vocal_tokens.squeeze(0),
                    bgm_tokens.squeeze(0)
                ], dim=0)

                # Prepare data to save
                save_data = {
                    'tokens': all_tokens.cpu(), # [3, T]
                    'lyrics': metadata.get('gt_lyric', ''),
                    'descriptions': metadata.get('descriptions', '')
                }

                # Save to .pt file
                save_path = os.path.join(output_dir, f"{metadata.get('idx', i)}.pt")
                torch.save(save_data, save_path)

                # Write to manifest
                manifest_entry = {
                    'audio_path': audio_path,
                    'token_path': save_path,
                    'idx': metadata.get('idx', i)
                }
                manifest_f.write(json.dumps(manifest_entry) + '\n')

            except Exception as e:
                print(f"Error processing item {metadata.get('idx', i)}: {e}")
                continue

    print(f"Data preparation complete. Prepared data saved to {output_dir}")
    print(f"Manifest file created at {manifest_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare audio data for SongGeneration model training.")
    parser.add_argument('--config', type=str, default='train_config.yaml',
                        help='Path to the training configuration YAML file.')
    args = parser.parse_args()

    prepare_data(args.config)
