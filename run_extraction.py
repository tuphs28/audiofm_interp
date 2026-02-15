import argparse
import numpy as np
import os
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm
import librosa
import hydra
from omegaconf import DictConfig, OmegaConf

from src.extractor import FeatureExtractor

@hydra.main(version_base="1.3", config_path="configs", config_name="extract")
def main(cfg: DictConfig):

    os.makedirs("features", exist_ok=True)
    random_init = (cfg.model.init_type == "random")
    extractor = FeatureExtractor(model_id=cfg.model.hf_id, random_init=random_init)

    print("--- loading dataset ---")
    dataset = load_dataset(
        path=cfg.dataset.hf_id, 
        name=cfg.dataset.get("subset", None), 
        split=cfg.dataset.split,
        streaming=False
    )

    shuffled_dataset = dataset.shuffle(seed=42)
    subset = shuffled_dataset.select(range(cfg.num_samples)) 

    all_hidden_states = defaultdict(list) 
    all_acoustic_targets = {feature_name: [] for feature_name in ["stft", "mel", "mfcc"]}
    subset = dataset.take(cfg.num_samples)

    print("--- extracting features ---")
    for i, batch in enumerate(tqdm(subset, total=cfg.num_samples)):
        audio_array = batch["audio"]["array"]
        hidden_states, acoustic_targets = extractor.get_aligned_hidden_states_and_targets(audio_array)
        for layer_idx, hidden_state in hidden_states.items():
            all_hidden_states[layer_idx].append(hidden_state.cpu().numpy())
        for feature_name in ["stft", "mel", "mfcc"]:
            all_acoustic_targets[feature_name].append(acoustic_targets[feature_name])

    print("--- saving features ---")
    final_hidden_states = {str(layer): np.concatenate(all_hidden_states[layer], axis=0) for layer in all_hidden_states.keys()}
    final_acoustic_targets = {target: np.concatenate(all_acoustic_targets[target], axis=0) for target in all_acoustic_targets.keys()}
    hidden_states_path = f"features/{cfg.model.name}_{cfg.model.init_type}_{cfg.dataset.name}_hidden_states.npz"
    acoustic_targets_path = f"features/{cfg.model.name}_{cfg.model.init_type}_{cfg.dataset.name}_acoustic_targets.npz"
    np.savez_compressed(hidden_states_path, **final_hidden_states)
    np.savez_compressed(acoustic_targets_path, **final_acoustic_targets)

    print("--- feature extraction complete ---")


if __name__ == "__main__":
    main()