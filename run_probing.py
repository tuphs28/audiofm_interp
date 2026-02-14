import argparse
import os
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.probing import LinearProbe

@hydra.main(version_base="1.3", config_path="configs", config_name="probe")
def main(cfg: DictConfig):

    if not os.path.exists("results"):
        os.makedirs("results")

    hidden_states_dict = np.load(cfg.data.hidden_states_path)
    acoustic_targets_dict = np.load(cfg.data.acoustic_targets_path)

    all_global_r2s = {}
    all_weights = {}

    for target_feature in cfg.probing.target_features:

        all_global_r2s[target_feature] = {}
        all_weights[target_feature] = {}
        targets = acoustic_targets_dict[target_feature]

        for layer_idx in tqdm(cfg.probing.layers, desc=f"Processing {target_feature}"):

            hidden_states = hidden_states_dict[str(layer_idx)]
            
            hidden_states_train, hidden_states_test, targets_train, targets_test = train_test_split(
                hidden_states, targets, test_size=0.2, shuffle=False
            )
            probe = LinearProbe(probe_id=cfg.probing.probe_id, alphas=cfg.probing.alphas)
            trained_probe_dict = probe.fit_and_score(
                hidden_states_train, hidden_states_test, targets_train, targets_test
            )

            all_global_r2s[target_feature][str(layer_idx)] = trained_probe_dict["global_r2"]
            all_weights[target_feature][f"layer_{layer_idx}_coefs"] = trained_probe_dict["probe_coefs"]
            all_weights[target_feature][f"layer_{layer_idx}_intercepts"] = trained_probe_dict["probe_intercepts"]
            all_weights[target_feature][f"layer_{layer_idx}_r2"] = trained_probe_dict["individual_r2"]

    results_dir = f"results/{cfg.model.name}_{cfg.model.init_type}_{cfg.probing.probe_id}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    globalr2s_path = f"{results_dir}/global_r2s.json"
    with open(globalr2s_path, "w") as f:
        json.dump(all_global_r2s, f)

    for target_feature in cfg.probing.target_features:
        weights_path = f"{results_dir}/{target_feature}_weights.npz"
        np.savez_compressed(weights_path, **all_weights[target_feature])

if __name__ == "__main__":
    main()

    
    

    



