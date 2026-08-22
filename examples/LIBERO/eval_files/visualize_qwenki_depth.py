"""Render reusable expert-depth attention overlays for a QwenKI_depth checkpoint.

This script intentionally keeps data loading minimal; pass a serialized sample
dict with keys: image, lang, optional state. The framework does the rest.
"""

import argparse
import pickle

from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--sample_pkl", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    if args.output_dir:
        cfg.framework.visualization.output_dir = args.output_dir
    cfg.framework.visualization.enabled = True
    model = build_framework(cfg).eval()
    with open(args.sample_pkl, "rb") as f:
        sample = pickle.load(f)
    output = model.predict_action([sample], return_visualization=True)
    visualization = output.get("visualization")
    if visualization is not None:
        for path in visualization.output_paths or []:
            print(path)


if __name__ == "__main__":
    main()
