from pathlib import Path
import torch
import argparse

import main as ca
from torch import nn
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(
            description="Test Script to generate emoji gif from pretrained model"
    )
    parser.add_argument(
            "-w",
            "--weight_file",
            type=str,
            help="pt file in the data/weights folder",
    )
    parser.add_argument(
            "-p",
            "--pool",
            type=bool,
            default=False,
            help="whether to generate pool gif from the training data result",
    )
    parser.add_argument(
            "-s",
            "--speed",
            type=float,
            default=5.0,
            help="the speed of the gif generation",
    )

    # set device
    ca.get_device()
    args = parser.parse_args()
    print(vars(args))
    weight_path = Path("data/weights") / args.weight_file
    print("Loading model from", weight_path)

    # load model
    model = ca.CANN(n_channels=16, cell_survival_rate=0.5).to(ca.device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    emoji = weight_path.stem.split("_")[-1]
    print("Received", emoji)

    # test model
    if args.speed < 1.0:
        print("ERROR: speed has to be greater than 1.0")
        return
    ca.test_loop(model, emoji, speed=args.speed, visualize_pool=args.pool)


if __name__ == "__main__":
    main()
