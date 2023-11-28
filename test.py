import torch
import argparse

import main as ca

def main():
    parser = argparse.ArgumentParser(
            description="Test Script to generate emoji gif from pretrained model"
    )
    parser.add_argument(
            "-p",
            "--gen_pool",
            type=bool,
            default=False,
            help="whether to generate pool gif from the training data result",
    )
    parser.add_argument(
            "-w",
            "--weight_path",
            type=str,
            default="data/CA_Model_TEAR.pt",
            help="path to the model weight",
    )
    # set device
    ca.get_device()
    args = parser.parse_args()
    print(vars(args))

    # load model
    model = ca.CANN(n_channels=16, cell_survival_rate=0.5).to(ca.device)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    emoji = args.weight_path.split("_")[-1].split(".")[0]
    print("Received", emoji)

    # test model
    print("\n preparing...")
    ca.test_loop(model, emoji, visualize_pool=args.gen_pool)


if __name__ == "__main__":
    main()
