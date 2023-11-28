import torch
import main as ca
from torch import nn
import torch.nn.functional as F

def main():
    # set device
    ca.get_device()

    # load model
    model = ca.CANN(n_channels=16, cell_survival_rate=0.5).to(ca.device)
    model.load_state_dict(torch.load('data/CA_Model_FINAL.pt'))
    model.eval()

    # test model
    emoji = ca.load_emoji("🤑")
    #emoji = F.pad(emoji, (1, 1, 1, 1), "constant", 0)
    ca.test_loop(model, emoji.shape[-1], epochs=8000)


if __name__ == "__main__":
    main()
