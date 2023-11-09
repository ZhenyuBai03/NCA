import torch
import main as ca

def main():
    # set device
    ca.get_device()

    # load model
    model = ca.CANN(n_channels=16, cell_update_chance=0.5).to(ca.device)
    model.load_state_dict(torch.load('data/CA_Model_FINAL.pt'))
    model.eval()

    # test model
    emoji = ca.load_emoji("🤑")
    ca.test_loop(model, emoji.shape[-1])


if __name__ == "__main__":
    main()
