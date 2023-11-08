# import local ca.py model
import torch
from torch.utils.data import DataLoader
import main as ca

def main():
    # set device
    ca.get_device()

    # load model
    model = ca.CANN().to(ca.device)
    model.load_state_dict(torch.load('data/CA_Model_FINAL.pt'))
    model.eval()

    # test model
    emoji = ca.load_emoji("🤑")
    dataset = ca.CellularAutomataDataset(emoji, num_items=1000)
    dataloader = DataLoader(dataset, batch_size=10)
    loss_fn = torch.nn.MSELoss()
    ca.test_loop(model, loss_fn, dataloader)


if __name__ == "__main__":
    main()
