import io
import requests
import numpy as np
import torch
from torch import nn
import torch.autograd.anomaly_mode as anomaly_mode
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import pathlib

def get_device():
    global device
    device = "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
# device = torch.device("cuda") if torch.cuda.is_available() else device
    print(f"Using device: {device}")

class CANN(nn.Module):
    """
    Cellular Automata Neural Net
    @description: Creates update rules for a CA that can generate an image
    @return: updated rgba + 12
    states:
        - a > .1 = mature    |
        - a <= .1 = growing  | LIVE
        - a == 0 = dead
    """

    def __init__(self, n_channels, cell_update_chance):
        super().__init__()
        self.n_channels = n_channels
        self.cell_update_chance = cell_update_chance

        self.seq = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, n_channels, kernel_size=1, bias=False),
        )

        # why do we need this?
        with torch.no_grad():
            self.seq[2].weight.zero_()

    def perceived_vector(self, X):
        # TODO: REVERT THIS TO OLD CODE
        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        identity_filter = torch.tensor(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=torch.float32,
        )
        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y])  # (3, 3, 3)
        filters = filters.repeat((16, 1, 1))  # (3 * n_channels, 3, 3)
        stacked_filters = filters[:, None, ...].to(device)

        perceived = F.conv2d(X, stacked_filters, padding=1, groups=self.n_channels)

        return perceived
    
    def update(self, X):
        return self.seq()
    
    def stochastic_update(self, X, cell_update_chance):
        mask = (torch.rand(X[:, :1, :, :].shape) <= cell_update_chance).to(device, torch.float32)
        return X * mask

    def live_cell_mask(self, X, alpha_threshold=0.1):
        return F.max_pool2d(X[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > alpha_threshold

    def forward(self, X):
        #X = X.permute(0, 3, 1, 2)
        pre_mask = self.live_cell_mask(X)
        y = self.perceived_vector(X)
        dx = self.seq(y)
        dx = self.stochastic_update(dx, self.cell_update_chance)

        X = X + dx

        post_mask = self.live_cell_mask(X)
        live_mask = (pre_mask & post_mask).to(torch.float32)
        # assert live_mask[:, 0, :, :].sum().item() != 0, "ERROR: No live cells"
        return X * live_mask

def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
    r = requests.get(url)
    return load_image(io.BytesIO(r.content))

def load_image(path: io.BytesIO, max_size=40) -> torch.Tensor:
    orig_im = Image.open(path)
    orig_im.thumbnail((max_size, max_size), resample=Image.BILINEAR)
    orig_im = np.float32(orig_im) / 255.0
    orig_im[..., :3] *= orig_im[..., 3:]
    orig_im = torch.from_numpy(orig_im).permute(2, 0, 1)[None, ...]
    return orig_im

def to_rgb(img_rgba):
    # convert our RGBA image into a rgb image tensor and a alpha value tensor
    # also make sure that alpha value is between 0 and 1
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[..., 3:], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)

def init_grid(size, n_channels):
    # Creates our initial image with a single black pixel
    # initializes all channels except the RGB to 1.0
    X = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    X[:, 3:, size // 2, size // 2] = 1
    return X

def train_loop_old(model, optimizer, loss_fn, data_loader, target, epochs=1000):
    batch_size = data_loader.batch_size
    target = target.to(device, torch.float32)
    target = target.unsqueeze(0)
    target = target.repeat(batch_size, 1, 1, 1)
    print("\n\nTraining with batch size: ", batch_size)
    for epoch in range(epochs):
        print(f"\n--------------------------")
        print(f"Epoch {epoch}")
        # exit()
        # X = init_grid(data_loader.shape[0]).to(device)  # FIXME: this is a hack (use image size)
        X = init_grid(40).to(device)  # FIXME: this is a hack (use image size)
        X = X.repeat(batch_size, 1, 1, 1)
        sX = X
        for y in data_loader:
            y = y.to(device)
            y_pred = model(sX)  # returns updated grid

            sX = y_pred.detach()

        loss = loss_fn(X[..., :3], target) # only use rgb for loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        live_cells = (X[..., 3] > 0.1).sum().item()
        growing_cells = (X[..., 3] > 0).sum().item() - live_cells
        print(f"Imature cells:     {growing_cells}")
        print(f"Mature cells:      {live_cells}")
        print(f"Total live cells:  {growing_cells + live_cells}")
        print(f'Loss:              {loss}')
        torch.save(X, f"data/train_model/{epoch}_CA_State.pt")
        save_img(X, name=f'train/{epoch}_CA_Image')


def train_loop(model, optimizer, loss_fn, data_loader, target, epochs=1000):
    return 0
    


def test_loop(model: CANN, data_loader, epochs=1000):
    with torch.no_grad():
        X = init_grid(40).to(device)
        print("\n\nTesting...")
        for epoch in range(epochs):
            print(f"\n--------------------------")
            print(f"Epoch {epoch}")
            for y in data_loader:
                y = y.to(device)
                y_pred = model(X)  # returns updated grid

                X = y_pred

            save_img(X, name=f'test/{epoch}_CA_Image')

def save_img(X, name="CA_Image"):
    transform = transforms.ToPILImage()
    tensor = X[0, :3, :, :]
    pil_image = transform(tensor)
    pil_image.save(f"data/{name}.png")

def main():
    # HYPERPARAMETERS
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 8000
    NUM_STEPS = 250
    POOL_SIZE = 1024

    # CONSTANTS
    N_CHANNELS = 16
    CELL_UPDATE_CHANCE = 0.5
    EMOJI_SIZE = 40

    # DEVICE MAKER
    get_device()

    # LOGGING FILES
    log_path = pathlib.Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    # add in padding to prevent weird edges with edges of emoji
    target_emoji_unpadded = load_emoji("🤑")
    target_emoji_unpadded = F.pad(target_emoji_unpadded,(1, 1, 1, 1), "constant", 0)
    target_emoji = target_emoji_unpadded.to(device)
    # create batch of emojis
    target_emoji = target_emoji.repeat(BATCH_SIZE, 1, 1, 1)

    # initialize model, optimizer, and loss function
    model = CANN(n_channels=N_CHANNELS, cell_update_chance=CELL_UPDATE_CHANCE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # initialize empty grid with 1 pixel at the center
    start_grid = init_grid(size=EMOJI_SIZE, n_channels=N_CHANNELS).to(device)
    # pad start grid
    start_grid = F.pad(start_grid,(1, 1, 1, 1), "constant", 0)
    # create pool of values
    pool_grid = start_grid.clone().repeat(POOL_SIZE, 1, 1, 1)

    for count in (range(NUM_EPOCHS)):
        print(f"EPOCH: {count}")

        batch_ids = np.random.choice(POOL_SIZE, BATCH_SIZE ,replace=False).tolist()
        X = pool_grid[batch_ids]
        for i in range(NUM_STEPS):
            X = model(X)

        loss = loss_fn(X[:, :4, ...], target_emoji)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Loss:              {loss}')
        writer.add_scalar("train/loss", loss, count)
        # TODO: FIX SAVE IMAGE
        save_img(X, name=f'train/{count}_CA_Image')

        # find which generation got the worst loss
        argmax_batch = loss.argmax().item()
        argmax_pool = batch_ids[argmax_batch]
        # remove the bad sample
        remaining_batch = [i for i in range(BATCH_SIZE) if i != argmax_batch]
        remaining_pool = [i for i in batch_ids if i != argmax_pool]

        # replace the bad growth with a new black pixel and try again
        pool_grid[argmax_pool] = start_grid.clone()
        pool_grid[remaining_pool] = X[remaining_batch].detach()
    # save model
    torch.save(model.state_dict(), "data/CA_Model_FINAL.pt")
    print("\nSaved model to data/CA_Model.pt\n\n")


if __name__ == "__main__":
    main()


# BEST
# num_items=1000
# batch_size=10
# epochs=1000
# learning_rate=0.002
# model code
#    nn.Conv2d(48, 128, kernel_size=1, bias=False),
#    nn.BatchNorm2d(128),
#    nn.Dropout(.25),
#    nn.ReLU(),
#    nn.Conv2d(128, n_channels, kernel_size=1, bias=False),
# Imature cells:     223
# Mature cells:      14663
# Total live cells:  14886
# Loss:              0.029477445408701897
# Lowest Loss:       0.029483851045370102
