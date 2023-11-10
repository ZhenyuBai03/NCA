import io
import requests
import pathlib

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from PIL import Image


def get_device():
    global device
    device = "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else device
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

    def __init__(self, n_channels, cell_survival_rate):
        super().__init__()
        self.n_channels = n_channels
        self.cell_survival_rate = cell_survival_rate

        self.seq = nn.Sequential(
            nn.Conv2d(n_channels * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, n_channels, kernel_size=1, bias=False),
        )

        # initialize weights to zero to prevent random noise
        with torch.no_grad():
            self.seq[0].weight.zero_()
            self.seq[2].weight.zero_()

    def perceived_vector(self, X):
        identity = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=X.device)
        identity[:, :, 1, 1] = 1

        sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32,
                device=X.device,
            ).view(1, 1, 3, 3)
            / 8.0
        )
        sobel_y = sobel_x.transpose(2, 3)

        identity = identity.repeat(self.n_channels, 1, 1, 1)
        sobel_x = sobel_x.repeat(self.n_channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(self.n_channels, 1, 1, 1)

        stacked_kernels = torch.cat((identity, sobel_x, sobel_y))

        perceived = F.conv2d(X, stacked_kernels, padding=1, groups=self.n_channels)

        return perceived

    def stochastic_update(self, X):
        mask = (torch.rand(X[:, :1, :, :].shape) < self.cell_survival_rate).to(
            device, torch.float32
        )
        return X * mask

    def live_cell_mask(self, X, alpha_threshold=0.1):
        return (
            F.max_pool2d(X[:, 3:4, :, :], kernel_size=3, stride=1, padding=1)
            > alpha_threshold
        )

    def forward(self, X):
        pre_mask = self.live_cell_mask(X)
        y = self.perceived_vector(X)
        dx = self.seq(y)
        dx = self.stochastic_update(dx)

        X = X + dx

        post_mask = self.live_cell_mask(X)
        live_mask = (pre_mask & post_mask).to(torch.float32)

        return X * live_mask


def load_emoji(emoji, size=40):
    code = hex(ord(emoji))[2:].lower()
    url = (
        "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"
        % code
    )
    r = requests.get(url)
    return load_image(io.BytesIO(r.content), max_size=size)


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


def test_loop(model: CANN, EMOJI_SIZE, N_CHANNELS=16, epochs=8000):
    images = []
    with torch.no_grad():
        X = init_grid(size=EMOJI_SIZE, n_channels=N_CHANNELS).to(device)
        print("\n\nTesting...")
        for epoch in range(epochs):
            print(f"Epoch {epoch:10}/{epochs}", end="\r")
            X = model(X)
            images.append(transforms.ToPILImage()(X[0, :3, :, :]))
            save_img(X, name=f"test/{epoch}_CA_Image")
        print("\n\n")
    make_gif(images)


def make_gif(images):
    print("Making gif...")
    images[0].save(
        "data/test.gif", save_all=True, append_images=images[1:], duration=100, loop=0
    )
    print("Saved gif to data/test.gif")


def save_img(X, name="CA_Image"):
    transform = transforms.ToPILImage()
    tensor = X[0, :3, :, :]
    pil_image = transform(tensor)
    pil_image.save(f"data/{name}.jpg")


def main():
    # HYPERPARAMETERS
    BATCH_SIZE = 8
    LEARNING_RATE = 0.002
    LEARNING_RATE_TWO = 0.0002
    LEARNING_RATE_THREE = 0.00002

    NUM_EPOCHS = 8000

    def NUM_STEPS():
        return np.random.randint(64, 96)

    POOL_SIZE = 1024

    # CONSTANTS
    N_CHANNELS = 16
    CELL_SURVIVAL_RATE = 0.5
    EMOJI_SIZE = 40

    # DEVICE MAKER
    get_device()

    # LOGGING FILES
    log_path = pathlib.Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    # add in padding to prevent weird edges with edges of emoji
    target_emoji_unpadded = load_emoji("💩", size=EMOJI_SIZE)
    # target_emoji_unpadded = load_emoji("🤑", size=EMOJI_SIZE)
    target_emoji_unpadded = F.pad(target_emoji_unpadded, (1, 1, 1, 1), "constant", 0)
    target_emoji = target_emoji_unpadded.to(device)
    # create batch of emojis
    target_emoji = target_emoji.repeat(BATCH_SIZE, 1, 1, 1)

    # initialize model, optimizer, and loss function
    model = CANN(n_channels=N_CHANNELS, cell_survival_rate=CELL_SURVIVAL_RATE).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # initialize empty grid with 1 pixel at the center
    start_grid = init_grid(size=EMOJI_SIZE, n_channels=N_CHANNELS).to(device)
    # pad start grid
    start_grid = F.pad(start_grid, (1, 1, 1, 1), "constant", 0)
    # create pool of values
    pool_grid = start_grid.clone().repeat(POOL_SIZE, 1, 1, 1)

    best_loss = 1
    for epoch in range(NUM_EPOCHS):
        batch_ids = np.random.choice(POOL_SIZE, BATCH_SIZE, replace=False).tolist()
        X = pool_grid[batch_ids]
        for _ in range(NUM_STEPS()):
            X = model(X)

        loss = loss_fn(X[:, :4, ...], target_emoji)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss, epoch)
        if loss < best_loss:
            best_loss = loss
        save_img(X, name=f"train/{epoch}_CA_Image")
        print(f"EPOCH {epoch}\n- Loss:      {loss}\n- Best Loss: {best_loss}\n")
        if epoch == 1000:
            for g in optimizer.param_groups:
                g["lr"] = LEARNING_RATE_TWO
            print(f"(LEARNING RATE CHANGED: {LEARNING_RATE_TWO})")
        if epoch == 3000:
            for g in optimizer.param_groups:
                g["lr"] = LEARNING_RATE_THREE
            print(f"(LEARNING RATE CHANGED: {LEARNING_RATE_THREE})")

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
