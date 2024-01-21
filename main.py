import io
import requests
from pathlib import Path
import argparse
import os
import shutil
import time


from subprocess import Popen, run
import platform
import signal

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from PIL import Image
import torchvision.utils as T

######################################################
# CONSTANTS
BATCH_SIZE = 8
PAD_SIZE = 1
LEARNING_RATE = 0.002
LEARNING_RATE_TWO = 0.0002
LEARNING_RATE_THREE = 0.00002

NUM_EPOCHS = 8000

def NUM_STEPS():
    return np.random.randint(64, 96)

POOL_SIZE = 1024
N_CHANNELS = 16
CELL_SURVIVAL_RATE = 0.5
EMOJI_SIZE = 40
TEMPERATURE = 1

# DEVICE MAKER
def get_device():
    global device
    device = "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else device
    print(f"Using device: {device}")
get_device()

######################################################
# MODEL
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
            self.seq[2].weight.zero_()

    def perceived_vector(self, X):
        sobel_filter = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter / scalar
        sobel_filter_y = sobel_filter.t() / scalar
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
        stacked_filters = filters[:, None, ...].to(device) # (3 * n_channels, 1, 3, 3)

        perceived = F.conv2d(X, stacked_filters, padding=1, groups=self.n_channels)

        return perceived

    def stochastic_update(self, X):
        mask = (torch.rand(X[:, :1, :, :].shape) <= self.cell_survival_rate).to(device, torch.float32)
        return X * mask

    def live_cell_mask(self, X, alpha_threshold=0.1):
        val = F.max_pool2d(X[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > alpha_threshold
        return (val)

    def forward(self, X):
        pre_mask = self.live_cell_mask(X)
        y = self.perceived_vector(X)
        dx = self.seq(y)
        dx = self.stochastic_update(dx)

        X = X + dx

        post_mask = self.live_cell_mask(X)
        live_mask = (pre_mask & post_mask).to(torch.float32)

        return X * live_mask

######################################################
# UTILS
#
def load_emoji(emoji, size=40):
    code = hex(ord(emoji))[2:].lower()
    print(f"Loading emoji: {emoji} ({code})")
    url = (
        "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"
        % code
    )
    r = requests.get(url)
    return load_image(io.BytesIO(r.content), max_size=size), code


def load_image(path: io.BytesIO, max_size=40) -> torch.Tensor:
    orig_im = Image.open(path)
    orig_im.thumbnail((max_size, max_size), resample=Image.LANCZOS)
    orig_im = np.float32(orig_im) / 255.0
    orig_im[..., :3] *= orig_im[..., 3:]
    orig_im = torch.from_numpy(orig_im).permute(2, 0, 1)[None, ...]
    return orig_im

def init_grid(size, n_channels):
    # Creates our initial image with a single black pixel
    # initializes all channels except the RGB to 1.0
    X = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    X[:, 3:, size // 2, size // 2] = 1
    return X

def to_rgb(img_rgba):
    if img_rgba.dim() == 3:  
        rgb, a = img_rgba[:3, :, :], torch.clamp(img_rgba[3:4, ...], 0, 1)
    else:
        rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:4, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)

def make_gif(images, img_name):
    print(f"Making {img_name} gif...")
    gif_dir = Path(f"SAMPLE/{img_name}.gif")
    images[0].save(
        gif_dir, save_all=True, append_images=images[1:], duration=100, loop=0
    )
    print(f"Saved gif to {str(gif_dir)}")


def save_img(X, save_dir, mode="normal"):
    mode_type = ["normal", "batch", "pool"]
    if mode not in mode_type:
        raise ValueError(f"mode must be one of {mode_type}")

    transform = transforms.ToPILImage()
    image_grid = to_rgb(X)
    if mode == "batch":
        image_grid = T.make_grid(image_grid, nrow=8)
    elif mode == "pool":
        w = int(np.ceil(np.sqrt(len(X))))
        image_grid = T.make_grid(image_grid, nrow=w)

    pil_image = transform(image_grid)
    pil_image.save(save_dir)

######################################################
# TEST LOOP
#
def test_loop(model: CANN, emoji_name, speed=1.0, N_CHANNELS=16, epochs=8000, visualize_pool=True):
    pool_images = []
    if visualize_pool:
        pool_dir = Path("data/train/")
        filepathes = sorted(pool_dir.glob("Pool_Image_*.png"), 
                            key=lambda item: item.name)

        print("Loading pool images...")
        for count, filepath in enumerate(filepathes):
            if count % 50 == 0 and count<2000:
                img = Image.open(filepath)
                img_tensor = transforms.ToTensor()(img)
                pool_images.append(transforms.ToPILImage()(img_tensor))
                img.close()
        make_gif(pool_images, "pool"+emoji_name)

    emoji_images = []
    with torch.no_grad():
        X = init_grid(size=EMOJI_SIZE, n_channels=N_CHANNELS).to(device)
        print("\n\nTesting...")
        img_step = 1 / speed
        for epoch in range(epochs):
            print(f"Epoch {epoch:10}/{epochs}", end="\r")
            X = model(X)
            rgb_X = to_rgb(X[0])
            if speed > 1 and epoch % speed == 0:
                emoji_images.append(transforms.ToPILImage()(rgb_X))

            else:
                for _ in range(int(img_step)):
                    emoji_images.append(transforms.ToPILImage()(rgb_X))
        print("\n\n")
    make_gif(emoji_images,emoji_name)

######################################################
# MAIN LOOP
#
def main():
    parser = argparse.ArgumentParser(
            description="Train Script for model to generate target emoji"
    )
    parser.add_argument(
        "-i",
        "--emoji",
        type=str,
        help="the emoji to generate",
    )
    parser.add_argument(
            "-d",
            "--to_data_path",
            type=bool,
            default=False,
            help="whether to generate intermidiate images in /data",
    )

    # set device
    args = parser.parse_args()
    print(vars(args))
    emoji = args.emoji

    # TARGET EMOJI
    # add in padding to prevent weird edges with edges of emoji
    target_emoji_unpadded, emoji_code = load_emoji(emoji, size=EMOJI_SIZE)
    target_emoji_unpadded = F.pad(target_emoji_unpadded, [PAD_SIZE]*4, "constant", 0)
    target_emoji = target_emoji_unpadded.to(device)
    # experiment 1
    target_emoji_pool = target_emoji.repeat(POOL_SIZE, 1, 1, 1)
    target_emoji = target_emoji.repeat(BATCH_SIZE, 1, 1, 1)

    target_dir = Path("data/target_img")
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(f"data/target_img/target_{emoji_code}.png")
    save_img(target_emoji[0], save_dir=save_path, mode="normal")

    # START GRID
    start_grid = init_grid(size=EMOJI_SIZE, n_channels=N_CHANNELS).to(device)
    # pad start grid
    start_grid = F.pad(start_grid, [PAD_SIZE]*4, "constant", 0)
    # create pool of values
    pool_grid = start_grid.clone().repeat(POOL_SIZE, 1, 1, 1)

    # initialize model, optimizer, and loss function
    model = CANN(n_channels=N_CHANNELS, cell_survival_rate=CELL_SURVIVAL_RATE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = 1

    def get_loss(X, target_emoji):
        return ((target_emoji - X[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])
    
    if args.to_data_path:
        batch_dir = Path("data/train/batch_img")
        pool_dir = Path("data/train/pool_img")
        probs_dir = Path("data/train/pool_probs")
        shutil.rmtree(batch_dir)
        shutil.rmtree(pool_dir)
        shutil.rmtree(probs_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)
        pool_dir.mkdir(parents=True, exist_ok=True)
        probs_dir.mkdir(parents=True, exist_ok=True)

    # open tensorboard automatically only on mac
    pwd = Path().resolve()
    macos_tb = None
    if platform.system() == "Darwin":
        run(["rm", "-r", "logs/"])
        macos_tb = Popen(
            [
                "/usr/bin/osascript",
                "-e",
                f'tell app "Terminal" to do script "cd {pwd} &&  python3 -m tensorboard.main --logdir=logs"',
            ]
        )
    # LOGGING FILES for tensorboard
    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)


    try:
        avg_exec_times = torch.zeros(((NUM_EPOCHS//100)-1, 2), dtype=torch.float32)
        total_exec_time = 0
        for epoch in range(NUM_EPOCHS):
            # show execution time of each epoch, check this to see if there is a problem with your pc
            # or if code is running slowly for whatever reason (eg. forgot to plug in craptop)
            if epoch != 0: 
                print("Previous epoch time: ", time.time() - start_time)
            start_time = time.time()

            # Get loss of all, multiply by -1 so lowest loss gets highest probability, divide by temperature, softmax
            sampling_probs = ((-1 * get_loss(pool_grid, target_emoji_pool)) / TEMPERATURE).cpu().softmax(dim=0).numpy()

            # sample a batch according to their probabilities
            batch_ids = np.random.choice(POOL_SIZE, BATCH_SIZE, replace=False, p=sampling_probs).tolist()

            # sort batch by loss
            X = pool_grid[batch_ids]
            loss_rank = get_loss(X, target_emoji).cpu().numpy().argsort()[::-1]
            batch_ids = np.array(batch_ids)[loss_rank]
            X = pool_grid[batch_ids]

            # replace the highest loss with the init grid
            X[0] = start_grid
            X0 = X.clone()

            for _ in range(NUM_STEPS()):
                X = model(X)

            loss = get_loss(X, target_emoji).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss, epoch)

            if loss < best_loss:
                best_loss = loss

            os.system('cls' if os.name == 'nt' else 'clear')
            print("TARGET EMOJI CODE:", emoji_code)
            print(f"EPOCH {epoch}\n- Loss:      {loss}\n- Best Loss: {best_loss}\n")
            if epoch == 1000:
                for g in optimizer.param_groups:
                    g["lr"] = LEARNING_RATE_TWO
                print(f"(LEARNING RATE CHANGED: {LEARNING_RATE_TWO})")
            if epoch == 3000:
                for g in optimizer.param_groups:
                    g["lr"] = LEARNING_RATE_THREE
                print(f"(LEARNING RATE CHANGED: {LEARNING_RATE_THREE})")

            # update pool
            pool_grid[batch_ids] = X.detach()
            if epoch % 100 == 0 and args.to_data_path:
                batch_grid = torch.cat([X0, X], dim=0).detach()
                save_img(batch_grid, save_dir="data/train/batch_img/{:04d}.png".format(epoch), mode="batch")
                save_img(pool_grid, save_dir="data/train/pool_img/{:04d}.png".format(epoch), mode="pool")
                # rounded to 7 decimals, unreadable otherwise
                np.savetxt("data/train/pool_probs/{:04d}.csv".format(epoch), sampling_probs, fmt='%.7f', delimiter = ";")
                avg_exec_times[(epoch//100)-1, 0] = epoch
                avg_exec_times[(epoch//100)-1, 1] = total_exec_time/100
                np.savetxt("data/train/avg_exec_times.csv", avg_exec_times, fmt='%.7f', delimiter = ";")
                total_exec_time = 0

            # open tensorboard automatically only on mac
            if epoch == 100 and macos_tb is not None:
                run(
                    [
                        "open",
                        "-a",
                        "Safari",
                        "http://localhost:6006/?darkMode=true#timeseries",
                    ]
                )
                            
            total_exec_time += time.time() - start_time

        if args.to_data_path:
            np.savetxt("data/train/avg_exec_times.csv", avg_exec_times, fmt='%.7f', delimiter = ";")


    except KeyboardInterrupt:
        print("\nTraining stopped by user")

    finally:
        writer.close()
        if macos_tb is not None:
            terminate = input("Terminate Tensorboard? (y/n): ")
            if terminate.lower() == "y":
                Popen(
                    [
                        "/usr/bin/osascript",
                        "-e",
                        'tell app "Terminal" to quit'
                    ]
                )
        # save model
        weight_path = f"data/weights/CA_Model_{emoji_code}.pt"
        torch.save(model.state_dict(), weight_path)
        print("\nSaved model to\n\n", weight_path)

if __name__ == "__main__":
    main()
