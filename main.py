import io
import requests
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

def get_device():
    global device
    device = "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
# device = torch.device("cuda") if torch.cuda.is_available() else device
    print(f"Using device: {device}")


class CellularAutomataDataset(Dataset):
    def __init__(self, target_image, num_items=10_000):
        self.grid_size = target_image.shape[0]
        self.target_image = target_image
        self.num_items = num_items
        # self.grids = np.array([self._init_grid()] * num_items) # TODO: remove

    def __len__(self):
        return self.num_items

    @staticmethod
    def _init_center_unit() -> np.ndarray:
        ca = np.zeros(16)  # rgba + 12
        ca[3:] = 1  # set to alive
        return ca

    def _init_grid(self):
        ca = np.zeros(
            (self.grid_size, self.grid_size, 16)
        )  # image: 160x160 with rgba + 12
        ca[:, :, :3] = 1
        center = self.grid_size // 2
        ca[center, center] = self._init_center_unit()
        return torch.from_numpy(ca)

    def __getitem__(self, idx): # TODO: replace with this
       return self.target_image
    # def __getitem__(self, idx):
    #     return self.grids[idx], self.target_image


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

    def __init__(self, n_channels=16, cell_update_chance=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.cell_update_chance = cell_update_chance
        self.seq = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(.25),
            nn.ReLU(),
            nn.Conv2d(128, n_channels, kernel_size=1, bias=False),
        )

    def perceived_vector(self, X):  # this can be learned
        #     # return a vector of the perception of the kernel
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

        stacked_kernels = torch.cat((sobel_x, sobel_y, identity))

        perceived = F.conv2d(X, stacked_kernels, padding=1, groups=self.n_channels)

        return perceived

    def live_cell_mask(self, X, alpha_threshold=0.1):
        # check surrounding 3x3 kernel for live cells
        # if no live cells, then cell dies
        alpha_channel = X[:, 3, :, :].unsqueeze(
            1
        )  # Shape [batch_size, 1, height, width]
        live_cells = (
            alpha_channel > alpha_threshold
        ).float()  # Get live cells as a binary mask
        # print((live_cells > alpha_threshold).sum().item())

        # Define a 3x3 kernel to count the live neighbors
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=X.device)
        # kernel[:, :, 1, 1] = 0  # Ignore the center cell itself

        # Count live neighbors using convolution
        live_neighbors = F.conv2d(live_cells, kernel, padding=1)
        # print((live_neighbors > alpha_threshold).sum().item())

        # exit(1)
        # If a cell has at least one live neighbor, it stays alive/comes to life
        return live_neighbors >= 1

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        pre_mask = self.live_cell_mask(X)

        p = self.perceived_vector(X)
        dx = self.seq(p)
        stochastic_update_mask = (
            torch.rand_like(dx) > self.cell_update_chance
        )  # stochastic update
        X += dx * stochastic_update_mask

        post_mask = self.live_cell_mask(X)
        live_mask = pre_mask & post_mask
        assert live_mask[:, 0, :, :].sum().item() != 0, "ERROR: No live cells"
        X = X * live_mask[:, 0:1, :, :].float()
        return X.permute(0, 2, 3, 1)


def np_to_image(arr):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img


def imshow(t):
    img = np_to_image(t.detach().numpy())
    img.format = "JPG"
    img.show()

def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
    r = requests.get(url)
    return load_image(io.BytesIO(r.content))


def load_image(path: str | io.BytesIO, max_size=40) -> torch.Tensor:
    orig_im = Image.open(path)
    orig_im.thumbnail((max_size, max_size), resample=Image.BILINEAR)
    # orig_im = np.asarray(orig_im)
    orig_im = np.float32(orig_im) / 255.0
    # orig_im = orig_im.reshape(160 * 160, 4)
    rgb, a = orig_im[..., :3], orig_im[..., 3]
    a = a[..., None]
    # multiply each a into rgb
    rgb = rgb * a  # shouldnt work but does
    # img = (rgb * a) + (rgb * (1 - a)) # should work but doesn't
    orig_im = torch.from_numpy(rgb)
    return orig_im


def init_grid(grid_size=40):
    ca = np.zeros(
        (grid_size, grid_size, 16), dtype=np.float32
    )  # image: 160x160 with rgba + 12
    center = grid_size // 2
    ca[center, center, 3:] = 1
    return torch.from_numpy(ca).unsqueeze(0)


def train_loop(model, optimizer, loss_fn, data_loader, epochs=1000):
    # X = initial CA state
    # y = original image
    # pred = calculated image from CA
    X = init_grid(40).to(device)  # FIXME: this is a hack (use image size)
    lowest_loss = 1
    last_saved = 1
    batch_size = data_loader.batch_size
    X = X.repeat(batch_size, 1, 1, 1)
    print("\n\nTraining with batch size: ", batch_size)
    for epoch in range(epochs):
        print(f"\n--------------------------")
        print(f"Epoch {epoch}")
        lowest_loss_loop = lowest_loss
        sX = X
        # if lowest_loss != 1:
        #     print(X.shape)
        #     print(sX.shape)
        #     print(sX[:, 79:82, 79:82, 3])
        #     exit(1)
        # kernels = get_kernels(ca) # probably want to replace this with a perception vector like in the paper
        # for batch_idx, (_, y) in enumerate(data_loader):
        for batch_idx, y in enumerate(data_loader):
            # X = state_grid
            y = y.to(device)

            y_pred = model(X.clone())  # returns updated grid

            loss = loss_fn(y_pred[..., :3], y)  # only use rgb for loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < lowest_loss_loop:
                lowest_loss_loop = loss
                sX = y_pred.detach()  # save best state_grid

            # if (batch_idx * batch_size) % 100 == 0:
            #     print(
            #         f"\tBatch {batch_idx * batch_size:4} Lowest {lowest_loss_loop:.8f}, Loss {loss}"
            #     )

        live_cells = (X[..., 3] > 0.1).sum().item()
        growing_cells = (X[..., 3] > 0).sum().item() - live_cells
        print(f"Imature cells:     {growing_cells}")
        print(f"Mature cells:      {live_cells}")
        print(f"Total live cells:  {growing_cells + live_cells}")
        print(f"Loss:              {lowest_loss_loop}")
        print(f"Lowest Loss:       {lowest_loss}")
        if lowest_loss_loop < lowest_loss:
            lowest_loss = lowest_loss_loop
            X = sX  # save best state_grid
            # gap = last_saved - lowest_loss
            # if gap > .0001:
            last_saved = lowest_loss
            torch.save(X, f"data/train_model/{epoch}_CA_State.pt")
            save_img(X, name=f'train/{epoch}_CA_Image')


# FIXME: not saving images or.. model not working
# def test_loop(data, model, loss_fn, epochs=1000):
def test_loop(model: CANN, data_loader, epochs=1000):
    with torch.no_grad():
        X = init_grid(40).to(device)
        # lowest_loss = 1
        batch_size = data_loader.batch_size
        X = X.repeat(batch_size, 1, 1, 1)
        print("\n\nTesting...")
        for epoch in range(epochs):
            print(f"\n--------------------------")
            print(f"Epoch {epoch}")
            # lowest_loss_loop = lowest_loss
            # sX = X
            for y in data_loader:
                y = y.to(device)

                y_pred = model(X)  # returns updated grid

                X = y_pred

                save_img(X, name=f'test/{epoch}_CA_Image')
            #     loss = loss_fn(y_pred[..., :3], y)  # only use rgb for loss
            #
            #
            #     if loss < lowest_loss_loop:
            #         lowest_loss_loop = loss
            #         sX = y_pred.detach()  # save best state_grid
            #
            #
            # live_cells = (X[..., 3] > 0.1).sum().item()
            # growing_cells = (X[..., 3] > 0).sum().item() - live_cells
            # print(f"Imature cells:     {growing_cells}")
            # print(f"Mature cells:      {live_cells}")
            # print(f"Total live cells:  {growing_cells + live_cells}")
            # print(f"Loss:              {lowest_loss_loop}")
            # print(f"Lowest Loss:       {lowest_loss}")
            # if lowest_loss_loop < lowest_loss:
            #     lowest_loss = lowest_loss_loop
            #     X = sX  # save best state_grid
            #     save_img(X, name=f'test/{epoch}_CA_Image')

def save_img(X, name="CA_Image"):
    img = np_to_image(X[0, :, :, :3].to("cpu").detach().numpy())
    img.save(f"data/{name}.png")

def main():
    get_device()
    emoji = load_emoji("🤑")
    # emoji = load_emoji("🥰")
    # emoji = load_image("res/money_mouth_face.png")

    model = CANN().to(device)
    loss_fn = nn.MSELoss()  # prob need to fix loss
    learning_rate = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = CellularAutomataDataset(emoji, num_items=1000)
    dataloader = DataLoader(dataset, batch_size=10)
    train_loop(model, optimizer, loss_fn, dataloader, epochs=1000)

    # save model
    torch.save(model.state_dict(), "data/CA_Model_FINAL.pt")
    print("\nSaved model to data/CA_Model.pt\n\n")

    test_loop(model, dataloader)



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
