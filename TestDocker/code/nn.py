import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

from tqdm import tqdm

from ds import generate_dataset

def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / union).item() if union > 0 else 1.0


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder (downsampling)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder forward pass
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)

        # Decoder forward pass
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Concatenation
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))


class FolderDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def train(
    model, dataloader, criterion, optimizer, device, epoch, test_after=True
):
    model.train()
    epoch_loss = 0
    n = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n += 1
        avg_loss = epoch_loss / n

        progress_bar.set_description(
            f"Epoch {epoch}, training Loss: {avg_loss:.4f}"
        )

    if test_after:
        test_loss, mean_iou = test(model, dataloader, criterion, device)
        print(f"Epoch {epoch}. Loss: {test_loss:.4f}, IoU: {mean_iou:.4f}")

    return avg_loss


def test(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    iou_scores = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            if torch.rand(1) < 0.5:
                images = transforms.functional.hflip(images)
                masks = transforms.functional.hflip(masks)

            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()

            for i in range(images.size(0)):
                iou_scores.append(calculate_iou(outputs[i], masks[i]))

    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    return epoch_loss / len(dataloader), mean_iou


# Example usage
if __name__ == "__main__":
    generate_dataset()
    model = UNet(in_channels=1, out_channels=1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = FolderDataset(
        "./data/train/images", "./data/train/masks", transform=transform
    )
    test_dataset = FolderDataset(
        "./data/test/images", "./data/test/masks", transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    for i in range(25):
        train(model, train_loader, criterion, optimizer, device, epoch=i + 1)
