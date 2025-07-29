from pathlib import Path
import numpy as np
from skimage.draw import ellipse
from scipy.ndimage import gaussian_filter
from skimage.morphology import (
    binary_erosion,
    binary_opening,
    binary_dilation,
    disk,
)
from skimage.util import random_noise


from tqdm import tqdm

from PIL import Image

import noise

# Image parameters
image_size = (512, 512)
num_cells_range = (20, 40)
cell_radius_range = (15, 40)
margin = 50
cell_intens_range = (0.2, 0.7)
bg_intens_range = (0, 0.3)


# Function to generate more realistic cell shapes
def generate_realistic_cell_mask(
    image_shape,
    num_cells_range,
    cell_rad_range,
    cell_intensity_range,
    bg_intensity_range,
    margin,
):
    bg_intens = np.random.uniform(*bg_intensity_range)
    img = np.full(image_shape, bg_intens, dtype=np.float32)
    gt_mask = np.zeros(image_shape, dtype=np.uint8)

    num_cells = np.random.randint(*num_cells_range)
    for label in range(1, num_cells + 1):
        radius = np.random.randint(*cell_rad_range)

        probs = (
            gt_mask[margin:-margin, margin:-margin]
            .flatten()
            .copy()
            .astype(np.float32)
        )
        probs = 1 / (probs + 1) ** 2
        if np.sum(gt_mask) > 0:
            probs /= np.sum(probs)
        else:
            probs = np.full(probs.shape, 1 / len(probs))

        idx = np.random.choice(len(probs), p=probs, size=1)[0]
        r = idx // (image_size[1] - 2 * margin) + margin
        c = idx % (image_size[1] - 2 * margin) + margin

        # Create a base ellipse with random aspect ratio
        major_axis = np.random.randint(radius // 2, radius)
        minor_axis = np.random.randint(radius // 2, radius)
        rr, cc = ellipse(r, c, major_axis, minor_axis, shape=image_shape)

        # Add random deformation to simulate irregular cell shape
        deformation = np.random.normal(0, radius // 8, size=(len(rr),))
        rr += deformation.astype(int)
        cc += deformation.astype(int)

        # Create a binary mask for the cell
        cell_mask = np.zeros(image_shape, dtype=np.uint8)
        cell_mask[rr, cc] = 1

        cell_mask = binary_dilation(cell_mask, disk(4))
        cell_mask = binary_opening(cell_mask, disk(9))

        border = cell_mask.astype(np.uint8) - binary_erosion(
            cell_mask, disk(4)
        ).astype(np.uint8)

        cell_intens = np.random.uniform(*cell_intensity_range)
        img[cell_mask == 1] = cell_intens * 1.3
        img[border == 1] = cell_intens
        gt_mask[cell_mask == 1] = 1

    return img, gt_mask


def perlin_noise_2d(
    shape, scale=10, octaves=4, persistence=0.5, lacunarity=2.0
):
    """
    Generate a 2D Perlin noise array.

    Parameters:
        shape (tuple): (height, width) of the noise map.
        scale (float): The frequency of the Perlin noise.
        octaves (int): Number of levels of detail.
        persistence (float): Amplitude multiplier per octave.
        lacunarity (float): Frequency multiplier per octave.

    Returns:
        np.ndarray: 2D array of Perlin noise values.
    """
    height, width = shape
    noise_map = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            noise_map[i, j] = noise.pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
            )

    # Normalize to range [0, 1]
    noise_map = (noise_map - noise_map.min()) / (
        noise_map.max() - noise_map.min()
    )
    return noise_map


def add_perlin_noise(image, scale=10, noise_intensity=0.3):
    """
    Add Perlin noise to an image.

    Parameters:
        image (np.ndarray): Input image (grayscale, values in [0,1]).
        scale (float): Scale of Perlin noise.
        noise_intensity (float): Strength of the noise effect.

    Returns:
        np.ndarray: Image with Perlin noise added.
    """
    perlin = perlin_noise_2d(image.shape, scale=scale)

    # Scale noise intensity
    noisy_image = np.clip(image + noise_intensity * (perlin - 0.5), 0, 1)

    return noisy_image


def generate_image_with_mask(
    image_size,
    num_cells_range,
    cell_rad_range,
    cell_intens_range,
    bg_intens_range,
    margin,
):
    cell_mask, gt_mask = generate_realistic_cell_mask(
        image_shape=image_size,
        num_cells_range=num_cells_range,
        cell_rad_range=cell_rad_range,
        cell_intensity_range=cell_intens_range,
        bg_intensity_range=bg_intens_range,
        margin=margin,
    )

    # Apply Gaussian blur for smoother cell shapes
    cell_mask = gaussian_filter(cell_mask, sigma=1)

    synthetic_image = add_perlin_noise(
        cell_mask, scale=20, noise_intensity=0.5
    )

    synthetic_image += random_noise(synthetic_image, mode="gaussian", var=0.3)

    # Normalize image
    synthetic_image = np.clip(
        (synthetic_image - synthetic_image.min())
        / (synthetic_image.max() - synthetic_image.min()),
        0,
        1,
    )
    return synthetic_image, gt_mask


if __name__ == "__main__":
    # Generate train set
    out_dir = Path("./data/train")
    (out_dir / "images").mkdir(exist_ok=True, parents=True)
    (out_dir / "masks").mkdir(exist_ok=True, parents=True)

    for i in tqdm(range(100)):

        # Generate realistic cell mask and GT mask
        img, mask = generate_image_with_mask(
            image_size=image_size,
            num_cells_range=num_cells_range,
            cell_rad_range=cell_radius_range,
            cell_intens_range=cell_intens_range,
            bg_intens_range=bg_intens_range,
            margin=margin,
        )

        Image.fromarray((img * 255).astype(np.uint8)).save(
            out_dir / "images" / f"{i}.png"
        )
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            out_dir / "masks" / f"{i}.png"
        )

    # Generate test set
    out_dir = Path("./data/test")
    (out_dir / "images").mkdir(exist_ok=True, parents=True)
    (out_dir / "masks").mkdir(exist_ok=True, parents=True)

    for i in tqdm(range(20)):

        # Generate realistic cell mask and GT mask
        img, mask = generate_image_with_mask(
            image_size=image_size,
            num_cells_range=num_cells_range,
            cell_rad_range=cell_radius_range,
            cell_intens_range=cell_intens_range,
            bg_intens_range=bg_intens_range,
            margin=margin,
        )

        Image.fromarray((img * 255).astype(np.uint8)).save(
            out_dir / "images" / f"{i}.png"
        )
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            out_dir / "masks" / f"{i}.png"
        )
