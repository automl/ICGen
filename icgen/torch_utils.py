import more_itertools
import torch
import torchvision.transforms as T
import numpy as np

from PIL import Image


class ICDataset:
    def __init__(self, data, transforms=None):
        examples, labels = more_itertools.unzip(data)
        self.data = list(examples)
        self.targets = list(labels)
        self.transforms = transforms

    def __getitem__(self, index):
        # Use as_tensor, since torch.from_numpy: The returned tensor is not resizable
        label = torch.as_tensor(self.targets[index])
        example = Image.fromarray(self.data[index])
        if self.transforms is not None:
            example = self.transforms(example)
        return example, label

    def __len__(self):
        return len(self.data)


def _downsample_fn(split: list, num_channels: int, resolution: int, old_min: int = 0,
                   old_max: int = 255) -> \
    (list, torch.Tensor, torch.Tensor, torch.Tensor, int, int):

    resizer = T.Resize(resolution)
    new_split = []
    npixels = torch.Tensor(0)
    sums = torch.tensor([0. for _ in range(num_channels)])
    sums_sq = torch.tensor([0. for _ in range(num_channels)])
    min_pixel = old_max
    max_pixel = old_min

    for idx, (image, label) in enumerate(more_itertools.unzip(split)):
        image: np.ndarray

        # Calculate stats according to the original, untouched images
        # Assume each image has format (H, W, C)
        npixels += image.shape[0] * image.shape[1]
        sums += image.sum(axis=[0, 1])
        sums_sq += (image ** 2).sum(axis=[0, 1])
        min_pixel, max_pixel = min(min_pixel, image.min()), max(max_pixel, image.max())

        img: Image = Image.fromarray(image)
        img: np.ndarray = np.array(resizer(img))
        new_split.append(img)

    return new_split, npixels, sums, sums_sq, min_pixel, max_pixel


def downsample_dataset(dev_split: list, test_split: list, info: dict,
                       resolution: int) -> (list, list, dict):

    old_min, old_max = info["min_pixel_value"], info["max_pixel_value"]
    nchannels = info["num_channels"]
    new_dev_split, npixels, sums, sums_sq, min_pixel, max_pixel = _downsample_fn(
        dev_split, nchannels, resolution, old_min, old_max)
    dev_means = sums / npixels
    dev_stds = torch.sqrt(sums_sq / npixels - dev_means ** 2)

    info["min_pixel_value"] = min_pixel
    info["max_pixel_value"] = max_pixel
    info["mean_pixel_value"] = dev_means / nchannels
    info["mean_std_pixel_value"] = torch.sqrt(
        (sums_sq.sum() / (npixels * nchannels))
        - info["mean_pixel_value"] ** 2)
    info["mean_pixel_value_per_channel"] = dev_means
    info["mean_std_pixel_value_per_channel"] = dev_stds
    info["is_uniform_across_examples"] = True
    info["is_square"] = True
    for s in info.keys():
        if "_dim" in s:
            if "std" in s:
                info[s] = 0
            else:
                info[s] = resolution

    new_test_split, _, _, _, _, _ = _downsample_fn(test_split, nchannels, resolution,
                                                   old_min, old_max)

    return new_dev_split, new_test_split, info
