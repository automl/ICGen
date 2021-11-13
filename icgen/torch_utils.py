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
    npixels = torch.Tensor([0])
    sums = torch.tensor([0. for _ in range(num_channels)])
    sums_sq = torch.tensor([0. for _ in range(num_channels)])
    min_pixel = old_max
    max_pixel = old_min

    for idx, (image, label) in enumerate(split):
        image_tensor: torch.Tensor = torch.Tensor(image)

        # Calculate stats according to the original, untouched images
        # Assume each image has format (H, W, C)
        npixels += image_tensor.shape[0] * image_tensor.shape[1]
        sums += image_tensor.sum([0, 1])
        sums_sq += (image_tensor ** 2).sum([0, 1])
        min_pixel = min(min_pixel, image_tensor.min())
        max_pixel = max(max_pixel, image_tensor.max())

        img: Image = Image.fromarray(image)
        img: np.ndarray = np.array(resizer(img))
        new_split.append((img, label))

    return new_split, npixels, sums, sums_sq, min_pixel, max_pixel


def downsample_dataset(dev_split: list, test_split: list, info: dict,
                       resolution: int) -> (list, list, dict):

    old_min, old_max = info["min_pixel_value"], info["max_pixel_value"]
    nchannels = info["num_channels"]
    new_dev_split, npixels, sums, sums_sq, min_pixel, max_pixel = _downsample_fn(
        dev_split, nchannels, resolution, old_min, old_max)
    dev_means = sums / npixels
    dev_stds = torch.sqrt(sums_sq / npixels - dev_means ** 2)

    new_info = dict(info)
    new_info["min_pixel_value"] = min_pixel.tolist()
    new_info["max_pixel_value"] = max_pixel.tolist()

    mean_pixel_value = dev_means.mean().squeeze()
    mean_std_pixel_value = torch.sqrt((sums_sq.sum() / (npixels * nchannels)) -
                                      mean_pixel_value ** 2, ).squeeze()
    new_info["mean_pixel_value"] = mean_pixel_value.tolist()
    new_info["mean_std_pixel_value"] = mean_std_pixel_value.tolist()

    new_info["mean_pixel_value_per_channel"] = dev_means.tolist()
    new_info["mean_std_pixel_value_per_channel"] = dev_stds.tolist()
    new_info["is_uniform_across_examples"] = True
    new_info["is_square"] = True
    for s in new_info.keys():
        if "_dim" in s:
            if "std" in s:
                new_info[s] = 0
            else:
                new_info[s] = resolution

    new_test_split, _, _, _, _, _ = _downsample_fn(test_split, nchannels, resolution,
                                                   old_min, old_max)

    return new_dev_split, new_test_split, new_info
