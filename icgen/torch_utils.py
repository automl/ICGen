import more_itertools
import torch
import torchvision.transforms as T
import numpy as np

from PIL import Image

downsampling_force_resize = False

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


def _downsample_fn(split: list, num_channels: int, resolution: int) -> \
    (list, int, torch.Tensor, torch.Tensor, int, int, int):
    """ Downsamples each image in a given dataset split (a list of 2-tuple, consisting of
    a numpy array and a label) to the given resolution. The function returns the
    downsampled split as well as updated dataset statistics - numper of pixels,
    per-channel sum of pixel values, per-channel sum of squared pixel values, minimum and
    maximum pixel values across channels and the number of images with an uneven size
    (i.e. not square). """

    # TODO: Better handling of stat calculation

    resizer = T.Resize(resolution)
    new_split = []
    npixels = torch.Tensor([0])
    sums = torch.tensor([0. for _ in range(num_channels)])
    sums_sq = torch.tensor([0. for _ in range(num_channels)])

    # PIL RGB images only support uint8 values i.e. integers in the range [0, 255].
    # The torchvision Transform ToTensor generates a Tensor with float values in the
    # range [0., 1.], but torch.Tensor can also be uint8 values.
    min_pixel = 255
    max_pixel = 0

    for idx, (image, label) in enumerate(split):
        # image is numpy.ndarray with dtype uint8
        # image_tensor will be a torch.Tensor with dtype uint8
        image_tensor: torch.Tensor = torch.Tensor(image)

        # Calculate stats according to the original, untouched images, thereby
        # effectively treating the downsampling operation as a proper pre-processing step
        # taken during model training even if the downsampled dataset was directly loaded
        # from disk to memory.
        # Assume each image has format (H, W, C)
        npixels += image_tensor.shape[0] * image_tensor.shape[1]

        # Reduce the first two dimensions - H and W, retain the third dimension - C
        sums += image_tensor.sum([0, 1])
        sums_sq += (image_tensor ** 2).sum([0, 1])
        nuneven = 0
        min_pixel = min(min_pixel, img.min())
        max_pixel = max(max_pixel, img.max())

        img: Image = Image.fromarray(image)
        resized_img: Image = resizer(img)

        # If either of the first two dimensions - H and W - are not exactly equal to the
        # desired resolution of the dataset, consider center cropping in order to force
        # this or note that the dataset is not uniform.
        if any([resized_img.size[i] != resolution for i in range(2)]):
            if downsampling_force_resize:
                resized_img = T.functional.center_crop(resized_img, resolution)
            nuneven += 1
        img: np.ndarray = np.array(resized_img)
        new_split.append((img, label))

    return new_split, npixels, sums, sums_sq, min_pixel, max_pixel, nuneven


def downsample_dataset(dev_split: list, test_split: list, info: dict,
                       resolution: int) -> (list, list, dict):

    old_min, old_max = info["min_pixel_value"], info["max_pixel_value"]
    nchannels = info["num_channels"]
    new_dev_split, npixels, sums, sums_sq, min_pixel, max_pixel, nuneven_train = \
        _downsample_fn(dev_split, nchannels, resolution)
    dev_means = sums / npixels
    dev_stds = torch.sqrt(sums_sq / npixels - dev_means ** 2)

    new_info = dict(info)
    new_info["min_pixel_value"] = min_pixel
    new_info["max_pixel_value"] = max_pixel

    mean_pixel_value = dev_means.mean().squeeze()
    mean_std_pixel_value = torch.sqrt((sums_sq.sum() / (npixels * nchannels)) -
                                      mean_pixel_value ** 2, ).squeeze()
    new_info["mean_pixel_value"] = mean_pixel_value.tolist()
    new_info["mean_std_pixel_value"] = mean_std_pixel_value.tolist()

    new_info["mean_pixel_value_per_channel"] = dev_means.tolist()
    new_info["mean_std_pixel_value_per_channel"] = dev_stds.tolist()
    # new_info["is_uniform_across_examples"] = True

    for s in new_info.keys():
        if "_dim" in s:
            if "std" in s:
                new_info[s] = 0
            else:
                new_info[s] = resolution

    new_test_split, _, _, _, _, _, nuneven_test = _downsample_fn(
        test_split, nchannels, resolution)

    new_info["is_square"] = nuneven_train == 0 or downsampling_force_resize
    new_info["num_uneven_train"] = nuneven_train
    new_info["num_uneven_test"] = nuneven_test

    return new_dev_split, new_test_split, new_info


def _pad_fn(split: list, padding: int) -> list:
    """ Given a split - a list of 2-tuples, consisting of a numpy.ndarray instance with
    dtype uint8 and a label - as well as an integer value 'padding', applies 'padding'
    pixels of padding to every image in the split in every direction. Therefore, the
    effective image size will increase from (H, W, C) to
    (H + 2*padding, W + 2*padding, C).
    """

    transformer = T.Pad(padding)
    new_split = []

    for idx, (image, label) in enumerate(split):
        img: Image = Image.fromarray(image)
        padded_img: Image = transformer(img)
        img: np.ndarray = np.array(padded_img)
        new_split.append((img, label))

    return new_split


def pad_images(dev_split: list, test_split: list, info: dict,
               padding: int) -> (list, list, dict):
    """ Pads the images in a given dataset by a given amount, does not alter any of the
    saved info statistics except for the min/max pixel values and the image size
    statistics. """

    new_dev_split = _pad_fn(dev_split, padding)
    new_test_split = _pad_fn(test_split, padding)
    new_info = dict(info)

    # TODO: Handle this better
    for k, v in info.items():
        if "_dim" in k:
            info[k] = v + 2 * padding

    return new_dev_split, new_test_split, new_info

