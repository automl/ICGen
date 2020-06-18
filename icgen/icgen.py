import json
import logging
import math
import random

from collections import defaultdict
from pathlib import Path

import tensorflow_datasets as tfds

from icgen.dataset_names import DATASETS as _DATASETS


logger = logging.getLogger(__name__)

# NOTE: dataset infos include test data in aggregates for pixel values


def _get_resolution_bracket(dim):
    return math.floor(math.log2(dim)) - 4


def _sample_resolution(mean_dim, min_resolution, max_resolution, max_log_res_deviation):
    # Work in discrete log resolutions
    min_bracket = _get_resolution_bracket(min_resolution)
    max_bracket = _get_resolution_bracket(max_resolution)
    real_bracket = _get_resolution_bracket(mean_dim)
    real_bracket = max(min(real_bracket, max_bracket), min_bracket)

    lower = max(real_bracket - max_log_res_deviation, min_bracket)
    upper = min(real_bracket + max_log_res_deviation, max_bracket)
    resolution_bracket = random.choice(list(range(lower, upper + 1)))
    return 2 ** (4 + resolution_bracket)


def _sample_num_classes(min_classes, max_classes, available_num_classes):
    lower = min(available_num_classes, min_classes)
    upper = min(available_num_classes, max_classes)
    return random.choice(list(range(lower, upper + 1)))


def _sample_classes(available_num_classes, num_classes):
    classes = random.sample(range(available_num_classes), k=num_classes)
    return set(classes)


def _sample_examples(
    max_examples_per_class,
    min_examples_per_class,
    examples_per_class,
    total_examples,
    classes,
):
    num_classes = len(classes)
    upper_test_per_class = (
        min(class_examples for class_examples in examples_per_class) // 2
    )
    upper_test_per_class = min(upper_test_per_class, total_examples // (10 * num_classes))
    lower_test_per_class = 5

    if upper_test_per_class <= lower_test_per_class:
        test_per_class = lower_test_per_class
    else:
        test_per_class = random.choice(
            list(range(lower_test_per_class, upper_test_per_class))
        )

    remaining_examples = total_examples - test_per_class * num_classes
    lower = max(min_examples_per_class * num_classes, remaining_examples // 10)
    upper = min(max_examples_per_class * num_classes, remaining_examples)
    sample_ratio = random.choice(list(range(lower, upper))) / total_examples

    class_to_dev_samples = dict()
    class_to_test_samples = dict()
    for class_id in classes:
        total_class_examples = examples_per_class[class_id]
        total_non_test_class_examples = total_class_examples - test_per_class

        num_class_samples = max(
            min_examples_per_class + test_per_class,
            int(sample_ratio * total_non_test_class_examples + test_per_class),
        )
        num_class_samples = min(num_class_samples, total_class_examples)

        samples = random.sample(range(total_class_examples), k=num_class_samples)

        class_to_test_samples[class_id] = set(samples[:test_per_class])
        class_to_dev_samples[class_id] = set(samples[test_per_class:])
    return class_to_dev_samples, class_to_test_samples


def _dataset_to_augmented_identifier(
    dataset,
    dataset_info,
    min_resolution=16,
    max_resolution=512,
    max_log_res_deviation=1,
    min_classes=2,
    max_classes=100,
    min_examples_per_class=20,
    max_examples_per_class=100_000,
):
    identifier = dict()

    # Dataset
    identifier["dataset"] = dataset

    # Resolution
    identifier["resolution"] = _sample_resolution(
        dataset_info["mean_dim"], min_resolution, max_resolution, max_log_res_deviation
    )

    # Classes
    num_classes = _sample_num_classes(
        min_classes, max_classes, dataset_info["num_classes"]
    )
    identifier["classes"] = _sample_classes(dataset_info["num_classes"], num_classes)

    # Actual samples
    class_to_dev_samples, class_to_test_samples = _sample_examples(
        max_examples_per_class,
        min_examples_per_class,
        dataset_info["examples_per_class"],
        dataset_info["num_examples"],
        identifier["classes"],
    )

    identifier["class_to_dev_samples"] = class_to_dev_samples
    identifier["class_to_test_samples"] = class_to_test_samples
    return identifier


def _dataset_to_identifier(dataset, dataset_info):
    identifier = dict()

    identifier["dataset"] = dataset

    # Use all classes and examples, valid_ratio classes statically
    identifier["classes"] = list(range(dataset_info["num_classes"]))

    num_classes = dataset_info["num_classes"]
    examples_per_class = dataset_info["examples_per_class"]
    total_examples = dataset_info["num_examples"]

    test_per_class = min(class_examples for class_examples in examples_per_class) // 2
    test_per_class = min(test_per_class, total_examples // (num_classes * 10))
    identifier["class_to_test_samples"] = {
        class_: set(range(test_per_class)) for class_ in range(num_classes)
    }
    identifier["class_to_dev_samples"] = {
        class_: set(range(test_per_class, examples_per_class[class_]))
        for class_ in range(num_classes)
    }

    # 1,...5, === 16 | 32 | 64 | 128 | 256 | 512
    mean_dim = dataset_info["mean_dim"]
    identifier["resolution"] = 2 ** (4 + _get_resolution_bracket(mean_dim))

    return identifier


def _identifier_to_data(identifier, dataset):
    label_mapping = {
        sampled_class: index for index, sampled_class in enumerate(identifier["classes"])
    }

    train_split, test_split = [], []
    class_to_count = defaultdict(int)
    for image, label in dataset:  # TODO: possible memory waste
        class_ = int(label)
        if class_ not in identifier["classes"]:
            continue

        image_id = class_to_count[class_]
        class_to_count[class_] += 1

        is_train_image = image_id in identifier["class_to_dev_samples"][class_]
        is_test_image = image_id in identifier["class_to_test_samples"][class_]
        if is_train_image:
            train_split.append((image, label_mapping[class_]))
        elif is_test_image:
            test_split.append((image, label_mapping[class_]))

    return train_split, test_split


def _load_dataset_info(dataset_key, info_path):
    with Path(info_path, f"{dataset_key}.json").open() as info_file:
        info = json.load(info_file)
    return info


def _load_dataset(dataset_key, data_path, info_path):
    dataset = tfds.load(
        dataset_key,
        split=None,  # return as dict: split_name -> valid_ratio
        data_dir=data_path,
        batch_size=None,  # return full datasets as tensors
        in_memory=False,
        shuffle_files=False,
        download=False,
        as_supervised=True,
    )
    splits = []
    for _, split in tfds.as_numpy(dataset).items():  # dict: split_name -> valid_ratio
        splits += list(split)
    info = _load_dataset_info(dataset_key, info_path)
    return splits, info


class ICDatasetGenerator:
    def __init__(
        self,
        data_path,
        min_resolution=16,
        max_resolution=512,
        max_log_res_deviation=1,
        min_classes=2,
        max_classes=100,
        min_examples_per_class=20,
        max_examples_per_class=100_000,
    ):
        self._data_path = data_path
        self._info_path = Path(__file__).parent / "infos"

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_log_res_deviation = max_log_res_deviation
        self.min_classes = min_classes
        self.max_classes = max_classes
        self.min_examples_per_class = min_examples_per_class
        self.max_examples_per_class = max_examples_per_class

        self._dataset_to_info = {
            dataset: _load_dataset_info(dataset, self._info_path) for dataset in _DATASETS
        }

    def __repr__(self):
        return (
            f"ICGen("
            f"data_path={self._data_path}, "
            f"min_resolution={self.min_resolution}, "
            f"max_resolution={self.max_resolution}, "
            f"max_log_res_deviation={self.max_log_res_deviation}, "
            f"min_classes={self.min_classes}, "
            f"max_classes={self.max_classes}, "
            f"min_examples_per_class={self.min_examples_per_class}, "
            f"max_examples_per_class={self.max_examples_per_class})"
        )

    def identifier_to_dataset(self, identifier):
        # Identifier includes:
        # - dataset
        # - classes
        # - class_to_dev_samples
        # - class_to_test_samples
        logger.info("Constructing data from identifier")
        dataset, dataset_info = _load_dataset(
            identifier["dataset"], self._data_path, self._info_path
        )
        dev_data, test_data = _identifier_to_data(identifier, dataset)
        return dev_data, test_data, dataset_info

    def get_identifier(self, dataset=None, datasets=None, augment=False):

        if datasets is None and dataset is None:
            raise ValueError("Need to specify either datasets or dataset")

        if datasets and dataset:
            raise ValueError("Do not specify datasets and dataset")

        if datasets:
            dataset = random.choice(datasets)
        dataset_info = self._dataset_to_info[dataset]
        logger.info(f"Sampling identifier for dataset {dataset}")

        if augment:
            return _dataset_to_augmented_identifier(dataset, dataset_info)
        else:
            return _dataset_to_identifier(dataset, dataset_info)

    def get_dataset(self, dataset=None, datasets=None, augment=False):
        identifier = self.get_identifier(
            dataset=dataset, datasets=datasets, augment=augment
        )
        return self.identifier_to_dataset(identifier)
