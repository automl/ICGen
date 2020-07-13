import argparse
import logging

from pathlib import Path

import tensorflow_datasets as tfds

import icgen


logger = logging.getLogger("icgen.download")


def download_dataset(dataset, data_dir):
    data_dir = Path(data_dir)
    builder = tfds.builder(dataset, data_dir=data_dir)
    config = tfds.download.DownloadConfig(
        extract_dir=data_dir / "_extract",
        download_mode=tfds.download.GenerateMode.REUSE_DATASET_IF_EXISTS,
    )
    builder.download_and_prepare(
        download_dir=data_dir / "_download", download_config=config
    )


def download_datasets(data_dir, datasets=None, dataset_group=None):
    if datasets is None and dataset_group is None:
        raise ValueError("Need to supply either datasets or datasets_group")

    if datasets is not None and dataset_group is not None:
        raise ValueError("Do not supply both: datasets and dataset_group")

    if datasets is None:
        if dataset_group == "all":
            datasets = icgen.DATASETS
        elif dataset_group == "train":
            datasets = icgen.TRAIN_DATASETS
        elif dataset_group == "val":
            datasets = icgen.VAL_DATASETS
        elif dataset_group == "test":
            datasets = icgen.TEST_DATASETS
        else:
            raise ValueError()

    for i, dataset in enumerate(datasets, 1):
        logger.info(f"Downloading {dataset} ({i}/{len(datasets)})")
        try:
            download_dataset(dataset, data_dir)
        except tfds.download.download_manager.NonMatchingChecksumError:
            logger.exception(
                "Checksum did not match, idealy we would restart the download for you, "
                "but as of now you need to remove all already downloaded files for "
                f"{dataset} and restart the download for this dataset"
            )
        except Exception:
            logger.exception(
                f"Exception occured during the download of {dataset}, please try to "
                "download it again or open an issue"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--dataset_group", default=None, choices=["all", "train", "val", "test"]
    )
    args = parser.parse_args()

    download_datasets(args.data_path, args.datasets, args.dataset_group)
