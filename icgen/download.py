import argparse

from pathlib import Path

import tensorflow_datasets as tfds

import icgen


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", required=True)
parser.add_argument("--datasets", nargs="+")
parser.add_argument("--dataset_group", default="all", choices=["all, train, val, test"])
args = parser.parse_args()

if args.datasets is None:
    if args.dataset_group == "all":
        datasets = icgen.DATASETS
    elif args.dataset_group == "train":
        datasets = icgen.TRAIN_DATASETS
    elif args.dataset_group == "val":
        datasets = icgen.VAL_DATASETS
    elif args.dataset_group == "test":
        datasets = icgen.TEST_DATASETS
    else:
        raise ValueError()
else:
    datasets = args.datasets


def _download(dataset, data_dir):
    data_dir = Path(data_dir)
    builder = tfds.builder(dataset, data_dir=data_dir)
    config = tfds.download.DownloadConfig(
        extract_dir=data_dir / "_extract",
        download_mode=tfds.download.GenerateMode.REUSE_DATASET_IF_EXISTS,
    )
    builder.download_and_prepare(
        download_dir=data_dir / "_download", download_config=config
    )


for i, dataset in enumerate(datasets, 1):
    print(f"Downloading {dataset} ({i}/{len(datasets)})")
    _download(dataset, args.data_path)
