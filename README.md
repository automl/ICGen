# ICGen

## Installation

```
pip install icgen
```

for a development installation see [CONTRIBUTING.md](CONTRIBUTING.md)

### Downloading the Datasets

To download datasets you can run

```
python -m icgen.download --data_path DATA_PATH --datasets D1 D2 D3
```

or directly download a complete group

```
python -m icgen.download --data_path DATA_PATH --dataset_group GROUP  # all, train, dev, test
```

For a list of available datasets you can run

```
python -m icgen.dataset_names
```


## Usage

### Sampling Tasks

```python
import icgen
dataset_generator = icgen.ICDatasetGenerator(
  data_path="datasets",  # Replace with the data_path you downloaded the datasets to
  min_resolution=16,
  max_resolution=512,
  max_log_res_deviation=1,  # Sample only 1 log resolution from the native one
  min_classes=2,
  max_classes=100,
  min_examples_per_class=20,
  max_examples_per_class=100_000,
)
dev_data, test_data, dataset_info = dataset_generator.get_dataset(dataset="cifar10", augment=True)
```

The `augment` parameter controls whether the original dataset is modified.

Options only affect sampling with `augment=True` and the min max ranges do not filter datasets.

The data is left at the original resolution, so it can be resized once by the user.

You can also sample from a list of datasets

```python
task = dataset_generator.get_dataset(datasets=["cifar100", "emnist/balanced"], augment=True)
```

We provide some lists of available datasets

```python
import icgen
icgen.DATASETS_TRAIN
icgen.DATASETS_VAL
icgen.DATASETS_TEST
icgen.DATASETS
```

### Reconstructing and Distributing Tasks

In distributed applications it may be necessary to sample datasets on one machine and then use them on another one.
Conversely, for reproducibility it may be necessary to store the exact dataset which was used.
For these cases icgen uses a dataset identifier which uniquely identifies datasets.


## License

[MIT](LICENSE)
