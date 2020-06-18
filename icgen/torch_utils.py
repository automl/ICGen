import more_itertools
import torch

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
