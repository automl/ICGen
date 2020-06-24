TRAIN_DATASETS = [
    "binary_alpha_digits",
    "caltech101",
    "caltech_birds2010",
    "caltech_birds2011",
    "cars196",
    "citrus_leaves",
    "cmaterdb/bangla",
    "cmaterdb/devanagari",
    "coil100",
    "colorectal_histology",
    "cycle_gan/apple2orange",
    "cycle_gan/maps",
    "cycle_gan/vangogh2photo",
    "cycle_gan/iphone2dslr_flower",
    "cats_vs_dogs",
    "deep_weeds",
    "kmnist",
    "emnist/byclass",
    "emnist/mnist",
    "eurosat/rgb",
    "fashion_mnist",
    "horses_or_humans",
    "imagenet_resized/32x32",
    "malaria",
    "mnist",
    "oxford_flowers102",
    "oxford_iiit_pet",
    "rock_paper_scissors",
    "svhn_cropped",
    "plant_village",
    "visual_domain_decathlon/daimlerpedcls",  # stats
    "visual_domain_decathlon/vgg-flowers",
    "visual_domain_decathlon/gtsrb",
    "uc_merced",
    "imagenette",
    "cycle_gan/facades",
    "cycle_gan/summer2winter_yosemite",
]
VAL_DATASETS = [
    "stanford_dogs",
    "cycle_gan/ukiyoe2photo",
    "cifar10",
    "omniglot",  # Contains small1 / small2 valid_ratio, is this a problem?
]
TEST_DATASETS = [
    "cassava",
    "cycle_gan/horse2zebra",
    "cifar100",
    "tf_flowers",
    "cmaterdb/telugu",
    "emnist/balanced",
    "visual_domain_decathlon/ucf101",  # stats
    "visual_domain_decathlon/dtd",
]
DATASETS = TRAIN_DATASETS + VAL_DATASETS + TEST_DATASETS

_ERROR = [
    "eurosat/all",  # High channels, KeyError: "image"
    "so2sat/all",  # High channels, ValueError: as_supervised=True but so2sat does not support a supervised (input, label) structure.
    "cycle_gan/cityscapes",  # NonMatchingChecksumError
    "plantae_k",  # tensorflow.python.framework.errors_impl.ResourceExhaustedError: + noisy
    "cifar10_1",  # Checksum makes problems
]

_LARGER_THAN_4GB = [
    "plant_leaves",  # Noisy during download
    "food101",
    "patch_camelyon",
    "dmlab",
]

_LARGER_THAN_10GB = [  # Could also be manual download
    "i_naturalist2017",
    "places365_small",
    "quickdraw_bitmap",
    "so2sat/rgb",  # Does not require very large RAM for stats?
    "imagenet_resized/64x64",
]

_KEEP_META_BALANCE = [  # Remove some cycle gan datasets
    "cycle_gan/monet2photo",
    "cycle_gan/cezanne2photo",
]

_MANUAL_DOWNLOAD = [  # Could also be too large
    "chexpert",
    "imagenet2012",
    "resisc45",
    "vgg_face2",
]


if __name__ == "__main__":
    print("Train Datasets:")
    print(15 * "=")
    print("\n".join(TRAIN_DATASETS))
    print()

    print("Val Datasets:")
    print(15 * "=")
    print("\n".join(VAL_DATASETS))
    print()

    print("Train Datasets:")
    print(15 * "=")
    print("\n".join(TEST_DATASETS))
    print()
