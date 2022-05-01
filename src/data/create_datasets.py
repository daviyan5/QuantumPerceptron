import fiftyone as fo
import fiftyone.zoo as foz


dataset_train = foz.load_zoo_dataset(
    "open-images-v6",
    split = "train",
    label_types = ["classifications"],
    classes = ["Cat"],
    max_samples = 500,
)
dataset_test = foz.load_zoo_dataset(
    "open-images-v6",
    split = "test",
    label_types = ["classifications"],
    classes = ["Cat"],
    max_samples = 500,
)
session = fo.launch_app(dataset_train)