import fiftyone as fo
import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset(
    "open-images-v6", 
    split="validation", 
    label_types=["detections", "classifications"], 
    classes=["Cat"],
    max_samples=500,
    seed=51,
    shuffle=True,
    dataset_name="open-images-cat-dog",
)

dog_subset = foz.load_zoo_dataset(
    "open-images-v6", 
    split="validation", 
    label_types=["detections", "classifications"], 
    classes=["Dog"],
    max_samples=500,
    seed=51,
    shuffle=True,
    dataset_name="dog-subset",
)