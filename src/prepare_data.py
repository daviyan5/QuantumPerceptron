from dependencies import *
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.splits as fous
from PIL import Image,ImageOps



def load_data_arrays():
    train_set_x = np.load("./params/train_set_x.npy")
    train_set_y = np.load("./params/train_set_y.npy")
    test_set_x = np.load("./params/test_set_x.npy")
    test_set_y = np.load("./params/test_set_y.npy")
    return train_set_x, train_set_y, test_set_x, test_set_y

dim = 16
def load_images(path):
    images = np.array([])
    images = images.reshape(0,dim * dim)
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            image_np = load_image(f)
            images = np.vstack((images,image_np[0]))
    return images

def load_image(f):
    image = ImageOps.grayscale(Image.open(f))
    image = image.resize((dim,dim))
    image_np = np.array(image)
    image_np = image_np.reshape(1,dim * dim)
    return image_np


def prepare_images_labels(dataset):
    set_y = np.zeros(len(dataset))
    set_x = np.array([])
    set_x = set_x.reshape(0,dim * dim)
    for pos,i in enumerate(dataset):
        image_np = load_image(i["filepath"])
        set_x = np.vstack((set_x,image_np[0]))
        pos_labels = i.positive_labels.to_dict()
        for j in range(len(pos_labels["classifications"])):
            if pos_labels["classifications"][j]['label'] == "Cat":
                set_y[pos] = 1
                break
    return set_x, set_y

def save_data_ararys(train_set_x, train_set_y, test_set_x, test_set_y):
    np.save("./params/train_set_x.npy", train_set_x)
    np.save("./params/train_set_y.npy", train_set_y)
    np.save("./params/test_set_x.npy", test_set_x)
    np.save("./params/test_set_y.npy", test_set_y)

def load_data(reload_data = False):
    if(reload_data is False):
        return load_data_arrays()
    

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

    dataset.merge_samples(dog_subset)
    dataset.shuffle(seed=51)
    fo.launch_app(dataset)
    dataset_test = fo.Dataset(name = "test")
    dataset_train = fo.Dataset(name = "train")
    for sample in dataset:
        if random() >= 0.2:
            dataset_train.add_sample(sample)
        if  random() >= 0.8:
            dataset_test.add_sample(sample)
    
    train_set_x, train_set_y = prepare_images_labels(dataset_train)
    test_set_x, test_set_y = prepare_images_labels(dataset_test)
    save_data_ararys(train_set_x, train_set_y, test_set_x, test_set_y)
    return train_set_x, train_set_y, test_set_x, test_set_y
    
train_set_x, train_set_y, test_set_x, test_set_y = load_data()

