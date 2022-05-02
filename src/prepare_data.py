from dependencies import *
import fiftyone as fo
import fiftyone.zoo as foz
from PIL import Image,ImageOps

from PIL import Image,ImageOps


def load_data_arrays():
    train_set_x = np.load("./params/train_set_x.npy")
    train_set_y = np.load("./params/train_set_y.npy")
    test_set_x = np.load("./params/test_set_x.npy")
    test_set_y = np.load("./params/test_set_y.npy")
    return train_set_x, train_set_y, test_set_x, test_set_y

def load_images(path):
    images = np.array([])
    images = images.reshape(0,16 * 16)
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            image_np = load_image(f)
            images = np.vstack((images,image_np[0]))
    return images

def load_image(f):
    image = ImageOps.grayscale(Image.open(f))
    image = image.resize((16,16))
    image_np = np.array(image)
    image_np = image_np.reshape(1,16 * 16)
    return image_np


def prepare_images_labels(dataset):
    set_y = np.zeros(len(dataset))
    set_x = np.array([])
    set_x = set_x.reshape(0,16 * 16)
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
        
    dataset_train = foz.load_zoo_dataset(
        "open-images-v6",
        split = "train",
        label_types = ["classifications"],
        classes = ["Cat"],
        max_samples = 500,
        dataset_dir= "./data/"
    )
    dataset_test = foz.load_zoo_dataset(
        "open-images-v6",
        split = "test",
        label_types = ["classifications"],
        classes = ["Cat"],
        max_samples = 500,
        dataset_dir= "./data/"
    )
    train_set_x, train_set_y = prepare_images_labels(dataset_train)
    test_set_x, test_set_y = prepare_images_labels(dataset_test)
    save_data_ararys(train_set_x, train_set_y, test_set_x, test_set_y)
    return train_set_x, train_set_y, test_set_x, test_set_y
    
train_set_x, train_set_y, test_set_x, test_set_y = load_data()

