from dependencies import *
from prepare_data import *
from model import *


def load_params():
    w = np.load("./params/w.npy")
    Hyperparameters = np.load("./params/Hyperparameters.npy")
    return w, Hyperparameters
    
def predict(test_set_y, w, Hyperparameters):
   A = get_activation(w,test_set_y)
   labels = np.array([1 if i > Hyperparameters["threshold"] else 0 for i in A])
   return labels

def main():
    train_set_x = load_images("./data/other")
    w, Hyperparameters = load_params()
    labels = predict(train_set_x, w, Hyperparameters)
    labels = np.array(["gato" if i == 1 else "nao-gato" for i in labels])
    for i in range(len(labels)):
        print("Previsão para imagem {} é {}".fomat(i, labels[i]))
    
if __name__ == "__main__":
    main()