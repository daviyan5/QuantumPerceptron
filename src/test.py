from dependencies import *
from prepare_data import *
from train import *


def load_params():
    w = np.load("./params/w.npy")
    acc = np.load("./params/acc.npy")
    return w, acc
    

def main():
    test_set_x = load_images("./data/other")
    w, Hyperparameters = load_params()
    labels = predict(test_set_x, w, Hyperparameters)
    labels = np.array(["gato" if i == 1 else "nao-gato" for i in labels])
    for i in range(len(labels)):
        print("Previsão para imagem {} é {}".fomat(i, labels[i]))
    
if __name__ == "__main__":
    main()