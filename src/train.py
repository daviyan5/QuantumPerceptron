from dependencies import *
from model import *
from prepare_data import *
from test import *
import sys


def initialize(train_set_x,sz):
    w = np.array([uniform(0,pi/2) for i in range(sz)])
    train_set_x = phase_normalize(train_set_x,255)
    return train_set_x, w

def train(train_set_x,train_set_y,test_set_x, test_set_y, w, n):
    Hyperparameters_min = {
    "iterations" : 100,
    "threshold" : 0.5,
    }
    cost_min = -1
    w_min = -1
    acc_min = 0
    for i in range(1):
        Hyperparameters = {
        "iterations" : 200 * (1 + i),
        "threshold" : 0.5 * (1 + i),
        }
        w, cost = optimize(w, train_set_x, train_set_y, Hyperparameters)
        print("\nTesting:")
        labels = predict(test_set_x, w, Hyperparameters)
        acc = (100 - np.mean(np.abs(labels - test_set_y)) * 100)
        print("Iteração {}, Acurácia {} Custo {} ------------------------------".format(i, acc, cost))
        if cost_min == -1 or cost < cost_min:
            cost_min = cost
            w_min = w
            acc_min = acc
            Hyperparameters_min = Hyperparameters
    
    return w_min, Hyperparameters_min, acc_min

def save_model(w,Hyperparameters,acc):
    np.save("./params/w.npy",w)
    np.save("./params/Hyperparameters.npy",Hyperparameters)
    np.save("./params/acc.npy",acc)

def main():
    reload_data = True if len(sys.argv) > 1 and sys.argv[1] == "reload" else False
    train_set_x, train_set_y, test_set_x, test_set_y = load_data(reload_data)
    m = len(train_set_x[0])
    n = ceil(log2(m))
    train_set_x, w = initialize(train_set_x,m)
    w, Hyperparameters, acc = train(train_set_x,train_set_y,test_set_x, test_set_y, w, n)
    save_model(w,Hyperparameters,acc)

if __name__ == "__main__":
    main()