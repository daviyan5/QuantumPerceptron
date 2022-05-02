from dependencies import *
from qcircuit import *
from train import save_model
import time
# tem que estar normalizado
def get_activation(w,train_set_x,num_shots = 8192):
    m = len(train_set_x[0])
    n = ceil(log2(m))
    A = np.array([])
    start = time.time()
    for pos,train_input in enumerate(train_set_x):
        model = PerceptronCircuit(train_input, w, n)
        qobj = assemble(model)   
        counts = sim.run(qobj,shots = num_shots).result().get_counts()
        end = time.time()
        print("Total Simulation Time for {} - {} (s)".format(pos,end-start))
        start = end
        if "1" in counts:
            A = np.append(A,counts["1"]/ num_shots)
        else:
            A = np.append(A,0.)
    return A

def get_dJ(w):
    u = train_set_x - w     # theta - phi
    c = np.cos(u)           # cos(theta - phi)
    C = c.sum(axis = 1).reshape(1,m).T
    s = np.sin(u)
    S = s.sum(axis = 1).reshape(1,m).T
    A = (C * C + S * S) / 2**(2 * n)
    return (2 * (s*C - c*S) / 2**(2 * n)) * A/set_size



def get_A(w,train_set_x,n):## Função exata de A
    u = train_set_x - w     # theta - phi
    c = np.cos(u)           # cos(theta - phi)
    C = c.sum(axis = 1).reshape(1,m).T
    s = np.sin(u)
    S = s.sum(axis = 1).reshape(1,m).T
    return (C * C + S * S) / 2**(2 * n)


def loss_func(w):
    A = get_activation(w,train_set_x_gb,8192)
    diff = A.reshape(set_size,1) - train_set_y_gb.reshape(set_size,1)
    cost = np.sum(diff ** 2) / set_size
    print("Custo: {}\n".format(cost))
    save_model(w, -1)
    return cost
    
def powerseries_f(eta=0.01, power=2, offset=0):
    k = 1
    while True:
        yield eta / ((k + offset) ** power)
        k += 1

def learning_rate_f():
    return powerseries_f(31.53679508386088, 0.602, 0)

def perturbation_f():
    return powerseries_f(0.2,  0.101)

def optimize(w,train_x,train_y, Hyperparameters):
    global train_set_x_gb
    global train_set_y_gb
    global set_size
    train_set_x_gb = train_x
    train_set_y_gb = train_y

    set_size = len(train_set_y_gb)


    spsa = SPSA(maxiter=Hyperparameters["iterations"],learning_rate = learning_rate_f, perturbation = perturbation_f)
    result = spsa.optimize(num_vars=1,objective_function=loss_func,gradient_function=get_dJ, initial_point = w)
    return result[0],result[1]