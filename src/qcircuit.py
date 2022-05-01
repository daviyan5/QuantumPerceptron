from dependencies import *


def Uw_phi(basis,phi,n):
  reg_q = QuantumRegister(n,"qr")
  circ_Uwp = QuantumCircuit(reg_q)
  neg = []
  for i,bit in enumerate(basis):
    if bit == "0":
      neg.append(i)
      circ_Uwp.x(i)
  circ_Uwp.append(MCPhaseGate(-phi,n-1),reg_q)
  for i in neg:
    circ_Uwp.x(i)
  circ_Uwp.barrier()
  return circ_Uwp

def Uw(W_phi, n):
  reg_q = QuantumRegister(n,"qr")
  circ_Uw = QuantumCircuit(reg_q)

  for i in range(1,len(W_phi)):
    basis = bin(i)[2:]
    while(len(basis) < n):
      basis = '0' + basis
    circ_Uw.compose(Uw_phi(basis,W_phi[i] - W_phi[0],n),reg_q,inplace = True)

  circ_Uw.h(reg_q)
  return circ_Uw

def Ui_theta(basis,theta,n):
  reg_q = QuantumRegister(n,"qr")
  circ_Uit = QuantumCircuit(reg_q)
  neg = []
  for i,bit in enumerate(basis):
    if bit == "0":
      neg.append(i)
      circ_Uit.x(i)
  circ_Uit.append(MCPhaseGate(theta,n-1),reg_q)
  for i in neg:
    circ_Uit.x(i)
  circ_Uit.barrier()
  return circ_Uit

def Ui(train_set_x, n):
  reg_q = QuantumRegister(n,"qr")
  circ_Ui = QuantumCircuit(reg_q)
  circ_Ui.h(reg_q)
  for i in range(1,len(train_set_x)):
    basis = bin(i)[2:]
    while(len(basis) < n):
      basis = '0' + basis
    circ_Ui.compose(Ui_theta(basis,train_set_x[i] - train_set_x[0],n),reg_q,inplace = True)
  return circ_Ui


def PerceptronCircuit(train_set_x, w_set, n):
    reg_q = QuantumRegister(n,"qr")
    reg_aux = QuantumRegister(1, "aux")
    cla_aux = ClassicalRegister(1)
    total = n + 1
    circ_perc = QuantumCircuit(reg_q,reg_aux,cla_aux)

    circ_perc.barrier()
    circ_perc.compose(Ui(train_set_x,n),range(n), inplace = True)
    circ_perc.compose(Uw(w_set,n),range(n), inplace = True)
    circ_perc.x(reg_q)
    circ_perc.mcx(reg_q,reg_aux)
    circ_perc.measure(reg_aux,cla_aux)
    return circ_perc