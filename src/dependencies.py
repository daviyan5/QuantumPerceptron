from qiskit import *
from qiskit.visualization import *
from qiskit.circuit.library import MCPhaseGate
from qiskit.algorithms.optimizers import SPSA

import qiskit.quantum_info as qi
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os


from random import *
from math import *

sim = Aer.get_backend('aer_simulator')
sim.set_options(device = 'GPU')

def phase_normalize(alfa,a):
  return alfa/a * (pi/2)

def phase_denormalize(norm,a):
  return norm * a/ (pi/2)