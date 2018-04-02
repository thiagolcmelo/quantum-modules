import warnings, shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from quantum_well import QuantumWell
warnings.filterwarnings('error')

eigenvalues = []
eigenvalues_per_length = []
for Vb in [1.0]:
    for L in np.linspace(10, 300, 1000):
        try:
            if int(L) % 100 == 0:
                shutil.copy('results/numeric_quantum_well_eigenvalues_by_well_length.csv',
                            'results/numeric_quantum_well_eigenvalues_by_well_length_%d.csv' % int(L))
                shutil.copy('results/numeric_quantum_well_number_of_eigenvalues_by_well_length.csv',
                            'results/numeric_quantum_well_number_of_eigenvalues_by_well_length_%d.csv' % int(L))
            
            d = QuantumWell(well_length=L, well_height=Vb, N=2048, dt=1e-18)
            d.evolve_imaginary(precision=1e-4)
            evs = d.eigenvalues
            for ev in evs:
                eigenvalues.append((L,ev))
            eigenvalues_per_length.append((L, len(evs)))
            
            print('Finalizamos L = %.2f' % L)
            
            L,e=tuple(zip(*eigenvalues))
            l_e = pd.DataFrame({'L':L, 'e':e})
            l_e.to_csv('results/numeric_quantum_well_eigenvalues_by_well_length.csv')

            L,n=tuple(zip(*eigenvalues_per_length))
            l_n = pd.DataFrame({'L':L, 'n':n})
            l_n.to_csv('results/numeric_quantum_well_number_of_eigenvalues_by_well_length.csv')
        except:
            pass