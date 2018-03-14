#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
in this module there is a class for calculating eigenvalues and
eigenstates for a quantum well based on Aluminium gallium arsenide
"""

# python standard
from multiprocessing import Pool, TimeoutError
from sklearn.preprocessing import StandardScaler
import os, time, logging

# python extended
import numpy as np
from scipy.special import hermite
from scipy.integrate import simps
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import gaussian
from scipy.spatial.distance import cdist
import scipy.constants as cte
import scipy.special as sp
from scipy.optimize import newton

class AlGaAsQuantumWell(object):
    """
    """

    def __init__(self, well_length=100, well_x=0.0, barrier_x=0.4, 
            N=2048, dt=1e-19, surround=1, gap_split=(0.7,0.3)):
        """
        """

        # AU of interest
        self.au_l = cte.value('atomic unit of length')
        self.au_t = cte.value('atomic unit of time')
        self.au_e = cte.value('atomic unit of energy')

        # other constants and relations of interest
        self.ev = cte.value('electron volt')
        self.au2ang = self.au_l / 1e-10
        self.au2ev = self.au_e / self.ev

        # constant effective mass
        self.m_eff = 0.067

        # set device
        self.set_device(well_length, well_x, barrier_x, N, 
            dt, surround, gap_split)

    def gap_energy_ev(self, x):
        """
        """
        return 1.519 + 1.447 * x - 0.15 * x**2

    def set_device(self, well_length=100, well_x=0.0, 
            barrier_x=0.4, N=2048, dt=1e-19, surround=1,
            gap_split=(0.7, 0.3)):
        """
        """
        self.well_length = W = well_length
        self.wl_au = self.well_length / self.au2ang
        self.well_x = well_x
        self.barrier_x = barrier_x
        self.device_length = L = (1.0 + 2.0 * surround) * well_length
        self.N = N
        self.z_ang = np.linspace(-L/2.0, L/2.0, self.N)
        self.z_au = self.z_ang / self.au2ang

        self.cond, self.vale = gap_split
        potential = np.vectorize(lambda z: self.gap_energy_ev(well_x) 
            if np.abs(z) < W/2 else self.gap_energy_ev(barrier_x))
        self.v_ev = potential(self.z_ang) - self.gap_energy_ev(well_x)
        self.v_ev *= self.cond
        self.v_au = self.v_ev / self.au2ev

        # the well's height
        self.vb_au = self.v_au[0]-self.v_au[int(N/2)]

        # time specifics
        self.dt = dt
        return self.set_time()
    
    def set_time(self, dt=None, imaginary=False):
        self.dt = dt or self.dt
        self.dt_au = self.dt / self.au_t
        if imaginary:
            self.dt_au *= -1.0j

        # split step
        self.dz_au = np.abs(self.z_au[1] - self.z_au[0])
        self.k_au = fftfreq(self.N, d=self.dz_au)
        self.exp_v2 = np.exp(-0.5j * self.v_au * self.dt_au)
        self.exp_t = np.exp(-0.5j * (2 * np.pi * self.k_au) ** 2 * \
            self.dt_au / self.m_eff)
        self.evolution_operator = lambda p: self.exp_v2 * \
            ifft(self.exp_t * fft(self.exp_v2 * p))
        
        return self

    def analytical_solution(self):
        """
        """
        eigenvalues = []
        self.vb_au

        # transcendental functions shown in most quantum mechanics
        # books, something like:
        # tan(k_w . L / 2) = k_b / k_w
        # cot(k_w . L / 2) = - k_b / k_w
        trans_tan = lambda e: np.tan(\
            np.sqrt(2*self.m_eff*e)*self.wl_au/2) - \
            np.sqrt(2*self.m_eff*(self.vb_au-e)) / \
            np.sqrt(2*self.m_eff*e)

        trans_cot = lambda e: 1.0 / np.tan(\
            np.sqrt(2*self.m_eff*e)*self.wl_au/2)+\
            np.sqrt(2*self.m_eff*(self.vb_au-e)) / \
            np.sqrt(2*self.m_eff*e)

        # vary from 0 to +vb we use a very good set of kickstart
        # values, the excepts bellow are not properly treated because
        # errors are mostly duo to the newton-raphson divergence
        for f in [trans_tan, trans_cot]:
            for e0 in np.linspace(-self.vb_au, self.vb_au, 1000):
                try:
                    root = newton(f, x0=e0)
                    if root > 0:
                        eigenvalues.append(root * self.au2ev)
                except:
                    pass
                    
        eigenvalues = np.array(list(sorted(set(eigenvalues))))
        
        # eigenvalues at this point is a huge list with almost
        # 2000 elements, many of them almost the same, duo to 
        # kickstart values leading to the same eigenvalue 
        # the code bellow identifies the proper ranges and get the
        # average of each, actually, the difference is only
        # the numerical error, which is about 1e-11 eV for an 
        # eigenvalue of order 0.01 eV
        offset = [0]
        for i in range(1, eigenvalues.size):
            if np.abs(eigenvalues[i] / \
                    np.average(eigenvalues[offset[-1]:i])-1.0) > 0.01:
                offset.append(i)
        offset.append(len(eigenvalues))
        eigenvalues = [np.average(eigenvalues[offset[i]:offset[i+1]]) \
            for i in range(len(offset)-1)]
        return eigenvalues
        

    def evolve_imaginary(self, nmax=3, precision=1e-4, 
            iterations=None, max_time=None, reset=False):
        """
        """
        # set time to be imaginary
        self.set_time(imaginary=True)
        analytic_values = self.analytical_solution()

        if reset or not hasattr(self, 'eigenvalues'):
            # initialize eigenvalues as zero
            self.eigenvalues = np.zeros(nmax)
            short_grid = np.linspace(-1, 1, self.N)

            # initialize eigenstates as a gaussian multiplied by
            # a legendry polynomial
            g = gaussian(self.N, std=int(self.N/100))
            self.eigenstates = np.array([g*sp.legendre(i)(short_grid) 
                for i in range(nmax)],dtype=np.complex_)
            
            # the counters will save how many operations were performed
            # on each state
            self.counters = np.zeros(nmax, dtype=np.int32)

            # the times will save how long (in seconds) were necessary
            # for achieving each state
            self.timers = np.zeros(nmax)

            # stores the precision achieved in each eigenvalue
            self.eigenvalues_precisions = np.zeros(nmax)

            # stores the precision achieved in each eigenstate
            self.eigenstates_precisions = np.zeros(nmax)

        
        # split step
        for s in range(nmax):
            while True:
                #state_before = np.copy(self.eigenstates[s])
                # time start here because the above operation
                # has a measurement purpose which we understand
                # not related to the method itself
                start_time = time.time()
                self.eigenstates[s] = \
                    self.evolution_operator(self.eigenstates[s])
                self.counters[s] += 1
                
                # gram-schmidt
                for m in range(s):
                    proj = simps(self.eigenstates[s] * \
                        self.eigenstates[m].conj(), self.z_au)
                    self.eigenstates[s] -= proj * self.eigenstates[m]
                    
                # normalize
                A = np.sqrt(simps(np.abs(self.eigenstates[s])**2, 
                    self.z_au))
                self.eigenstates[s] /= A
                self.timers[s] += time.time() - start_time
                
                if (iterations and self.counters[s] >= iterations) \
                    or (max_time and self.timers[s] >= max_time) \
                    or self.counters[s] % 1000 == 0:
                    # second derivative
                    derivative2 = (self.eigenstates[s][:-2] - \
                        2* self.eigenstates[s][1:-1] + \
                        self.eigenstates[s][2:]) / self.dz_au**2
                    psi = self.eigenstates[s][1:-1]
                    # <Psi|H|Psi>
                    p_h_p = simps(psi.conj() * (-0.5 * derivative2 / \
                        self.m_eff + self.v_au[1:-1] * psi), \
                        self.z_au[1:-1])
                    # divide by <Psi|Psi> 
                    p_h_p /= A**2

                    value_before = self.eigenvalues[s] or 1.0
                    self.eigenvalues[s] = p_h_p.real * self.au2ev # eV
                    self.eigenvalues_precisions[s] = np.abs(1.0 - \
                    #    self.eigenvalues[s] / analytic_values[s])
                        self.eigenvalues[s] / value_before)
                    
                    if (iterations and self.counters[s] >= iterations) \
                        or (max_time and self.timers[s] >= max_time) \
                        or (not iterations and not max_time and \
                            self.eigenvalues_precisions[s] < precision):
                        # XA = [self.eigenstates[s]]
                        # XB = [state_before]
                        # self.eigenstates_precisions[s] = \
                        #     cdist(XA, XB, 'sqeuclidean')[0][0]
                        print("%s: N=%.10e - A=%.10e, iter=%d" % (s, self.eigenvalues[s], analytic_values[s], self.counters[s]))
                        break

        return self
