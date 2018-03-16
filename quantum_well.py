#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for simulating quantum well. It can calculate:
- eigenvalues, using a pseudo analytical approach.
- eigenvalues and eigenstates, using a numerical approach.
"""

# python standard
import time

# python extended
import numpy as np
import scipy.constants as cte
from scipy.integrate import simps
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import gaussian
from scipy.special import legendre
from scipy.optimize import newton

class QuantumWell(object):
    """Simulates a quantum well"""


    def __init__(self, well_length=100.0, well_height=1.0, N=2048, 
            dt=1e-19, surround=1):
        """Constructor
        
        Keyword Arguments:
            well_length {float} -- the well's length in Angstrom
                (default: {100.0})
            well_height {float} -- the well's height in eV
                (default: {1.0})
            N {int} -- the number of grid points (default: {2048})
            dt {float} -- the delta time step (default: {1e-19})
            surround {int} -- It is how long the full system will be.
                For instance, if the well is 50 Angstrom wide, then:
                - surround=1 means that the whole system will be 150 
                    Angstrom (50 Angstrom on each side)
                - surround=2 means that the whole system will be 250 
                    Angstrom (100 Angstrom on each side)
                (default: {1})
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
        self.m_eff = 1.0

        # set device
        self.set_device(well_length, well_height, N, dt, surround)

    def set_device(self, well_length=100.0, well_height=1.0, N=2048, 
            dt=(1e-19), surround=1):
        """Sets the device's properties even after its creation
        
        Keyword Arguments:
            well_length {float} -- the well's length in Angstrom
                (default: {100.0})
            well_height {float} -- the well's height in eV
                (default: {1.0})
            N {int} -- the number of grid points (default: {2048})
            dt {float} -- the delta time step (default: {1e-19})
            surround {int} -- It is how long the full system will be.
                For instance, if the well is 50 Angstrom wide, then:
                - surround=1 means that the whole system will be 150 
                    Angstrom (50 Angstrom on each side)
                - surround=2 means that the whole system will be 250 
                    Angstrom (100 Angstrom on each side)
                (default: {1})
        
        Returns:
            QuantumWell -- the object itself to be used in chain calls
        """
        self.well_length = W = well_length
        self.well_height = well_height
        self.vb_au = well_height / self.au2ev
        self.wl_au = well_length / self.au2ang
        
        self.device_length = L = (1.0 + 2.0 * surround) * well_length
        self.N = N
        self.z_ang = np.linspace(-L/2.0, L/2.0, self.N)
        self.z_au = self.z_ang / self.au2ang
        self.dz_au = np.abs(self.z_au[1] - self.z_au[0])

        potential = np.vectorize(lambda z: 0.0 if np.abs(z) < W/2 
            else well_height)
        self.v_ev = potential(self.z_ang)
        self.v_au = self.v_ev / self.au2ev

        # time specifics
        return self._set_time(dt)
    
    def analytical_solution(self):
        """The allowed energy levels for the current device
        
        Returns:
            QuantumWell -- the object itself to be used in chain calls
        """
        import warnings
        warnings.filterwarnings('error')
        eigenvalues = []
        
        # transcendental functions shown in most quantum mechanics
        # books, something like:
        # tan(k_w . L / 2) = k_b / k_w
        # cot(k_w . L / 2) = - k_b / k_w
        trans_tan = lambda e: np.tan(
                np.sqrt(2*self.m_eff*e)*self.wl_au/2
            ) - np.sqrt(self.vb_au/e - 1.0)

        trans_tan_der = lambda e: 1.0 / np.cos(
                np.sqrt(2*self.m_eff*e)*self.wl_au/2
            )**2 * (
                self.m_eff * self.wl_au / (2 * np.sqrt(2*self.m_eff*e))
            ) + self.vb_au / (2.0 * e**2 * np.sqrt(self.vb_au/e - 1.0))

        trans_cot = lambda e: 1.0 / np.tan(
                np.sqrt(2*self.m_eff*e)*self.wl_au/2
            ) + np.sqrt(self.vb_au/e - 1.0)

        trans_cot_der = lambda e: -1.0 / np.sin(
                np.sqrt(2*self.m_eff*e)*self.wl_au/2
            )**2 * (
                self.m_eff * self.wl_au / (2 * np.sqrt(2*self.m_eff*e))
            ) - self.vb_au / (2.0 * e**2 * np.sqrt(self.vb_au/e - 1.0))

        # vary from -0.1vb to +1.1vb we use a very good set of kickstart
        # values, the excepts bellow are not properly treated because
        # errors are mostly duo to the newton-raphson divergence
        t_functions = [
            (trans_tan,trans_tan_der),
            (trans_cot, trans_cot_der)
        ]
        for f,fp in t_functions:
            for e0 in np.linspace(-self.vb_au/10.0, self.vb_au, 10000):
                try:
                    root = newton(f, x0=e0, fprime=fp)
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

    def evolve_imaginary(self, nmax=None, precision=1e-4, 
            iterations=None, max_time=None, reset=False):
        """Calculate eigenvalues and eigenstates using a numerical
        approach (Pseudo-Espectral: Split-Step FFT)
        
        Keyword Arguments:
            nmax {int} -- the max number of states to be generated, 
                if the well allows less then nmax or nmax is not set, 
                the actual max is used instead
                (default: {None})
            precision {float} -- The minimum precision for the 
                eigenvalues. Is is not related to the analytical 
                values, but to the numerical convergence. 
                1e-4 stands for 0.01 %
                (default: {1e-4})
            iterations {int} -- the maximum number of iterations
                per level, to be used instead of precision
                (default: {None})
            max_time {float} -- the maximum time to wait for each
                level instead of precision or max iterations
                (default: {None})
            reset {bool} -- if True, the current eigenstates will be
                erased, which means the evolution will start again,
                otherwise, it continues from the last call
                (default: {False})
        
        Returns:
            QuantumWell -- the object itself to be used in chain calls
        """

        # set time to be imaginary
        self._set_time(imaginary=True)
        analytic_values = self.analytical_solution()
        if nmax:
            nmax = min(nmax, len(analytic_values))
        else:
            nmax = len(analytic_values)

        if reset or not hasattr(self, 'eigenvalues'):
            # initialize eigenvalues as zero
            self.eigenvalues = np.zeros(nmax)
            short_grid = np.linspace(-1, 1, self.N)

            # initialize eigenstates as a gaussian multiplied by
            # a legendry polynomial
            g = gaussian(self.N, std=int(self.N/100))
            self.eigenstates = np.array([g*legendre(i)(short_grid) 
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
                        2 * self.eigenstates[s][1:-1] + \
                        self.eigenstates[s][2:]) / self.dz_au**2
                    psi = self.eigenstates[s][1:-1]
                    # <Psi|H|Psi>
                    p_h_p = simps(psi.conj() * (-0.5 * derivative2 / \
                        self.m_eff + self.v_au[1:-1] * psi), \
                        self.z_au[1:-1])
                    # divide by <Psi|Psi> 
                    #p_h_p /= A**2

                    value_before = self.eigenvalues[s] or 1.0
                    self.eigenvalues[s] = p_h_p.real * self.au2ev # eV
                    self.eigenvalues_precisions[s] = np.abs(1.0 - \
                    #    self.eigenvalues[s] / analytic_values[s])
                        self.eigenvalues[s] / value_before)
                    
                    if (iterations and self.counters[s] >= iterations) \
                        or (max_time and self.timers[s] >= max_time) \
                        or (not iterations and not max_time and \
                            self.eigenvalues_precisions[s] < precision):
                        
                        print("""Energy [{0}]:
                                Numeric={1:.10e}
                                Analytic={2:.10e}
                                iterations={3}
                                --------------""".format(
                                    s,
                                    self.eigenvalues[s],
                                    analytic_values[s],
                                    self.counters[s]))
                        break

        return self

    def _set_time(self, dt=None, imaginary=False):
        """It sets the value of the time and whether it is real or
        imaginary. This method initializes the split-step operator.
        
        Keyword Arguments:
            dt {float} -- The time step in seconds (default: {None})
            imaginary {bool} -- True for imaginary, real otherwise
                (default: {False})
        
        Returns:
            QuantumWell -- the object itself to be used in chain calls
        """

        self.dt = dt or self.dt
        self.dt_au = self.dt / self.au_t
        if imaginary:
            self.dt_au *= -1.0j
        

        # # split step
        self.k_au = fftfreq(self.N, d=self.dz_au)
        self.exp_v2 = np.exp(-0.5j * self.v_au * self.dt_au)
        self.exp_t = np.exp(-0.5j * (2 * np.pi * self.k_au) ** 2 * \
            self.dt_au / self.m_eff)
        self.evolution_operator = lambda p: self.exp_v2 * \
            ifft(self.exp_t * fft(self.exp_v2 * p))
        
        return self