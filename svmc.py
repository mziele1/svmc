import pandas as pd
import numpy as np
from scipy.interpolate import splrep, splev
import dimod


def load_dwave_schedule(path):
    """Load a scheudle provided by D-Wave and extract the necessary values
    .
    The excel file located at the provided path should have two sheets.
    The second sheet should have four columns containing values for s, A, B, and C.
    Recall D-Wave implements A/2, B/2.
    
    Args:
        path: Path to the schedule file.
    
    Returns:
        s: Schedule s.
        A: Schedule A.
        B: Schedule B.
    """
    schedule = pd.read_excel(path, sheet_name=1)
    s = schedule.iloc[:, 0].values
    A = schedule.iloc[:, 1].values / 2
    B = schedule.iloc[:, 2].values / 2
    return s, A, B


def theta_to_spins(theta):
    """Convert angles to spins [-1, 1]
    
    Args:
        theta: A list of angles.
        
    Returns:
        spins: A list of spins.
    """
    if isinstance(theta, np.ndarray):
        spins = np.ones(len(theta), dtype=np.int8)
        spin_down_idx = np.cos(theta) < 0
        spins[spin_down_idx] = -1
    elif isinstance(theta, dict):
        spins = {k: 1 if np.cos(v) >= 0 else -1}
    else:
        raise TypeError("Only np arrays and dictionaries are supported.")
    return spins


class SVMC:
    """A class that uses spin vector Monte Carlo to simulate quantum annealing.
    
    https://arxiv.org/abs/1401.7087
    https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.15.014029
    https://arxiv.org/abs/1409.3827
    """
    def __init__(self, qpu, model, seed=None):
        """
        Args:
            qpu: A QPU object.
            model: A Model object.
            seed: Seed for rng.
        """
        self.qpu = qpu
        self.model = model
        self.rng =  np.random.default_rng(seed=seed)
        
    def sample(self, sweep_schedule):
        """Performs SVMC according to the input schedule and produces an array of angles.
        
        Args:
            sweep_schedule: A schedule returned by QPU.get_sweep_schedule.
            
        Returns:
            theta: A list of angles.
        """
        theta = self.get_random_angles()
        for s, A, B in sweep_schedule:
            theta_prime = self.get_updated_angles(theta, A, B)
            delta_e = self.model.delta_E(theta, theta_prime, A, B)
            acceptance_idx = self.get_acceptance_idx(delta_e)
            theta[acceptance_idx] = theta_prime[acceptance_idx]
        return theta
    
    def get_acceptance_idx(self, delta_e):
        acceptance_probs = np.exp(-delta_e/self.qpu.temperature)
        acceptance_thresholds = self.rng.random(self.model.bqm.num_variables)
        acceptance_idx = acceptance_probs > acceptance_thresholds
        return acceptance_idx
        
    def get_random_angles(self):
        """Return random angles in the range [0, pi).
        
        Returns:
            theta_prime: Randomly generated angles.
        """
        theta_prime = self.rng.random(self.model.bqm.num_variables) * np.pi
        return theta_prime
    
    def get_updated_angles(self, theta, A, B):
        """Return updated angles in the range [0, pi).
        For SVMC, this is the same as get_random_angles.
        
        Args:
            theta: Unused.
            A: Unused.
            B: Unused.
        Returns:
            theta_prime: Updated angles.
        """
        return self.get_random_angles()        
    

class SVMCTF(SVMC):
    def get_updated_angles(self, theta, A, B):
        """Return updated angles in the range  [0, pi).
        In SVMCTF, angles are updated rather than completely replaced with a new randon angle.
        Updated angles are selected from a range that is centered around the current value and whose width depends on A and B.
        
        Args:
            theta: Current angles.
            A: A value for sweep.
            B: B value for sweep.
        Returns:
            theta_prime: Updated angles.
        """
        ABrat = A/B
        half_range = min(1, ABrat) * np.pi
        theta_prime = theta + ((self.rng.random(self.model.bqm.num_variables) * 2 * half_range) - half_range)
        theta_prime = np.minimum(theta_prime, np.pi)
        theta_prime = np.maximum(theta_prime, 0)
        return theta_prime
    

class Model:
    """A class for modeling the computational problem to be solved.
    """
    def __init__(self, bqm, qpu, normalize=True):
        """
        Args:
            bqm: A dimod bqm representing the problem to be solved.
            qpu: The QPU that the problem will be solved on. This is used to rescale the BQM weights, as is done prior to sampling on the QPU.
            normalize: Whether or not to call bqm.normalize.
        """
        bqm = bqm.copy()
        if not isinstance(bqm, dimod.BQM):
            raise TypeError("Input model must be a dimod BQM.")
        if bqm.vartype != dimod.Vartype.SPIN:
            bqm = dimod.BQM.from_ising(*bqm.to_ising())
        if normalize:
            bqm.normalize(bias_range=qpu.h_range, quadratic_range=qpu.J_range)
        relabeled_bqm, relabel_mapping = bqm.relabel_variables_as_integers(inplace=False)
                
        linear, quadratic, offset = relabeled_bqm.to_ising()
        h = np.zeros(relabeled_bqm.num_variables)
        for i, bias in linear.items():
            h[i] = bias
        J = np.zeros((relabeled_bqm.num_variables, relabeled_bqm.num_variables))
        for (i, j), coupling in quadratic.items():
            J[min(i, j), max(i, j)] = coupling
        
        self.original_bqm = bqm
        self.bqm = relabeled_bqm
        self.relabel_mapping = relabel_mapping
        self.inv_relabel_mapping = {v: k for k, v in relabel_mapping.items()}
        self.h, self.J, self.offset = h, J, offset
        # a symmetric J matrix can be used to reduce computation time
        self.Jsym = np.zeros((self.bqm.num_variables, self.bqm.num_variables), dtype=np.float64)
        self.Jsym += self.J
        self.Jsym += self.J.T
    
    def delta_E(self, theta, theta_prime, A, B):
        """Calculate the energy impact of switching theta to theta_prime.
        
        Args:
            theta: Current angles.
            theta_prime: Proposed angles.
            A: A value for the sweep.
            B: B value for the sweep.
        
        Returns:
            energy: Change in energy.
        """
        sin_theta = np.sin(theta)
        sin_theta_prime = np.sin(theta_prime)
        H_D = sin_theta_prime - sin_theta
        cos_theta = np.cos(theta)
        cos_theta_prime = np.cos(theta_prime)
        H_P = self.h * (cos_theta_prime - cos_theta)
        H_P += (self.Jsym * cos_theta).sum(axis=1) * (cos_theta_prime - cos_theta)
        energy = (-A * H_D) + (B * H_P)
        return energy
    
    def label_spins(self, spins):
        """Assign original bqm variable labels to spins.
        
        Args:
            spins: An iterable of spin values.
        
        Returns:
            labeled_spins: Spins with labels.
        """            
        if self.relabel_mapping != {}:
            labeled_spins = {qubit: spins[i] for i, qubit in self.relabel_mapping.items()}
        else:
            labeled_spins = {i: spin for i, spin in enumerate(spins)}
        return labeled_theta


class QPU:
    """A class that models a QPU.    
    """
    def __init__(self, s, A, B, temperature, h_range, J_range):
        """Init and generate interpolations of A and B from annealing schedule values.
        
        Args:
            s: Annealing schedule s values.
            A: Annealing schedule A (GHz) values.
            B: Annealing schedule B (GHz) values.
            temperature: The reported operating temperature (mK) of the annealer.
            h_range: The range of supported h values.
            J_range: The range of supported J values.
        """
        self.temperature = temperature * 1e-3 * 2.083661912e10 * 1e-9 # -> Kelvin -> Hz -> GHz
        self.h_range = h_range
        self.J_range = J_range
        
        self.Atck = splrep(s, A)
        self.Btck = splrep(s, B)
    
    def get_sweep_schedule(self, annealing_sweeps, pause_sweeps=None, pause_location=None):
        """Returns an annealing schedule of annealing_sweeps sweeps. Optional pause of pause_sweeps sweeps at s=pause_location.
                
        Args:
            annealing_sweeps: Number of discrete steps in the schedule.
            pause_sweeps: Number of steps to pause for.
            pause_location: The s location to pause at.
            
        Returns:
            sweep_schedule: A (sweeps, 3)-shaped array of (s, A, B) values.
        """
        if (pause_sweeps is not None and pause_location is None) or (pause_sweeps is None and pause_location is not None):
            raise TypeError("Both pause_duration and pause_location are required for a pause.")
               
        s = np.linspace(0, 1, annealing_sweeps)
        A = splev(s, self.Atck)
        B = splev(s, self.Btck)
               
        if pause_sweeps is not None:
            pause_A = splev(pause_location, self.Atck)
            pause_B = splev(pause_location, self.Btck)
            idx = np.searchsorted(s, pause_location, side="right")
            A = np.insert(A, idx, np.repeat(pause_A, pause_sweeps))
            B = np.insert(B, idx, np.repeat(pause_B, pause_sweeps))
            s = np.insert(s, idx, np.repeat(pause_location, pause_sweeps))
        sweep_schedule = np.stack([s, A, B], axis=1)
        return sweep_schedule
