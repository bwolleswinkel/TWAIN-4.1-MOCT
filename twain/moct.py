"""
This is the high-level module for the multi-objective control toolbox (MOCT) of the TWAIN environment.
"""

from typing import TypeVar, TypeAlias, List
from pathlib import Path

import numpy as np
import pandas as pd
import numpy.typing as npt

# ------ TYPE ALIASES ------

T = TypeVar('T', int, float, complex, bool)
NPArray: TypeAlias = npt.NDArray[np.dtype[T]]

# ------ CLASSES ------


class Scenario:
    """
    Class which stores a scenario.
    """
    DEFAULT_WT_MODEL = 'DTU10MW'
    
    def __init__(self, wf_layout: NPArray[float] = None, U_inf: float = None, theta: float = None, model: tuple[str, ...] = None) -> None:
        self.wf_layout = pd.DataFrame(wf_layout, columns=['x', 'y'])
        self.U_inf = U_inf
        self.theta = theta
        self.model = model if model is not None else [self.DEFAULT_WT_MODEL for _ in range(wf_layout.shape[0])]
        self.n_wt = wf_layout.shape[0]


class OptProblem:
    """
    Class which stores an optimization problem.
    """
    def __init__(self, scenario: Scenario, metrics: dict | list, opt_type: str) -> None:
        self.scenario = scenario
        self.metrics = metrics
        self.opt_type = opt_type
        #: Flags
        self.is_multi_objective = len(metrics) > 1
        self.is_solved = False

    def solve(self) -> None:
        """
        Function to solve the optimization problem.
        """
        match self.opt_type:
            case 'downregulation':
                #: Run the optimization method
                opt_results = optimal_downregulation(self)
                #: Return the result
                return opt_results
            case 'wake_steering':
                raise NotImplementedError("Wake steering optimization not yet implemented")
            case _:
                raise ValueError(f"Unrecognized optimization type '{self.opt_type}'")


class ControlSetpoints:
    """
    Class which stores the control setpoints.
    """
    def __init__(self, yaw_angles: NPArray[float] = None, power_setpoints: NPArray[float] = None) -> None:
        self.yaw_angles = yaw_angles
        self.power_setpoints = power_setpoints


class WindFarmModel:
    """
    Class which stores the wind farm model.
    """
    DEFAULT_WAKE_SURROGATE = 'FLORIS'  # Default wake surrogate model, either 'FLORIS' or 'PyWake'

    def __init__(self, scenario: Scenario, wake_surrogate: str = None) -> None:
        self.scenario = scenario
        self.wake_surrogate = wake_surrogate if wake_surrogate is not None else DEFAULT_WAKE_SURROGATE

    def impact_control_variables(self, control_setpoints: ControlSetpoints) -> NPArray[float]:
        """
        Function to compute the impact of the control variables on the wind farm.
        """
        #: Create something random
        wt_power, wt_load, wt_noise = np.random.rand(3, self.scenario.wf_layout.shape[0])
        #: Return the results
        return wt_power, wt_load, wt_noise
    

class Metrics:
    """
    Class which stores the metrics.
    """
    def __init__(self) -> None:
        pass

    def compute_aep(self, wt_power: NPArray[float]) -> float:
        """
        Function to compute the annual energy production.
        """
        #: Compute the annual power production
        aep = np.sum(wt_power)
        #: Return the result
        return aep


# ------ FUNCTIONS ------


def optimal_downregulation(problem: OptProblem, N_iter: int = 100) -> ControlSetpoints:
    """
    Function to solve the downregulation optimization problem.
    """
    #: Create the initial set of control setpoints, greedy control
    control_setpoints = ControlSetpoints(yaw_angles=np.zeros(problem.scenario.n_wt), power_setpoints=np.ones(problem.scenario.n_wt))
    #: Create a surrogate wind farm and metrics model
    wf_model, metrics = WindFarmModel(problem.scenario, wake_surrogate='FLORIS'), Metrics()
    #: Initialize the variables
    aep = np.zeros(N_iter) 
    #: Perform iteration
    for iter in range(N_iter):
        #: Compute the power production
        wt_power, _, _ = wf_model.impact_control_variables(control_setpoints)
        #: Map the power production to metrics
        aep[iter] = metrics.compute_aep(wt_power)
        #: Based on the AEP, compute the next control setpoints
        control_setpoints = ControlSetpoints(yaw_angles=np.zeros(problem.scenario.n_wt), power_setpoints=np.random.uniform(0, 1, problem.scenario.n_wt))
    #: Return the results
    return control_setpoints


