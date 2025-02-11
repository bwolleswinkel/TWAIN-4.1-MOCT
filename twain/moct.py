"""
This is the high-level module for the multi-objective control toolbox (MOCT) of the TWAIN environment.
"""

from typing import TypeVar, TypeAlias, List
from pathlib import Path

import numpy as np
import scipy as sp
import pandas as pd
import numpy.typing as npt
from floris import FlorisModel

# ------ TYPE ALIASES ------

T = TypeVar('T', int, float, complex, bool)
NPArray: TypeAlias = npt.NDArray[np.dtype[T]]

# ------ CLASSES ------


class Scenario:
    """
    Class which stores a scenario.
    """
    DEFAULT_WT_MODEL = 'nrel_5MW'
    
    def __init__(self, wf_layout: NPArray[float] = None, U_inf: float = None, theta: float = None, TI: float = None, model: tuple[str, ...] = None, perimeter: NPArray[float] = None) -> None:
        self.wf_layout = pd.DataFrame(wf_layout, columns=['x', 'y'])
        self.U_inf = U_inf
        self.theta = theta
        self.TI = TI
        self.wt_model = model if model is not None else [self.DEFAULT_WT_MODEL for _ in range(wf_layout.shape[0])]
        self.n_wt = wf_layout.shape[0]
        self.wind_rose = None
        if perimeter is None:
            convex_hull_indices = sp.spatial.ConvexHull(wf_layout).vertices
            self.perimeter = np.column_stack([wf_layout[convex_hull_indices, 0], wf_layout[convex_hull_indices, 1]])
        else:
            self.perimeter = perimeter


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
        self.wake_surrogate = wake_surrogate if wake_surrogate is not None else self.DEFAULT_WAKE_SURROGATE

    def impact_control_variables(self, control_setpoints: ControlSetpoints, N_grid: int = 100) -> NPArray[float]:
        """
        Function to compute the impact of the control variables on the wind farm.
        """
        #: Calculate the power
        match self.wake_surrogate:
            case 'FLORIS':
                #: Construct the wind farm model from a template
                fmodel = FlorisModel('./config/floris/config_floris_farm.yaml')
                fmodel.set(layout_x=self.scenario.wf_layout['x'], layout_y=self.scenario.wf_layout['y'])
                fmodel.set(turbulence_intensities=[self.scenario.TI], wind_directions=[self.scenario.theta], wind_speeds=[self.scenario.U_inf])
                # FIXME: This gives a persistent warning 'floris.floris_model.FlorisModel WARNING turbine_type has been changed without specifying a new reference_wind_height. reference_wind_height remains 90.00 m. Consider calling `FlorisModel.assign_hub_height_to_ref_height` to update the reference wind height to the turbine hub height.'
                # fmodel.set_wt_yaw(turbine_type=self.scenario.wt_model)
                if self.scenario.wind_rose is not None:
                    fmodel.set(wind_data=self.scenario.wind_rose)
                #: Set the control variables
                fmodel.set(yaw_angles=[control_setpoints.yaw_angles])
                if control_setpoints.power_setpoints is True:
                    fmodel.set_operation_model("simple-derating")
                    fmodel.set(power_setpoints=[control_setpoints.power_setpoints])
                #: Run the model
                fmodel.run()
                #: Extract the powers and local wind speeds
                wt_power, wt_wind_speed = fmodel.get_turbine_powers(), np.zeros(self.scenario.n_wt)
                # TEMP
                #
                import floris.layout_visualization as layoutviz
                from floris.flow_visualization import visualize_cut_plane
                x, y = fmodel.get_turbine_layout()
                print("     x       y")
                for _x, _y in zip(x, y):
                    print(f"{_x:6.1f}, {_y:6.1f}")
                horizontal_plane = fmodel.calculate_horizontal_plane(height=90.0)
                import matplotlib.pyplot as plt
                layoutviz.plot_turbine_points(fmodel)
                visualize_cut_plane(horizontal_plane, title="270 - Aligned")
                plt.show()
                #
                #: Construct the load surrogate model
                lmodel = LoadSurrogateModel()
                #: Extract the loads
                wt_load = np.zeros(self.scenario.n_wt)
                # FIXME: Now, we need to add braces everywhere, because FLORIS accepts multiple wind directions
                for idx in range(self.scenario.n_wt):
                    wt_load[idx] = lmodel.get_turbine_loads(wt_power[0, idx], wt_wind_speed[idx])
            case 'PyWake':
                raise NotImplementedError("PyWake wake surrogate not yet implemented")
            case _:
                raise ValueError(f"Unrecognized wake surrogate model '{self.wake_surrogate}'")
        #: Create something random
        wt_load = None
        #: Extract the perimeter of the wind farm
        perimeter = self.scenario.perimeter
        #: Create a meshgrid
        (X, Y), Z = np.meshgrid(np.linspace(perimeter[:, 0].min(), perimeter[:, 0].max(), N_grid), np.linspace(perimeter[:, 1].min(), perimeter[:, 1].max(), N_grid), indexing='xy'), np.zeros((N_grid, N_grid))
        #: Convert the wind-farm layout to a numpy array
        wf_array_layout = self.scenario.wf_layout.to_numpy()
        #: Compute the noise field
        for wt in wf_array_layout:
            #: Compute the distribution
            Z += placeholder_oscilation_decay_2d(X, Y, wt, self.scenario.U_inf, self.scenario.theta)
        wf_noise = Z
        #: Cap the noise
        wf_noise = 4 * np.clip(wf_noise + 12, 0, Z.max())
        #: Return the results
        return wt_power, wt_load, (X, Y, wf_noise)
    

class LoadSurrogateModel:
    """
    Class which stores the load surrogate model.
    """
    def __init__(self) -> None:
        pass

    def get_turbine_loads(self, wt_power: NPArray[float], wt_wind_speed: NPArray[float]) -> NPArray[float]:
        """
        Function to compute the loads on the wind turbines.
        """
        #: Create something random
        wt_load = np.random.uniform(0, 1)
        #: Return the result
        return wt_load


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
    
    def compute_lcoe(self, scenario: Scenario, wt_power: NPArray[float]) -> float:
        """
        Function to compute the levelized cost of energy. This also takes into account loads.
        """
        #: Compute the annual energy production
        aep = self.compute_aep(wt_power)
        #: Compute the levelized cost of energy
        lcoe = 1.0 * aep
        #: Return the result
        return lcoe


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


# ------ PLACEHOLDERS ------


def placeholder_oscilation_decay_2d(X: NPArray[float], Y: NPArray[float], wt_loc: NPArray[float], U_inf: float, theta: float) -> NPArray[float]:
    """
    Function to compute the 2D draft oscillation decay.
    """
    #: Compute the distance
    R = np.sqrt((X - wt_loc[0]) ** 2 + (Y - wt_loc[1]) ** 2)
    #: Compute the angle
    alpha = np.arctan2(Y - wt_loc[1], X - wt_loc[0])
    #: Compute the decay
    decay = np.exp(-R / 100)
    #: Compute the oscillation
    oscillation = np.sin(2 * np.pi * R / 100)
    #: Compute the draft
    draft = U_inf * decay * oscillation
    #: Return the result
    return draft


