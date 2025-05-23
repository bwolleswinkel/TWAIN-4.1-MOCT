"""
This is the high-level module for the multi-objective control toolbox (MOCT) of the TWAIN environment.
"""

from __future__ import annotations
from typing import TypeVar, TypeAlias, Any

import numpy as np
import scipy as sp
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from floris import FlorisModel, WindRose as FlorisWindRose
from floris.optimization.layout_optimization.layout_optimization_scipy import LayoutOptimizationScipy
from floris.optimization.yaw_optimization.yaw_optimizer_geometric import YawOptimizationGeometric

# ------ TYPE ALIASES ------

T = TypeVar('T', int, float, complex, bool)
NPArray: TypeAlias = npt.NDArray[np.dtype[T]]

# ------ CLASSES ------


class Scenario:
    """
    Class which stores a scenario.
    """
    DEFAULT_WT_MODEL = 'nrel_5MW'
    
    def __init__(self, wf_layout: NPArray[float] = None, U_inf: float = None, theta: float = None, TI: float = None, model: tuple[str, ...] = None, perimeter: NPArray[float] = None, n_wt: int = None, wt_names: list = None, power_lines: list = None) -> None:
        self.wf_layout = pd.DataFrame(wf_layout, columns=['x', 'y'])
        self.U_inf = U_inf
        self.theta = theta
        self.TI = TI
        # TODO: We should really add a lot of check, i.e., check consistency between the number of wind turbines, size of array_layout, number of names, etc.
        self.n_wt = n_wt if n_wt is not None else wf_layout.shape[0]
        self.wt_model = WindTurbineModel(model) if model is not None else WindTurbineModel(self.DEFAULT_WT_MODEL)
        # TODO: Combine wind turbine names and wind farm layout in a single DataFrame
        self.wt_names = wt_names if wt_names is not None else [f'{i:03}' for i in range(self.n_wt)]
        self.wind_rose = None
        if perimeter is None and wf_layout is not None:
            convex_hull_indices = sp.spatial.ConvexHull(wf_layout).vertices
            self.perimeter = np.column_stack([wf_layout[convex_hull_indices, 0], wf_layout[convex_hull_indices, 1]])
        else:
            self.perimeter = perimeter
        self.power_lines = power_lines

    def __str__(self) -> str:
        return f"<class: {self.__class__.__name__}> | Scenario object with {self.n_wt if self.n_wt is not None and self.n_wt > 0 else 'unspecified number of'} wind turbines\n" + f"+------------+\n" + f"{''.join(['|            |' for _ in range(4)])}" + f"+------------+"


class WindRose(FlorisWindRose):
    """
    Class which stores a wind rose. This class also 'eats up' the other classes 
    """
    def __init__(self, wind_rose: dict | pd.DataFrame) -> None:
        #: Match the type of file
        match wind_rose:
            case dict():
                #: Convert the dictionary to a DataFrame
                wind_speeds = np.array(wind_rose['wind_speeds'])
                # FIXME: This does not actually work, does not have the right information
                df_wind_rose = pd.DataFrame(wind_rose['wind_rose'], columns=['direction', 'frequencies'])
                self.wind_rose = df_wind_rose
                self.wind_speeds = wind_speeds
            case pd.DataFrame():
                #: Save the dataframe
                self.wind_rose = wind_rose
                #: Check the number of bins
                n_bins_wd, n_bins_ws = np.unique(wind_rose['wind_dir']).size, np.unique(wind_rose['wind_speed']).size
                self.n_bins_wd = n_bins_wd
                self.n_bins_ws = n_bins_ws
                #: Create an array with all the prevalence values
                # FIXME: We need to flip thus up/down because the wind speeds need to be in increasing order
                self.data = np.flipud(wind_rose['prevalence'].to_numpy().reshape((n_bins_ws, n_bins_wd), order='F'))
            case _:
                raise ValueError(f"Unrecognized wind rose format '{type(wind_rose)}'")

    @classmethod
    def from_ts(cls, wind_direction: NPArray[float], wind_speed: NPArray[float], n_bins_wd: int = None, n_bins_ws: int = None) -> WindRose:
        #: Compute a histogram
        hist, _, _ = np.histogram2d(wind_speed, wind_direction, bins=[n_bins_ws, n_bins_wd], range=[[0, 25], [0, 360]])
        #: Normalize the histogram
        hist = hist / np.sum(hist)
        #: Create a DataFrame
        T, R = np.meshgrid(np.linspace(0, 360, n_bins_wd), np.linspace(0, 25, n_bins_ws), indexing='xy')
        df_wind_rose = pd.DataFrame(np.column_stack((T.flatten(order='F'), R.flatten(order='F'), hist.flatten(order='F'))), columns=['wind_dir', 'wind_speed', 'prevalence'])
        #: Create a wind rose object
        wind_rose = cls(df_wind_rose)
        #: Return the wind rose object
        return wind_rose
    
    def __str__(self) -> str:
        return f"<class: {self.__class__.__name__}> | Wind rose object with {self.n_bins_wd} direction bins and {self.n_bins_ws} wind speed bins\n" + f"{self.wind_rose.__str__()}"


class OptProblem:
    """
    Class which stores an optimization problem.
    """
    def __init__(self, scenario: Scenario, metrics: dict | list, opt_type: str, opt_method: str = None, params: Any = None) -> None:
        self.scenario = scenario
        self.metrics = metrics
        self.opt_type = opt_type
        self.opt_method = opt_method
        self.params = params
        #: Flags
        self.is_multi_objective = len(metrics) > 1
        self.is_solved = False

    def solve(self, verbose: int = 0) -> None:
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
                #: Run the optimization method
                opt_results = optimal_wake_steering(self, verbose=verbose)
                #: Return the result
                return opt_results
            case 'layout':
                #: Run the optimization method
                control_setpoints, scenario = optimal_greedy_layout(self)
                #: Return the result
                return control_setpoints, scenario
            case 'power_lines':
                #: Extract the groups
                groups = self.params
                #: Run the optimization method
                vertices_power_lines = optimal_power_lines(self, groups)
                #: Update the scenario
                self.scenario.power_lines = vertices_power_lines
                #: Return the result
                return None, self.scenario
            case _:
                raise ValueError(f"Unrecognized optimization type '{self.opt_type}'")


class ControlSetpoints:
    """
    Class which stores the control setpoints.
    """
    def __init__(self, yaw_angles: NPArray[float] = None, power_setpoints: NPArray[float] = None) -> None:
        self.yaw_angles = yaw_angles
        self.power_setpoints = power_setpoints


class WindTurbineModel:
    """
    Class which stores the wind turbine model.
    """
    # TODO: This really inherit from PyWake and FLORIS, i.e., superclasses
    def __init__(self, model: str) -> None:
        self.model = model
        match model:
            case 'nrel_5MW':
                self.hub_height = 90.0
                self.rotor_diameter = 63.0
            case _:
                raise ValueError(f"Unrecognized wind turbine model '{model}'")


class WindFarmModel:
    """
    Class which stores the wind farm model.
    """
    DEFAULT_WAKE_SURROGATE = 'FLORIS'  # Default wake surrogate model, either 'FLORIS' or 'PyWake'

    def __init__(self, scenario: Scenario, wake_surrogate: str = None) -> None:
        self.scenario = scenario
        self.wake_surrogate = wake_surrogate if wake_surrogate is not None else self.DEFAULT_WAKE_SURROGATE

    # TODO: This should really be separate functions, if we're interested in separate 'impacts'
    def impact_control_variables(self, control_setpoints: ControlSetpoints, xy_range: tuple[tuple[float, float], tuple[float, float]] = None, N_grid: int = 1000) -> NPArray[float]:
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
        if xy_range is None:
            (X, Y), Z = np.meshgrid(np.linspace(perimeter[:, 0].min(), perimeter[:, 0].max(), N_grid), np.linspace(perimeter[:, 1].min(), perimeter[:, 1].max(), N_grid), indexing='xy'), np.zeros((N_grid, N_grid))
        else:
            (X, Y), Z = np.meshgrid(np.linspace(*xy_range[0], N_grid), np.linspace(*xy_range[1], N_grid), indexing='xy'), np.zeros((N_grid, N_grid))
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
    
    def get_flow_field(self, control_setpoints: ControlSetpoints, xy_range: tuple[tuple[float, float], tuple[float, float]], N_points: int = 1000) -> NPArray[float]:
        #: Calculate the power
        match self.wake_surrogate:
            case 'FLORIS':
                #: Construct the wind farm model from a template
                fmodel = FlorisModel('./config/floris/config_floris_farm.yaml')
                #: Set the scenario parameters
                fmodel.set(layout_x=self.scenario.wf_layout['x'], layout_y=self.scenario.wf_layout['y'])
                fmodel.set(turbulence_intensities=[self.scenario.TI], wind_directions=[self.scenario.theta], wind_speeds=[self.scenario.U_inf])
                #: Set the control variables
                fmodel.set(yaw_angles=[control_setpoints.yaw_angles])
                if control_setpoints.power_setpoints is not None:
                    # TEMP
                    #
                    print(f"Power setpoints: {control_setpoints.power_setpoints}")
                    #
                    fmodel.set_operation_model("simple-derating")
                    fmodel.set(power_setpoints=[control_setpoints.power_setpoints * 1E7])
                #: Run the model
                fmodel.run()
                #: Extract the flow field
                (X, Y), Z = np.meshgrid(np.linspace(*xy_range[0], N_points), np.linspace(*xy_range[1], N_points), indexing='xy'), np.full(N_points ** 2, 90.0)
                flow_field = fmodel.sample_flow_at_points(X.flatten(order='F'), Y.flatten(order='F'), Z)
                flow_field = flow_field.reshape(N_points, N_points, order='F')
            case 'PyWake':
                raise NotImplementedError("PyWake wake surrogate not yet implemented")
            case _:
                raise ValueError(f"Unrecognized wake surrogate model '{self.wake_surrogate}'")
        #: Return the results
        return X, Y, flow_field
    

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
        aep = np.sum(wt_power) * 31_536_000
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
    

class SpatialArray:
    """ Class which stores an array value f, based on coordinates x, y (and z), and has special slicing operations.

    # TODO: Subclass a numpy array, and implement the __getitem__ method to allow for slicing
    # FROM: https://numpy.org/doc/stable/user/basics.subclassing.html
    """
    
    def __init__(self, spatial: list[NPArray[float], ...], data: NPArray[float]) -> SpatialArray:
        if len(spatial) == 2:
            self.X, self.Y = spatial
        elif len(spatial) == 3:
            self.X, self.Y, self.Z = spatial
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __getitem__(self, key) -> NPArray:
        if isinstance(key, int) or (isinstance(key, tuple) and all(isinstance(elem, int) for elem in key)):
            return self.data[key]
        elif isinstance(key, float) or (isinstance(key, tuple) and all(isinstance(elem, float) for elem in key)):
            if len(key) == 2:
                indices = (np.argmin(np.abs(self.X - key[0]), axis=1)[0], np.argmin(np.abs(self.Y - key[1]), axis=0)[0])
            elif len(key) == 3:
                raise NotImplementedError(f"Searching in a 3D Spatial array is not implemented")
            return self.data[indices]
        elif isinstance(key, slice):
            raise NotImplementedError("Slicing not yet implemented")
        else:
            raise ValueError(f"Unrecognized slicing operation '{key}'")


# ------ FUNCTIONS ------


def optimal_downregulation(problem: OptProblem, N_iter: int = 100) -> ControlSetpoints:
    """
    Function to solve the downregulation optimization problem.
    """
    #: Match the metric
    match problem.metrics:
        case ['aep']:
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
        case ['noise']:
            #: Create the initial set of control setpoints, greedy control
            control_setpoints = ControlSetpoints(yaw_angles=np.zeros(problem.scenario.n_wt), power_setpoints=np.ones(problem.scenario.n_wt))
            #: Create a surrogate wind farm and metrics model
            wf_model, metrics = WindFarmModel(problem.scenario, wake_surrogate='FLORIS'), Metrics()
            #: Create the max-noise variable
            max_noise = np.nan
            #: Start the iteration
            while np.isnan(max_noise) or max_noise > 0.1:
                #: Compute the noise field
                _, _, (X_noise, Y_noise, wf_noise) = wf_model.impact_control_variables(control_setpoints)
                #: Retrieve the mask
                X_mask, Y_mask, noise_mask = problem.params[0]
                #: Convert to SpatialArray
                noise_mask = SpatialArray((X_mask, Y_mask), noise_mask)
                #: Loop over all values in the noise field
                for idx, _ in np.ndenumerate(wf_noise):
                    #: Retrieve the x- and y-values
                    x, y = X_noise[idx], Y_noise[idx]
                    #: Check if the point is within the mask
                    wf_noise[idx] = wf_noise[idx] if noise_mask[x, y] else np.nan
                # FIXME: Something is wrong, we have to 'flip' the matrix
                wf_noise = np.rot90(wf_noise, k=1)
                #: Compute the maximum noise
                max_noise = np.nanmax(wf_noise)
                #: Compute the new control setpoints, by halving the power
                control_setpoints = ControlSetpoints(yaw_angles=np.zeros(problem.scenario.n_wt), power_setpoints=0.5 * control_setpoints.power_setpoints)
        case _:
            raise ValueError(f"Unrecognized metric '{problem.metrics}'")


def optimal_wake_steering(problem: OptProblem, N_iter: int = 100, N_swarm: int = 100, w_vel_old: float = 0.5, w_vel_best_local: float = 0.1, w_vel_best_global: float = 0.1, verbose: int = 0) -> ControlSetpoints:
    """
    Function to solve the downregulation optimization problem.
    """
    YAW_LOWER_LIMIT, YAW_UPPER_LIMIT = -30, 30

    #: Select the method
    match problem.opt_method:
        case 'pso':
            #: Check if additional parameters are provided
            if problem.params is not None:
                if 'N_swarm' in problem.params:
                    N_swarm = problem.params['N_swarm']
                if 'N_iter' in problem.params:
                    N_iter = problem.params['N_iter']
            #: Create a surrogate wind farm and metrics model
            wf_model, metrics = WindFarmModel(problem.scenario, wake_surrogate='FLORIS'), Metrics()
            #: Initialize the swarm
            yaw_now, cost_now, idx_iter_best = np.zeros((problem.scenario.n_wt, N_swarm, N_iter)), np.zeros((N_swarm, N_iter)), np.zeros((N_swarm, N_iter), dtype=int)
            yaw_now[:, :, 0] = np.random.uniform(YAW_LOWER_LIMIT, YAW_UPPER_LIMIT, (problem.scenario.n_wt, N_swarm))
            #: Initialize the velocity
            velocity_now = np.random.uniform(-abs(YAW_UPPER_LIMIT - YAW_LOWER_LIMIT), abs(YAW_UPPER_LIMIT - YAW_LOWER_LIMIT), (problem.scenario.n_wt, N_swarm))
            #: Loop over all the iterations
            for iter in tqdm(range(N_iter), desc="Iterations", colour='cyan', disable=verbose == 0):
                #: Loop over all the particles
                for particle in tqdm(range(N_swarm), desc="Particles", colour='magenta', leave=False, disable=verbose == 0):
                    #: Compute the power production
                    wt_power, _, _ = wf_model.impact_control_variables(ControlSetpoints(yaw_angles=yaw_now[:, particle, iter], power_setpoints=np.ones(problem.scenario.n_wt)))
                    #: Map the power production to metrics
                    # FIXME: This is a in=place measure, results are garbage
                    cost_now[particle, iter] = metrics.compute_aep(wt_power) * 1E-14
                    #: Check if this is the particles best score since now
                    if cost_now[particle, iter] > cost_now[particle, idx_iter_best[particle, max([0, iter - 1])]]:
                        idx_iter_best[particle, iter] = iter
                    else:
                        idx_iter_best[particle, iter] = idx_iter_best[particle, iter - 1]
                #: Compute the global best
                best_costs = np.array([cost_now[particle, idx_iter_best[particle, iter]] for particle in range(N_swarm)])
                particle_best = np.argmax(best_costs)
                global_best = yaw_now[:, particle_best, idx_iter_best[particle_best, iter]]
                #: Loop over all the particles
                for particle in range(N_swarm):
                    #: Select random scaling
                    rand_local, rand_global = np.random.uniform(0, 1), np.random.uniform(0, 1)
                    #: Update the velocity
                    velocity_now[:, particle] = w_vel_old * velocity_now[:, particle] + rand_local * w_vel_best_local * (yaw_now[:, particle, idx_iter_best[particle, iter]] - yaw_now[:, particle, iter]) + rand_global * w_vel_best_global * (global_best - yaw_now[:, particle, iter])
                    #: Update the yaw
                    if iter < N_iter - 1:
                        yaw_now[:, particle, iter + 1] = np.clip(yaw_now[:, particle, iter] + velocity_now[:, particle], YAW_LOWER_LIMIT, YAW_UPPER_LIMIT)
            #: Create the operating setpoints
            control_setpoints = ControlSetpoints(yaw_angles=global_best, power_setpoints=np.ones(problem.scenario.n_wt))
        case 'geometric':
            #: Create a surrogate wind farm and metrics model
            # TODO: Move this to the class WindFarmModel
            fmodel = FlorisModel('./config/floris/config_floris_farm.yaml')
            fmodel.set(layout_x=problem.scenario.wf_layout['x'], layout_y=problem.scenario.wf_layout['y'])
            fmodel.set(turbulence_intensities=[problem.scenario.TI], wind_directions=[problem.scenario.theta], wind_speeds=[problem.scenario.U_inf])
            #: Compute the optimal yaw angles
            # FROM: https://nrel.github.io/floris/examples/examples_control_optimization/006_compare_yaw_optimizers.html  # nopep8
            yaw_opt_geo = YawOptimizationGeometric(fmodel)
            df_opt_geo = yaw_opt_geo.optimize()
            yaw_angles_opt_geo = np.vstack(df_opt_geo.yaw_angles_opt).flatten()
            #: Create the operating setpoints
            control_setpoints = ControlSetpoints(yaw_angles=yaw_angles_opt_geo, power_setpoints=np.ones(problem.scenario.n_wt))
        case _:
            raise ValueError(f"Unrecognized optimization method '{problem.opt_method}'")
    # TEMP: plot iterations of PSO
    #
    if problem.opt_method == 'pso':
        cost_best = np.array([[cost_now[particle, idx_iter_best[particle, iter]] for particle in range(N_swarm)] for iter in range(N_iter)]).T
        import matplotlib.pyplot as plt
        plt.plot(np.max(cost_best, axis=0))
        plt.show()
    #
    #: Return the results
    return control_setpoints


def optimal_greedy_layout(problem: OptProblem, N_iter: int = 100) -> tuple[ControlSetpoints, Scenario]:
    """
    Function to solve the greedy layout optimization problem.
    """
    #: Select which method to use
    match problem.opt_method:
        case 'scipy':
            # NOTE: This assumes we use FLORIS for the wind farm surrogate model
            #: Create a FLORIS model
            fmodel = FlorisModel('./config/floris/config_floris_farm.yaml')
            #: Set the minimum separation distance
            min_dist = 2 * problem.scenario.wt_model.rotor_diameter
            #: Set the number of wind turbines
            fmodel.set(layout_x=np.random.uniform(0, 500, problem.scenario.n_wt), layout_y=np.random.uniform(0, 500, problem.scenario.n_wt))
            opt_problem = LayoutOptimizationScipy(fmodel, [vertex for vertex in problem.scenario.perimeter], min_dist=min_dist)
            #: Retrieve the optimal locations
            optimal_layout = opt_problem.optimize()
            #: Set the optimal layout for the scenario
            problem.scenario.wf_layout = pd.DataFrame([list(x) for x in zip(*optimal_layout)], columns=['x', 'y'])
            #: Set the control variables
            control_setpoints = ControlSetpoints(yaw_angles=np.zeros(problem.scenario.n_wt), power_setpoints=np.random.uniform(0, 1, problem.scenario.n_wt))
        case _:
            raise ValueError(f"Unrecognized optimization method '{problem.opt_method}'")
    #: Return the results
    return control_setpoints, problem.scenario


def optimal_power_lines(problem: OptProblem, groups: list) -> list:
    """
    This computes the optimal power line placement based on the groups.

    Parameters
    ----------
    problem : OptProblem
        all the parameters from the optimization problem, which includes the scenario
    groups : list
        a nested list of the format [group_1, group_2, ...], where each group is a list of wind turbine names, i.e., group_i = ['001', '002', ...]

    Returns
    -------
    vertices_power_lines : list
        a list of length len(groups) containing the optimal power line vertices for each group. Within each group, there is a main power line [the first element], and then a list with additional power lines [the rest of the elements]
    """
    #: Call a dummy function
    vertices_power_lines = placeholder_power_lines_llstq(problem.scenario, groups)
    #: Return the results
    return vertices_power_lines


def ti_from_ws(TI: float) -> float:
    pass


# ------ PLACEHOLDERS ------


def placeholder_oscilation_decay_2d(X: NPArray[float], Y: NPArray[float], wt_loc: NPArray[float], U_inf: float, theta: float, f_decay: float = 500) -> NPArray[float]:
    """
    Function to compute the 2D draft oscillation decay.
    """
    # FROM: GitHub Copilot GPT 4o
    #: Compute the distance
    R = np.sqrt((X - wt_loc[0]) ** 2 + (Y - wt_loc[1]) ** 2)
    #: Compute the angle
    alpha = np.arctan2(Y - wt_loc[1], X - wt_loc[0])
    #: Compute the decay
    decay = np.exp(-R / f_decay)
    #: Compute the oscillation
    oscillation = np.sin(2 * np.pi * R / f_decay)
    #: Compute the draft
    draft = 0.80 * U_inf * decay * oscillation
    #: Return the result
    return draft


def placeholder_power_lines_llstq(scenario: Scenario, groups: list) -> list:
    """
    This computes the optimal power line placement based on the groups.
    """

    def point_on_line(a, b, p):
        # FROM: https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y  # nopep8
        ap = p - a
        ab = b - a
        result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
        return result

    #: Initialize an empty list 
    vertices_power_lines = []
    #: Loop over all the groups
    for group in groups:
        #: Get the indices of these wind farms
        indices_wt = [i for i, wt_name in enumerate(scenario.wt_names) if wt_name in group]
        #: Extract the wind turbine locations
        wt_locations = scenario.wf_layout.iloc[indices_wt].to_numpy()
        n_wt_group = wt_locations.shape[0]
        #: Perform LLSTQ
        coeffs = np.linalg.lstsq(np.column_stack([np.ones(n_wt_group), wt_locations[:, 0]]), wt_locations[:, 1])[0]
        #: Compute the endpoints
        end_left, end_right = np.array([0, coeffs[0]]), np.array([500, 500 * coeffs[1] + coeffs[0]])
        #: Project all the points to this line
        wt_group_intersect = np.zeros((n_wt_group, 2))
        for idx in range(n_wt_group):
            wt_group_intersect[idx, :] = point_on_line(end_left, end_right, wt_locations[idx])
        #: Select the new right power line
        end_right = wt_group_intersect[np.argmax(wt_group_intersect[:, 0]), :]
        #: Construct the array
        vertices_power_lines += [[np.row_stack([end_left, end_right]), [np.row_stack([wt_group_intersect[idx, :], wt_locations[idx, :]]) for idx in range(n_wt_group)]]]
    #: Return the results
    return vertices_power_lines


