"""This is a object-oriented implementations with a lot of wrappers, so we can have dummy functions.

"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import warnings

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from floris import FlorisModel
import floris.layout_visualization as layoutviz
from floris.flow_visualization import visualize_cut_plane
from floris.optimization.yaw_optimization.yaw_optimizer_geometric import YawOptimizationGeometric
from floris.optimization.yaw_optimization.yaw_optimizer_scipy import YawOptimizationScipy

# ====== DECORATORS AND WRAPPERS ======


class partialabstractproperty(property):
    """A decorator that allows for partial abstract properties. For instance, it allows the getter method of an inheriting class to be an abstractmethod, whilst the setter method for that same property is set by the abstract class itself.

    # FROM: https://stackoverflow.com/questions/19715507/inheriting-setter-overwriting-getter-in-python-abstract-class
    
    """
    
    def getter(self, func):
        if getattr(func, '__isabstractmethod__', False) or getattr(self.fset, '__isabstractmethod__', False):
            p = partialabstractproperty(func, self.fset)
        else:
            p = property(func, self.fset)
        return p

    def setter(self, func):
        if getattr(self.fget, '__isabstractmethod__', False) or getattr(func, '__isabstractmethod__', False):
            p = partialabstractproperty(self.fget, func)
        else:
            p = property(self.fset, func)
        return p
    

class ClassPropertyDescriptor(object):
    """Allows the construction of class properties.

    # FIXME: Move to a utilities class? This kind of impairs readability...
    # FIXME: I have the feeling a lot of this can be removed, and is redundant

    # FROM: https://stackoverflow.com/questions/5189699/how-to-make-a-class-property

    """

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("Can't set attribute.")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self
    

def classproperty(func):
    """for the classproperty decorator

    # FROM: https://stackoverflow.com/questions/5189699/how-to-make-a-class-property

    """
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


class NoSetMeta(type):
    """A metaclass that prevents the setting of certain attributes.
    
    # FROM: Microsoft Copilot GPT-4.1
    
    """
    # Map attribute names to error messages
    _readonly_fields = {}

    def __setattr__(cls, name, value):
        if name in cls._readonly_fields:
            raise AttributeError(cls._readonly_fields[name])
        super().__setattr__(name, value)


class CombinedMeta(NoSetMeta, ABC):
    """A new class that combines the functionality of NoSetMeta and ABC.
    
    # FROM: Microsoft Copilot GPT-4.1
    
    """
    pass


class ExtendedABC(metaclass=CombinedMeta):
    pass


# ====== ABSTRACT CLASSES ======


class WindTurbineModel(ABC):
    """Upper-level class which defines a wind turbine model.

    # FIXME: Actually, this should also be just classbased methods and properties... Just like Metrics
    
    """

    @abstractmethod
    def __init__(self):
        pass

    @partialabstractproperty
    @abstractmethod
    def model_name(self) -> str:
        pass

    @model_name.setter
    def model_name(self, value: str):
        raise ValueError("The model name of a wind turbine is fixed and cannot be changed")
    
    @partialabstractproperty
    @abstractmethod
    def rotor_diameter(self) -> str:
        pass

    @rotor_diameter.setter
    def rotor_diameter(self, value: float):
        raise ValueError("The rotor diameter of a wind turbine is fixed and cannot be changed")
    
    @partialabstractproperty
    @abstractmethod
    def hub_height(self) -> str:
        pass

    @hub_height.setter
    def hub_height(self, value: float):
        raise ValueError("The hub height of a wind turbine is fixed and cannot be changed")

    def __str__(self) -> str:
        return f"WindTurbineModel(model_name={self.model_name})"


class WindFarmModel(ABC):
    """Upper-level class which defines a wind farm model.
    
    """
    @abstractmethod
    def __init__(self, wind_turbine_model: WindTurbineModel):
        pass

    @abstractmethod
    def get_turbine_power(self, amb_cond: AmbientConditions, op_setpoints: OperationalControlSetpoints):
        """Get the power output of the turbines in the wind farm.

        Returns
        -------
        power : np.ndarray[floats]
            An array of size n_wt of electrical power output for each turbine, in W.

        """
        raise NotImplementedError("This function should be overwritten by the inheritance class, but it is not implemented correctly")

    @abstractmethod
    def get_load_channels(self, amb_cond: AmbientConditions, op_setpoints: OperationalControlSetpoints):
        """Get the fatigue loads on the turbines in the wind farm.

        Returns
        -------
        loads : np.ndarray[floats]
            An array of size n_wt x 18 of fatigue loads for each turbine, in DELs.

        """
        raise NotImplementedError("This function should be overwritten by the inheritance class, but it is not implemented correctly")

    @abstractmethod
    def get_noise_levels(self, amb_cond: AmbientConditions, op_setpoints: OperationalControlSetpoints, obs_loc: npt.ArrayLike):
        """Get the noise levels for several observation locations.

        Parameters
        ----------
        amb_cond : AmbientCondition
            The current ambient conditions for the wind farm.
        op_setpoints : OperationalControlSetpoints
            The operational control setpoints for the wind farm.
        obs_loc : np.ndarray
            An array if size N x 3 containing the x, y, and z coordinates of the N observation locations.

        Returns
        -------
        noise : np.ndarray[floats]
            An array of size n_wt x FIXME of decibel levels in the third-octave band (20 Hz to 10 kHz).

        """
        raise NotImplementedError("This function should be overwritten by the inheritance class, but it is not implemented correctly")
    

class Metric(ExtendedABC):
    """Class representing a metric for the performance of operational control setpoints and ambient conditions

    # FIXME: Maybe make a distinction between instantaneous metrics and accumulated metrics (also to do with constraints)

    """
    _is_coupled = None  # NOTE: These fields need to be set in the inheriting class
    _is_numeric = None
    _units = None
    _readonly_fields = {
        'is_coupled': "Class attribute 'is_coupled' is read-only.",
        'is_numeric': "Class attribute 'is_numeric' is read-only.",
        'units': "Class attribute 'units' is read-only."
    }

    def __new__(cls, *args, **kwargs):
        # FIXME: What we could do instead... is make it such that `__new__` actually functions as `__call__` (or `eval` for that matter)? So it doesn't require instantiation, but actually it does not give use the freedom to do multiple calculations...
        raise TypeError("Cannot instantiate an object of the superclass Metric. Instead, use the (instance)methods of the subclass directly.")
    
    @classproperty
    def is_coupled(cls):
        if cls._is_coupled is None:
            raise AttributeError("The variable '_is_coupled' needs to be set in the inheriting class, and cannot be None.")
        return cls._is_coupled
    
    @classproperty
    def is_numeric(cls):
        if cls._is_numeric is None:
            raise AttributeError("The variable '_is_numeric' needs to be set in the inheriting class, and cannot be None.")
        return cls._is_numeric

    @classproperty
    def units(cls):
        if cls._units is None:
            raise AttributeError("The variable '_units' needs to be set in the inheriting class, and cannot be None.")
        return cls._units
    
    @classproperty
    def name(self) -> str:
        return self.__name__

    @name.setter
    def name(self, value: str):
        raise ValueError("The name of a metric is fixed and cannot be changed.")

    # FIXME: Maybe this should not be a class-method..., maybe it should be a staticmethod? No actually it should not be, it should be a classmethod.
    @classmethod
    @abstractmethod
    def eval(cls, a, b, c) -> float:
        """Evaluate the metric for a wind farm model, a list of ambient conditions, and a list of operational control setpoints.

        """
        raise NotImplementedError(f"This method should have been overwritten by the inheriting class '{cls.__name__}' but was not implemented.")
    
    @classmethod
    def eval_decoupled(cls, a, b, c) -> float:
        pass
    

class Constraint():
    """Class representing a constraint for the optimization problem.

    # FIXME: I think its overly restrictive to make these properties read-only

    """
    @abstractmethod
    def __init__(self, metric: Metric, limit: float, ineq: str = '<='):
        self._metric = metric
        self._ineq = ineq

    @property
    def metric(self) -> Metric:
        return self._metric
    
    @metric.setter
    def metric(self, value: str):
        raise ValueError("The metric type is fixed and cannot be changed.")
    
    @property
    def ineq(self) -> str:
        return self._ineq
    
    @ineq.setter
    def ineq(self, value: str):
        raise ValueError("The inequality type of a constraint is fixed and cannot be changed.")


class Optimizer(ExtendedABC):
    """Class representing a generic optimizer.
    
    """
    @abstractmethod
    def __init__(self, wfm: WindFarmModel, amb_cond: AmbientConditions, metric: Metric, constraints: list[Constraint] = None):
        self.wfm = wfm
        self.amb_cond = amb_cond
        self.metric = metric
        self.constraints = constraints
        #: Check arguments
        if not isinstance(wfm, WindFarmModel):
            raise ValueError("The wind farm model should be an instance of the WindFarmModel class.")
        if not issubclass(metric, Metric):
            raise ValueError("The metric should be a subclass of the Metric class.")

    @abstractmethod
    def solve(self, model: WindFarmModel, amb_cond: AmbientConditions) -> list[float, OperationalControlSetpoints]:
        """Optimize the operational control setpoints for the wind farm model given the ambient conditions.

        """
        pass


# ====== CLASSES ======
    

class OperationalControlSetpoints():
    """Class representing the n_scn operational control setpoints for the wind farm. These are for a single ambient conditions, i.e., they contain one setpoint for each independent setpoint for each wind turbine.

    Attributes
    ----------
    yaw_setpoints : np.ndarray[floats]
        An array of size n_wt of yaw setpoints for each turbine, in degrees. Whilst not enforced, they should be in the range [-30, 30]
    power_setpoints : np.ndarray[floats]
         An array of size n_wt of power setpoints for each turbine, in % of the maximum available power. Whilst not enforced, they should be in the range [0, 1]

    """
    def __init__(self, yaw_setpoints: npt.ArrayLike = None, power_setpoints: npt.ArrayLike = None):
        self.yaw_setpoints = yaw_setpoints  # NOTE: `None` means greedy control, no yaw offset
        self.power_setpoints = power_setpoints  # NOTE: `None` means greedy control, maximal power generation
        self.noise_curtailment_mode = None
        self.operational_mode = None  # NOTE: Should be in {'idling', 'power_production', 'start_up', 'shut_down'}


class AmbientConditions():
    """Class representing the n_scn ambient conditions for the wind farm. note that these might be wind time-series data, or distributional data

    Attributes
    ----------
    wind_speed : float
        The ambient wind speed, in m/s.
    wind_direction : float
        The mean wind direction, in degrees. Whilst not enforced, should be in the range [0, 360)
    turbulence_intensity : float
        The turbulence intensity, unitless (% / 100).

    # FIXME: Should this class also contain frequency information? Or duration of that ambient condition?

    """
    def __init__(self, wind_speeds: npt.ArrayLike, wind_directions: npt.ArrayLike, turbulence_intensities: npt.ArrayLike, data_type: str):
        self._amb_cond = np.column_stack((wind_speeds, wind_directions, turbulence_intensities))
        self.n_scn = wind_speeds.size
        self.data_type = data_type  # NOTE: Should be in {'time_series', 'distribution'}

    def __getitem__(self, item: int) -> AmbientConditions:
         return AmbientConditions(self.wind_speeds[item], self.wind_directions[item], self.turbulence_intensities[item], self.data_type)

    @property
    def wind_speeds(self) -> npt.ArrayLike:
        return self._amb_cond[:, 0]
    
    @property
    def wind_directions(self) -> npt.ArrayLike:
        return self._amb_cond[:, 1]
    
    @property
    def turbulence_intensities(self) -> npt.ArrayLike:
        return self._amb_cond[:, 2]

    def __str__(self) -> str:
        return f"AmbientConditions(wind_speeds={self.wind_speeds}, wind_directions={self.wind_directions}, turbulence_intensities={self.turbulence_intensities}, data_type={self.data_type})"


# ====== SUBCLASSES ======


class FLORISModel(WindFarmModel):
    """FLORIS (FLOw Redirection and Induction in Steady State) model for wind farm optimization.

    """
    def __init__(self, turbine_positions: npt.ArrayLike, turbine_model: WindTurbineModel | list[WindTurbineModel], config: dict | Path = Path('./config/floris/config_floris_farm.yaml')):
        self.layout: npt.ArrayLike = np.asarray(turbine_positions)
        self.n_wt: int = turbine_positions.shape[0]
        self.turbine_model: list[WindTurbineModel] = turbine_model
        if isinstance(turbine_model, WindTurbineModel):
            self.is_homogeneous: bool = True  # NOTE: If every wind turbine is of the same type
        else:
            self.is_homogeneous: bool = False
        #: Create FLORIS model object
        self.fmodel = FlorisModel(config)
        # FIXME: Should we use simple numpy arrays, or panda data frames, or xarray?
        self.fmodel.set(layout_x=self.layout[:, 0], layout_y=self.layout[:, 1])

    def get_turbine_power(self, amb_cond: AmbientConditions, op_setpoints: OperationalControlSetpoints):
        #: Set the ambient conditions
        self.fmodel.set(wind_directions=[wd for wd in amb_cond.wind_directions], wind_speeds=[ws for ws in amb_cond.wind_speeds], turbulence_intensities=[ti for ti in amb_cond.turbulence_intensities])
        #: Set the control variables
        if op_setpoints.yaw_setpoints is not None:
            self.fmodel.set(yaw_angles=[ys for ys in op_setpoints.yaw_setpoints])
        if op_setpoints.power_setpoints is not None:
            self.fmodel.set_operation_model('simple-derating')
            # FIXME: These are now based on the rated power, but should they be based on the available power? Actually, that's really hard to compute... because we don't know the available power beforehand (i.e., before running the simulation)
            if self.is_homogeneous:
                self.fmodel.set(power_setpoints=[op_setpoints.power_setpoints * self.turbine_model.rated_power])
            else:
                self.fmodel.set(power_setpoints=[op_setpoints.power_setpoints[wt_idx] * self.turbine_model[wt_idx].rated_power for wt_idx in range(self.n_wt)])
        #: Run the model
        self.fmodel.run()
        #: Extract the powers
        wt_power = self.fmodel.get_turbine_powers()
        #: Return the result
        return wt_power
    
    def get_load_channels(self, amb_cond, op_setpoints):
        raise NotImplementedError("This function is not yet implemented as the load surrogates are not available.")
    
    def get_noise_levels(self, amb_cond, op_setpoints, obs_loc):
        raise NotImplementedError("This function is not yet implemented as the noise propagation model is not available.")
    
    def plot(self, wd: float, ws: float, ti: float, yaw_setpoints: npt.ArrayLike = None, power_setpoints: npt.ArrayLike = None, x_res: int = 100, y_res: int = 100, label: str = None, ax: Axes = None) -> Axes:
        self.fmodel.set(wind_directions=[wd], wind_speeds=[ws], turbulence_intensities=[ti])
        if yaw_setpoints is not None:
            self.fmodel.set(yaw_angles=[yaw_setpoints])
        if power_setpoints is not None:
            raise NotImplementedError("Power setpoints are not yet implemented in the plotting function.")
        if self.is_homogeneous:
            height = self.turbine_model.hub_height
        else:
            warnings.warn("The hub height is not well-defined for heterogeneous wind farms. Using the hub height of the first turbine.")
            height = self.turbine_model[0].hub_height
        horizontal_plane = self.fmodel.calculate_horizontal_plane(x_resolution=x_res, y_resolution=y_res, height=height)
        if ax is None:
            fig, ax = plt.subplots()
        visualize_cut_plane(horizontal_plane, ax=ax, label_contours=False, title=label)
        #: Plot the turbine rotors
        layoutviz.plot_turbine_rotors(self.fmodel, ax=ax)
        layoutviz.plot_turbine_labels(self.fmodel, ax=ax, turbine_names=None)
        #: Return the results
        return ax


class PyWakeModel(WindFarmModel):
    pass


class NREL5MW(WindTurbineModel):
    """Turbine which implements the NREL 5-MW reference turbine model.
    
    """
    def __init__(self):
        self._model_name = 'nrel5mw'
        self._rotor_diameter = 126.0
        self._hub_height = 90.0

    # FIXME: This is kind of ugly... but possibly the best we can do
    @WindTurbineModel.model_name.getter
    def model_name(self) -> str:
        return self._model_name
    
    @WindTurbineModel.rotor_diameter.getter
    def rotor_diameter(self) -> float:
        return self._rotor_diameter

    @WindTurbineModel.hub_height.getter
    def hub_height(self) -> float:
        return self._hub_height


class IEA22MWRWT(WindTurbineModel):
    """Turbine which implements the IEA 22-MW reference turbine model.

    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError("This function should be overwritten by the inheritance class, but it is not implemented correctly")
    

class DTU10MWRWT(WindTurbineModel):
    """Turbine which implements the DTU 10-MW reference turbine model.

    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError("This function should be overwritten by the inheritance class, but it is not implemented correctly")
    

class AnnualEnergyProduction(Metric):
    """Calculates the annual energy production
    
    """
    _is_coupled = False
    _is_numeric = True
    _units = "Wh/year"

    # FIXME: Is this actually automatically copied (from the superclass Metric) as a @staticmethod? How does that work?
    def eval(wfm: WindFarmModel, amb_conds: list[AmbientConditions], op_conds: list[OperationalControlSetpoints]) -> float:
        #: Check whether the data is distribution of time-series based
        match amb_conds.data_type:
            case 'time_series':
                #: Retrieve the powers from the wind farm
                powers = wfm.get_turbine_power(amb_conds, op_conds)  # NOTE: Should be of size (n_scn n_wt)
                #: Aggregate all the power and multiple by duration
                # NOTE: This implicitly assumes the time series cover an entire year
                aep = np.sum(31_536_000 / amb_conds.n_scn * np.sum(powers, axis=1), axis=0)  # NOTE: 31 536 000 is the number of seconds in a year
                return aep
            case 'distribution':
                return 0


class LifetimeEnergyProduction(Metric):
    pass


class PsychoacousticAnnoyance(Metric):
    pass


class MarginalDisplacementFactor(Metric):
    pass


class AnnualWildlifeImpact(Metric):
    pass


class YawOptimizerGeometric(Optimizer):
    """Solve for the optimal yaw angle w.r.t AEP maximization (i.e., power maximization) using the FLORIS wind farm model.
    
    """
    def __init__(self, wfm: WindFarmModel, amb_cond: AmbientConditions, metric: Metric = AnnualEnergyProduction, constraints: list[callable] = None):
        super().__init__(wfm, amb_cond, metric, constraints)
        #: Check the arguments
        if wfm.__class__ is not FLORISModel:
            raise ValueError("Only the FLORIS wind farm model is supported for geometric wake steering.")
        if metric is not AnnualEnergyProduction:
            raise ValueError("Only the AnnualEnergyProduction metric is supported for geometric wake steering.")
        if constraints is not None:
            raise ValueError("Constraints are not supported for geometric wake steering.")
        #: Set the ambient conditions
        # FIXME: Should we do that here or in the solve section?
        self.wfm.fmodel.set(wind_directions=[wd for wd in self.amb_cond.wind_directions], wind_speeds=[ws for ws in self.amb_cond.wind_speeds], turbulence_intensities=[ti for ti in self.amb_cond.turbulence_intensities])
    
    def solve(self, solver: str = 'geometric_yaw', verbose: int = 0) -> list[float, OperationalControlSetpoints]:
        if verbose > 0:
            print(f"Starting optimization with {solver}...")
        #: Compute the optimal yaw angles
        # FROM: https://nrel.github.io/floris/examples/examples_control_optimization/006_compare_yaw_optimizers.html  # nopep8
        yaw_opt = YawOptimizationGeometric(self.wfm.fmodel)
        df_opt = yaw_opt.optimize()
        yaw_angles_opt = np.squeeze(np.vstack(df_opt.yaw_angles_opt))
        opt_cont = OperationalControlSetpoints(yaw_setpoints=yaw_angles_opt)
        opt_cost = AnnualEnergyProduction.eval(self.wfm, self.amb_cond, opt_cont)
        return opt_cost, opt_cont
    

class YawOptimizerSciPy(Optimizer):
    """Solve for the optimal yaw angle w.r.t AEP maximization (i.e., power maximization) using the FLORIS wind farm model.
    
    """
    def __init__(self, wfm: WindFarmModel, amb_cond: AmbientConditions, metric: Metric = AnnualEnergyProduction, constraints: list[callable] = None):
        super().__init__(wfm, amb_cond, metric, constraints)
        #: Check the arguments
        if wfm.__class__ is not FLORISModel:
            raise ValueError("Only the FLORIS wind farm model is supported for geometric wake steering.")
        if metric is not AnnualEnergyProduction:
            raise ValueError("Only the AnnualEnergyProduction metric is supported for geometric wake steering.")
        if constraints is not None:
            raise ValueError("Constraints are not supported for geometric wake steering.")
        #: Set the ambient conditions
        # FIXME: Should we do that here or in the solve section?
        self.wfm.fmodel.set(wind_directions=[wd for wd in self.amb_cond.wind_directions], wind_speeds=[ws for ws in self.amb_cond.wind_speeds], turbulence_intensities=[ti for ti in self.amb_cond.turbulence_intensities])
    
    def solve(self, solver: str = 'geometric_yaw', verbose: int = 0) -> list[float, OperationalControlSetpoints]:
        if verbose > 0:
            print(f"Starting optimization with {solver}...")
        #: Compute the optimal yaw angles
        # FROM: https://nrel.github.io/floris/examples/examples_control_optimization/006_compare_yaw_optimizers.html  # nopep8
        yaw_opt = YawOptimizationScipy(self.wfm.fmodel)
        df_opt = yaw_opt.optimize()
        yaw_angles_opt = np.squeeze(np.vstack(df_opt.yaw_angles_opt))
        opt_cont = OperationalControlSetpoints(yaw_setpoints=yaw_angles_opt)
        opt_cost = AnnualEnergyProduction.eval(self.wfm, self.amb_cond, opt_cont)
        return opt_cost, opt_cont


class GenericSerialRefinementOptimizer(Optimizer):
    pass


class SerialRefinementWakeSteeringSingleAmbient(Optimizer):
    """This is serial refinement using an arbitrary metric, but only for a single ambient condition and wake steering.

    """
    def __init__(self, wfm: WindFarmModel, amb_cond: AmbientConditions, metric: Metric = AnnualEnergyProduction, constraints: list[callable] = None):
        super().__init__(wfm, amb_cond, metric, constraints)
        #: Check the arguments
        if amb_cond.n_scn != 1:
            raise ValueError("Only a single scenario is supported for serial refinement.")
        
    def solve(self, verbose: int = 0) -> list[float, OperationalControlSetpoints]:

        def f_obj(yaw: npt.ArrayLike) -> float:
            return self.metric.eval(self.wfm, self.amb_cond, OperationalControlSetpoints(yaw_setpoints=np.atleast_2d(yaw)))

        # FIXME: This is temporary
        import sys
        serial_refinement_wp4_dir = '/Users/bartwolleswink/surfdrive - Bart Wolleswinkel@surfdrive.surf.nl/TWAIN/Task 4.1 | Multi-Objective Optimisation Methodologies/Serial Refinement/WP4'
        print(serial_refinement_wp4_dir)
        sys.path.append(serial_refinement_wp4_dir)
        from WP4_yaw_optimizer.yaw_optimizer_WP4 import YawOptimizer_SR
        optimizer = YawOptimizer_SR(self.wfm.layout[:, 0], self.wfm.layout[:, 1], self.amb_cond.wind_directions[0], f_obj)
        optimizer.optimize()
        opt_cont = OperationalControlSetpoints(yaw_setpoints=optimizer.yaw_opt)
        opt_cost = 0
        return opt_cost, opt_cont


# ====== MAIN ======
            

def main():

    # Set a plotting index
    plot_idx = 2

    # Create a random number of wind conditions
    wind_speeds = np.array([5, 5, 6, 12, 11])
    wind_directions = np.array([0, 10, 270, 5, 90])  # NOTE: In degrees
    turbulence_intensities = np.array([0.1, 0.1, 0.2, 0.3, 0.4])  # NOTE: In %

    # Create ambient conditions
    amb_conds = AmbientConditions(wind_speeds, wind_directions, turbulence_intensities, data_type='time_series')
    print(amb_conds)

    # Create a turbine model
    turbine_model = NREL5MW()
    print(turbine_model.model_name)
    # turbine_model.model_name = 'nrel5mw'  # NOTE: This should raise an error
    print(turbine_model)

    # Create a wind farm object
    layout = np.array([[608, 500], [1500, 500], [2392, 500]])
    wfm = FLORISModel(turbine_positions=layout, turbine_model=turbine_model)

    # Plot the original layout
    wfm.plot(wd=amb_conds.wind_directions[plot_idx], ws=amb_conds.wind_speeds[plot_idx], ti=amb_conds.turbulence_intensities[plot_idx], label="Greedy control")

    # Create operational conditions
    op_conds = OperationalControlSetpoints()

    # Calculate AEP
    # NOTE: Here the WFM is actually modified in-place, and such the ambient conditions are maintained!
    # aep_obj = AnnualEnergyProduction()  # NOTE: This should throw an error
    aep = AnnualEnergyProduction.eval(wfm, amb_conds, op_conds)
    # Test if @staticmethod is actually inherited correctly
    import inspect
    # FIXME: This inheritance does not seem to work correctly...
    print(f"Is the method 'eval' a static method? {isinstance(inspect.getattr_static(AnnualEnergyProduction, 'eval'), staticmethod)}")
    print(f"AEP: {aep} Wh/year")

    # Create a constraint
    cons_1 = Constraint(AnnualEnergyProduction, limit=30.0, ineq='<=')
    print(cons_1.ineq)
    print(cons_1.metric.name)
    
    opt_prob_gws = YawOptimizerGeometric(wfm, amb_conds, AnnualEnergyProduction)
    opt_cost_gws, opt_control_variables_gws = opt_prob_gws.solve(verbose=1)

    opt_prob_spws = YawOptimizerSciPy(wfm, amb_conds, AnnualEnergyProduction)
    opt_cost_spws, opt_control_variables_spws = opt_prob_spws.solve(verbose=1)

    opt_prob_sr = SerialRefinementWakeSteeringSingleAmbient(wfm, amb_conds[plot_idx], AnnualEnergyProduction, constraints=[cons_1])
    opt_cost_sr, opt_control_variables_sr = opt_prob_sr.solve(verbose=1)

    # Print the results
    print(f"Optimal Cost: {opt_cost_gws / 3600 * 1E-9:.2f} (in GWh)")
    print(f"Optimal Control Variables (geometric wake steering):\n{opt_control_variables_gws.yaw_setpoints}")
    print(f"Optimal Control Variables (SciPy wake steering):\n{opt_control_variables_spws.yaw_setpoints}")

    # Plot the results
    wfm.plot(wd=amb_conds.wind_directions[plot_idx], ws=amb_conds.wind_speeds[plot_idx], ti=amb_conds.turbulence_intensities[plot_idx], yaw_setpoints=opt_control_variables_gws.yaw_setpoints[plot_idx, :], label="Optimal Yaw Angles Using Geometric Wake Steering")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
