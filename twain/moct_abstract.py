"""This is a object-oriented implementations with a lot of wrappers, so we can have dummy functions.

"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

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

    def get_turbine_power(self, amb_cond: AmbientConditions, op_setpoints: OperationalControlSetpoints):
        """Get the power output of the turbines in the wind farm.

        Returns
        -------
        power : np.ndarray[floats]
            An array of size n_wt of electrical power output for each turbine, in W.

        """
        raise NotImplementedError("This function should be overwritten by the inheritance class, but it is not implemented correctly")

    def get_load_channels(self, amb_cond: AmbientConditions, op_setpoints: OperationalControlSetpoints):
        """Get the fatigue loads on the turbines in the wind farm.

        Returns
        -------
        loads : np.ndarray[floats]
            An array of size n_wt x 18 of fatigue loads for each turbine, in DELs.

        """
        raise NotImplementedError("This function should be overwritten by the inheritance class, but it is not implemented correctly")

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

    """
    _is_coupled = None  # NOTE: These fields need to be set in the inheriting class
    _is_numeric = None
    _units = None
    _readonly_fields = {
        'is_coupled': "Class attribute 'is_coupled' is read-only.",
        'is_numeric': "Class attribute 'is_numeric' is read-only.",
        'units': "Class attribute 'units' is read-only."
    }

    # FIXME: Maybe this should not be a class-method...
    @classmethod
    @abstractmethod
    def eval(cls, a, b, c) -> float:
        """Evaluate the metric for a wind farm model, a list of ambient conditions, and a list of operational control setpoints.

        """
        raise NotImplementedError(f"This method should have been overwritten by the inheriting class '{cls.__name__}' but was not implemented.")
    
    @classmethod
    def eval_decoupled(cls, a, b, c) -> float:
        pass
    
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
    

class Optimizer(ABC):
    """Class representing a generic optimizer.
    
    """
    @abstractmethod
    def __init__(self, cost_function: callable, constraints: list[callable]):
        pass

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
    def __init__(self):
        self.yaw_setpoints = None
        self.power_setpoints = None
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
        self.wind_speeds = np.asarray(wind_speeds)
        self.wind_directions = np.asarray(wind_directions)
        self.turbulence_intensities = np.asarray(turbulence_intensities)
        self.n_scn = wind_speeds.size
        self.data_type = data_type  # NOTE: Should be in {'time_series', 'distribution'}

    def __str__(self) -> str:
        return f"AmbientConditions(wind_speeds={self.wind_speeds}, wind_directions={self.wind_directions}, turbulence_intensities={self.turbulence_intensities}, data_type={self.data_type})"


# ====== SUBCLASSES ======


class FLORISModel(WindFarmModel):
    """FLORIS (FLOw Redirection and Induction in Steady State) model for wind farm optimization.

    """
    def __init__(self, turbine_positions: npt.ArrayLike, turbine_model: list[WindTurbineModel]):
        self.layout: npt.ArrayLike = np.asarray(turbine_positions)
        self.n_wt: int = turbine_positions.shape[0]
        self.is_homogeneous: bool = None  # NOTE: If every wind turbine is of the same type

    def get_turbine_power(self, amb_cond: AmbientConditions, op_setpoints: OperationalControlSetpoints):
        return np.random.rand(amb_cond.n_scn, self.n_wt)  # NOTE: Should be of size (n_scn n_wt)
    

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


class GeometricYawOptimizer(Optimizer):
    def __init__(self, wfm: WindFarmModel, metric: Metric, constraints: list[callable]):
        self.wfm = wfm
        self.metric = metric
        self.constraints = constraints
        #: Check the arguments
        if wfm.__class__ is not FLORISModel:
            raise ValueError("Only the FLORIS wind farm model is supported for geometric wake steering")

    def solve(self, solver: str = 'geometric_yaw', n_iter: int = 1000, verbose: int = 0) -> list[float, OperationalControlSetpoints]:
        if verbose > 0:
            print(f"Starting optimization with {solver}...")
        return None, None

# ====== MAIN ======
            

def main():

    #: Create a random number of wind conditions
    wind_speeds = np.array([5, 5, 6, 12, 11])
    wind_directions = np.array([0, 10, 350, 5, 355])  # NOTE: In degrees
    turbulence_intensities = np.array([0.1, 0.1, 0.2, 0.3, 0.4])  # NOTE: In %

    #: Create ambient conditions
    amb_conds = AmbientConditions(wind_speeds, wind_directions, turbulence_intensities, data_type='time_series')
    print(amb_conds)

    #: Create a turbine model
    turbine_model = NREL5MW()
    print(turbine_model.model_name)
    # turbine_model.model_name = 'nrel5mw'  # NOTE: This should raise an error
    print(turbine_model)

    #: Create a wind farm object
    layout = np.array([[608, 500], [1500, 500], [2392, 500]])
    wfm = FLORISModel(turbine_positions=layout, turbine_model=turbine_model)

    #: Create operational conditions
    op_conds = OperationalControlSetpoints()

    #: Calculate AEP
    aep = AnnualEnergyProduction.eval(wfm, amb_conds, op_conds)
    print(AnnualEnergyProduction.is_coupled)
    print(AnnualEnergyProduction.is_numeric)
    print(AnnualEnergyProduction.units)
    print(f"AEP: {aep} Wh/year")

    #: Construct an optimization function
    def cost_function(wfm: WindFarmModel, amb_conds: AmbientConditions, op_conds: OperationalControlSetpoints) -> float:
        return -AnnualEnergyProduction.eval(wfm, amb_conds, op_conds)
    
    def constraint1(wfm: WindFarmModel, amb_conds: AmbientConditions, op_conds: OperationalControlSetpoints) -> bool:
        return 0.0 <= 30
    
    def constraint2(wfm: WindFarmModel, amb_conds: AmbientConditions, op_conds: OperationalControlSetpoints) -> bool:
        return 0.0 <= 1.0
    
    opt_prob = GeometricYawOptimizer(wfm, AnnualEnergyProduction, constraints=[constraint1, constraint2])
    opt_cost, opt_control_variables = opt_prob.solve(verbose=1)

    #: Print the results
    print(f"Optimal Cost: {opt_cost}")
    print(f"Optimal Control Variables: {opt_control_variables}")


if __name__ == "__main__":
    main()
