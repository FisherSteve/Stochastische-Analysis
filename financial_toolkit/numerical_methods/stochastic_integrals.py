# --- File: financial_toolkit/numerical_methods/__init__.py ---
# This file can be empty. It indicates that 'numerical_methods' is a Python package.


# --- File: financial_toolkit/numerical_methods/stochastic_integrals.py ---
import numpy as np
from scipy.interpolate import interp1d # For midpoint_time variant if interpolating process X

class StochasticIntegralApproximator:
    """
    Approximates stochastic integrals of the form integral_0^t f(X_u) dX_u
    or integral_0^t f(u, X_u) dX_u using different rules.
    """
    def __init__(self, integrand_f, process_X_path, time_axis_t):
        """
        Initializes the approximator.

        Args:
            integrand_f (callable): The function f. 
                                    If f(X_u), it should take one argument (value of X_u).
                                    If f(u, X_u), it should take two arguments (time u, value of X_u).
                                    The approximator will try to call with two args first.
            process_X_path (np.ndarray): 1D array representing a single path of the integrator process X_t.
            time_axis_t (np.ndarray): 1D array of time points corresponding to process_X_path.
        """
        if not callable(integrand_f):
            raise TypeError("integrand_f must be a callable function.")
        if not isinstance(process_X_path, np.ndarray) or process_X_path.ndim != 1:
            raise TypeError("process_X_path must be a 1D NumPy array.")
        if not isinstance(time_axis_t, np.ndarray) or time_axis_t.ndim != 1:
            raise TypeError("time_axis_t must be a 1D NumPy array.")
        if len(process_X_path) != len(time_axis_t):
            raise ValueError("process_X_path and time_axis_t must have the same length.")
        if len(process_X_path) < 2:
            raise ValueError("Process path must have at least 2 points to calculate increments.")

        self.integrand_f = integrand_f
        self.X_path = process_X_path
        self.t_axis = time_axis_t
        self.N = len(time_axis_t)
        self.dX = np.diff(self.X_path) # Increments dX_i = X_{t_{i+1}} - X_{t_i}

    def _evaluate_f(self, t_eval, X_eval):
        """Helper to evaluate f, trying f(t,X) then f(X)."""
        try:
            return self.integrand_f(t_eval, X_eval)
        except TypeError: # integrand_f might only take X
            try:
                return self.integrand_f(X_eval)
            except TypeError:
                raise TypeError("integrand_f signature not compatible. Expected f(X) or f(t,X).")


    def approximate_integral_process(self, variant='ito_left'):
        """
        Calculates the approximated stochastic integral process.
        Integral_process_t = sum_{k=0}^{i-1} (approximated_integrand_term_k * dX_k), for t = t_i.

        Args:
            variant (str): The approximation rule. Options:
                           'ito_left': Ito integral (left endpoint rule for f).
                           'stratonovich': Stratonovich integral (midpoint of f values).
                           'right_endpoint': Right endpoint rule for f.
                           'midpoint_time': f evaluated at X(t_midpoint), where t_midpoint is midpoint of time interval.
                                           Requires interpolation of X_path.

        Returns:
            tuple: (time_axis, integral_process_path)
                   integral_process_path is a 1D NumPy array. Integral(0)=0.
        """
        f_terms = np.zeros(self.N - 1) # N-1 terms for N-1 increments dX

        if variant == 'ito_left':
            # f(t_i, X_{t_i}) or f(X_{t_i})
            f_eval_points = self._evaluate_f(self.t_axis[:-1], self.X_path[:-1])
            f_terms = f_eval_points
        
        elif variant == 'right_endpoint':
            # f(t_{i+1}, X_{t_{i+1}}) or f(X_{t_{i+1}})
            f_eval_points = self._evaluate_f(self.t_axis[1:], self.X_path[1:])
            f_terms = f_eval_points

        elif variant == 'midpoint_time':
            # f( (t_i+t_{i+1})/2, X((t_i+t_{i+1})/2) )
            t_mid_points = (self.t_axis[:-1] + self.t_axis[1:]) / 2.0
            # Interpolate X_path at these mid time points
            X_interpolator = interp1d(self.t_axis, self.X_path, kind='linear', assume_sorted=True)
            X_at_t_mid = X_interpolator(t_mid_points)
            f_eval_points = self._evaluate_f(t_mid_points, X_at_t_mid)
            f_terms = f_eval_points
            
        elif variant == 'stratonovich':
            # (f(t_i,X_{t_i}) + f(t_{i+1},X_{t_{i+1}})) / 2
            f_left = self._evaluate_f(self.t_axis[:-1], self.X_path[:-1])
            f_right = self._evaluate_f(self.t_axis[1:], self.X_path[1:])
            f_terms = (f_left + f_right) / 2.0
            
        else:
            raise ValueError(f"Unknown variant '{variant}'. Choose from 'ito_left', 'right_endpoint', 'midpoint_time', 'stratonovich'.")

        # Element-wise product for sum: sum f_terms_i * dX_i
        integral_increments = f_terms * self.dX
        
        # Cumulatively sum the increments
        cumulative_integral = np.cumsum(integral_increments)
        
        # Prepend a zero for the integral value at t=0
        integral_process_path = np.concatenate(([0.0], cumulative_integral))
        
        return self.t_axis, integral_process_path

