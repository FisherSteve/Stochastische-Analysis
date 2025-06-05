# --- File: financial_toolkit/stochastic_processes/__init__.py ---
# This file can be empty. It indicates that 'stochastic_processes' is a Python package.

# --- File: financial_toolkit/stochastic_processes/ito_processes.py ---
import numpy as np
from scipy.integrate import cumulative_trapezoid # For path-dependent theoretical QV

class ArithmeticBrownianMotion:
    """
    Implements Arithmetic Brownian Motion (BM with drift and volatility).
    Process: dX_t = mu*dt + sigma*dW_t
    """
    def __init__(self, x0=0.0, mu=0.05, sigma=0.2, T=1.0, N=1001, paths=1, seed=None):
        if N < 1: raise ValueError("N must be at least 1.")
        if N == 1 and T != 0.0: raise ValueError("If N=1, T must be 0.0.")

        self.x0_param = x0
        self.mu = mu
        self.sigma = sigma
        self.T = float(T)
        self.N = int(N)
        self.paths = int(paths)
        self.seed = seed

        if self.N == 1:
            self.t_values = np.array([0.0])
            self.dt = 0.0
        else:
            self.t_values = np.linspace(0, self.T, self.N)
            self.dt = self.T / (self.N - 1)
        
        self.X = np.zeros((self.paths, self.N))

        if self.seed is not None:
            np.random.seed(self.seed)

    def generate_paths(self):
        current_x0_values = np.zeros(self.paths)
        if callable(self.x0_param):
            for i in range(self.paths): current_x0_values[i] = self.x0_param()
        else:
            current_x0_values[:] = self.x0_param

        if self.N <= 1:
            for i in range(self.paths): self.X[i, 0] = current_x0_values[i]
            return self.X

        sqdt = np.sqrt(self.dt) if self.dt > 0 else 0.0
        for i in range(self.paths):
            self.X[i, 0] = current_x0_values[i]
            if self.dt > 0: 
                for j in range(self.N - 1):
                    Z = np.random.standard_normal() 
                    self.X[i, j+1] = self.X[i, j] + self.mu * self.dt + self.sigma * sqdt * Z
        return self.X

    def get_realized_quadratic_variation_process(self):
        if self.N <= 1: return np.zeros_like(self.X)
        increments = np.diff(self.X, axis=1)
        squared_increments = increments**2
        cumulative_qv = np.cumsum(squared_increments, axis=1)
        return np.concatenate((np.zeros((self.paths, 1)), cumulative_qv), axis=1)

    def get_theoretical_quadratic_variation_process(self):
        """Theoretical QV_t for dX_t = mu*dt + sigma*dW_t is sigma^2 * t."""
        if self.N <= 1: return np.array([0.0])
        return self.sigma**2 * self.t_values.copy()

class GeometricBrownianMotion:
    """
    Implements Geometric Brownian Motion (GBM).
    SDE: dS_t = mu*S_t*dt + sigma*S_t*dW_t
    """
    def __init__(self, S0=100.0, mu=0.05, sigma=0.2, T=1.0, N=1001, paths=1, seed=None):
        if N < 1: raise ValueError("N must be at least 1.")
        if N == 1 and T != 0.0: raise ValueError("If N=1, T must be 0.0.")

        self.S0_param = S0 
        self.mu = mu
        self.sigma = sigma
        self.T = float(T)
        self.N = int(N)
        self.paths = int(paths)
        self.seed = seed

        if self.N == 1:
            self.t_values = np.array([0.0])
            self.dt = 0.0
        else:
            self.t_values = np.linspace(0, self.T, self.N)
            self.dt = self.T / (self.N - 1)
        
        self.S = np.zeros((self.paths, self.N))

        if self.seed is not None:
            np.random.seed(self.seed)

    def generate_paths(self):
        current_S0_values = np.zeros(self.paths)
        if callable(self.S0_param):
            for i in range(self.paths): 
                s0_val = self.S0_param()
                if s0_val <= 0: raise ValueError("S0 from callable must be positive for GBM.")
                current_S0_values[i] = s0_val
        else:
            if self.S0_param <= 0: raise ValueError("Fixed S0 must be positive for GBM.")
            current_S0_values[:] = self.S0_param
        
        if self.N <= 1:
            for i in range(self.paths): self.S[i, 0] = current_S0_values[i]
            return self.S

        std_W = np.zeros((self.paths, self.N))
        sqdt = np.sqrt(self.dt) if self.dt > 0 else 0.0
        if self.dt > 0: 
            for i in range(self.paths):
                increments = sqdt * np.random.standard_normal(self.N - 1)
                std_W[i, 1:] = np.cumsum(increments)
        
        exponent_drift = (self.mu - 0.5 * self.sigma**2) * self.t_values 
        for i in range(self.paths):
            exponent_stochastic = self.sigma * std_W[i, :] 
            self.S[i, :] = current_S0_values[i] * np.exp(exponent_drift + exponent_stochastic)
        return self.S

    def get_realized_quadratic_variation_process(self):
        if self.N <= 1: return np.zeros_like(self.S)
        increments = np.diff(self.S, axis=1)
        squared_increments = increments**2
        cumulative_qv = np.cumsum(squared_increments, axis=1)
        return np.concatenate((np.zeros((self.paths, 1)), cumulative_qv), axis=1)

    def get_theoretical_quadratic_variation_logS_process(self):
        """Theoretical QV_t for log(S_t) is sigma^2 * t."""
        if self.N <= 1: return np.array([0.0])
        return self.sigma**2 * self.t_values.copy()
        
    def get_path_dependent_theoretical_qv_process(self):
        """
        Calculates theoretical QV process for S_t: integral_0^t (sigma * S_u)^2 du.
        Returns an array of QV processes, one for each path.
        """
        if self.N <= 1 or self.S.shape[1] < 2 : return np.zeros_like(self.S)
        
        integrand_values = (self.sigma * self.S)**2
        theoretical_qv_process = cumulative_trapezoid(integrand_values, self.t_values, axis=1, initial=0)
        return theoretical_qv_process

class OrnsteinUhlenbeckProcess:
    """
    Implements Ornstein-Uhlenbeck (OU) Process.
    SDE: dX_t = theta*(mu - X_t)*dt + sigma*dW_t
    """
    def __init__(self, x0=0.0, theta=1.0, mu_ou=0.0, sigma=0.1, T=1.0, N=1001, paths=1, seed=None):
        if N < 1: raise ValueError("N must be at least 1.")
        if N == 1 and T != 0.0: raise ValueError("If N=1, T must be 0.0.")
        if theta < 0: print("Warning: OU theta (mean reversion speed) is negative. Process may diverge.")

        self.x0_param = x0
        self.theta = theta 
        self.mu_ou = mu_ou       
        self.sigma = sigma 
        self.T = float(T)
        self.N = int(N)
        self.paths = int(paths)
        self.seed = seed

        if self.N == 1:
            self.t_values = np.array([0.0])
            self.dt = 0.0
        else:
            self.t_values = np.linspace(0, self.T, self.N)
            self.dt = self.T / (self.N - 1)
        
        self.X = np.zeros((self.paths, self.N))

        if self.seed is not None:
            np.random.seed(self.seed)

    def generate_paths(self):
        current_x0_values = np.zeros(self.paths)
        if callable(self.x0_param):
            for i in range(self.paths): current_x0_values[i] = self.x0_param()
        else:
            current_x0_values[:] = self.x0_param

        if self.N <= 1:
            for i in range(self.paths): self.X[i, 0] = current_x0_values[i]
            return self.X

        sqdt = np.sqrt(self.dt) if self.dt > 0 else 0.0
        for i in range(self.paths):
            self.X[i, 0] = current_x0_values[i]
            if self.dt > 0: 
                for j in range(self.N - 1):
                    Z = np.random.standard_normal()
                    self.X[i, j+1] = self.X[i, j] + \
                                     self.theta * (self.mu_ou - self.X[i, j]) * self.dt + \
                                     self.sigma * sqdt * Z
        return self.X

    def get_realized_quadratic_variation_process(self):
        if self.N <= 1: return np.zeros_like(self.X)
        increments = np.diff(self.X, axis=1)
        squared_increments = increments**2
        cumulative_qv = np.cumsum(squared_increments, axis=1)
        return np.concatenate((np.zeros((self.paths, 1)), cumulative_qv), axis=1)

    def get_theoretical_quadratic_variation_process(self):
        """Theoretical QV_t for dX_t = ... + sigma*dW_t is sigma^2 * t."""
        if self.N <= 1: return np.array([0.0])
        return self.sigma**2 * self.t_values.copy()

class CoxIngersollRossProcess:
    """
    Implements Cox-Ingersoll-Ross (CIR) Process.
    SDE: dX_t = theta*(mu - X_t)*dt + sigma*sqrt(X_t)*dW_t
    """
    def __init__(self, x0=0.05, theta=1.0, mu_cir=0.05, sigma=0.1, T=1.0, N=1001, paths=1, seed=None):
        if N < 1: raise ValueError("N must be at least 1.")
        if N == 1 and T != 0.0: raise ValueError("If N=1, T must be 0.0.")
        
        self.x0_param = x0
        self.theta = theta
        self.mu_cir = mu_cir 
        self.sigma = sigma
        self.T = float(T)
        self.N = int(N)
        self.paths = int(paths)
        self.seed = seed
        
        if 2 * self.theta * self.mu_cir < self.sigma**2:
            print(f"Warning: CIR Feller condition not met (2*{self.theta}*{self.mu_cir} < {self.sigma**2:.4f}).")

        if self.N == 1:
            self.t_values = np.array([0.0])
            self.dt = 0.0
        else:
            self.t_values = np.linspace(0, self.T, self.N)
            self.dt = self.T / (self.N - 1)
        
        self.X = np.zeros((self.paths, self.N))

        if self.seed is not None:
            np.random.seed(self.seed)

    def generate_paths(self):
        current_x0_values = np.zeros(self.paths)
        if callable(self.x0_param):
            for i in range(self.paths): 
                x0_val = self.x0_param()
                if x0_val < 0: raise ValueError("x0 from callable must be non-negative for CIR.")
                current_x0_values[i] = x0_val
        else:
            if self.x0_param < 0: raise ValueError("Fixed x0 must be non-negative for CIR.")
            current_x0_values[:] = self.x0_param
        
        if self.N <= 1:
            for i in range(self.paths): self.X[i, 0] = current_x0_values[i]
            return self.X

        sqdt = np.sqrt(self.dt) if self.dt > 0 else 0.0
        for i in range(self.paths):
            self.X[i, 0] = current_x0_values[i]
            if self.dt > 0: 
                for j in range(self.N - 1):
                    Z = np.random.standard_normal()
                    sqrt_X_term = np.sqrt(np.maximum(self.X[i, j], 0.0)) 
                    
                    self.X[i, j+1] = self.X[i, j] + \
                                     self.theta * (self.mu_cir - self.X[i, j]) * self.dt + \
                                     self.sigma * sqrt_X_term * sqdt * Z
                    self.X[i, j+1] = np.maximum(self.X[i, j+1], 0.0) 
        return self.X

    def get_realized_quadratic_variation_process(self):
        if self.N <= 1: return np.zeros_like(self.X)
        increments = np.diff(self.X, axis=1)
        squared_increments = increments**2
        cumulative_qv = np.cumsum(squared_increments, axis=1)
        return np.concatenate((np.zeros((self.paths, 1)), cumulative_qv), axis=1)
        
    def get_path_dependent_theoretical_qv_process(self):
        """
        Calculates theoretical QV process for CIR: integral_0^t (sigma * sqrt(X_u))^2 du 
                                                 = sigma^2 * integral_0^t X_u du.
        """
        if self.N <= 1 or self.X.shape[1] < 2: return np.zeros_like(self.X)
        
        integrand_values = self.sigma**2 * self.X 
        theoretical_qv_process = cumulative_trapezoid(integrand_values, self.t_values, axis=1, initial=0)
        return theoretical_qv_process

class HestonModel:
    """
    Implements the Heston Stochastic Volatility Model.
    Includes methods for MC option pricing and Greeks estimation.
    """
    def __init__(self, S0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04, xi=0.1, rho=-0.7,
                 T=1.0, N=1001, paths=1, seed=None, dividend_yield=0.0): 
        if N < 1: raise ValueError("N must be at least 1.")
        if N == 1 and T != 0.0: raise ValueError("If N=1, T must be 0.0.")
        if v0 < 0: raise ValueError("Initial variance v0 must be non-negative.")
        if S0 <= 0: raise ValueError("Initial asset price S0 must be positive.")
        if not (-1 <= rho <= 1): raise ValueError("Correlation rho must be between -1 and 1.")

        self.S0_param = S0 
        self.v0_param = v0 
        self.mu = mu         
        self.kappa = kappa   
        self.theta = theta   
        self.xi = xi         
        self.rho = rho       
        self.q = dividend_yield 

        self.T = float(T)
        self.N = int(N)
        self.paths = int(paths)
        self.seed = seed

        if self.N == 1:
            self.t_values = np.array([0.0])
            self.dt = 0.0
        else:
            self.t_values = np.linspace(0, self.T, self.N)
            self.dt = self.T / (self.N - 1)
        
        self.S = np.zeros((self.paths, self.N)) 
        self.V = np.zeros((self.paths, self.N)) 

        if self.seed is not None:
            np.random.seed(self.seed)
            
        if 2 * self.kappa * self.theta < self.xi**2:
            print(f"Warning: Heston Feller condition (2*kappa*theta >= xi^2) not met.")


    def _generate_single_path(self, s0_val, v0_val, mu_val, q_val, T_val, N_val, current_dt, current_sqdt, ZS_path_increments, Zv_ind_path_increments):
        s_path = np.zeros(N_val)
        v_path = np.zeros(N_val)
        s_path[0] = s0_val
        v_path[0] = v0_val

        if N_val > 1 and current_dt > 0:
            for j in range(N_val - 1):
                Z_S_step = ZS_path_increments[j] 
                Z_v_ind_step = Zv_ind_path_increments[j]
                
                dW_S_step = current_sqdt * Z_S_step
                dW_v_step = current_sqdt * (self.rho * Z_S_step + np.sqrt(1 - self.rho**2) * Z_v_ind_step)
                
                S_t_curr = s_path[j]
                v_t_curr = v_path[j]
                sqrt_v_t_safe = np.sqrt(np.maximum(v_t_curr, 0.0))

                s_path[j+1] = S_t_curr + (mu_val - q_val) * S_t_curr * current_dt + sqrt_v_t_safe * S_t_curr * dW_S_step
                v_path[j+1] = v_t_curr + self.kappa * (self.theta - v_t_curr) * current_dt + \
                                 self.xi * sqrt_v_t_safe * dW_v_step
                v_path[j+1] = np.maximum(v_path[j+1], 0.0) 
        return s_path, v_path

    def generate_paths(self, S0_override=None, v0_override=None, mu_override=None, q_override=None, T_override=None, N_override=None, paths_override=None):
        _S0 = S0_override if S0_override is not None else self.S0_param
        _v0 = v0_override if v0_override is not None else self.v0_param
        _mu = mu_override if mu_override is not None else self.mu
        _q = q_override if q_override is not None else self.q 
        _T = T_override if T_override is not None else self.T
        _N = N_override if N_override is not None else self.N
        _paths = paths_override if paths_override is not None else self.paths

        _current_t_values = np.linspace(0, _T, _N) if _N > 1 else np.array([0.0])
        _current_dt = _T / (_N - 1) if _N > 1 else 0.0
        
        S_paths = np.zeros((_paths, _N))
        V_paths = np.zeros((_paths, _N))

        current_S0_values = np.zeros(_paths)
        if callable(_S0):
            for i in range(_paths): 
                s0 = _S0(); 
                if s0 <=0: raise ValueError("S0 from callable must be positive.")
                current_S0_values[i] = s0
        else:
            if _S0 <=0: raise ValueError("S0 must be positive.")
            current_S0_values[:] = _S0

        current_v0_values = np.zeros(_paths)
        if callable(_v0):
            for i in range(_paths): 
                v0 = _v0(); 
                if v0 < 0: raise ValueError("v0 from callable must be non-negative.")
                current_v0_values[i] = v0
        else:
            if _v0 < 0: raise ValueError("v0 must be non-negative.")
            current_v0_values[:] = _v0

        if _N <= 1:
            for i in range(_paths):
                S_paths[i, 0] = current_S0_values[i]
                V_paths[i, 0] = current_v0_values[i]
            if paths_override is None: 
                self.S, self.V = S_paths, V_paths
                self.t_values, self.dt = _current_t_values, _current_dt
            return S_paths, V_paths

        sqdt = np.sqrt(_current_dt) if _current_dt > 0 else 0.0
        
        # Preserve current random state to restore after potentially using a fixed seed for this call
        # This is important if the main simulation uses a global seed and we want these calls to also be deterministic
        # without advancing the global state unpredictably from the caller's perspective.
        # However, for Greeks, we need the *same* Zs for up/down bumps.
        # The price_european_option_mc and Greek methods will handle their own seed management if needed.
        # This generate_paths should use the instance's seed or the global state if instance seed is None.
        
        for i in range(_paths):
            ZS_path_i_incs = np.random.standard_normal(_N - 1 if _N > 1 else 0)
            Zv_ind_path_i_incs = np.random.standard_normal(_N - 1 if _N > 1 else 0)
            
            S_paths[i,:], V_paths[i,:] = self._generate_single_path(
                current_S0_values[i], current_v0_values[i], _mu, _q,
                _T, _N, _current_dt, sqdt,
                ZS_path_i_incs, Zv_ind_path_i_incs
            )
        
        if paths_override is None: 
            self.S, self.V = S_paths, V_paths
            self.t_values, self.dt = _current_t_values, _current_dt
        return S_paths, V_paths


    def price_european_option_mc(self, K, T_option, r, option_type='call', 
                                 num_simulations=10000, N_steps_option=None, 
                                 use_antithetic=False, use_control_variate=False,
                                 S0_override=None, v0_override=None, # For Greek calculations
                                 fixed_Z_S_paths=None, fixed_Z_v_ind_paths=None): # For Greek consistency
        """
        Prices a European option using Monte Carlo under the Heston model.
        Allows overriding S0, v0 and using fixed random number paths for Greek calculations.
        """
        if T_option <= 1e-9: 
            _s0_val = S0_override if S0_override is not None else (self.S0_param() if callable(self.S0_param) else self.S0_param)
            payoff = np.maximum(_s0_val - K, 0) if option_type.lower() == 'call' else np.maximum(K - _s0_val, 0)
            return payoff, 0.0 

        _N_option_sim = N_steps_option
        if _N_option_sim is None:
            _N_option_sim = max(2, int(252 * T_option)) 
        
        risk_neutral_drift_S = r - self.q 

        S_T_values = np.zeros(num_simulations)
        num_primary_sims = num_simulations
        
        if use_antithetic:
            if num_simulations % 2 != 0:
                num_simulations +=1 
                S_T_values = np.zeros(num_simulations) 
            num_primary_sims = num_simulations // 2
            
        _current_dt_mc = T_option / (_N_option_sim - 1) if _N_option_sim > 1 else 0.0
        _current_sqdt_mc = np.sqrt(_current_dt_mc) if _current_dt_mc > 0 else 0.0

        _s0_mc = S0_override if S0_override is not None else (self.S0_param() if callable(self.S0_param) else self.S0_param)
        _v0_mc = v0_override if v0_override is not None else (self.v0_param() if callable(self.v0_param) else self.v0_param)

        # Manage random number generation for consistency in Greek calculations
        # If fixed_Z paths are provided, use them. Otherwise, generate new ones.
        # The fixed_Z arrays should be of shape (num_primary_sims, _N_option_sim - 1)

        for i in range(num_primary_sims):
            ZS_path_i_incs = fixed_Z_S_paths[i] if fixed_Z_S_paths is not None else np.random.standard_normal(_N_option_sim - 1 if _N_option_sim > 1 else 0)
            Zv_ind_path_i_incs = fixed_Z_v_ind_paths[i] if fixed_Z_v_ind_paths is not None else np.random.standard_normal(_N_option_sim - 1 if _N_option_sim > 1 else 0)

            s_path1, _ = self._generate_single_path(
                _s0_mc, _v0_mc, risk_neutral_drift_S, self.q, 
                T_option, _N_option_sim, _current_dt_mc, _current_sqdt_mc,
                ZS_path_i_incs, Zv_ind_path_i_incs
            )
            
            if use_antithetic:
                S_T_values[2*i] = s_path1[-1]
                s_path2, _ = self._generate_single_path(
                    _s0_mc, _v0_mc, risk_neutral_drift_S, self.q,
                    T_option, _N_option_sim, _current_dt_mc, _current_sqdt_mc,
                    -ZS_path_i_incs, -Zv_ind_path_i_incs 
                )
                S_T_values[2*i + 1] = s_path2[-1]
            else:
                S_T_values[i] = s_path1[-1]
        
        payoffs_Y = np.maximum(S_T_values - K, 0) if option_type.lower() == 'call' else np.maximum(K - S_T_values, 0)
        discounted_payoffs_Y = np.exp(-r * T_option) * payoffs_Y
        
        if use_control_variate:
            control_C = np.exp(-r * T_option) * S_T_values
            expected_C = _s0_mc * np.exp(-self.q * T_option) 
            
            cov_YC = np.cov(discounted_payoffs_Y, control_C, ddof=1)[0,1]
            var_C = np.var(control_C, ddof=1)
            b_optimal = cov_YC / var_C if var_C > 1e-9 else 0.0
            adjusted_payoffs = discounted_payoffs_Y - b_optimal * (control_C - expected_C)
            mc_price = np.mean(adjusted_payoffs)
            std_error = np.std(adjusted_payoffs, ddof=1) / np.sqrt(num_simulations)
        else:
            mc_price = np.mean(discounted_payoffs_Y)
            std_error = np.std(discounted_payoffs_Y, ddof=1) / np.sqrt(num_simulations)
        
        return mc_price, std_error

    def get_delta_mc(self, K, T_option, r, option_type='call', 
                     num_simulations=10000, N_steps_option=None, 
                     dS_ratio=0.01, use_antithetic=False, use_control_variate=False):
        """Calculates Delta using Monte Carlo (finite difference / pathwise)."""
        s0_orig = self.S0_param() if callable(self.S0_param) else self.S0_param
        dS = s0_orig * dS_ratio

        # Ensure consistent random numbers for up and down paths
        # We generate them once for the "up" path and reuse (with negation for antithetic)
        num_primary_sims = num_simulations
        if use_antithetic:
            if num_simulations % 2 != 0: num_simulations +=1
            num_primary_sims = num_simulations // 2
        
        _N_mc = N_steps_option if N_steps_option is not None else max(2, int(252 * T_option))
        
        # Generate Zs once to be used for C_plus, C_minus (and C_center for Gamma)
        # These are the dW/sqrt(dt) terms for each step of each path
        fixed_Z_S_paths = np.random.standard_normal((num_primary_sims, _N_mc - 1 if _N_mc > 1 else 0))
        fixed_Z_v_ind_paths = np.random.standard_normal((num_primary_sims, _N_mc - 1 if _N_mc > 1 else 0))
        
        price_plus, _ = self.price_european_option_mc(
            K, T_option, r, option_type, num_simulations, N_steps_option,
            use_antithetic, use_control_variate, S0_override=s0_orig + dS,
            fixed_Z_S_paths=fixed_Z_S_paths, fixed_Z_v_ind_paths=fixed_Z_v_ind_paths)
        
        price_minus, _ = self.price_european_option_mc(
            K, T_option, r, option_type, num_simulations, N_steps_option,
            use_antithetic, use_control_variate, S0_override=s0_orig - dS,
            fixed_Z_S_paths=fixed_Z_S_paths, fixed_Z_v_ind_paths=fixed_Z_v_ind_paths)
            
        delta = (price_plus - price_minus) / (2 * dS)
        return delta

    def get_vega_v0_mc(self, K, T_option, r, option_type='call', 
                       num_simulations=10000, N_steps_option=None, 
                       dv0_ratio=0.01, use_antithetic=False, use_control_variate=False):
        """Calculates Vega w.r.t v0 (dV/dv0) using Monte Carlo."""
        v0_orig = self.v0_param() if callable(self.v0_param) else self.v0_param
        dv0 = v0_orig * dv0_ratio 
        if dv0 < 1e-7: dv0 = 1e-7 # Ensure dv0 is not too small / zero

        num_primary_sims = num_simulations
        if use_antithetic:
            if num_simulations % 2 != 0: num_simulations +=1
            num_primary_sims = num_simulations // 2
        _N_mc = N_steps_option if N_steps_option is not None else max(2, int(252 * T_option))
        fixed_Z_S_paths = np.random.standard_normal((num_primary_sims, _N_mc - 1 if _N_mc > 1 else 0))
        fixed_Z_v_ind_paths = np.random.standard_normal((num_primary_sims, _N_mc - 1 if _N_mc > 1 else 0))

        price_plus, _ = self.price_european_option_mc(
            K, T_option, r, option_type, num_simulations, N_steps_option,
            use_antithetic, use_control_variate, v0_override=max(1e-9, v0_orig + dv0), # Ensure v0_plus > 0
            fixed_Z_S_paths=fixed_Z_S_paths, fixed_Z_v_ind_paths=fixed_Z_v_ind_paths)
        
        price_minus, _ = self.price_european_option_mc(
            K, T_option, r, option_type, num_simulations, N_steps_option,
            use_antithetic, use_control_variate, v0_override=max(1e-9, v0_orig - dv0), # Ensure v0_minus > 0
            fixed_Z_S_paths=fixed_Z_S_paths, fixed_Z_v_ind_paths=fixed_Z_v_ind_paths)
            
        vega_v0 = (price_plus - price_minus) / (2 * dv0)
        return vega_v0

    def get_gamma_mc(self, K, T_option, r, option_type='call', 
                     num_simulations=10000, N_steps_option=None, 
                     dS_ratio=0.01, use_antithetic=False, use_control_variate=False):
        """Calculates Gamma using Monte Carlo (finite difference / pathwise)."""
        s0_orig = self.S0_param() if callable(self.S0_param) else self.S0_param
        dS = s0_orig * dS_ratio
        if dS < 1e-7 : dS = 1e-7

        num_primary_sims = num_simulations
        if use_antithetic:
            if num_simulations % 2 != 0: num_simulations +=1
            num_primary_sims = num_simulations // 2
        _N_mc = N_steps_option if N_steps_option is not None else max(2, int(252 * T_option))
        fixed_Z_S_paths = np.random.standard_normal((num_primary_sims, _N_mc - 1 if _N_mc > 1 else 0))
        fixed_Z_v_ind_paths = np.random.standard_normal((num_primary_sims, _N_mc - 1 if _N_mc > 1 else 0))

        price_center, _ = self.price_european_option_mc(
            K, T_option, r, option_type, num_simulations, N_steps_option,
            use_antithetic, use_control_variate, S0_override=s0_orig, # Center price
            fixed_Z_S_paths=fixed_Z_S_paths, fixed_Z_v_ind_paths=fixed_Z_v_ind_paths)

        price_plus, _ = self.price_european_option_mc(
            K, T_option, r, option_type, num_simulations, N_steps_option,
            use_antithetic, use_control_variate, S0_override=s0_orig + dS,
            fixed_Z_S_paths=fixed_Z_S_paths, fixed_Z_v_ind_paths=fixed_Z_v_ind_paths)
        
        price_minus, _ = self.price_european_option_mc(
            K, T_option, r, option_type, num_simulations, N_steps_option,
            use_antithetic, use_control_variate, S0_override=s0_orig - dS,
            fixed_Z_S_paths=fixed_Z_S_paths, fixed_Z_v_ind_paths=fixed_Z_v_ind_paths)
            
        gamma = (price_plus - 2 * price_center + price_minus) / (dS**2)
        return gamma


    def get_realized_quadratic_variation_S_process(self):
        if self.N <= 1: return np.zeros_like(self.S)
        increments = np.diff(self.S, axis=1)
        squared_increments = increments**2
        cumulative_qv = np.cumsum(squared_increments, axis=1)
        return np.concatenate((np.zeros((self.paths, 1)), cumulative_qv), axis=1)
        
    def get_realized_quadratic_variation_V_process(self):
        if self.N <= 1: return np.zeros_like(self.V)
        increments = np.diff(self.V, axis=1)
        squared_increments = increments**2
        cumulative_qv = np.cumsum(squared_increments, axis=1)
        return np.concatenate((np.zeros((self.paths, 1)), cumulative_qv), axis=1)

    def get_path_dependent_theoretical_qv_S_process(self):
        if self.N <= 1 or self.S.shape[1] < 2: return np.zeros_like(self.S)
        integrand_values = self.V * (self.S**2) 
        theoretical_qv_process = cumulative_trapezoid(integrand_values, self.t_values, axis=1, initial=0)
        return theoretical_qv_process

    def get_path_dependent_theoretical_qv_V_process(self):
        if self.N <= 1 or self.V.shape[1] < 2: return np.zeros_like(self.V)
        integrand_values = (self.xi**2) * self.V 
        theoretical_qv_process = cumulative_trapezoid(integrand_values, self.t_values, axis=1, initial=0)
        return theoretical_qv_process
