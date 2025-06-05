import numpy as np
from scipy.optimize import minimize, differential_evolution 
import time

# Attempt to import HestonModel from the stochastic_processes package
try:
    from ..stochastic_processes.ito_processes import HestonModel
except (ImportError, ValueError):
    try:
        from financial_toolkit.stochastic_processes.ito_processes import HestonModel
    except ImportError:
        print("CRITICAL WARNING in heston_calibration.py: HestonModel not found. "
              "Ensure 'financial_toolkit.stochastic_processes.ito_processes' is accessible.")
        class HestonModel: # Dummy for parsing
            def __init__(self, *args, **kwargs): print("Dummy HestonModel initialized.")
            def price_european_option_mc(self, *args, **kwargs): return 0.0, 0.0


CALIBRATION_MC_SEED_OFFSET = 1301 

def heston_calibration_objective_function(
    params_to_calibrate_values, 
    param_names_to_calibrate, 
    fixed_heston_params_dict, 
    market_option_prices_arr, 
    N_mc_calibration_obj, 
    N_steps_calibration_obj,
    feller_penalty_factor=0.0 # New argument for penalty
    ):
    """
    Objective function for Heston model calibration.
    Includes an optional penalty for Feller condition violation.
    """
    
    trial_heston_params_full = fixed_heston_params_dict.copy()
    for i, name in enumerate(param_names_to_calibrate):
        trial_value = params_to_calibrate_values[i]
        
        # Apply basic validation based on typical parameter ranges
        if name == "v0": trial_value = max(1e-5, trial_value)
        if name == "xi": trial_value = max(1e-5, trial_value)
        if name == "kappa": trial_value = max(1e-5, trial_value)
        if name == "theta": trial_value = max(1e-5, trial_value)
        if name == "rho": trial_value = np.clip(trial_value, -0.9999, 0.9999)

        trial_heston_params_full[name] = trial_value

    model_prices_trial = []
    
    current_mc_seed = fixed_heston_params_dict.get("base_seed", None)
    if current_mc_seed is not None:
        current_mc_seed += CALIBRATION_MC_SEED_OFFSET
        
    heston_pricer_instance = HestonModel(
        S0=trial_heston_params_full["S0"], 
        v0=trial_heston_params_full["v0"],
        mu=trial_heston_params_full["r"] - trial_heston_params_full["q"], 
        kappa=trial_heston_params_full["kappa"], 
        theta=trial_heston_params_full["theta"],
        xi=trial_heston_params_full["xi"], 
        rho=trial_heston_params_full["rho"],
        T=trial_heston_params_full["T_option"], 
        N=N_steps_calibration_obj, 
        paths=1, 
        seed=current_mc_seed, 
        dividend_yield=trial_heston_params_full["q"]
    )

    for k_strike in fixed_heston_params_dict["strikes"]:
        price, _ = heston_pricer_instance.price_european_option_mc(
            K=k_strike, 
            T_option=fixed_heston_params_dict["T_option"], 
            r=fixed_heston_params_dict["r"], 
            option_type=fixed_heston_params_dict["option_type"],
            num_simulations=N_mc_calibration_obj, 
            N_steps_option=N_steps_calibration_obj, 
            use_antithetic=True, 
            use_control_variate=False 
        )
        model_prices_trial.append(price)
    
    model_prices_trial = np.array(model_prices_trial)
    safe_market_prices = np.maximum(market_option_prices_arr, 1e-6) 
    error = np.sum(((model_prices_trial - market_option_prices_arr) / safe_market_prices)**2) 
    
    # Add Feller condition penalty
    if feller_penalty_factor > 0:
        # Parameters needed for Feller: kappa, theta, xi
        # These come from trial_heston_params_full
        kappa_trial = trial_heston_params_full["kappa"]
        theta_trial = trial_heston_params_full["theta"]
        xi_trial = trial_heston_params_full["xi"]
        
        feller_value = 2 * kappa_trial * theta_trial - xi_trial**2
        if feller_value < 0: # Condition violated
            # Penalty could be proportional to the magnitude of violation
            penalty = feller_penalty_factor * (-feller_value) # -feller_value is positive
            error += penalty
            print(f"DEBUG Objective: Params={dict(zip(param_names_to_calibrate, params_to_calibrate_values))}, PriceError={error-penalty:.6e}, FellerPenalty={penalty:.6e}, TotalError={error:.6e}")

    else:
        print(f"DEBUG Objective: Params={dict(zip(param_names_to_calibrate, params_to_calibrate_values))}, Error={error:.6e}")
    return error


def calibrate_heston_params(
    market_prices, 
    S0, K_array, T_option, r, q, option_type, 
    initial_guesses_dict, 
    params_to_calibrate_names, 
    fixed_heston_params_dict_base, 
    param_bounds_dict, 
    N_mc_calibration=5000, 
    N_steps_calibration=50,
    base_seed=None, 
    optimizer_choice='L-BFGS-B', 
    optimizer_options={'disp': True, 'maxiter': 50, 'ftol': 1e-7},
    feller_penalty_factor=0.0 # New argument for penalty factor
    ):
    """
    Calibrates specified Heston model parameters to market option prices.
    Allows choice of optimizer and Feller condition penalty.
    """
    
    initial_params_arr = [initial_guesses_dict[name] for name in params_to_calibrate_names]
    bounds_list_for_optimizer = [param_bounds_dict[name] for name in params_to_calibrate_names]

    all_fixed_params_for_obj_func = fixed_heston_params_dict_base.copy()
    all_fixed_params_for_obj_func.update({
        "S0": S0, "strikes": K_array, "T_option": T_option, 
        "r": r, "q": q, "option_type": option_type,
        "base_seed": base_seed 
    })
    
    args_for_objective = (
        params_to_calibrate_names, 
        all_fixed_params_for_obj_func, 
        market_prices, 
        N_mc_calibration, 
        N_steps_calibration,
        feller_penalty_factor # Pass penalty factor to objective
    )

    print(f"\nStarting Heston calibration using '{optimizer_choice}' optimizer.")
    print(f"Parameters to calibrate: {params_to_calibrate_names}")
    print(f"Initial guesses: {dict(zip(params_to_calibrate_names, initial_params_arr))}")
    if feller_penalty_factor > 0:
        print(f"Feller condition penalty factor: {feller_penalty_factor}")
    
    calibration_start_time = time.time()
    
    if optimizer_choice.lower() == 'differential_evolution':
        de_options = optimizer_options.copy() # Start with general options
        de_options.setdefault('maxiter', 100) 
        de_options.setdefault('popsize', 15)
        de_options.setdefault('tol', 0.01)
        de_options.setdefault('polish', True)
        print(f"Differential Evolution options: {de_options}")
        calibration_result = differential_evolution(
            func=heston_calibration_objective_function, 
            bounds=bounds_list_for_optimizer, 
            args=args_for_objective,
            **de_options
        )
    elif optimizer_choice in ['L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'TNC', 'COBYLA']:
        calibration_result = minimize(
            fun=heston_calibration_objective_function, 
            x0=initial_params_arr, 
            args=args_for_objective,
            method=optimizer_choice,
            bounds=bounds_list_for_optimizer if optimizer_choice in ['L-BFGS-B', 'TNC', 'SLSQP'] else None,
            options=optimizer_options
        )
    else:
        raise ValueError(f"Unsupported optimizer_choice: {optimizer_choice}")

    calibration_end_time = time.time()
    total_time = calibration_end_time - calibration_start_time
    print(f"Calibration process completed in {total_time:.2f} seconds.")
    
    if hasattr(calibration_result, 'message'): print(f"Optimizer message: {calibration_result.message}")
    if hasattr(calibration_result, 'success'): print(f"Success: {calibration_result.success}")
    if hasattr(calibration_result, 'fun'): print(f"Final objective value: {calibration_result.fun:.6e}")

    return calibration_result
