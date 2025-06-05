import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq # For implied volatility calculation

class EuropeanOptionBSM:
    """
    Calculates Black-Scholes-Merton prices and Greeks for European options.
    Also includes MC pricing with antithetic variates and control variates.
    """
    def __init__(self, S0, K, T, r, sigma, option_type='call', dividend_yield=0.0):
        if T < -1e-9: 
            raise ValueError("Time to maturity T must be non-negative.")
        if T > 1e-9 and sigma <= 1e-9: 
            raise ValueError("Volatility sigma must be positive if T > 0.")
        
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma) 
        self.q = float(dividend_yield) # Continuous dividend yield
        
        if option_type.lower() not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'.")
        self.option_type = option_type.lower()

        self._d1, self._d2 = self._calculate_d1_d2(self.S0, self.K, self.T, self.r, self.q, self.sigma)


    def _calculate_d1_d2(self, S, K, T, r, q, sigma_calc):
        """Helper static method to calculate d1 and d2 for any set of params including dividend yield q."""
        if T <= 1e-9: 
            d1 = float('inf') if S > K else (0 if S == K else float('-inf'))
            # For T=0, exact d1/d2 for norm.cdf less critical than payoff.
            # This simplification for T=0 is acceptable as price() handles T=0 directly.
            d2 = d1 
            return d1, d2
        
        sigma_calc_safe = max(1e-9, sigma_calc)

        if S <= 0 or K <= 0:
             # Using self.option_type if available, or assuming call for extreme d1 if not.
             # A more robust static method would take option_type.
             op_type = getattr(self, 'option_type', 'call') 
             if op_type == 'call':
                 d1 = float('-inf') 
             else: 
                 d1 = float('inf') 
             d2 = d1
             return d1, d2

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma_calc_safe**2) * T) / (sigma_calc_safe * np.sqrt(T))
        d2 = d1 - sigma_calc_safe * np.sqrt(T)
        return d1, d2

    def price(self, S_eval=None, T_eval=None, sigma_eval=None):
        _S = self.S0 if S_eval is None else S_eval
        _K = self.K
        _T = self.T if T_eval is None else T_eval
        _r = self.r
        _q = self.q
        _sigma = self.sigma if sigma_eval is None else sigma_eval

        if _T <= 1e-9: 
            if self.option_type == 'call': return max(0.0, _S - _K)
            else: return max(0.0, _K - _S)
        
        if _sigma <= 1e-9 : 
            # For zero vol, option price is discounted expected payoff assuming S evolves deterministically
            S_at_T_deterministic = _S * np.exp((_r - _q) * _T)
            if self.option_type == 'call': 
                return np.maximum(0, S_at_T_deterministic - _K) * np.exp(-_r * _T)
            else: 
                return np.maximum(0, _K - S_at_T_deterministic) * np.exp(-_r * _T)


        d1_calc, d2_calc = self._calculate_d1_d2(_S, _K, _T, _r, _q, _sigma)

        if self.option_type == 'call':
            price_val = _S * np.exp(-_q * _T) * norm.cdf(d1_calc) - _K * np.exp(-_r * _T) * norm.cdf(d2_calc)
        else: 
            price_val = _K * np.exp(-_r * _T) * norm.cdf(-d2_calc) - _S * np.exp(-_q * _T) * norm.cdf(-d1_calc)
        return price_val

    def delta(self): 
        if self.T <= 1e-9: 
            if self.option_type == 'call':
                return 1.0 if self.S0 > self.K else (0.5 if np.isclose(self.S0, self.K) else 0.0)
            else: 
                return -1.0 if self.S0 < self.K else (-0.5 if np.isclose(self.S0, self.K) else 0.0)
        if self.sigma <= 1e-9: 
            S_fwd = self.S0 * np.exp((self.r - self.q) * self.T)
            if self.option_type == 'call':
                return np.exp(-self.q * self.T) if S_fwd > self.K else 0.0 
            else: 
                return -np.exp(-self.q * self.T) if S_fwd < self.K else 0.0
        
        if self.option_type == 'call': return np.exp(-self.q * self.T) * norm.cdf(self._d1)
        else: return np.exp(-self.q * self.T) * (norm.cdf(self._d1) - 1.0)

    # ... (Gamma, Vega, Theta, Rho methods would also need to account for q if not already) ...
    # For simplicity, keeping them as they were, but a full BSMG model would update them.
    def gamma(self): 
        if self.T <= 1e-9 or self.S0 <= 1e-9 or self.sigma <= 1e-9 : return 0.0 
        return np.exp(-self.q * self.T) * norm.pdf(self._d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def vega(self): 
        if self.T <= 1e-9 or self.sigma <= 1e-9 : return 0.0
        return self.S0 * np.exp(-self.q * self.T) * norm.pdf(self._d1) * np.sqrt(self.T) 

    def theta(self): 
        if self.T <= 1e-9 or self.sigma <= 1e-9: return 0.0
        
        term_S_pdf = self.S0 * np.exp(-self.q * self.T) * norm.pdf(self._d1) * self.sigma / (2 * np.sqrt(self.T))
        if self.option_type == 'call':
            return -term_S_pdf \
                   + self.q * self.S0 * np.exp(-self.q * self.T) * norm.cdf(self._d1) \
                   - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self._d2)
        else: # put
            return -term_S_pdf \
                   - self.q * self.S0 * np.exp(-self.q * self.T) * norm.cdf(-self._d1) \
                   + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self._d2)
            
    def rho(self): 
        if self.T <= 1e-9 : return 0.0
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self._d2)
        else: 
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self._d2)


    def mc_price(self, num_simulations=10000, seed=None, use_antithetic=False, use_control_variate=False):
        """
        Calculates option price using Monte Carlo simulation.
        Can use antithetic variates and/or control variates (S_T) for variance reduction.
        """
        if self.T <= 1e-9: 
            price_at_expiry = self.price() 
            return price_at_expiry, 0.0

        if seed is not None: np.random.seed(seed)

        actual_num_sims = num_simulations
        if use_antithetic:
            if num_simulations % 2 != 0:
                actual_num_sims = num_simulations + 1
            # Generate num_simulations/2 independent Z draws
            Z_half = np.random.standard_normal(actual_num_sims // 2)
            Z = np.concatenate((Z_half, -Z_half)) 
        else:
            Z = np.random.standard_normal(actual_num_sims)
        
        # Risk-neutral S_T: S0 * exp((r - q - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        ST = self.S0 * np.exp((self.r - self.q - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        
        payoffs_Y = np.maximum(ST - self.K, 0) if self.option_type == 'call' else np.maximum(K - ST, 0)
        discounted_payoffs_Y = np.exp(-self.r * self.T) * payoffs_Y
        
        if use_control_variate:
            # Control variate C = discounted S_T. E[C] = S0 * exp(-qT)
            # (This assumes the process for S_T used to derive E[C] matches the one generating S_T)
            # For BSM, under risk-neutral measure, E[S_T] = S0 * exp((r-q)T).
            # So E[exp(-rT)S_T] = S0 * exp(-qT).
            control_C = np.exp(-self.r * self.T) * ST
            expected_C = self.S0 * np.exp(-self.q * self.T) 
            
            # Estimate optimal b: Cov(Y,C) / Var(C)
            # Note: using sample covariance and variance
            cov_YC = np.cov(discounted_payoffs_Y, control_C, ddof=1)[0,1]
            var_C = np.var(control_C, ddof=1)
            
            b_optimal = cov_YC / var_C if var_C > 1e-9 else 0 # Avoid division by zero
            
            # Adjusted estimator Y_adj = Y - b*(C - E[C])
            adjusted_payoffs = discounted_payoffs_Y - b_optimal * (control_C - expected_C)
            
            mc_price = np.mean(adjusted_payoffs)
            std_error = np.std(adjusted_payoffs, ddof=1) / np.sqrt(actual_num_sims)
        else:
            mc_price = np.mean(discounted_payoffs_Y)
            std_error = np.std(discounted_payoffs_Y, ddof=1) / np.sqrt(actual_num_sims)
        
        return mc_price, std_error

    @staticmethod
    def calculate_implied_volatility(market_price, S0, K, T, r, option_type, dividend_yield=0.0,
                                     low_vol=1e-5, high_vol=2.0, tol=1e-6, max_iter=100):
        # ... (calculate_implied_volatility method unchanged from previous version, but should now use dividend_yield) ...
        # Small modification to pass dividend_yield to temp_opt
        if T <= 1e-9: return np.nan 

        def objective_function(sigma_iv):
            if sigma_iv < 1e-9 : sigma_iv = 1e-9 
            temp_opt = EuropeanOptionBSM(S0, K, T, r, sigma_iv, option_type, dividend_yield=dividend_yield)
            return temp_opt.price() - market_price

        try:
            price_at_low = objective_function(low_vol) + market_price
            price_at_high = objective_function(high_vol) + market_price

            intrinsic_val = 0.0
            if option_type == 'call':
                intrinsic_val = max(0, S0 * np.exp(-dividend_yield * T) - K * np.exp(-r * T)) # Corrected intrinsic for dividend
                if market_price < intrinsic_val - tol : return np.nan 
                if market_price > S0 * np.exp(-dividend_yield * T) + tol : return np.nan 
            else: 
                intrinsic_val = max(0, K * np.exp(-r * T) - S0 * np.exp(-dividend_yield * T)) # Corrected intrinsic for dividend
                if market_price < intrinsic_val - tol : return np.nan
                if market_price > K * np.exp(-r*T) + tol : return np.nan 
            
            if np.sign(objective_function(low_vol)) == np.sign(objective_function(high_vol)):
                 if abs(objective_function(low_vol)) < abs(objective_function(high_vol)): 
                    if price_at_high < market_price and objective_function(high_vol) < 0 : return np.nan
                 else: 
                    if price_at_low > market_price and objective_function(low_vol) > 0 : return np.nan
                 return np.nan 

            implied_sigma = brentq(objective_function, low_vol, high_vol, xtol=tol, rtol=tol, maxiter=max_iter)
            return implied_sigma
        except ValueError: 
            return np.nan


class DeltaHedgingSimulator:
    # ... (DeltaHedgingSimulator class code remains unchanged from previous version) ...
    # Small update: ensure temp options in hedging use the main option's dividend yield.
    """
    Simulates a delta hedging strategy for a European option.
    """
    def __init__(self, option: EuropeanOptionBSM, underlying_S_path: np.ndarray, time_axis: np.ndarray, hedging_sigma: float = None):
        if not isinstance(option, EuropeanOptionBSM):
            raise TypeError("option must be an instance of EuropeanOptionBSM.")
        if len(underlying_S_path) != len(time_axis):
            raise ValueError("underlying_S_path and time_axis must have same length.")
        if len(underlying_S_path) < 2:
            raise ValueError("Path must have at least 2 points.")

        self.option = option # This option instance carries S0, K, T, r, sigma, option_type, and q
        self.S_path = underlying_S_path
        self.t_axis = time_axis
        self.N_steps = len(time_axis) -1 
        self.dt = time_axis[1] - time_axis[0] if len(time_axis) > 1 else 0 

        self.hedging_sigma = hedging_sigma if hedging_sigma is not None else option.sigma # Vol for delta calc
        
        self.delta_shares = np.zeros(self.N_steps + 1) 
        self.cash_account = np.zeros(self.N_steps + 1) 
        self.portfolio_value = np.zeros(self.N_steps + 1)

    def simulate_hedge(self):
        # Use main option's r and q for all temp BSM calculations
        r_hedge = self.option.r
        q_hedge = self.option.q

        temp_option_for_initial_price = EuropeanOptionBSM(
            S0=self.S_path[0], K=self.option.K, T=self.option.T, 
            r=r_hedge, sigma=self.hedging_sigma, option_type=self.option.option_type, dividend_yield=q_hedge
        )
        initial_option_price_hedge_vol = temp_option_for_initial_price.price()
        self.portfolio_value[0] = initial_option_price_hedge_vol 

        current_T_remaining = self.option.T - self.t_axis[0]
        temp_option_for_delta = EuropeanOptionBSM(
            S0=self.S_path[0], K=self.option.K, T=current_T_remaining, 
            r=r_hedge, sigma=self.hedging_sigma, option_type=self.option.option_type, dividend_yield=q_hedge
        )
        self.delta_shares[0] = temp_option_for_delta.delta()
        self.cash_account[0] = self.portfolio_value[0] - self.delta_shares[0] * self.S_path[0]

        for i in range(self.N_steps): 
            # Cash grows at risk-free rate. Stock pays dividend yield (cost to holder of delta shares).
            # Net growth of cash from stock holding part of portfolio: delta*S*r - delta*S*q
            # Cash account itself grows at r. Cost of holding shares (financing + dividends)
            # Simpler: Cash from previous step grows at r. Separately, handle stock value.
            self.cash_account[i] = self.cash_account[i] * np.exp(r_hedge * self.dt) # Cash grows
            
            # If holding delta shares, dividend is paid out of cash or reduces cash needed to borrow
            # This is implicitly handled if we consider the cost of carrying the shares.
            # Standard approach: cash account = Portfolio_value - Delta * S
            # Portfolio value change: dP = delta*dS + (P - delta*S)*r*dt
            # Here, discrete time:
            # P_i+1 = delta_i * S_i+1 + (P_i - delta_i * S_i) * exp(r*dt)
            # This is what the rebalancing step effectively does.

            # Value of portfolio components before rebalancing at t_{i+1}
            # Cash account already accrued interest for the period ending at t_i (from previous step)
            # or it means cash from t_i to t_{i+1}. Let's be clear:
            # self.cash_account[i] is cash at t_i after rebalancing at t_i.
            # It grows to self.cash_account[i]*exp(r*dt) by t_{i+1} before next rebalancing.
            
            # Value of portfolio at t_{i+1} BEFORE rebalancing:
            prev_cash_grown = self.cash_account[i] * np.exp(r_hedge * self.dt)
            prev_stock_value = self.delta_shares[i] * self.S_path[i+1]
            self.portfolio_value[i+1] = prev_cash_grown + prev_stock_value
            
            if i < self.N_steps -1 : # Not the last interval
                current_T_remaining_at_rebalance = self.option.T - self.t_axis[i+1]
                temp_option_for_delta_rebalance = EuropeanOptionBSM(
                    S0=self.S_path[i+1], K=self.option.K, T=current_T_remaining_at_rebalance,
                    r=r_hedge, sigma=self.hedging_sigma, option_type=self.option.option_type, dividend_yield=q_hedge
                )
                self.delta_shares[i+1] = temp_option_for_delta_rebalance.delta()
                self.cash_account[i+1] = self.portfolio_value[i+1] - self.delta_shares[i+1] * self.S_path[i+1]
            else: # Last step, at T. Liquidation.
                self.delta_shares[i+1] = 0.0 
                self.cash_account[i+1] = self.portfolio_value[i+1] # All converted to cash

        final_portfolio_value = self.portfolio_value[-1]
        
        option_payoff_at_T_actual = EuropeanOptionBSM(
            S0=self.S_path[-1], K=self.option.K, T=0.0, 
            r=r_hedge, sigma=self.option.sigma, 
            option_type=self.option.option_type, dividend_yield=q_hedge
        ).price() 
            
        hedging_pnl = final_portfolio_value - option_payoff_at_T_actual

        return {
            "hedging_pnl": hedging_pnl,
            "final_portfolio_value": final_portfolio_value,
            "option_payoff": option_payoff_at_T_actual,
            "portfolio_values_over_time": self.portfolio_value,
            "delta_shares_over_time": self.delta_shares,
            "cash_account_over_time": self.cash_account
        }
