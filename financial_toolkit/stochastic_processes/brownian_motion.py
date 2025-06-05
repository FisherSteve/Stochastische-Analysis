
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid # For path-dependent theoretical QV


try:
    from ..utils.plotting import plot_lines, plot_histogram_with_pdf
except (ImportError, ValueError): 
    try:
        from utils.plotting import plot_lines, plot_histogram_with_pdf 
    except ImportError:
        print("CRITICAL WARNING: Plotting utilities not found in brownian_motion.py. Ensure 'utils' directory is correctly placed or PYTHONPATH is set.")
        def plot_lines(*args, **kwargs): raise ImportError("plot_lines utility not found.")
        def plot_histogram_with_pdf(*args, **kwargs): raise ImportError("plot_histogram_with_pdf utility not found.")


class BrownianMotion:
    """
    Implements 1D Brownian Motion (Wiener Process) starting from x0.
    The starting point x0 can be a fixed value or a callable function
    (e.g., a lambda for drawing from a distribution).
    Property demonstration methods operate on a standard BM (x0=0) internally.
    """
    def __init__(self, x0=0.0, T=10.0, N=10000, paths=1, seed=None):
        if N < 1:
            raise ValueError("N (number of time points) must be at least 1.")
        if N == 1 and T != 0.0:
            raise ValueError("If N=1 (single time point), T must be 0.0.")

        self.x0_param = x0 
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
        
        self.sqdt = np.sqrt(self.dt) if self.dt > 0 else 0.0
        self.X = np.zeros((self.paths, self.N)) 
        
        if self.seed is not None:
            np.random.seed(self.seed)

    def generate_paths(self):
        current_x0_values = np.zeros(self.paths)
        if callable(self.x0_param):
            for i in range(self.paths):
                current_x0_values[i] = self.x0_param()
        else: 
            current_x0_values[:] = self.x0_param

        if self.N <= 1: 
            for i in range(self.paths):
                self.X[i, 0] = current_x0_values[i]
            return self.X
            
        for i in range(self.paths):
            dW_std = self.sqdt * np.random.standard_normal(self.N - 1)
            self.X[i, 0] = current_x0_values[i] 
            standard_W_part = np.cumsum(dW_std)
            self.X[i, 1:] = current_x0_values[i] + standard_W_part
        return self.X

    def get_realized_quadratic_variation_process(self):
        """
        Calculates the realized quadratic variation process for each generated path.
        QV_realized_t = sum_{k=0}^{i-1} (X_{t_{k+1}} - X_{t_k})^2, for t = t_i
        Returns an array of the same shape as self.X. QV(0) = 0.

        Returns:
            numpy.ndarray: Array of realized quadratic variation processes.
        """
        if self.N <= 1:
            return np.zeros_like(self.X)
        
        increments = np.diff(self.X, axis=1) 
        squared_increments = increments**2
        
        # Cumulatively sum squared increments
        cumulative_qv = np.cumsum(squared_increments, axis=1)
        
        # Prepend a zero for QV at t=0
        realized_qv_process = np.concatenate((np.zeros((self.paths, 1)), cumulative_qv), axis=1)
        return realized_qv_process

    def get_theoretical_quadratic_variation_process(self):
        """
        Returns the theoretical quadratic variation process for X_t = x0 + W_t.
        The process dX_t = dW_t has sigma = 1.
        Theoretical QV_t = t. This is independent of x0.
        Returns an array of shape (N,) corresponding to self.t_values.
        """
        if self.N <= 1: 
            return np.array([0.0])
        # For BM, QV_t = t
        return self.t_values.copy()


    # --- Property demonstration methods (operate on standard BM, x0=0 internally) ---
    # These remain unchanged as they focus on path properties, not QV.

    def get_self_similarity_data(self, c_values=None):
        # print("DEBUG: BM.get_self_similarity_data called (demonstrates for standard BM x0=0)")
        T_orig = self.T
        N_orig = self.N
        t_plot_axis = np.linspace(0, T_orig, N_orig) if N_orig > 1 else np.array([0.0])

        if c_values is None: 
            c_values = [0.25, 0.5, 1.0, 2.0, 4.0] 
        else: 
            if 1.0 not in c_values: c_values = sorted(list(set(c_values + [1.0]))) 
            else: c_values = sorted(list(set(c_values))) 

        valid_c_values = [c for c in c_values if c > 0]

        if not valid_c_values :
            print("Warning: No valid positive c_values for self-similarity. Cannot generate data.")
            return None 

        max_c_val = max(valid_c_values) 
        T_base = max(T_orig, max_c_val * T_orig) 
        
        N_base = N_orig 
        if N_orig > 1 and T_orig > 0:
            dt_orig = T_orig / (N_orig -1)
            N_base = int(np.ceil(T_base / dt_orig)) + 1 if dt_orig > 1e-9 else N_orig * int(np.ceil(max_c_val if max_c_val > 1 else 1.0))
        N_base = max(N_base, 2) if T_base > 0 else 1


        bm_std_helper = BrownianMotion(x0=0.0, T=T_base, N=N_base, paths=1, seed=self.seed)
        W_base_path = bm_std_helper.generate_paths()[0, :] 
        t_base_for_interp = bm_std_helper.t_values 

        paths_data_list = []
        labels_list = []
        
        for c_val in valid_c_values:
            args_for_W_base_interp = c_val * t_plot_axis 
            W_base_at_ct = np.interp(args_for_W_base_interp, t_base_for_interp, W_base_path)
            V_t_scaled = (1 / np.sqrt(c_val)) * W_base_at_ct
            paths_data_list.append(V_t_scaled)
            if np.isclose(c_val, 1.0):
                 labels_list.append('$W_t$ (Std. Ref. $c=1.00$)')
            else:
                 labels_list.append(f'$(1/\\sqrt{{{c_val:.2f}}}) W_{{{c_val:.2f}t}}$ (Std.)')
        
        return {
            't_axis': t_plot_axis,
            'paths_data': paths_data_list,
            'labels': labels_list,
            'T_orig': T_orig,
            'note': 'Demonstrates property for a standard BM (x0=0)'
        }

    def get_time_reversal_data(self, path_index_for_std_bm=0): 
        # print("DEBUG: BM.get_time_reversal_data called (demonstrates for standard BM x0=0)")
        if self.N == 0: 
            print("Warning: Cannot get time reversal data with N=0 points.")
            return None

        bm_std_helper = BrownianMotion(x0=0.0, T=self.T, N=self.N, paths=1, seed=self.seed)
        W_std_orig = bm_std_helper.generate_paths()[0, :] 

        V_t_transformed = np.zeros_like(W_std_orig)
        if self.N > 0: 
            W_T_val = W_std_orig[-1] 
            W_T_minus_t_values = W_std_orig[::-1] 
            V_t_transformed = W_T_val - W_T_minus_t_values
        
        return {
            't_axis': self.t_values.copy(), 
            'W_original': W_std_orig, 
            'V_transformed': V_t_transformed,
            'path_index': 0, 
            'note': 'Demonstrates property for a standard BM (x0=0)'
        }

    def get_time_inversion_data(self, path_index_for_std_bm=0):
        # print("DEBUG: BM.get_time_inversion_data called (demonstrates for standard BM x0=0)")
        
        if self.N <= 1: 
            note_str = f'Time inversion V_t = tW(1/t) not applicable for N={self.N} (requires t>0 for standard BM demo).'
            bm_std_trivial = BrownianMotion(x0=0.0, T=0.0, N=1, seed=self.seed)
            W_std_trivial = bm_std_trivial.generate_paths()[0,:]
            return {
                't_axis': bm_std_trivial.t_values.copy(),
                'W_original': W_std_trivial,
                'V_transformed': W_std_trivial.copy(),
                'path_index': 0,
                'note': note_str
            }
        
        if self.dt <= 1e-9: 
            note_str = f"Time step dt ({self.dt:.2e}) is zero or near-zero. Standard BM Time inversion problematic."
            bm_std_problematic = BrownianMotion(x0=0.0, T=self.T, N=self.N, seed=self.seed)
            W_std_problematic = bm_std_problematic.generate_paths()[0,:]
            return {
                't_axis': self.t_values.copy(),
                'W_original': W_std_problematic,
                'V_transformed': W_std_problematic.copy(),
                'path_index': 0,
                'note': note_str
            }
            
        t_for_V_calc_values = self.t_values[1:]
        T_base_for_helper = 1.0 / self.dt 
        inversion_N_refinement_factor = 60 
        N_base_for_helper = max(2, int(self.N * inversion_N_refinement_factor)) 
        
        helper_seed = self.seed + 1 if self.seed is not None else None 
        bm_W_helper = BrownianMotion(x0=0.0, T=T_base_for_helper, N=N_base_for_helper, paths=1, seed=helper_seed)
        W_helper_path = bm_W_helper.generate_paths()[0, :]
        t_helper_for_interp = bm_W_helper.t_values 
        
        args_for_W_interpolation = 1.0 / t_for_V_calc_values 
        W_one_over_t = np.interp(args_for_W_interpolation, t_helper_for_interp, W_helper_path)
        V_t_values_nonzero_t = t_for_V_calc_values * W_one_over_t
        
        V_t_final = np.zeros(self.N) 
        V_t_final[0] = 0.0 
        if self.N > 1: V_t_final[1:] = V_t_values_nonzero_t

        bm_std_orig_compare = BrownianMotion(x0=0.0, T=self.T, N=self.N, paths=1, seed=self.seed)
        W_std_orig_compare_path = bm_std_orig_compare.generate_paths()[0,:]

        return {
            't_axis': self.t_values.copy(),
            'W_original': W_std_orig_compare_path, 
            'V_transformed': V_t_final,         
            'path_index': 0,
            'note': 'Demonstrates property for a standard BM (x0=0)'
        }

    def get_symmetry_data(self, path_index_for_std_bm_plot=0, num_paths_for_hist=10000):
        # print("DEBUG: BM.get_symmetry_data called (demonstrates for standard BM x0=0)")
        
        bm_std_plot = BrownianMotion(x0=0.0, T=self.T, N=self.N, paths=1, seed=self.seed)
        if bm_std_plot.N > 1 and bm_std_plot.T > 0 and np.allclose(bm_std_plot.X[0,1:], 0.0) and not np.allclose(bm_std_plot.X[0,0],0.0):
            bm_std_plot.generate_paths()
        elif bm_std_plot.X.shape[1] == 0 : 
            bm_std_plot.generate_paths()

        W_std_single = bm_std_plot.X[0,:].copy() 
        minus_W_std_single = -W_std_single
        
        path_comparison_data = {
            't_axis': bm_std_plot.t_values.copy(),
            'W_t': W_std_single,
            '-W_t': minus_W_std_single,
            'path_index': 0 
        }
        
        W_T_dist_data = {'WT_values': np.array([]), 'T_param': self.T, 'note': ''}
        if self.N > 0 : 
            target_hist_paths = num_paths_for_hist
            if num_paths_for_hist <=0 : target_hist_paths = 1 
            
            hist_seed = self.seed + 2 if self.seed is not None else None
            bm_hist_helper = BrownianMotion(x0=0.0, T=self.T, N=self.N, 
                                           paths=max(1,target_hist_paths), 
                                           seed=hist_seed)
            all_std_paths_for_hist = bm_hist_helper.generate_paths()
            if all_std_paths_for_hist.shape[1] > 0: 
                W_T_dist_data['WT_values'] = all_std_paths_for_hist[:, -1] 
            
            if W_T_dist_data['WT_values'].size == 0 and target_hist_paths > 0:
                 W_T_dist_data['note'] = 'Resulting WT_values array is empty despite target_hist_paths > 0.'
            elif num_paths_for_hist <=0:
                W_T_dist_data['note'] = 'num_paths_for_hist was zero or negative.'
        else: 
            W_T_dist_data['note'] = 'Symmetry of W_T distribution not applicable for N=0.'
            
        return {
            'path_comparison': path_comparison_data,
            'WT_distribution': W_T_dist_data,
            'note': 'Demonstrates property for a standard BM (x0=0)'
        }
