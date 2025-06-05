import matplotlib.pyplot as plt
import numpy as np # Needed for np.linspace in plot_lines color generation

def plot_lines(t_values, paths_data_list, path_labels_list, title, xlabel, ylabel, figsize=(7.5, 5), ax=None, colors=None, linestyles=None, linewidths=None, legend_loc='best', grid=True):
    """
    Centralized function to plot one or more lines.
    If ax is None, creates a new figure and axes. Otherwise, plots on the provided ax.

    Args:
        t_values (array-like): X-axis data (e.g., time).
        paths_data_list (list of array-like): List of Y-axis data series.
        path_labels_list (list of str): List of labels for each data series.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        figsize (tuple): Figure size, used if ax is None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, new figure/axes created.
        colors (list of str, optional): Colors for the lines.
        linestyles (list of str, optional): Linestyles for the lines.
        linewidths (list of float, optional): Linewidths for the lines.
        legend_loc (str): Location of the legend.
        grid (bool): Whether to display a grid.

    Returns:
        matplotlib.axes.Axes: The axes object the plot was drawn on.
    """
    if ax is None:
        fig, current_ax = plt.subplots(figsize=figsize)
    else:
        current_ax = ax
        # fig = current_ax.figure # Get the figure from the axes, not strictly needed here

    num_series = len(paths_data_list)
    
    # Provide defaults if not specified
    if colors is None:
        # Generate distinct colors if many paths, simple blue if one
        if num_series > 1:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            default_colors = prop_cycle.by_key()['color']
            colors = [default_colors[i % len(default_colors)] for i in range(num_series)]
        else:
            colors = ['blue']

    if linestyles is None:
        linestyles = ['-'] * num_series
    if linewidths is None:
        linewidths = [1.0] * num_series
    if path_labels_list is None:
        path_labels_list = [None] * num_series


    for i in range(num_series):
        current_ax.plot(t_values, paths_data_list[i],
                        label=path_labels_list[i] if i < len(path_labels_list) else None,
                        color=colors[i % len(colors)],
                        linestyle=linestyles[i % len(linestyles)],
                        linewidth=linewidths[i % len(linewidths)])

    current_ax.set_title(title)
    current_ax.set_xlabel(xlabel)
    current_ax.set_ylabel(ylabel)
    
    if grid:
        current_ax.grid(True)
    
    # Show legend only if there are actual labels provided
    if any(label for label in path_labels_list if label is not None):
        current_ax.legend(loc=legend_loc)

    # If ax was None, this implies we created the figure and might want to show it if running interactively.
    # However, in a script/notebook, plt.show() is usually called at the end.
    # For utility functions, it's often better not to call plt.show() directly.
    
    return current_ax


def plot_histogram_with_pdf(data, bins, title, xlabel, ylabel, 
                            pdf_x=None, pdf_y=None, pdf_label=None, 
                            hist_label=None, figsize=(7.5, 5), ax=None, 
                            hist_color='green', pdf_color='red', grid=True, density=True):
    """
    Centralized function to plot a histogram and optionally a PDF overlay.
    If ax is None, creates a new figure and axes. Otherwise, plots on the provided ax.
    
    Args:
        data (array-like): Data for the histogram.
        bins (int or sequence): Number of bins or bin edges.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        pdf_x (array-like, optional): X-values for the PDF overlay.
        pdf_y (array-like, optional): Y-values for the PDF overlay.
        pdf_label (str, optional): Label for the PDF line.
        hist_label (str, optional): Label for the histogram.
        figsize (tuple): Figure size, used if ax is None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, new figure/axes created.
        hist_color (str): Color of the histogram.
        pdf_color (str): Color of the PDF line.
        grid (bool): Whether to display a grid.
        density (bool): Whether the histogram should be normalized to form a probability density.

    Returns:
        matplotlib.axes.Axes: The axes object the plot was drawn on.
    """
    if ax is None:
        fig, current_ax = plt.subplots(figsize=figsize)
    else:
        current_ax = ax
        # fig = current_ax.figure

    current_ax.hist(data, bins=bins, density=density, label=hist_label, color=hist_color, alpha=0.7)
    if pdf_x is not None and pdf_y is not None:
        current_ax.plot(pdf_x, pdf_y, color=pdf_color, lw=2, label=pdf_label)

    current_ax.set_title(title)
    current_ax.set_xlabel(xlabel)
    current_ax.set_ylabel(ylabel)
    
    if grid:
        current_ax.grid(True)
    
    if hist_label or pdf_label:
        current_ax.legend(loc='best')
    
    return current_ax


# --- File: financial_toolkit/stochastic_processes/__init__.py ---
# This file can be empty. It indicates that 'stochastic_processes' is a Python package.


# --- File: financial_toolkit/stochastic_processes/brownian_motion.py ---
import numpy as np
import matplotlib.pyplot as plt # Still needed for plt.subplots, plt.show, plt.suptitle
from scipy.stats import norm
# Assuming plotting.py is in a directory 'utils' at the same level as 'stochastic_processes'
# For package structure: from ..utils.plotting import plot_lines, plot_histogram_with_pdf
# If running this file directly for testing, you might need to adjust sys.path or use a simpler import
# For this example, let's assume the structure is handled by the execution environment (e.g. IDE, PYTHONPATH)
# If utils is in the same directory (not typical for packages but simpler for single file context):
try:
    from utils.plotting import plot_lines, plot_histogram_with_pdf
except ImportError: # Fallback for direct execution if utils is a sibling directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add parent directory
    from utils.plotting import plot_lines, plot_histogram_with_pdf


class BrownianMotion:
    """
    Implements 1D Brownian Motion (Wiener Process) and demonstrates its properties.
    Relies on centralized plotting functions from utils.plotting.
    """
    def __init__(self, T=10.0, N=10000, paths=1, seed=None):
        if N <= 1:
            raise ValueError("N (number of time points) must be greater than 1.")
        self.T = float(T)
        self.N = int(N)
        self.paths = int(paths)
        self.seed = seed
        self.t_values = np.linspace(0, self.T, self.N)
        self.dt = self.T / (self.N - 1)
        self.sqdt = np.sqrt(self.dt)
        self.W = np.zeros((self.paths, self.N))
        if self.seed is not None:
            np.random.seed(self.seed)

    def generate_paths(self):
        for i in range(self.paths):
            increments = self.sqdt * np.random.standard_normal(self.N - 1)
            self.W[i, 0] = 0.0
            self.W[i, 1:] = np.cumsum(increments)
        return self.W

    def plot_generated_paths(self, num_to_plot=None, title_suffix="", figsize=(7.5, 5)):
        """
        Plots a selection of the generated Brownian motion paths using the centralized plotting function.
        """
        if self.W.shape[1] == 0 or np.all(self.W == 0):
             print("Paths not generated or all zero. Call generate_paths() first.")
             # self.generate_paths() # Optionally generate if not done
             return

        if num_to_plot is None:
            num_to_plot = min(self.paths, 5) # Default to plotting up to 5 paths
        else:
            num_to_plot = min(num_to_plot, self.paths)

        paths_to_plot_data = [self.W[i, :] for i in range(num_to_plot)]
        labels = [f'Path {i+1}' for i in range(num_to_plot)] if num_to_plot > 1 else [None]
        
        plot_title = f"{num_to_plot} Brownian Motion Path(s)"
        if title_suffix:
            plot_title += f" - {title_suffix}"
        
        plot_lines(self.t_values, 
                   paths_data_list=paths_to_plot_data, 
                   path_labels_list=labels,
                   title=plot_title, 
                   xlabel="Time (t)", 
                   ylabel="W(t)", 
                   figsize=figsize)
        plt.show() # Show the plot after calling the utility

    def demonstrate_self_similarity(self, path_index=0, c_values=None, figsize=(7.5, 5)):
        if path_index < 0 or path_index >= self.paths:
            raise ValueError(f"path_index must be between 0 and {self.paths - 1}")
        if self.W.shape[1] == 0 or np.all(self.W[path_index,:] == 0):
            print("Paths not generated. Call generate_paths() first.")
            self.generate_paths()

        W_orig = self.W[path_index, :]
        t_orig = self.t_values
        if c_values is None:
            c_values = [0.25, 0.5, 2.0, 4.0]

        num_subplots = len(c_values) + 1
        fig_height_per_plot = figsize[1] * 0.7 # Adjusted for better spacing
        fig, axes = plt.subplots(num_subplots, 1, 
                                 figsize=(figsize[0], fig_height_per_plot * num_subplots), 
                                 sharex=False)
        if num_subplots == 1: axes = [axes] 

        plot_lines(ax=axes[0], t_values=t_orig, paths_data_list=[W_orig], 
                   path_labels_list=[f'$W_t$ (Original Path {path_index+1})'],
                   title=f'Original Brownian Motion $W_t$', 
                   xlabel='Time $t \\in [0, T_{orig}]$', ylabel='$W_t$')

        for i, c_val in enumerate(c_values):
            ax_current = axes[i+1]
            if c_val <= 0:
                ax_current.text(0.5, 0.5, f'c={c_val:.2f} is invalid (must be > 0)', ha='center', va='center')
                ax_current.set_title(f'Invalid c={c_val:.2f}')
                ax_current.set_xticks([])
                ax_current.set_yticks([])
                continue

            t_V_domain_end = self.T / c_val
            s_values_for_V = np.linspace(0, t_V_domain_end, self.N)
            args_for_W = c_val * s_values_for_V
            W_at_cs = np.interp(args_for_W, t_orig, W_orig)
            V_s = (1 / np.sqrt(c_val)) * W_at_cs
            
            plot_lines(ax=ax_current, t_values=s_values_for_V, paths_data_list=[V_s],
                       path_labels_list=[f'$V_s = (1/\\sqrt{{{c_val:.2f}}}) W_{{{c_val:.2f}s}}$'],
                       title=f'Scaled Process $V_s$ with $c={c_val:.2f}$',
                       xlabel=f'Time $s \\in [0, {t_V_domain_end:.2f}]$', ylabel='$V_s$')
        
        plt.suptitle("Self-Similarity (Brownian Scaling)", fontsize=14, y=1.0) # Adjust y for suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust for suptitle
        plt.show()

    def demonstrate_time_reversal(self, path_index=0, figsize=(7.5, 5)):
        if path_index < 0 or path_index >= self.paths:
            raise ValueError(f"path_index must be between 0 and {self.paths - 1}")
        if self.W.shape[1] == 0 or np.all(self.W[path_index,:] == 0):
            self.generate_paths()

        W_orig = self.W[path_index, :]
        t_orig = self.t_values
        W_T_minus_t_values = W_orig[::-1]
        V_t = W_orig[-1] - W_T_minus_t_values

        fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.2), sharex=True, sharey=True)
        plot_lines(ax=axes[0], t_values=t_orig, paths_data_list=[W_orig],
                   path_labels_list=[f'$W_t$ (Original Path {path_index+1})'],
                   title='Original Brownian Motion $W_t$', xlabel=None, ylabel='$W_t$')
        plot_lines(ax=axes[1], t_values=t_orig, paths_data_list=[V_t],
                   path_labels_list=['$V_t = W_T - W_{T-t}$'],
                   title='Time-Reversed Process $V_t$', xlabel='Time $t$', ylabel='$V_t$')
        
        plt.suptitle("Time Reversal Property", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def demonstrate_time_inversion(self, path_index=0, figsize=(7.5, 5)):
        if path_index < 0 or path_index >= self.paths:
            self.generate_paths()

        t_for_V = self.t_values[1:]
        max_arg_for_W_base = (self.N - 1) / self.T
        base_bm_seed = self.seed + 1 if self.seed is not None else None
        bm_base = BrownianMotion(T=max_arg_for_W_base, N=self.N, paths=1, seed=base_bm_seed)
        W_base_path = bm_base.generate_paths()[0, :]
        t_base = bm_base.t_values
        args_for_W_interpolation = 1.0 / t_for_V
        W_one_over_t = np.interp(args_for_W_interpolation, t_base, W_base_path)
        V_t_values_nonzero_t = t_for_V * W_one_over_t
        V_t_final = np.concatenate(([0.0], V_t_values_nonzero_t))

        fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.2), sharex=True, sharey=True)
        plot_lines(ax=axes[0], t_values=self.t_values, paths_data_list=[self.W[path_index, :]],
                   path_labels_list=[f'$W_t$ (Original Path {path_index+1})'],
                   title='Original Brownian Motion $W_t$', xlabel=None, ylabel='$W_t$')
        plot_lines(ax=axes[1], t_values=self.t_values, paths_data_list=[V_t_final],
                   path_labels_list=['$V_t = t W_{1/t}$ (with $V_0=0$)'],
                   title='Time-Inverted Process $V_t$', xlabel='Time $t$', ylabel='$V_t$')

        plt.suptitle("Time Inversion Property", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def demonstrate_symmetry(self, path_index_for_plot=0, num_paths_for_hist=10000, figsize=(7.5, 5)):
        if path_index_for_plot < 0 or path_index_for_plot >= self.paths:
            self.generate_paths()
        
        W_orig_single = self.W[path_index_for_plot, :]
        minus_W_single = -W_orig_single
        
        if self.paths < num_paths_for_hist:
            temp_seed = self.seed + 2 if self.seed is not None else None
            bm_temp = BrownianMotion(T=self.T, N=self.N, paths=num_paths_for_hist, seed=temp_seed)
            W_T_values = bm_temp.generate_paths()[:, -1]
        else:
            W_T_values = self.W[:num_paths_for_hist, -1]

        fig_width = figsize[0]
        fig_height = figsize[1]
        fig, axes = plt.subplots(1, 2, figsize=(fig_width * 1.8, fig_height * 1.1))

        plot_lines(ax=axes[0], t_values=self.t_values, 
                   paths_data_list=[W_orig_single, minus_W_single],
                   path_labels_list=[f'$W_t$ (Path {path_index_for_plot+1})', f'$-W_t$'],
                   title='$W_t$ and its negative $-W_t$', xlabel='Time $t$', ylabel='Value',
                   colors=['blue', 'orange']) # Specify colors for clarity

        mu_theory = 0
        sigma_theory = np.sqrt(self.T)
        x_norm = np.linspace(mu_theory - 4*sigma_theory, mu_theory + 4*sigma_theory, 200)
        pdf_theory = norm.pdf(x_norm, mu_theory, sigma_theory)
        
        plot_histogram_with_pdf(ax=axes[1], data=W_T_values, bins=75,
                                title=f'Distribution of $W_T$ (T={self.T:.1f})',
                                xlabel='$W_T$ Value', ylabel='Density',
                                hist_label=f'Simulated $W_T$\n({len(W_T_values)} paths)',
                                pdf_x=x_norm, pdf_y=pdf_theory, 
                                pdf_label=f'Theoretical $N(0, T={self.T:.1f})$')

        plt.suptitle("Symmetry Properties", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


