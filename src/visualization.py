"""
Visualization functionality for wall models
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import r2_score
import torch
import mplcursors


black   = "#000000"
blue    = "#065279"
red     = "#9d2933"
orange  = "#ca6924"
cyan    = "#426666"
purple  = "#686789"
yellow  = "#c89b40"
green   = "#75878a"
grey    = "#36282b"

plt.rcParams.update({
    "text.usetex": True,
})
plt.rcParams['axes.labelsize'] = 32  # Adjust as desired
plt.rcParams['xtick.labelsize'] = 24  # Adjust as desired
plt.rcParams['ytick.labelsize'] = 24  # Adjust as desired
plot_size = (12, 12)  # Default plot size
markersize = 18  # Default marker size
cmap = 'viridis'

class WallModelVisualization:
    """
    Handles visualization and result analysis for wall models
    """
    
    def __init__(self, dataset_labels: Optional[Dict[str, str]] = None):
        """
        Initialize visualization manager
        
        Args:
            dataset_labels: Optional dictionary mapping dataset keys to human-readable labels
        """
        self.dataset_labels = dataset_labels or {}
        
    
    def plot_training_results(self, 
                             output_train: np.ndarray, 
                             outputs_train_predict: np.ndarray,
                             output_valid: np.ndarray,
                             outputs_valid_predict: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot the regression results for training and validation data
        
        Args:
            output_train: True outputs for training data
            outputs_train_predict: Predicted outputs for training data
            output_valid: True outputs for validation data
            outputs_valid_predict: Predicted outputs for validation data
            save_path: Optional path to save the figure
        """
        # Convert to numpy if tensors
        if torch.is_tensor(output_train):
            output_train = output_train.cpu().detach().numpy()
        if torch.is_tensor(outputs_train_predict):
            outputs_train_predict = outputs_train_predict.cpu().detach().numpy()
        if torch.is_tensor(output_valid):
            output_valid = output_valid.cpu().detach().numpy()
        if torch.is_tensor(outputs_valid_predict):
            outputs_valid_predict = outputs_valid_predict.cpu().detach().numpy()
        
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        
        # Plot training data
        ax[0].scatter(output_train, outputs_train_predict, label="Data", rasterized=True, color=blue, s=8.5)
        
        # Plot validation data
        ax[1].scatter(output_valid, outputs_valid_predict, label="Data", rasterized=True, color=blue, s=8.5)
        
        # Calculate R² scores
        r2_train = r2_score(output_train, outputs_train_predict)
        r2_valid = r2_score(output_valid, outputs_valid_predict)
        
        # Add titles
        ax[0].set_title(f'Training; $R^2: {r2_train:.2f}$')
        ax[1].set_title(f'Validation; $R^2: {r2_valid:.2f}$')
        
        # Add labels and diagonal line
        for a in ax:
            xlim = a.get_xlim()
            a.set_xlabel(r'True $\Pi_{\text{out}}$')
            a.set_ylabel(r'Predicted $\Pi_{\text{out}}$')
            a.plot(xlim, xlim, 'r--')
            a.legend()
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f"{save_path}/regression_results.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/regression_results.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_results_scatter_error(self, 
                                 inputs: np.ndarray,
                                 output: np.ndarray,
                                 output_true: np.ndarray,
                                 dataset: str,
                                 log_scale: bool = False,
                                 abs_err: bool = False,
                                 bin_loc: Tuple[float, float, float, float] = (0.4, 0.1, 0.5, 0.35),
                                 unnormalized_inputs: Optional[np.ndarray] = None,
                                 flow_type: Optional[np.ndarray] = None,
                                 weighted_utau: bool = False,
                                 tauw: bool = False,
                                 save_path: Optional[str] = None,
                                 mask_threshold: Optional[float] = None) -> Tuple[Optional[float], float, float, float, float]:
        """
        Plot scatter plot colored by error
        
        Args:
            inputs: Input features
            output: Model predictions
            output_true: True values
            dataset: Dataset name
            log_scale: Whether to use log scale for error
            abs_err: Whether to use absolute error
            bin_loc: Location for histogram inset
            unnormalized_inputs: Unnormalized input features
            flow_type: Flow type information
            weighted_utau: Whether to weight by friction velocity
            tauw: Whether to plot wall shear stress
            save_path: Optional path to save plots
            mask_threshold: Optional threshold for masking
            
        Returns:
            Tuple of (max_abs_error, mean_abs_error, std_abs_error, mean_rel_error, std_rel_error)
        """
        if inputs.shape[1] == 1:
            inputs = np.hstack((inputs, inputs))
        
        # Calculate tauw predictions
        nu = unnormalized_inputs[:,2]
        y  = unnormalized_inputs[:,0]
        utau = unnormalized_inputs[:,3]

        # NOTE: here that outputs have been normalized by nu/y
        output_tauw = output ** 2
        # Calculate wall shear stress
        output_true_tauw = output_true ** 2

        # Apply masking if requested
        if mask_threshold is not None:
            kept_idx = np.where(np.abs(output_true_tauw) > mask_threshold)
            other_idx = np.where(np.abs(output_true_tauw) <= mask_threshold)
            
            # Plot separation region if needed
            if len(other_idx[0]) > 0:
                inputs_sep = inputs[other_idx]
                output_tauw = output_tauw[other_idx]
                output_true_tauw = output_true_tauw[other_idx]
                flow_type_sep = flow_type[other_idx]
                unnormalized_inputs_sep = unnormalized_inputs[other_idx]
                
                # Calculate absolute error for separation region
                err_abs = np.abs(output_tauw - output_true_tauw).squeeze()
                
                # Plot separation region results
                fig_abs, ax_abs = plt.subplots(figsize=plot_size)
                sc_abs = ax_abs.scatter(inputs_sep[:, 0], inputs_sep[:, 1], c=err_abs, # edgecolors=black, linewidths=0.05,
                                    cmap=cmap, s=markersize, rasterized=True)
                # if save_path is None:
                #     plt.colorbar(sc_abs, ax=ax_abs, label='Absolute Error', orientation='vertical')
                
                # Set labels and title
                ax_abs.set_xlabel(r'$u_1n_1/\nu$')
                ax_abs.set_ylabel(r'$u_pn_1/\nu$')
                if dataset is not None and save_path is None:
                    ax_abs.set_title(f'{self.dataset_labels.get(dataset, dataset)} using BFM \n (near separation) [Mean error: {np.mean(err_abs):.2e}]')
                
                # Save or show plot
                if save_path is not None:
                    case_name = dataset.replace("-", "")
                    fig_abs.savefig(f"{save_path}/{case_name}_bfm_abs.pdf", dpi=300)
                    fig_abs.savefig(f"{save_path}/{case_name}_bfm_abs.png", dpi=300)
                    plt.close(fig_abs)
                else:
                    fig_abs.show()
            
            # Keep non-separation indices
            inputs = inputs[kept_idx]
            output = output[kept_idx]
            output_true = output_true[kept_idx]
            flow_type = flow_type[kept_idx]
            unnormalized_inputs = unnormalized_inputs[kept_idx]

        # Create figure
        # fig, ax = plt.subplots(figsize=plot_size)
        fig = plt.figure(figsize=plot_size) # Adjust figure size as needed
        # NOTE: Create gridspec for better layout
        # Top: Histogram inset; Bottom: Scatter plot
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.25) # Increased hspace slightly

        axins = fig.add_subplot(gs[0, 0]) # NOTE: Histogram
        ax      = fig.add_subplot(gs[1, 0]) # NOTE: Scatter plot
        
        # Calculate errors
        if abs_err:
            err = np.abs(output - output_true).squeeze()
            if weighted_utau and flow_type is not None:
                utau = np.array([float(flow_type[i, 1]) for i in range(len(flow_type))])
                err = err * utau
        else:
            err = np.abs((output - output_true) * 100 / output_true).squeeze()
            if weighted_utau and flow_type is not None:
                utau = np.array([float(flow_type[i, 1]) for i in range(len(flow_type))])
                err = err * utau
        
        # Plot scatter
        sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=err, cmap=cmap, s=markersize,  rasterized=True, # edgecolors=black, linewidths=0.05,
                        norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=100))
        
        # Add colorbar
        # if save_path is None:
        #     if abs_err:
        #         plt.colorbar(sc, ax=ax, label='Absolute Error', orientation='vertical')
        #     else:
        #         plt.colorbar(sc, ax=ax, label='Relative Error (%)', orientation='vertical')

        ##############################################################
        # NOTE: Add hover information
        cursor = mplcursors.cursor(sc, hover=True)
        y_delta = unnormalized_inputs[:,0]/np.array([float(flow_type[i, 3]) for i in range(len(flow_type))])
        cursor.connect("add", lambda sel: sel.annotation.set_text(f'True,Pre,Err: {output_true[sel.index]:.3e},{output[sel.index]:.3e},{err[sel.index]:.2e}% \n at x,y/delta: {float(flow_type[sel.index,2]):.2e},{y_delta[sel.index]:.3f}; utau {unnormalized_inputs[sel.index,3]:.5e}; up {unnormalized_inputs[sel.index,4]:.5e}'))
        ##############################################################
        
        # Create histogram inset
        # axins = inset_axes(ax, width="80%", height="70%", 
        #                   bbox_to_anchor=bin_loc,
        #                   bbox_transform=ax.transAxes)
        
        # Add semi-opaque background
        # axins.patch.set_facecolor('white')
        # axins.patch.set_alpha(0.9)
        
        if err is not None and err.ndim > 0 and len(err) > 0:
            # Plot histogram
            sorted_err = np.sort(err)
            counts, bins, _ = axins.hist(sorted_err, bins=50, color=purple, 
                                        weights=np.zeros_like(sorted_err) + 100. / sorted_err.size, 
                                        range=(0, 100))
            
            # Set histogram properties
            axins.set_xticks(np.linspace(0, 100, 11))
            axins.set_xlim(0, 100)
            axins.set_xlabel(rf'Relative Error (\%)', fontsize=24)
            axins.set_ylabel(rf'Relative\\ Frequency (\%)', fontsize=20)
            max_count = np.ceil(max(counts))
            axins.set_yticks(np.linspace(0, max_count, 6))

            axins.tick_params(axis='x', labelsize=13) # Even smaller tick labels for inset
            axins.tick_params(axis='y', labelsize=13)
            
            # Calculate statistics
            n_outside = np.sum(err > 100)
            outside_percent = n_outside / len(err) * 100
            below_20_percent = np.mean(err <= 20) * 100
            
            # Add annotations
            if outside_percent > 0.01:
                axins.text(0.8, 0.75, fr'$>{100}\%: {outside_percent:.1f}\%$',
                        transform=axins.transAxes, fontsize=18, fontweight='bold',
                        bbox=dict(facecolor='none', alpha=0.5, edgecolor='red'))
            
            axins.axvline(x=20, color=green, linestyle='--', alpha=0.9)
            axins.text(22, axins.get_ylim()[1] * 0.7, fr'$\le 20\%: {below_20_percent:.1f}\%$',
                    fontsize=18, fontweight='bold',
                    bbox=dict(facecolor='none', alpha=0.7, edgecolor=green))
            
            # Set labels and title
            ax.set_xlabel(rf'$u_1n_1/\nu$')
            ax.set_ylabel(rf'$u_pn_1/\nu$')
            # if dataset is not None and save_path is None:
            #     ax.set_title(f'{self.dataset_labels.get(dataset, dataset)}')
        
        # Save or show plot
        if save_path is not None:
            case_name = dataset.replace("-", "")
            plt.tight_layout()
            fig.savefig(f"{save_path}/{case_name}_wm.pdf", dpi=300, bbox_inches='tight')
            fig.savefig(f"{save_path}/{case_name}_wm.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:

            fig.show()
        
        # Return metrics
        if mask_threshold is not None and len(other_idx[0]) > 0:
            if len(kept_idx[0]) > 0:
                return np.max(err_abs), np.mean(err_abs), np.std(err_abs), np.mean(err), np.std(err)
            else:
                return np.max(err_abs), np.mean(err_abs), np.std(err_abs), 0, 0
        else:
            return None, 0, 0, np.mean(err), np.std(err)
    
    def plot_results_scatter_error_loglaw(self, 
                                        inputs: np.ndarray,
                                        log_predictions: np.ndarray,
                                        output_true: np.ndarray,
                                        dataset: str,
                                        log_scale: bool = False,
                                        bin_loc: Tuple[float, float, float, float] = (0.4, 0.1, 0.5, 0.35),
                                        unnormalized_inputs: Optional[np.ndarray] = None,
                                        flow_type: Optional[np.ndarray] = None,
                                        save_path: Optional[str] = None,
                                        mask_threshold: Optional[float] = None,
                                        max_model_err: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        Plot scatter plot colored by error for log law baseline
        
        Args:
            inputs: Input features
            output_true: True values
            dataset: Dataset name
            log_scale: Whether to use log scale for error
            bin_loc: Location for histogram inset
            unnormalized_inputs: Unnormalized input features
            flow_type: Flow type information
            save_path: Optional path to save plots
            mask_threshold: Optional threshold for masking
            max_model_err: Maximum error from model for comparison
            
        Returns:
            Tuple of (mean_abs_error, std_abs_error, mean_rel_error, std_rel_error)
        """
        if inputs.shape[1] == 1:
            inputs = np.hstack((inputs, inputs))

        # Extract flow parameters
        utau = unnormalized_inputs[:, 3]
        
        # Calculate log law predictions
        output_log = log_predictions
        
        # Calculate wall shear stress
        output_true_tauw = (utau)**2
        
        # Apply masking if requested
        if mask_threshold is not None:
            kept_idx = np.where(np.abs(output_true_tauw) > mask_threshold)
            other_idx = np.where(np.abs(output_true_tauw) <= mask_threshold)
            
            # Plot separation region if needed
            if len(other_idx[0]) > 0:
                inputs_sep = inputs[other_idx]
                output_log_sep = output_log[other_idx]
                output_true_sep = output_true_tauw[other_idx]
                flow_type_sep = flow_type[other_idx]
                unnormalized_inputs_sep = unnormalized_inputs[other_idx]
                
                # Calculate absolute error for separation region
                err_abs = np.abs(output_log_sep - output_true_sep).squeeze()
                
                # Plot separation region results
                fig_abs, ax_abs = plt.subplots(figsize=plot_size)
                sc_abs = ax_abs.scatter(inputs_sep[:, 0], inputs_sep[:, 1], c=err_abs, rasterized=True, # edgecolors=black, linewidths=0.05,
                                       cmap=cmap, s=markersize, vmax=max_model_err)
                # if save_path is None:
                #     plt.colorbar(sc_abs, ax=ax_abs, label='Absolute Error', orientation='vertical')
                
                # Set labels and title
                ax_abs.set_xlabel(r'$u_1n_1/\nu$')
                ax_abs.set_ylabel(r'$u_pn_1/\nu$')
                if dataset is not None and save_path is None:
                    ax_abs.set_title(f'{self.dataset_labels.get(dataset, dataset)} using Log Law \n (near separation) [Mean error: {np.mean(err_abs):.2e}]')
                
                # Save or show plot
                if save_path is not None:
                    case_name = dataset.replace("-", "")
                    plt.tight_layout()
                    fig_abs.savefig(f"{save_path}/{case_name}_loglaw_abs.pdf", dpi=300, bbox_inches='tight')
                    fig_abs.savefig(f"{save_path}/{case_name}_loglaw_abs.png", dpi=300, bbox_inches='tight')
                    plt.close(fig_abs)
                else:
                    fig_abs.show()
            
            # Keep non-separation indices
            inputs = inputs[kept_idx]
            output_log = output_log[kept_idx]
            output_true_tauw = output_true_tauw[kept_idx]
            flow_type = flow_type[kept_idx]
            unnormalized_inputs = unnormalized_inputs[kept_idx]

        # Create figure
        # fig, ax = plt.subplots(figsize=plot_size)
        fig = plt.figure(figsize=plot_size) # Adjust figure size as needed
        # NOTE: Create gridspec for better layout
        # Top: Histogram inset; Bottom: Scatter plot
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.25) # Increased hspace slightly

        axins = fig.add_subplot(gs[0, 0]) # NOTE: Histogram
        ax      = fig.add_subplot(gs[1, 0]) # NOTE: Scatter plot
        
        # Calculate relative error
        err = np.abs((output_log - output_true_tauw) * 100 / output_true_tauw).squeeze()
        
        if err is not None and err.ndim > 0 and len(err) > 0:
            # Plot scatter
            sc = ax.scatter(inputs[:, 0], inputs[:, 1], c=err, cmap=cmap, s=markersize, rasterized=True, # edgecolors=black, linewidths=0.05
                        norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=100))
            
            # Plot histogram
            sorted_err = np.sort(err)
            counts, bins, _ = axins.hist(sorted_err, bins=50, color=purple, 
                                        weights=np.zeros_like(sorted_err) + 100. / sorted_err.size, 
                                        range=(0, 100))
            
            # Set histogram properties
            axins.set_xticks(np.linspace(0, 100, 11))
            axins.set_xlim(0, 100)
            axins.set_xlabel(rf'Relative Error (\%)', fontsize=24)
            axins.set_ylabel(rf'Relative\\ Frequency (\%)', fontsize=20)
            max_count = np.ceil(max(counts))
            axins.set_yticks(np.linspace(0, max_count, 6))
            
            axins.tick_params(axis='x', labelsize=13) # Even smaller tick labels for inset
            axins.tick_params(axis='y', labelsize=13)
            
            # Calculate statistics
            n_outside = np.sum(err > 100)
            outside_percent = n_outside / len(err) * 100
            below_20_percent = np.mean(err <= 20) * 100
            
            # Add annotations
            if outside_percent > 0.005:
                axins.text(0.8, 0.75, fr'$>{100}\%: {outside_percent:.1f}\%$',
                        transform=axins.transAxes, fontsize=18, fontweight='bold',
                        bbox=dict(facecolor='none', alpha=0.5, edgecolor='red'))
            
            axins.axvline(x=20, color=green, linestyle='--', alpha=0.9)
            axins.text(22, axins.get_ylim()[1] * 0.7, fr'$\le 20\%: {below_20_percent:.1f}\%$',
                    fontsize=18, fontweight='bold',
                    bbox=dict(facecolor='none', alpha=0.7, edgecolor=green))
            
            # Set labels and title
            ax.set_xlabel(rf'$u_1n_1/\nu$')
            ax.set_ylabel(rf'$u_pn_1/\nu$')
        

        # Save or show plot
        if save_path is not None:
            case_name = dataset.replace("-", "")
            plt.tight_layout()
            fig.savefig(f"{save_path}/{case_name}_loglaw.pdf", dpi=300, bbox_inches='tight')
            fig.savefig(f"{save_path}/{case_name}_loglaw.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:

            fig.show()
        
        # Return metrics
        if mask_threshold is not None and len(other_idx[0]) > 0:
            if len(kept_idx[0]) > 0:
                return np.mean(err_abs), np.std(err_abs), np.mean(err), np.std(err)
            else:
                return np.mean(err_abs), np.std(err_abs), 0, 0
        else:
            return 0, 0, np.mean(err), np.std(err)
    
    def plot_results_fixed_height(self, 
                                fixed_height: float,
                                output_pred: np.ndarray,
                                output_true: np.ndarray,
                                dataset: str,
                                unnormalized_inputs: np.ndarray,
                                flow_type: np.ndarray,
                                save_path: Optional[str] = None,
                                abs_err: bool = False) -> None:
        """
        Plot results at a fixed height ratio
        
        Args:
            fixed_height: Fixed height ratio
            output_pred: Model predictions
            output_true: True values
            dataset: Dataset name
            unnormalized_inputs: Unnormalized input features
            flow_type: Flow type information
            save_path: Optional path to save plots
            abs_err: Whether to use absolute error
        """
        # Create figure
        fig, ax = plt.subplots(figsize=plot_size)
        
        # Extract flow parameters
        nu = unnormalized_inputs[:, 2]
        u = unnormalized_inputs[:, 1]
        y = unnormalized_inputs[:, 0]
        utau = unnormalized_inputs[:, 3]
        
        # Find corresponding y values
        delta = np.array([float(flow_type[i, 3]) for i in range(len(flow_type))])
        fixed_height_delta = fixed_height * delta
        
        # Find unique x values
        x = np.array([float(flow_type[i, 2]) for i in range(len(flow_type))])
        x_unique = np.sort(np.unique(x))
        
        # Find samples at fixed height
        y_idx = np.zeros_like(x_unique, dtype=int)
        for x_idx, x_val in enumerate(x_unique):
            sample_idx = np.argwhere(x == x_val).squeeze()
            if len(sample_idx) > 0:
                y_idx[x_idx] = sample_idx[np.argmin(np.abs(y[sample_idx] - fixed_height_delta[sample_idx]))]
        
        # Extract samples at fixed height
        output_true_fixed = output_true[y_idx]
        output_pred_fixed = output_pred[y_idx]
        y_fixed = y[y_idx]
        nu_fixed = nu[y_idx]
        u_fixed = u[y_idx]
        utau_fixed = utau[y_idx]
        
        # Calculate log law predictions
        output_log = np.zeros_like(output_true_fixed)
        for idx in range(len(output_true_fixed)):
            output_log[idx] = (self._eqwm_solve(y_fixed[idx], nu_fixed[idx], abs(u_fixed[idx])))**2
            output_pred_fixed[idx] = (output_pred_fixed[idx] * nu_fixed[idx] / y_fixed[idx])**2
        
        # Calculate wall shear stress
        output_true_tauw = (utau_fixed)**2
        
        # Calculate errors
        if abs_err:
            err_log = np.abs(output_log - output_true_tauw).squeeze()
            err_pred = np.abs(output_pred_fixed - output_true_tauw).squeeze()
        else:
            err_log = np.abs((output_log - output_true_tauw) * 100 / output_true_tauw).squeeze()
            err_pred = np.abs((output_pred_fixed - output_true_tauw) * 100 / output_true_tauw).squeeze()
        
        # Plot errors
        ax.plot(x_unique, err_log, '-o', color='green', label='Log Law')
        ax.plot(x_unique, err_pred, '-o', color='blue', label='Model')
        ax.legend()

        
        # Set labels and title
        if abs_err:
            ax.set_ylabel(r'Absolute Error ($|\tau_{w, \text{true}} - \tau_{w, \text{pred}}|$)')
        else:
            ax.set_ylabel(r'Relative Error (%)')
            ax.set_ylim(-5, 20)
        
        if dataset is not None:
            ax.set_title(fr'{self.dataset_labels.get(dataset, dataset)} with matching location $y/\delta$ = {fixed_height}')
        
        if 'naca' in dataset:
            ax.set_xlabel(r'$x/c$')
        else:
            ax.set_xlabel(r'$x/\theta_{in}$')
        
        # Save or show plot
        if save_path is not None:
            case_name = dataset.replace("-", "")
            plt.tight_layout()
            plt.savefig(f"{save_path}/{case_name}_y_{fixed_height}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/{case_name}_y_{fixed_height}.png", dpi=200, bbox_inches='tight')
            plt.close()
        else:
            fig.show()
    
    def plot_comparison_bar_chart(self, 
                                model_mean: float,
                                model_std: float,
                                loglaw_mean: float,
                                loglaw_std: float,
                                dataset: str,
                                save_path: Optional[str] = None) -> None:
        """
        Plot comparison bar chart between model and log law
        
        Args:
            model_mean: Mean error for model
            model_std: Standard deviation of error for model
            loglaw_mean: Mean error for log law
            loglaw_std: Standard deviation of error for log law
            dataset: Dataset name
            save_path: Optional path to save plots
        """
        # Create figure
        fig_bar, ax = plt.subplots(figsize=(6, 6))
        
        # Plot bars
        ax.bar(['Model', 'Log Law'], [model_mean, loglaw_mean], 
               yerr=[model_std, loglaw_std], 
               color=[blue, purple], 
               error_kw=dict(capsize=5))
        
        # Set y-axis limit
        # ax.set_ylim(0, max(10, max(model_mean + model_std, loglaw_mean + loglaw_std)))
        # ax.set_ylim(0, max(10, max(model_mean, loglaw_mean)))  # IH: removed std for better visibility
        ax.set_ylim(0, 100)  # IH: removed std for better visibility

        # Set labels and title
        ax.set_ylabel(rf'Mean relative error (\%)')
        ax.set_title(f'{self.dataset_labels.get(dataset, dataset)}')
        
        # Save or show plot
        if save_path is not None:
            case_name = dataset.replace("-", "")
            plt.tight_layout()
            plt.savefig(f"{save_path}/{case_name}_comparison.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}/{case_name}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            fig_bar.show()
