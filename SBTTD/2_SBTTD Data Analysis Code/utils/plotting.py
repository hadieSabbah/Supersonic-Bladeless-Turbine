import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import re





# Defining all figure sizes #

def plotter(x, y, x_string, y_string, unit_x, unit_y, save = True, return_axes = False):
    # Set publication-quality parameters
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']  # Or 'DejaVu Serif'
    mpl.rcParams['font.size'] = 18  # Base font size
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 21
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.titlesize'] = 21

    # Line widths
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['grid.linewidth'] = 0.5

    # DPI for screen and saving
    mpl.rcParams['figure.dpi'] = 150  # Screen display
    mpl.rcParams['savefig.dpi'] = 600  # Save at high resolution
    
    # Defining labels #
    x_string_label = x_string + " " + unit_x
    y_string_label = y_string + " " + unit_y
    figure_string_title = x_string + " Vs " + y_string
    
    
    # Plotting figure # 
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(x, y)
    ax.set_title(figure_string_title)
    ax.set_xlabel(x_string_label)
    ax.set_ylabel(y_string_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save with tight layout
    plt.tight_layout()
    
    # Showing the plot # 
    if not return_axes:
        plt.show()
    
    if save == True:
        
        # Defining the directory and the figure name #
        dirc = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study"
        figName = f"{x_string}Vs{y_string}"
        
        # Saving the figures # 
        plt.savefig(rf'{dirc}\{figName}.png', dpi=600, bbox_inches='tight')
        plt.savefig(rf'{dirc}\{figName}.pdf', bbox_inches='tight')  # Vector format (best!)
    if return_axes:
        return fig, ax
        


def subplotter(nrows, ncols, x_data, y_data, x_strings, y_strings, 
               unit_x, unit_y, subplot_titles=None, figsize=None, 
               save=True, overall_title=None):
    """
    Create publication-quality subplots.
    
    Parameters
    ----------
    nrows : int
        Number of rows of subplots
    ncols : int
        Number of columns of subplots
    x_data : list of arrays
        List of x data for each subplot (length = nrows*ncols)
    y_data : list of arrays
        List of y data for each subplot (length = nrows*ncols)
    x_strings : list of str or str
        X-axis variable names. If single string, applies to all subplots
    y_strings : list of str or str
        Y-axis variable names. If single string, applies to all subplots
    unit_x : list of str or str
        X-axis units. If single string, applies to all subplots
    unit_y : list of str or str
        Y-axis units. If single string, applies to all subplots
    subplot_titles : list of str, optional
        Title for each subplot. If None, auto-generated from x_strings vs y_strings
    figsize : tuple, optional
        Figure size. If None, auto-calculated based on nrows and ncols
    save : bool, optional
        Whether to save the figure (default True)
    overall_title : str, optional
        Overall figure title (suptitle)
        
    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects
    """
    
    # Set publication-quality parameters
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 11  # Smaller for subplots
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['figure.titlesize'] = 21
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 600
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        width = 3.5 * ncols if ncols <= 2 else 7
        height = 2.5 * nrows
        figsize = (width, height)
    
    # Convert single strings to lists
    n_plots = nrows * ncols
    if isinstance(x_strings, str):
        x_strings = [x_strings] * n_plots
    if isinstance(y_strings, str):
        y_strings = [y_strings] * n_plots
    if isinstance(unit_x, str):
        unit_x = [unit_x] * n_plots
    if isinstance(unit_y, str):
        unit_y = [unit_y] * n_plots
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes array for easier iteration
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if nrows > 1 or ncols > 1 else axes
    
    # Plot each subplot
    for idx, ax in enumerate(axes_flat):
        if idx < len(x_data):
            # Create labels
            x_label = f"{x_strings[idx]} {unit_x[idx]}"
            y_label = f"{y_strings[idx]} {unit_y[idx]}"
            
            # Create title
            if subplot_titles is not None:
                title = subplot_titles[idx]
            else:
                title = f"{x_strings[idx]} vs {y_strings[idx]}"
            
            # Plot data
            ax.plot(x_data[idx], y_data[idx], 'k-', linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
        else:
            # Hide extra subplots if data doesn't fill all slots
            ax.set_visible(False)
    
    # Add overall title if provided
    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    # Save if requested
    if save:
        dirc = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study"
        
        # Create figure name from first subplot
        figName = f"{x_strings[0]}Vs{y_strings[0]}_subplots"
        
        plt.savefig(rf'{dirc}\{figName}.png', dpi=600, bbox_inches='tight')
        plt.savefig(rf'{dirc}\{figName}.pdf', bbox_inches='tight')
    
    return fig, axes
    
def plotter_multiPerCase(x_dict, y_dict, x_string, y_string, unit_x, unit_y,
                     filter_param, filter_value, vary_param='mach',
                     labels=None, cmap_name='cividis', save=False, title=None):
    """
    Plot multiple cases filtered by one parameter, showing variation in another.
    
    Parameters
    ----------
    x_dict : dict
        Dictionary of x data {case_name: x_array}
    y_dict : dict
        Dictionary of y data {case_name: y_array}
    x_string : str
        X-axis variable name
    y_string : str
        Y-axis variable name
    unit_x : str
        X-axis units
    unit_y : str
        Y-axis units
    filter_param : str
        Parameter to filter by ('h_l', 'p0', 'mach')
    filter_value : float or str
        Value of filter parameter (e.g., 0.02 for h_l, '1' for p0_1bar)
    vary_param : str, optional
        Parameter that varies ('mach', 'p0', 'h_l') - used for labeling
    labels : dict, optional
        Custom labels. If None, auto-generated from vary_param
    cmap_name : str, optional
        Colormap name (default 'cividis')
    save : bool, optional
        Whether to save the figure (default True)
    title : str, optional
        Custom title. If None, auto-generated
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
        
    Examples
    --------
    # Plot all Mach numbers at h/l = 0.02
    plotter_filtered(x, tau_x, 'x', '$\\tau_x$', '[m]', '[Pa]',
                     filter_param='h_l', filter_value=0.02, vary_param='mach')
    
    # Plot all h/l values at p0 = 1 bar
    plotter_filtered(x, P, 'x', 'P', '[m]', '[Pa]',
                     filter_param='p0', filter_value='1', vary_param='h_l')
    """
    
    # Set publication-quality parameters
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 21
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['figure.titlesize'] = 21
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 600
    
    # Filter cases based on the filter parameter
    filtered_keys = []
    
    for key in x_dict.keys():
        if filter_param == 'h_l':
            # Match h_l_0.02 pattern
            match = re.search(r'h_l_([\d.]+)', key)
            if match and float(match.group(1)) == float(filter_value):
                filtered_keys.append(key)
        
        elif filter_param == 'p0':
            # Match p0_1bar pattern
            match = re.search(r'p0_(\d+)bar', key)
            if match and match.group(1) == str(filter_value):
                filtered_keys.append(key)
        
        elif filter_param == 'mach':
            # Match mach_2.5 or M2.5 pattern
            match = re.search(r'(?:mach|M)_([\d.]+)', key)
            if match and float(match.group(1)) == float(filter_value):
                filtered_keys.append(key)
    
    if len(filtered_keys) == 0:
        print(f"No cases found with {filter_param} = {filter_value}")
        return None, None
    
    # Sort keys to get consistent ordering
    filtered_keys.sort()
    
    # Get colormap
    cmap = cm.get_cmap(cmap_name, len(filtered_keys))
    
    # Create labels
    x_string_label = f"{x_string} {unit_x}"
    y_string_label = f"{y_string} {unit_y}"
    
    # Auto-generate title if not provided
    if title is None:
        # Format filter parameter nicely (h_l -> h/l)
        if filter_param == 'h_l':
            filter_str = f"h/l = {filter_value}"
        elif filter_param == 'p0':
            filter_str = f"$P_0$ = {filter_value} bar"
        elif filter_param == 'mach':
            filter_str = f"M = {filter_value}"
        else:
            filter_str = f"{filter_param} = {filter_value}"
        
        figure_string_title = f"{x_string} vs {y_string} ({filter_str})"
    else:
        figure_string_title = title
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Plot each filtered case
    for i, key in enumerate(filtered_keys):
        # Auto-generate label if not provided
        if labels is None:
            if vary_param == 'mach':
                match = re.search(r'(?:mach|M|Mach)_?([\d.]+)', key)
                if match:
                    label = f"M = {match.group(1)}"
                else:
                    label = key
            elif vary_param == 'h_l':
                match = re.search(r'h_l_([\d.]+)', key)
                label = f"h/l = {match.group(1)}" if match else key
            elif vary_param == 'p0':
                match = re.search(r'p0_(\d+)bar', key)
                label = f"$P_0$ = {match.group(1)} bar" if match else key
            else:
                label = key
        else:
            label = labels[key] if isinstance(labels, dict) else labels[i]
        
        # Plot with color from colormap
        ax.plot(x_dict[key], y_dict[key], color=cmap(i), 
                linewidth=1.5, label=label)
    
    # Set labels and formatting
    ax.set_title(figure_string_title)
    ax.set_xlabel(x_string_label)
    ax.set_ylabel(y_string_label)
    ax.legend(frameon=False, bbox_to_anchor=(1.12,1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save if requested
    if save == True:
        dirc = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study"
        figName = f"{x_string}Vs{y_string}_{filter_param}{filter_value}"
        
        plt.savefig(rf'{dirc}\{figName}.png', dpi=600, bbox_inches='tight')
        plt.savefig(rf'{dirc}\{figName}.pdf', bbox_inches='tight')
    
    return fig, ax


def plotter_multi_all(x_dict, y_dict, x_string, y_string, unit_x, unit_y, 
                  labels=None, cmap_name='cividis', save=False, title=None):
    """
    Plot multiple cases on the same axes with different colors.
    
    Parameters
    ----------
    x_dict : dict
        Dictionary of x data {case_name: x_array}
    y_dict : dict
        Dictionary of y data {case_name: y_array}
    x_string : str
        X-axis variable name
    y_string : str
        Y-axis variable name
    unit_x : str
        X-axis units
    unit_y : str
        Y-axis units
    labels : dict or list, optional
        Custom labels for legend. If None, uses case keys
    cmap_name : str, optional
        Colormap name (default 'cividis')
    save : bool, optional
        Whether to save the figure (default True)
    title : str, optional
        Custom title. If None, auto-generated
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    
    # Set publication-quality parameters
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 21
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 9
    mpl.rcParams['figure.titlesize'] = 21
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 600
    
    # Get colormap
    case_keys = list(x_dict.keys())
    cmap = cm.get_cmap(cmap_name, len(case_keys))
    
    # Create labels
    x_string_label = f"{x_string} {unit_x}"
    y_string_label = f"{y_string} {unit_y}"
    figure_string_title = title if title else f"{x_string} vs {y_string}"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Plot each case
    for i, key in enumerate(case_keys):
        # Determine label
        if labels is not None:
            if isinstance(labels, dict):
                label = labels[key]
            else:
                label = labels[i]
        else:
            label = key
        
        # Plot with color from colormap
        ax.plot(x_dict[key], y_dict[key], color=cmap(i), 
                linewidth=2, label=label)
    
    # Set labels and formatting
    ax.set_title(figure_string_title)
    ax.set_xlabel(x_string_label)
    ax.set_ylabel(y_string_label)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save if requested
    if save:
        dirc = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study"
        figName = f"{x_string}Vs{y_string}_multi"
        
        plt.savefig(rf'{dirc}\{figName}.png', dpi=600, bbox_inches='tight')
        plt.savefig(rf'{dirc}\{figName}.pdf', bbox_inches='tight')
    
    return fig, ax