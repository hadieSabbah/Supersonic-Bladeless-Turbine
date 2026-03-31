import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pandas as pd
import re
from pathlib import Path
import tecplot as tp
from tecplot.constant import PlotType, GeomShape, Color


# Defining all figure sizes #

def plotter(x, y, x_string, y_string, unit_x, unit_y, save = False, return_axes = False):
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
    mpl.rcParams['figure.dpi'] = 600  # Screen display
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

        

def plot_BL_thickness_subplots(delta_n_dict, x_start_dict, save=False,
                                save_dir=Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study")):
    """
    All h/l cases in one figure — one subplot per h/l, curves = Mach numbers.

    Parameters
    ----------
    delta_n_dict  : dict  {case_key: np.ndarray}  BL thickness [mm]
    x_start_dict  : dict  {case_key: np.ndarray}  rake x-positions [m]
    save          : bool
    save_dir      : Path
    """

    mpl.rcParams['font.family']     = 'serif'
    mpl.rcParams['font.serif']      = ['Times New Roman']
    mpl.rcParams['font.size']       = 24
    mpl.rcParams['axes.labelsize']  = 34
    mpl.rcParams['axes.titlesize']  = 34
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['figure.dpi']      = 1200
    mpl.rcParams['savefig.dpi']     = 600
    mpl.rcParams['axes.linewidth']  = 1
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['grid.linewidth']  = 0.5

    # --- Extract and sort unique h/l values ---
    hl_set = set()
    for key in delta_n_dict:
        m = re.search(r'h_l_([\d.x]+)', key)
        if m:
            hl_set.add(m.group(1))
    hl_values = sorted(hl_set, key=lambda v: float('inf') if v == 'x' else float(v))

    n_hl  = len(hl_values)
    ncols = 3
    nrows = int(np.ceil(n_hl / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat  = axes.flatten()

    for ax_idx, hl in enumerate(hl_values):
        ax = axes_flat[ax_idx]

        keys_for_hl = sorted(
            [k for k in delta_n_dict if re.search(rf'h_l_{re.escape(hl)}_', k)],
            key=lambda k: float(re.search(r'Mach_([\d.]+)', k).group(1))
        )

        n_curves = len(keys_for_hl)
        cmap     = cm.get_cmap('cividis', n_curves)

        for i, key in enumerate(keys_for_hl):
            mach_match = re.search(r'Mach_([\d.]+)', key)
            label      = f"M = {mach_match.group(1)}" if mach_match else key
            ax.plot(x_start_dict[key], delta_n_dict[key], color=cmap(i), label=label)

        ax.set_title(f"h/l = {hl}")
        ax.set_xlabel(r"X [m]")
        ax.set_ylabel(r"BL $\delta$ [mm]")
        ax.grid(True, alpha=0.3)

    # Hide unused subplot slots
    for ax_idx in range(n_hl, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # Single figure-level legend from the first subplot
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center left',
               bbox_to_anchor=(1.01, 0.5),
               frameon=False,
               fontsize=12,
               title='Mach Number',
               title_fontsize=13)

    fig.suptitle("Boundary Layer Thickness — All Cases", fontsize=14)
    plt.tight_layout()
    plt.show()

    if save:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "BL_thickness_all_cases.png", dpi=600, bbox_inches='tight')
        fig.savefig(save_dir / "BL_thickness_all_cases.pdf",            bbox_inches='tight')

    plt.close(fig)
    
    
def plot_BL_location_tecplot(edge_x_dict, edge_y_dict, file_paths, ds_by_case,
                              save_dir=Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\BL_Location_Tecplot")):
    import tecplot as tp
    from tecplot.constant import PlotType, SymbolType,GeomShape, Color, ContourType
    
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tp.session.connect()

    for idx, key in enumerate(ds_by_case.keys()):

        edge_x = edge_x_dict.get(key)
        edge_y = edge_y_dict.get(key)

        if edge_x is None or len(edge_x) == 0:
            print(f"Skipping {key} — no BL edge points found.")
            continue

        tp.new_layout()
        tp.data.load_tecplot(file_paths[idx].as_posix())
        fr   = tp.active_frame()
        fr.plot_type = PlotType.Cartesian2D
        test = fr.dataset
        plot = fr.plot()
        
        # Enable contour display on all flow field zones
        plot.show_contour = True
        plot.show_scatter = True
        
        for i in range(test.num_zones):
            try:
                fm = plot.fieldmap(i)
                fm.show         = True
                fm.contour.show = True
                fm.mesh.show    = False
            except:
                pass
        
        # Add BL edge zone
        bl_zone = test.add_ordered_zone('BL_Edge', len(edge_x))
        bl_zone.values('X')[:] = edge_x
        bl_zone.values('Y')[:] = edge_y
        bl_zone.values('Z')[:] = np.zeros(len(edge_x))
        
        
        # Enable plot-level layers
        plot.show_contour = True
        plot.show_scatter = True
        
        # Assign contour variable and levels
        contour = plot.contour(0)
        contour.variable          = test.variable('U')
        contour.colormap_name     = 'Diverging - Blue/Red'
        contour.levels.reset_to_nice()
        
        # Flood contour on all existing flow zones, scatter OFF
        for i in range(test.num_zones):
            try:
                fm = plot.fieldmap(i)
                fm.show                              = True
                fm.scatter.show                      = False    # <-- turn off scatter for flow zones
                fm.contour.show                      = True
                fm.contour.contour_type              = ContourType.Flood
                fm.contour.flood_contour_group_index = 0
                fm.mesh.show                         = False
            except:
                pass
            
    
    
    
        # Configure scatter on BL zone only #
        fieldmap                      = plot.fieldmap(bl_zone)
        fieldmap.scatter.show         = True
        fieldmap.scatter.symbol_type  = SymbolType.Geometry
        fieldmap.scatter.symbol().shape = GeomShape.Circle
        fieldmap.scatter.color        = Color.Green
        fieldmap.scatter.size         = 0.5
        fieldmap.mesh.show            = False
        fieldmap.contour.show         = False
        
        plot.view.fit()

        # Zoom into BL region #
        plot.axes.x_axis.min = 0.0
        plot.axes.x_axis.max = 0.1   # adjust to your geometry length
        plot.axes.y_axis.min = -0.02
        plot.axes.y_axis.max = 0.07
        
        # Outputting file # 
        out_path = save_dir / f"BL_location_{key}.png"
        tp.export.save_png(out_path.as_posix(), width=1920)
        print(f"Saved: {out_path}")
        
        




def plot_BL_and_separation_contours(delta_n_dict, x, x_sep,
                                     sep_length_nonDim,
                                     save=False,
                                     save_dir=Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study")):
    """
    Parameters
    ----------
    delta_n_dict      : dict  {key: np.ndarray}  BL thickness [mm]
    x                 : dict  {key: np.ndarray}  wall x-coordinates [m]
    x_sep             : dict  {key: np.ndarray}  separation x-locations from find_sepLength
    sep_length_nonDim : dict  {key: float}       normalized separation length from find_sepLength
    """

    mpl.rcParams['font.family']    = 'serif'
    mpl.rcParams['font.serif']     = ['Times New Roman']
    mpl.rcParams['font.size']      = 18
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['figure.dpi']     = 600
    mpl.rcParams['savefig.dpi']    = 600

    hl_bl    = []
    mach_bl  = []
    bl_vals  = []

    hl_sep   = []
    mach_sep = []
    sep_vals = []

    for key in delta_n_dict:
        hl_m   = re.search(r'h_l_([\d.x]+)', key)
        mach_m = re.search(r'Mach_([\d.]+)', key)
        if not hl_m or not mach_m or hl_m.group(1) == 'x':
            continue

        hl   = float(hl_m.group(1))
        mach = float(mach_m.group(1))

        # --- Plot 2: separation % from find_sepLength (all cases) ---
        sep_nd = sep_length_nonDim.get(key, 0.0)
        sep_nd = 0.0 if (sep_nd is None or np.isnan(sep_nd)) else sep_nd
        hl_sep.append(hl)
        mach_sep.append(mach)
        sep_vals.append(sep_nd * 100.0)



        # --- Plot 1: BL height before separation ---
        # Use x_sep from find_sepLength — already correctly computed on curved wall
        x_sep_pts = x_sep.get(key, np.array([]))
        if x_sep_pts.size == 0:
            continue  # no separation
        


        # First separation x-location
        x_sep_first = float(x_sep_pts[0])



        # Find the index in x[key] closest to x_sep_first
        x_wall    = x[key]
        sep_idx   = int(np.argmin(np.abs(x_wall - x_sep_first)))



        # delta_n_dict has one value per wall point — get last valid before sep_idx
        delta_arr     = delta_n_dict[key]
        pre_sep_delta = delta_arr[:sep_idx]
        valid         = np.where(np.isfinite(pre_sep_delta))[0]
        if len(valid) == 0:
            continue



        bl_val = float(pre_sep_delta[valid[-1]])
        hl_bl.append(hl)
        mach_bl.append(mach)
        bl_vals.append(bl_val)

    hl_bl    = np.array(hl_bl)
    mach_bl  = np.array(mach_bl)
    bl_vals  = np.array(bl_vals)
    hl_sep   = np.array(hl_sep)
    mach_sep = np.array(mach_sep)
    sep_vals = np.array(sep_vals)

    # Only h/l values with at least one separation case
    hl_with_sep = set(hl_bl)
    mask_bl     = np.array([hl in hl_with_sep for hl in hl_bl])



    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # sc1 = axes[0].scatter(hl_bl[mask_bl], mach_bl[mask_bl],
    #                      c=bl_vals[mask_bl],
     #                     cmap='cividis', s=200, zorder=5,
      #                    edgecolors='k', linewidths=0.5)
      
      
    # Normalize colormap to data range (exclude outliers using percentiles)
    vmin_bl = np.percentile(bl_vals[mask_bl], 5)
    vmax_bl = np.percentile(bl_vals[mask_bl], 95)

    sc1 = axes[0].scatter(hl_bl[mask_bl], mach_bl[mask_bl],
                          c=bl_vals[mask_bl],
                          cmap='cividis', s=200, zorder=5,
                          edgecolors='k', linewidths=0.5,
                          vmin=vmin_bl, vmax=vmax_bl)
    
      
      
    fig.colorbar(sc1, ax=axes[0], label=r'BL $\delta$ before sep. [mm]')
    axes[0].set_xlabel('h/l')
    axes[0].set_ylabel('Mach')
    axes[0].set_title('BL Height Before Separation')
    axes[0].grid(True, alpha=0.3)

    sc2 = axes[1].scatter(hl_sep, mach_sep,
                          c=sep_vals,
                          cmap='cividis', s=200, zorder=5,
                          edgecolors='k', linewidths=0.5)
    fig.colorbar(sc2, ax=axes[1], label='Separation [%]')
    axes[1].set_xlabel('h/l')
    axes[1].set_ylabel('Mach')
    axes[1].set_title('Separation Percentage')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    if save:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "BL_separation_contours.png", dpi=600, bbox_inches='tight')
        fig.savefig(save_dir / "BL_separation_contours.pdf",            bbox_inches='tight')

    plt.close(fig)



    
    
def plot_BL_thickness(delta_n_dict, x_start_dict, save=False,
                      save_dir=Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study")):
    """
    One figure per h/l value, one curve per Mach number.

    Parameters
    ----------
    delta_n_dict  : dict  {case_key: np.ndarray}  BL thickness [mm]
    x_start_dict  : dict  {case_key: np.ndarray}  rake x-positions [m]
    save          : bool
    save_dir      : Path
    """

    mpl.rcParams['font.family']     = 'serif'
    mpl.rcParams['font.serif']      = ['Times New Roman']
    mpl.rcParams['font.size']       = 18
    mpl.rcParams['axes.labelsize']  = 10
    mpl.rcParams['axes.titlesize']  = 11
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['figure.dpi']      = 600
    mpl.rcParams['savefig.dpi']     = 600
    mpl.rcParams['axes.linewidth']  = 1
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['grid.linewidth']  = 0.5

    # --- Extract unique h/l values ----
    hl_set = set()
    for key in delta_n_dict:
        m = re.search(r'h_l_([\d.x]+)', key)
        if m:
            hl_set.add(m.group(1))
    
    hl_values = sorted(
        hl_set,
        key=lambda v: float('inf') if v == 'x' else float(v)
    )

    # --- One figure per h/l ---
    for hl in hl_values:

        # Collect all keys that match this h/l, sorted by Mach
        keys_for_hl = sorted(
            [k for k in delta_n_dict if re.search(rf'h_l_{re.escape(hl)}_', k)],
            key=lambda k: float(re.search(r'Mach_([\d.]+)', k).group(1))
        )

        n_curves = len(keys_for_hl)
        cmap     = cm.get_cmap('cividis', n_curves)

        fig, ax = plt.subplots(figsize=(3.5, 2.5))

        for i, key in enumerate(keys_for_hl):
            x_arr     = x_start_dict[key]
            delta_arr = delta_n_dict[key]

            # Drop NaN (separation-skipped points)
            # With this:
            x_plot     = x_arr          # keep full length
            delta_plot = delta_arr      # NaNs stay in — matplotlib breaks line there

            mach_match = re.search(r'Mach_([\d.]+)', key)
            label      = f"M = {mach_match.group(1)}" if mach_match else key

            ax.plot(x_plot, delta_plot, color=cmap(i), label=label)

        ax.set_title(f"BL Thickness — h/l = {hl}")
        ax.set_xlabel(r"X [m]")
        ax.set_ylabel(r"BL $\delta$ [mm]")
        ax.legend(frameon=False, fontsize=6)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        if save:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fig_name = f"BL_thickness_hl_{hl}"
            fig.savefig(save_dir / f"{fig_name}.png", dpi=600, bbox_inches='tight')
            fig.savefig(save_dir / f"{fig_name}.pdf",            bbox_inches='tight')

        plt.close(fig)
        


def subplotter_multiPerCase(x_dict, y_dict, x_string, y_string, unit_x, unit_y,
                            filter_param, filter_values, vary_param='mach',
                            hl_filter=None,
                            cmap_name='cividis', figsize=None, save=False, 
                            overall_title=None):
    """
    Create subplots, each filtered by a different value of filter_param.
    
    Parameters
    ----------
    filter_values : list
        List of values to filter by (one subplot per value)
    hl_filter : list or None
        Optional list of h/l values to include as subplots.
        e.g. hl_filter=[0.02, 0.05, 0.10] shows only those h/l subplots.
        If None, all values in filter_values are plotted.
    """
    
    # Set publication-quality parameters
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 34
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['axes.linewidth'] = 3
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 600

    # Apply optional h/l filter to select which subplots to show
    if hl_filter is not None:
        hl_filter_float = [float(v) for v in hl_filter]
        filter_values = [v for v in filter_values if float(v) in hl_filter_float]
    
    # Determine subplot grid
    n_plots = len(filter_values)
    ncols = min(2, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten() if n_plots > 1 else [axes]
    
    for idx, filter_value in enumerate(filter_values):
        ax = axes_flat[idx]
        
        # Filter cases by filter_param
        filtered_keys = []
        for key in x_dict.keys():
            if filter_param == 'h_l':
                match = re.search(r'h_l_([\d.]+)', key)
                if match and float(match.group(1)) == float(filter_value):
                    filtered_keys.append(key)
            elif filter_param == 'mach':
                match = re.search(r'(?:mach|M|Mach)_?([\d.]+)', key)
                if match and float(match.group(1)) == float(filter_value):
                    filtered_keys.append(key)
        
        filtered_keys.sort()

        if not filtered_keys:
            ax.set_visible(False)
            continue
        
        cmap = cm.get_cmap(cmap_name, len(filtered_keys))
        
        for i, key in enumerate(filtered_keys):
            # Auto-generate label
            if vary_param == 'mach':
                match = re.search(r'(?:mach|M|Mach)_?([\d.]+)', key)
                label = f"M = {match.group(1)}" if match else key
            elif vary_param == 'h_l':
                match = re.search(r'h_l_([\d.]+)', key)
                label = f"h/l = {match.group(1)}" if match else key
            else:
                label = key
            
            ax.plot(x_dict[key], y_dict[key], color=cmap(i), lw=5, label=label)
        
        # Format subplot title
        if filter_param == 'h_l':
            title = f"h/l = {filter_value}"
        elif filter_param == 'mach':
            title = f"M = {filter_value}"
        else:
            title = f"{filter_param} = {filter_value}"
        
        ax.set_title(title, fontsize=34)
        ax.set_xlabel(f"{x_string} {unit_x}")
        ax.set_ylabel(f"{y_string} {unit_y}")
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    if overall_title:
        fig.suptitle(overall_title, fontsize=34)

    # Single figure-level legend on the right, taken from the first visible subplot
    first_ax = next(ax for ax in axes_flat if ax.get_visible())
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center left',
               bbox_to_anchor=(1.01, 0.5),
               frameon=False,
               fontsize=18,
               title='Mach Number' if vary_param == 'mach' else 'h/l',
               title_fontsize=18)

    plt.tight_layout()
    plt.show()
    
    if save:
        dirc = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study"
        figName = f"{x_string}Vs{y_string}_{filter_param}_subplots"
        plt.savefig(rf'{dirc}\{figName}.png', dpi=600, bbox_inches='tight')
        plt.savefig(rf'{dirc}\{figName}.pdf', bbox_inches='tight')
    
    return fig, axes

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
    mpl.rcParams['figure.dpi'] = 600
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

    
def plot_scaled_axialForce_vs_hl(results_dict, h_l_values, save=False):
    """
    Plot Scaled Axial Force vs h/l for varying Mach numbers.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with keys like 'h_l_0.02_Mach_1.5' and values as scaled axial force
    h_l_values : array
        Array of h/l values
    save : bool
        Whether to save the figure
    
    Returns
    -------
    fig, ax
    """
    
    # Set publication-quality parameters
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['legend.title_fontsize'] = 8
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['figure.dpi'] = 600
    mpl.rcParams['savefig.dpi'] = 600
    
    # Extract unique Mach numbers from keys
    mach_numbers = set()
    for key in results_dict.keys():
        match = re.search(r'Mach_([\d.]+)', key)
        if match:
            mach_numbers.add(float(match.group(1)))
    mach_numbers = sorted(mach_numbers)
    
    # Get colormap
    cmap = cm.get_cmap('cividis', len(mach_numbers))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot each Mach number
    for i, mach in enumerate(mach_numbers):
        # Collect data for this Mach number
        forces = []
        h_l_plot = []
        
        for h_l in h_l_values:
            # Construct key
            case_key = f"h_l_{h_l:.2f}_Mach_{mach:.1f}"
            
            if case_key in results_dict:
                forces.append(results_dict[case_key])
                h_l_plot.append(h_l)
        
        # Convert to arrays
        forces = np.array(forces)
        h_l_plot = np.array(h_l_plot)
        
        color = cmap(i)
        label = f"M = {mach:.1f}"
        
        # Plot line with markers
        ax.plot(h_l_plot, forces, 'o-', color=color, 
                linewidth=1.5, markersize=6, label=label)
    
    # Labels and formatting
    ax.set_xlabel('h/l')
    ax.set_ylabel(r'$F_{RANS} / F_{Small Pert}$')
    ax.set_title('Scaled Axial Force vs h/l\n(Varying Mach Number)')
    ax.legend(title='Mach Number', frameon=False,bbox_to_anchor = (1.05,1) , borderaxespad = 0.1)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=1 for reference
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Perfect Agreement')
    
    plt.tight_layout()
    plt.show()
    
    # Save if requested
    if save:
        dirc = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study"
        plt.savefig(rf'{dirc}\scaledAxialForce_vs_hl.png', dpi=600, bbox_inches='tight')
        plt.savefig(rf'{dirc}\scaledAxialForce_vs_hl.pdf', bbox_inches='tight')
    
    return fig, ax


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
    mpl.rcParams['figure.dpi'] = 600
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
    ax.legend(frameon=False, bbox_to_anchor=(1.05,1), borderaxespad = 0.1, loc = "best")
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
    mpl.rcParams['figure.dpi'] = 600
    mpl.rcParams['savefig.dpi'] = 600
    

    
    # 
    if isinstance(x_dict, dict) and x_dict:
        case_keys = list(x_dict.keys())
    elif isinstance(y_dict, dict) and y_dict:
        case_keys = list(y_dict.keys())
    else:
        raise ValueError("Neither x_dict nor y_dict is a valid non-empty dictionary")
            
    # Get colormap   
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







### Plotting residuals ####


def load_residuals(path, skiprows=3, ncols=4):
    # Read all columns as strings, whitespace-separated
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skiprows, engine="python")
    # Coerce anything non-numeric to NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    # Keep only numeric columns
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < ncols:
        raise ValueError(f"Found only {num.shape[1]} numeric columns in {path}, need {ncols}.")
    # Most residual files have iteration/index columns first; residuals are usually the last 4 numeric cols
    resid = num.iloc[:, -ncols:].to_numpy()
    return resid



# Root directory to import mcfd_tec.bin files # 

def residual_plotter(rootDir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Results\Mach Study 2"), Resid_labels = ["energy", "mass", "x-momentum", "y-momentum"]):
    subDirs1 = [p for p in rootDir.iterdir() if p.is_dir()]
    fileName = "mcfd.rhsgi"
    subDirs2_rhsgi = [p for d in subDirs1 for p in d.iterdir() if p.is_dir()]  # flattened
    file_paths_rhsgi = [p / fileName for p in subDirs2_rhsgi]
    

    
    for file_path_rhsgi in file_paths_rhsgi:
        try: 
            resid = load_residuals(file_path_rhsgi.as_posix(), skiprows=3, ncols=4)
        
            # Normalize each column by its first entry (avoid divide-by-zero)
            denom = resid[0, :].copy()
            denom[denom == 0] = 1.0
            resid = resid / denom
        
            iterations = np.arange(1, resid.shape[0] + 1)
        
            plt.figure(figsize=(8, 6))
            for j in range(resid.shape[1]):
                plt.semilogy(iterations, resid[:, j], linewidth=2)
            plt.title(f"Residuals Vs Iterations: {file_path_rhsgi.parent.name}", fontsize=24)
            plt.legend(Resid_labels)
            plt.xlabel("Iterations")
            plt.ylabel("Residuals")
            plt.grid(True, which="both")
            plt.tick_params(axis='both', which='major', labelsize=18)
            plt.show()
        except: 
            print(f"Couldn't Find rhsgi file for {file_path_rhsgi.parent.name}\n")
    
    
    
        residMAX = max(resid[-1])
        print(f"Maximum residual: {residMAX:.2e} \n")
        
        

def mass_flux_analyzer(root_dir=None, file_name="minfo1_e1", mass_flux_criterion=0.8, 
                       plot=True, save=False):
    """
    Analyze mass flux convergence from ANSYS Fluent monitor files.
    
    Parameters
    ----------
    root_dir : str or Path, optional
        Root directory containing parametric study solution files
        If None, uses default directory
    file_name : str, optional
        Name of the mass flux monitor file (default: "minfo1_e1")
    mass_flux_criterion : float, optional
        Convergence criterion for net mass flux (default: 0.8)
    plot : bool, optional
        Whether to plot each case (default: True)
    save : bool, optional
        Whether to save the plots (default: False)
        
    Returns
    -------
    mass_flux_end : dict
        Dictionary of final mass flux values {case_name: final_mass_flux}
    max_key : str
        Case name with highest mass flux
    max_val : float
        Highest mass flux value
    """
    
    # Set default root directory if not provided
    if root_dir is None:
        root_dir = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\8_Mach_Sweep_Study_2(Solution)\4_Mach_Reruns"
    
    # Root directory to import files
    rootDir_flux = Path(root_dir)
    subDirs1_flux = [p for p in rootDir_flux.iterdir() if p.is_dir()]
    
    # Flatten directory structure
    subDirs2_flux = [p for d in subDirs1_flux for p in d.iterdir() if p.is_dir()]
    file_paths_flux = [p / file_name for p in subDirs2_flux]
    
    # Load data and analyze
    mass_flux_end = {}
    
    for file_path_flux in file_paths_flux:
        # Read CSV file
        df = pd.read_csv(file_path_flux, sep=r"\s+", comment="#")
        df.rename(columns={"mass_flux": "misc", 
                          "infout1": "iterations", 
                          "d": "mass_flux"}, inplace=True)
        
        # Plot if requested
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(df["iterations"][:-3], df["mass_flux"][:-3])
            
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))
            
            plt.xlabel("Iterations")
            plt.ylabel("Net Mass Flux")
            plt.title(f"Net Mass Flux Vs Iterations: {file_path_flux.parent.name}")
            plt.grid(True, which="both")
            plt.tick_params(axis='both', which='major', labelsize=18)
            
            if save:
                save_dir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Convergence")
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir / f"mass_flux_{file_path_flux.parent.name}.png", 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
        
        # Store final mass flux value
        mass_flux_end[file_path_flux.parent.name] = df["mass_flux"][:-3].iloc[-1]
    
    # Find maximum mass flux
    mass_fluxMAX_key = max(mass_flux_end, key=mass_flux_end.get)
    mass_fluxMAX_val = mass_flux_end[mass_fluxMAX_key]
    print(f"Highest last iteration mass flux: {mass_fluxMAX_val:.2e} at {mass_fluxMAX_key}\n")
    
    # Check convergence criteria
    RED = '\033[31m'
    RESET = '\033[0m'
    
    for key, value in mass_flux_end.items():
        if value > mass_flux_criterion:
            diff = value - mass_flux_criterion
            if diff > 1.0:
                print(f"{key} does not meet criteria (net mass flux < {mass_flux_criterion}). "
                      f"Off by {RED}{diff:.1f}{RESET} and net mass flux is {RED}{value:.2e}{RESET}\n")
            else:
                print(f"{key} does not meet criteria (net mass flux < {mass_flux_criterion}). "
                      f"Off by {diff:.1f} and net mass flux is {value:.2e}\n")
    
    return mass_flux_end, mass_fluxMAX_key, mass_fluxMAX_val






"""
#------------------------------------------------------------------------------------------------------------------------------------#
     Getting all residuals and all convergence criterions(net mass flow) to be able to see if the simuilations converged properly
#------------------------------------------------------------------------------------------------------------------------------------#
""" 
def mass_flux_imbalance_analyzer(root_dir=None, file_name_total="minfo1_e1", 
                                 file_name_inlet="minfo1_e3", plot=True, 
                                 save_csv=True, csv_filename="mass_flux_imbalance_last.csv"):
    """
    Analyze mass flux imbalance between total (net) and inlet mass flux.
    
    Parameters
    ----------
    root_dir : str or Path, optional
        Root directory containing case folders with minfo files
        If None, uses default directory
    file_name_total : str, optional
        Name of total/net mass flux file (default: "minfo1_e1")
    file_name_inlet : str, optional
        Name of inlet mass flux file (default: "minfo1_e3")
    plot : bool, optional
        Whether to plot imbalance vs iterations (default: True)
    save_csv : bool, optional
        Whether to save results to CSV (default: True)
    csv_filename : str, optional
        Name of CSV output file (default: "mass_flux_imbalance_last.csv")
        
    Returns
    -------
    results : list
        List of tuples (case_name, last_iteration, last_imbalance_percent)
    common_cases : list
        List of case names that have both e1 and e3 files
    """
    
    # Set default root directory if not provided
    if root_dir is None:
        root_dir = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Results\Mach Study 2\h_l_0.02"
    
    # Get subdirectories
    rootDir_info = Path(root_dir)
    subDirs_info = [p for p in rootDir_info.iterdir() if p.is_dir()]
    
    # Build file paths
    file_paths_total = [p / file_name_total for p in subDirs_info]
    file_paths_inlet = [p / file_name_inlet for p in subDirs_info]
    
    def read_minfo_flexible(path: Path):
        """
        Flexible reader for minfo1_* files.
        - Skips comment/blank lines.
        - On each data line, parses *numeric* tokens only.
        - If >=3 numbers: first is 'iter', last is 'val'.
          If exactly 2 numbers: treat last as 'val' and auto-iterate (1,2,3,...).
        Returns arrays (iter, val) as float.
        """
        iters, vals = [], []
        auto_i = 0
        try:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    nums = []
                    for tok in s.replace(",", " ").split():
                        try:
                            nums.append(float(tok))
                        except ValueError:
                            continue
                    if len(nums) >= 3:
                        iters.append(nums[0])
                        vals.append(nums[-1])
                    elif len(nums) == 2:
                        auto_i += 1
                        iters.append(float(auto_i))
                        vals.append(nums[-1])
                    else:
                        continue
        except FileNotFoundError:
            return np.array([]), np.array([])
        
        iters = np.asarray(iters, dtype=float)
        vals = np.asarray(vals, dtype=float)
        
        # Sort by iteration, drop duplicate iters (keep last occurrence)
        if iters.size:
            order = np.argsort(iters)
            iters, vals = iters[order], vals[order]
            # Remove duplicates
            uniq, idx = np.unique(iters, return_index=True)
            iters, vals = iters[idx], vals[idx]
        return iters, vals
    
    # Map case folder -> file path for e1 (net) and e3 (inlet)
    total_by_case = {p.parent.name: p for p in file_paths_total if p.exists()}
    inlet_by_case = {p.parent.name: p for p in file_paths_inlet if p.exists()}
    
    common_cases = sorted(set(total_by_case) & set(inlet_by_case))
    
    if not common_cases:
        print("No case has BOTH minfo1_e1 and minfo1_e3 under root directory.")
        return [], []
    
    # Plotting
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = cm.get_cmap("viridis", len(common_cases))
        plotted = 0
        
        for i, case in enumerate(common_cases):
            p_e1 = total_by_case[case]   # minfo1_e1 (net)
            p_e3 = inlet_by_case[case]   # minfo1_e3 (inlet)
            
            it_e1, v_e1 = read_minfo_flexible(p_e1)
            it_e3, v_e3 = read_minfo_flexible(p_e3)
            
            if it_e1.size == 0 or it_e3.size == 0:
                print(f"Skipping {case}: empty data (e1:{it_e1.size}, e3:{it_e3.size})")
                continue
            
            # Align by iteration (inner join)
            d_e1 = dict(zip(it_e1, v_e1))
            d_e3 = dict(zip(it_e3, v_e3))
            it_common = np.array(sorted(set(d_e1.keys()) & set(d_e3.keys())), dtype=float)
            
            if it_common.size == 0:
                print(f"Skipping {case}: no overlapping iterations")
                continue
            
            net = np.array([d_e1[it] for it in it_common], dtype=float)
            inlet = np.array([d_e3[it] for it in it_common], dtype=float)
            
            mask = np.isfinite(net) & np.isfinite(inlet) & (np.abs(inlet) > 0)
            if not np.any(mask):
                print(f"Skipping {case}: all denom invalid/zero")
                continue
            
            it_plot = it_common[mask]
            pct = 100.0 * np.abs(net[mask]) / np.abs(inlet[mask])
            
            ax.plot(it_plot, pct, lw=2, color=cmap(i), label=case)
            plotted += 1
        
        ax.set_title("Mass-flux Imbalance vs Iterations")
        ax.set_xlabel("Iterations")
        ax.set_ylabel(r"Imbalance [%]  =  $|\dot m_{\rm total}| / |\dot m_{\rm inlet}| \times 100$")
        ax.grid(True, which="both", alpha=0.35)
        
        if plotted:
            ax.legend(title="Cases", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad = 0.1)
        else:
            print("No curves plotted — check the messages above for which cases were skipped.")
        
        fig.tight_layout()
        plt.show()
    
    # Calculate last-iteration imbalance
    print("\n=== Last-iteration mass-flux imbalance (|e1|/|e3| * 100) ===")
    
    results = []
    for case in common_cases:
        p_e1 = total_by_case[case]   # minfo1_e1 (net/total)
        p_e3 = inlet_by_case[case]   # minfo1_e3 (inlet)
        
        it_e1, v_e1 = read_minfo_flexible(p_e1)
        it_e3, v_e3 = read_minfo_flexible(p_e3)
        
        if it_e1.size == 0 or it_e3.size == 0:
            print(f"Skipping {case}: empty data")
            continue
        
        # Align by iteration (inner join via dicts)
        d1 = dict(zip(it_e1, v_e1))
        d3 = dict(zip(it_e3, v_e3))
        it_common = np.array(sorted(set(d1) & set(d3)), dtype=float)
        
        if it_common.size == 0:
            print(f"Skipping {case}: no overlapping iterations")
            continue
        
        net = np.array([d1[it] for it in it_common], dtype=float)
        inlet = np.array([d3[it] for it in it_common], dtype=float)
        mask = np.isfinite(net) & np.isfinite(inlet) & (np.abs(inlet) > 0)
        
        if not np.any(mask):
            print(f"Skipping {case}: no valid ratio (zero/NaN inlet)")
            continue
        
        # Last valid sample (highest iteration with a valid ratio)
        last_idx = np.where(mask)[0][-1]
        last_it = it_common[last_idx]
        last_pct = 100.0 * np.abs(net[last_idx]) / np.abs(inlet[last_idx])
        
        results.append((case, last_it, last_pct))
    
    # Pretty print (sorted by case name)
    for case, last_it, last_pct in sorted(results, key=lambda t: t[0]):
        it_disp = int(round(last_it)) if abs(last_it - round(last_it)) < 1e-6 else last_it
        print(f"{case:35s}  iter {it_disp:>7}  ->  {last_pct:8.3f}%")
    
    # Save to CSV if requested
    if save_csv and results:
        import pandas as pd
        df = pd.DataFrame(results, columns=["case", "last_iter", "last_pct"])
        df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to {csv_filename}")
    
    return results, common_cases

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_mcfd_info1(root_dir, outlet_selector=2):
    """
    Parse iCFD++ mcfd.info1 boundary flux files across multiple case folders.
    Returns dicts compatible with plotter_multi_all() and subplotter_multiPerCase().

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing case subfolders, each with mcfd.info1
    outlet_selector : int
        Selector number for the outlet boundary (default: 2)

    Returns
    -------
    iter_dict : dict
        {case_name: np.array of iteration numbers}
    flux_dict : dict
        {case_name: np.array of outlet mass flux values}
    """
    root_dir = Path(root_dir)
    case_dirs = sorted([p for p in root_dir.rglob("mcfd.info1")])

    iter_dict = {}
    flux_dict = {}

    for info1_path in case_dirs:
        case_name = info1_path.parent.name
        iters, fluxes = [], []
        current_iter = None
        current_sel  = None
        in_nondim    = False
        sel_flux     = {}

        with open(info1_path) as f:
            for line in f:
                line = line.rstrip()

                m = re.match(r'^nt\s+(\d+)', line)
                if m:
                    if current_iter is not None:
                        iters.append(current_iter)
                        fluxes.append(sel_flux.get(outlet_selector, np.nan))
                    current_iter = int(m.group(1))
                    sel_flux, current_sel, in_nondim = {}, None, False
                    continue

                m = re.match(r'^For selector\s+(\d+)', line)
                if m:
                    current_sel = int(m.group(1))
                    in_nondim   = False
                    continue

                if 'nondimensional' in line:
                    in_nondim = True
                    continue
                if 'dimensional' in line and 'non' not in line:
                    in_nondim = False
                    continue

                if in_nondim and current_sel is not None and 'mass   flux' in line:
                    if current_sel not in sel_flux:   # first = nondim total
                        sel_flux[current_sel] = float(line.split()[2])

        # flush last iteration
        if current_iter is not None:
            iters.append(current_iter)
            fluxes.append(sel_flux.get(outlet_selector, np.nan))

        iter_dict[case_name] = np.array(iters)
        flux_dict[case_name] = np.abs(np.array(fluxes))

    return iter_dict, flux_dict

def load_info0(path):
    """
    Parse iCFD++ mcfd.info0 residual file.
    Columns: iter | dt | L2(rho) | L2(rhou) | L2(rhoe) | CFL | tau | misc
    Returns dict of numpy arrays.
    """
    data = np.loadtxt(path)
    return {
        "iter":    data[:, 0],
        "L2_rho":  data[:, 2],
        "L2_rhou": data[:, 3],
        "L2_rhoe": data[:, 4],
        "tau":     data[:, 6],
    }


def load_info1(path, outlet_selector=2):
    """
    Parse iCFD++ mcfd.info1 boundary flux file.
    Extracts outlet mass flux (nondimensional) per iteration.
    
    Selector 1 = inlet (fixed BC), 2 = outlet (tracks convergence), 3 = wall (~0)
    """
    iters, outlet_flux = [], []
    current_iter  = None
    current_sel   = None
    in_nondim     = False
    sel_flux      = {}

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            m = re.match(r'^nt\s+(\d+)', line)
            if m:
                if current_iter is not None:
                    iters.append(current_iter)
                    outlet_flux.append(sel_flux.get(outlet_selector, np.nan))
                current_iter = int(m.group(1))
                sel_flux, current_sel, in_nondim = {}, None, False
                continue

            m = re.match(r'^For selector\s+(\d+)', line)
            if m:
                current_sel = int(m.group(1))
                in_nondim   = False
                continue

            if 'nondimensional' in line:
                in_nondim = True;  continue
            if 'dimensional' in line and 'non' not in line:
                in_nondim = False; continue

            if in_nondim and current_sel is not None and 'mass   flux' in line:
                if current_sel not in sel_flux:          # first = nondim total
                    sel_flux[current_sel] = float(line.split()[2])

    if current_iter is not None:                         # flush last iteration
        iters.append(current_iter)
        outlet_flux.append(sel_flux.get(outlet_selector, np.nan))

    return {
        "iter":         np.array(iters),
        "outlet_mflux": np.abs(np.array(outlet_flux)),
    }


def icfd_convergence_plotter(root_dir, case_labels=None, save=False):
    """
    Plot iCFD++ convergence (residuals + mass flux) for all cases under root_dir.
    Mirrors the style of residual_plotter() and mass_flux_analyzer().

    Expects structure:
        root_dir/
            case_A/   <- contains mcfd.info0 and mcfd.info1
            case_B/
            ...

    Parameters
    ----------
    root_dir   : str or Path
    case_labels: dict, optional  {folder_name: display_label}
    save       : bool
    """
    root_dir  = Path(root_dir)
    case_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])

    if not case_dirs:
        print("No subdirectories found.")
        return

    # ── collect all data first ──────────────────────────────────────────────
    cases = []
    for d in case_dirs:
        p0 = d / "mcfd.info0"
        p1 = d / "mcfd.info1"
        if not p0.exists() or not p1.exists():
            print(f"Skipping {d.name}: missing info0 or info1")
            continue
        label = case_labels.get(d.name, d.name) if case_labels else d.name
        cases.append({
            "label": label,
            "res":   load_info0(p0),
            "flux":  load_info1(p1),
        })

    if not cases:
        print("No valid cases found.")
        return

    cmap = cm.get_cmap("cividis", len(cases))

    # ── 1. Separate plot per case ────────────────────────────────────────────
    for i, c in enumerate(cases):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
        color = cmap(i)

        # Residuals
        ax = axes[0]
        ax.semilogy(c["res"]["iter"], c["res"]["L2_rho"],  color=color,          lw=2, label=r"L2($\rho$)")
        ax.semilogy(c["res"]["iter"], c["res"]["L2_rhou"], color=color, ls="--", lw=2, label=r"L2($\rho u$)")
        ax.semilogy(c["res"]["iter"], c["res"]["L2_rhoe"], color=color, ls=":",  lw=2, label=r"L2($\rho e$)")
        ax.set_ylabel("L2 Residual")
        ax.set_title(f"Residuals — {c['label']}", fontsize=14)
        ax.legend(frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=12)

        # Mass flux
        ax = axes[1]
        inlet_ref = c["flux"]["outlet_mflux"][0]
        ax.plot(c["flux"]["iter"], c["flux"]["outlet_mflux"], color=color, lw=2, label="|ṁ| outlet")
        ax.axhline(inlet_ref, color="gray", ls=":", lw=1.5, label=f"Inlet ref = {inlet_ref:.3f}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|ṁ| [kg/s]")
        ax.set_title(f"Mass Flux Convergence — {c['label']}", fontsize=14)
        ax.legend(frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=12)

        plt.tight_layout()

        if save:
            dirc = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Convergence")
            dirc.mkdir(parents=True, exist_ok=True)
            plt.savefig(dirc / f"convergence_{c['label']}.png", dpi=600, bbox_inches="tight")
            plt.savefig(dirc / f"convergence_{c['label']}.pdf", bbox_inches="tight")

        plt.show()

    # ── 2. Combined plot — all cases together ────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for i, c in enumerate(cases):
        color = cmap(i)
        lbl   = c["label"]
        axes[0].semilogy(c["res"]["iter"],  c["res"]["L2_rho"],          color=color, lw=2, label=f"{lbl} — L2(ρ)")
        axes[0].semilogy(c["res"]["iter"],  c["res"]["L2_rhou"], ls="--", color=color, lw=2)
        axes[0].semilogy(c["res"]["iter"],  c["res"]["L2_rhoe"], ls=":",  color=color, lw=2)
        axes[1].plot(c["flux"]["iter"], c["flux"]["outlet_mflux"],        color=color, lw=2, label=lbl)

    axes[0].set_ylabel("L2 Residual");     axes[0].set_title("Residuals — All Cases", fontsize=14)
    axes[1].set_ylabel("|ṁ| [kg/s]");     axes[1].set_title("Mass Flux — All Cases", fontsize=14)
    axes[1].set_xlabel("Iteration")

    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=12)

    plt.tight_layout()

    if save:
        plt.savefig(dirc / "convergence_combined.png", dpi=600, bbox_inches="tight")
        plt.savefig(dirc / "convergence_combined.pdf", bbox_inches="tight")

    plt.show()


def load_mcfd_net_mass_flux(root_dir, inlet_selector=1, outlet_selector=2):
    """
    Parse mcfd.info1 and compute net mass flux per iteration.
    
    Net mass flux = (ṁ_outlet + ṁ_inlet) / A_outlet
    Approaches 0 as simulation converges.

    Returns
    -------
    iter_dict : dict  {case_name: iterations array}
    flux_dict : dict  {case_name: net mass flux array [kg/m²·s]}
    """
    root_dir = Path(root_dir)
    iter_dict, flux_dict = {}, {}

    for info1_path in sorted(root_dir.rglob("mcfd.info1")):
        case_name = info1_path.parent.name
        iters, net_fluxes = [], []

        current_iter = None
        current_sel  = None
        in_nondim    = False
        sel_flow     = {}   # selector -> mass flow [kg/s]
        sel_area     = {}   # selector -> area [m²]

        with open(info1_path) as f:
            for line in f:
                line = line.rstrip()

                m = re.match(r'^nt\s+(\d+)', line)
                if m:
                    if current_iter is not None:
                        ṁ_in  = sel_flow.get(inlet_selector,  np.nan)
                        ṁ_out = sel_flow.get(outlet_selector, np.nan)
                        A_out = sel_area.get(outlet_selector, np.nan)
                        iters.append(current_iter)
                        net_fluxes.append((ṁ_out + ṁ_in) / A_out)
                    current_iter = int(m.group(1))
                    sel_flow, sel_area = {}, {}
                    current_sel, in_nondim = None, False
                    continue

                m = re.match(r'^For selector\s+(\d+)', line)
                if m:
                    current_sel = int(m.group(1))
                    in_nondim   = False
                    continue

                if 'nondimensional' in line:
                    in_nondim = True
                    continue
                if 'dimensional' in line and 'non' not in line:
                    in_nondim = False
                    continue

                if in_nondim and current_sel is not None:
                    if 'mass   flux' in line and current_sel not in sel_flow:
                        sel_flow[current_sel] = float(line.split()[2])
                    if line.strip().startswith('areas') and current_sel not in sel_area:
                        sel_area[current_sel] = abs(float(line.split()[-1]))

        # flush last iteration
        if current_iter is not None:
            ṁ_in  = sel_flow.get(inlet_selector,  np.nan)
            ṁ_out = sel_flow.get(outlet_selector, np.nan)
            A_out = sel_area.get(outlet_selector, np.nan)
            iters.append(current_iter)
            net_fluxes.append((ṁ_out + ṁ_in) / A_out)

        iter_dict[case_name] = np.array(iters)
        flux_dict[case_name] = np.array(net_fluxes)

    return iter_dict, flux_dict






def export_mach_contours(
    study_dir,
    png_dest,
    layout_dest,
    file_name="mcfd_tec.bin",
    res_number=4096,
):
    """
    Export Mach contour images for all cases in a parametric study using a Tecplot layout.

    Parameters
    ----------
    study_dir   : str or Path
        Root directory containing the CFD++ result files.
    png_dest    : str or Path
        Destination directory for the exported PNG contour images.
    layout_dest : str or Path
        Full path to the Tecplot layout file (e.g. machLayout.lay).
    file_name   : str, optional
        Name of the Tecplot binary file to load (default: 'mcfd_tec.bin').
    res_number  : int, optional
        Image resolution in pixels (default: 4096).
    """
    from tecplot.constant import ReadDataOption

    study_dir   = Path(study_dir)
    png_dest    = Path(png_dest)
    layout_dest = Path(layout_dest)

    # Recursively find all matching files under study_dir
    file_paths = list(study_dir.rglob(file_name))

    if not file_paths:
        print(f"No '{file_name}' files found under {study_dir}")
        return

    # Connect to open Tecplot session
    tp.session.connect()

    for file_path in file_paths:
        tp.new_layout()
        tp.load_layout(layout_dest.as_posix())
        tp.data.load_tecplot(
            file_path.as_posix(),
            read_data_option=ReadDataOption.Replace,
            reset_style=False
        )

        # Get plot and dataset objects
        frame   = tp.active_frame()
        dataset = frame.dataset
        plot    = frame.plot()

        # Compute actual Mach data range across all zones
        mach_var = dataset.variable('M')
        mach_min = min(zone.values(mach_var).min() for zone in dataset.zones())
        mach_max = max(zone.values(mach_var).max() for zone in dataset.zones())

        # Reset contour levels to actual data range
        plot.contour(0).levels.reset_levels(mach_min, mach_max, (mach_max - mach_min) / 10)

        tp.macro.execute_command(f'''$!GlobalContour 1
            ColorMapFilter
            {{
                ColorMapDistribution = Continuous
                ContinuousColor
                {{
                    CMin = {mach_min}
                    CMax = {mach_max}
                }}
            }}''')

        plot.contour(0).levels.reset_to_nice(num_levels=10)

        # Export PNG
        out_path = png_dest / f"{file_path.parent.name}.png"
        tp.export.save_png(out_path.as_posix(), res_number, supersample=3)
        print(f"Saved: {out_path.name}")

    print("\nExport complete!")
    
    
    
    



def plot_mach_contours_per_hl(
    hl_value: float,
    viscous_dir: str,
    save: bool = False,
    save_dir: str = None,
    figsize: tuple = None,
    ncols: int = 3,
):
    """
    For a single h/l value, load all viscous Mach contour PNG images
    and display them in a subplot grid (one panel per Mach number).

    Parameters
    ----------
    hl_value : float
        The h/l value to plot (e.g., 0.02, 0.03, ...).
    viscous_dir : str
        Path to the folder containing viscous PNG files.
        Naming convention: h_l_{hl}_Mach_{mach}.png
    save : bool
        Whether to save the figure.
    save_dir : str or None
        Directory to save the figure. Required if save=True.
    figsize : tuple or None
        Figure size. Auto-calculated if None.
    ncols : int
        Number of columns in the subplot grid (default 3).
    """
    # --- Publication rcParams (consistent with your codebase) ---
    mpl.rcParams['font.family']      = 'serif'
    mpl.rcParams['font.serif']       = ['Times New Roman']
    mpl.rcParams['axes.titlesize']   = 10
    mpl.rcParams['axes.labelsize']   = 9
    mpl.rcParams['xtick.labelsize']  = 7
    mpl.rcParams['ytick.labelsize']  = 7
    mpl.rcParams['figure.dpi']       = 150   # screen preview
    mpl.rcParams['savefig.dpi']      = 600

    viscous_path = Path(viscous_dir)

    # --- Glob and sort matching files ---
    # Pattern: h_l_0.03_Mach_1.5.png  (hl formatted to 2 decimal places)
    hl_str   = f"{float(hl_value):.2f}"
    pattern  = f"h_l_{hl_str}_Mach_*.png"
    img_files = sorted(
        viscous_path.glob(pattern),
        key=lambda p: float(re.search(r'Mach_([\d.]+)', p.stem).group(1))
    )

    if not img_files:
        print(f"[plot_mach_contours_per_hl] No files found for h/l = {hl_str} in {viscous_dir}")
        return

    n     = len(img_files)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    for idx, img_path in enumerate(img_files):
        ax    = axes_flat[idx]
        img   = plt.imread(str(img_path))           # loads as float [0,1] RGBA or RGB
        ax.imshow(img)
        ax.axis('off')                               # images have their own axes baked in

        # Extract Mach number for the subplot title
        mach_match = re.search(r'Mach_([\d.]+)', img_path.stem)
        title = f"M = {mach_match.group(1)}" if mach_match else img_path.stem
        ax.set_title(title, fontsize=10)

    # Hide any unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Mach Contours — Viscous — h/l = {hl_str}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()

    if save:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        stem = f"MachContours_viscous_hl_{hl_str}"
        fig.savefig(save_path / f"{stem}.png", dpi=600, bbox_inches='tight')
        fig.savefig(save_path / f"{stem}.pdf",            bbox_inches='tight')

    plt.close(fig)
    return fig, axes


def plot_viscous_vs_inviscid_contours(
    hl_value: float,
    viscous_dir: str,
    inviscid_dir: str,
    save: bool = False,
    save_dir: str = None,
    figsize: tuple = None,
    mach_range: list = None,
):
    """
    For a single h/l value, plot viscous (top row) vs inviscid (bottom row)
    Mach contour images. Each column = one Mach number.

    Layout (6 Mach numbers):
        Row 0 [Viscous ]:  M1.5  M2.0  M2.5  M3.0  M3.5  M4.0
        Row 1 [Inviscid]:  M1.5  M2.0  M2.5  M3.0  M3.5  M4.0

    Parameters
    ----------
    hl_value : float
        The h/l value to compare (e.g., 0.02, 0.05, ...).
    viscous_dir : str
        Path to folder containing viscous PNG files.
    inviscid_dir : str
        Path to folder containing inviscid PNG files.
        Both use the same naming convention: h_l_{hl}_Mach_{mach}.png
    save : bool
        Whether to save the figure.
    save_dir : str or None
        Directory to save. Required if save=True.
    figsize : tuple or None
        Figure size. Auto-calculated if None.
    mach_range : list or None
        Optional list of Mach numbers to include, e.g. [1.5, 2.0, 3.0].
        If None, all available Mach numbers are used.
    """
    mpl.rcParams['font.family']    = 'serif'
    mpl.rcParams['font.serif']     = ['Times New Roman']
    mpl.rcParams['axes.titlesize'] = 24
    mpl.rcParams['figure.dpi']     = 600
    mpl.rcParams['savefig.dpi']    = 600

    viscous_path  = Path(viscous_dir)
    inviscid_path = Path(inviscid_dir)

    hl_str  = f"{float(hl_value):.2f}"
    pattern = f"h_l_{hl_str}_Mach_*.png"

    # --- Collect and sort viscous files by Mach number ---
    viscous_files = sorted(
        viscous_path.glob(pattern),
        key=lambda p: float(re.search(r'Mach_([\d.]+)', p.stem).group(1))
    )

    # --- Filter to requested Mach numbers if mach_range is provided ---
    if mach_range is not None:
        mach_range_float = [float(m) for m in mach_range]
        viscous_files = [
            p for p in viscous_files
            if float(re.search(r'Mach_([\d.]+)', p.stem).group(1)) in mach_range_float
        ]

    if not viscous_files:
        print(f"[plot_viscous_vs_inviscid_contours] No viscous files found for h/l = {hl_str}")
        return

    n_mach = len(viscous_files)
    nrows  = 2
    ncols  = n_mach

    if figsize is None:
        figsize = (4.5 * ncols, 3.2* nrows)
 
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if ncols == 1:
        axes = axes[:, np.newaxis]

    axes[0, 0].set_ylabel("Viscous",  fontsize=24, fontweight='bold',
                           rotation=90, labelpad=4)
    axes[1, 0].set_ylabel("Inviscid", fontsize=24, fontweight='bold',
                           rotation=90, labelpad=4)

    for col, v_path in enumerate(viscous_files):
        mach_match = re.search(r'Mach_([\d.]+)', v_path.stem)
        mach_str   = mach_match.group(1) if mach_match else "?"

        ax_v = axes[0, col]
        ax_v.imshow(plt.imread(str(v_path)))
        ax_v.axis('off')
        ax_v.set_title(f"M = {mach_str}", fontsize=21)

        i_path = inviscid_path / v_path.name
        ax_i   = axes[1, col]
        ax_i.axis('off')
        if i_path.exists():
            ax_i.imshow(plt.imread(str(i_path)))
        else:
            ax_i.text(0.5, 0.5, f"M={mach_str}\nNot found",
                      ha='center', va='center',
                      transform=ax_i.transAxes, fontsize=24, color='gray')

    fig.suptitle(f"Viscous vs Inviscid — Mach Contours — h/l = {hl_str}",
                 fontsize=24, y=1.01)
    plt.tight_layout()
    plt.show()

    if save:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        stem = f"ViscVsInv_MachContours_hl_{hl_str}"
        fig.savefig(save_path / f"{stem}.png", dpi=600, bbox_inches='tight')
        fig.savefig(save_path / f"{stem}.pdf",            bbox_inches='tight')

    plt.close(fig)
    return fig, axes