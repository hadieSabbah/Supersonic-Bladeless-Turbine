import re
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from pathlib import Path

# Conversion factors: multiply a value in Pa by this factor to get the target unit.
# Add new units here as needed.
PRESSURE_UNIT_FROM_PA = {
    'Pa':   1.0,
    'kPa':  1e-3,
    'MPa':  1e-6,
    'bar':  1e-5,
    'atm':  1.0 / 101325.0,
    'psi':  1.0 / 6894.757,
}

PRESSURE_UNIT_LABELS = {
    'Pa': 'Pa', 'kPa': 'kPa', 'MPa': 'MPa',
    'bar': 'bar', 'atm': 'atm', 'psi': 'psi',
}


def _convert_pressure(value_pa, target_unit):
    """Convert a pressure value (or array) from Pa to target_unit."""
    if target_unit not in PRESSURE_UNIT_FROM_PA:
        valid = ', '.join(PRESSURE_UNIT_FROM_PA.keys())
        raise ValueError(f"Unknown pressure unit '{target_unit}'. Valid options: {valid}")
    return value_pa * PRESSURE_UNIT_FROM_PA[target_unit]


def set_publication_style(dpi=300):
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['axes.titlesize'] = 21
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['figure.titlesize'] = 21
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['savefig.dpi'] = dpi


def exp_cleaner(dirc):
    """Read an experimental CSV and return structured data split by sensor type.

    Parameters
    ----------
    dirc : str or Path
        Path to the CSV file.

    Returns
    -------
    dict with keys:
        'sensor_map' : column metadata
        'meta'       : non-sensor columns (Frame, FTime, etc.)
        'P'          : {channel_number: np.array} for pressure sensors
        'T'          : {channel_number: np.array} for temperature sensors
    """
    with open(dirc, 'r', encoding='utf8') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        headers = dict_reader.fieldnames

    old_data = np.genfromtxt(dirc, delimiter=',', skip_header=1)

    pattern = re.compile(r'(SN\d+)-([PT])(\d+)')
    sensor_map = {}

    for col_idx, header in enumerate(headers):
        match = pattern.search(header)
        if match:
            sensor_map[col_idx] = {
                'serial': match.group(1),
                'type': match.group(2),
                'channel': int(match.group(3)),
                'header': header.strip()
            }

    meta_idx = [i for i in range(len(headers)) if i not in sensor_map]
    meta_keys = [headers[i].strip() for i in meta_idx]

    p_cols = {v['channel']: k for k, v in sensor_map.items() if v['type'] == 'P'}
    t_cols = {v['channel']: k for k, v in sensor_map.items() if v['type'] == 'T'}

    meta_raw = {k: old_data[:, i] for k, i in zip(meta_keys, meta_idx)}
    if 'FTime[ns]' in meta_raw:
        meta_raw['FTime[s]_from_ns'] = meta_raw['FTime[ns]'] / 1e9

    clean_data = {
        'sensor_map': sensor_map,
        'meta': {k: old_data[:, i] for k, i in zip(meta_keys, meta_idx)},
        'P': {ch: old_data[:, col] for ch, col in sorted(p_cols.items())},
        'T': {ch: old_data[:, col] for ch, col in sorted(t_cols.items())},
    }

    return clean_data


def detect_test_window(clean_data, k_sigma=10, n_baseline=50):
    """Auto-detect the test window from pressure data using a sigma-threshold.

    Parameters
    ----------
    clean_data : dict
        Output from exp_cleaner().
    k_sigma : float
        Number of standard deviations above the baseline mean to set the threshold.
    n_baseline : int
        Number of initial frames used to estimate the noise floor.

    Returns
    -------
    i_start, i_end : int
        Frame indices bounding the test window.
    t_start, t_end : float
        Corresponding times in seconds.
    """
    stag_ch = max(clean_data['P'], key=lambda ch: np.mean(clean_data['P'][ch]))
    trigger = clean_data['P'][stag_ch]
    print(f"  Trigger channel: P{stag_ch:02d}  (mean = {np.mean(trigger):.4f} Pa)")

    baseline = trigger[:n_baseline]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline, ddof=1)
    threshold = baseline_mean + k_sigma * baseline_std
    print(f"  Baseline mean = {baseline_mean:.6f} Pa,  std = {baseline_std:.6f} Pa")
    print(f"  Threshold     = {threshold:.6f} Pa  ({k_sigma}σ above baseline)")

    above = np.where(trigger > threshold)[0]
    if len(above) == 0:
        raise ValueError("No frames exceeded the threshold — check k_sigma or n_baseline.")

    i_start = above[0]
    i_end = above[-1]

    t = clean_data['meta']['FTime[s]']
    t_start, t_end = t[i_start], t[i_end]
    print(f"  Test window: frame {i_start} → {i_end}  "
          f"({t_start:.3f}s → {t_end:.3f}s, duration = {t_end - t_start:.3f}s)")

    return i_start, i_end, t_start, t_end


def exp_plotter(x_data, y_dict, y_name, label_name, y_unit, show=True):
    """Plot time-series data for each sensor channel.

    Parameters
    ----------
    x_data : np.array
        Time array (x-axis).
    y_dict : dict
        {channel_number: np.array} data to plot.
    y_name : str
        Name for the y-axis (e.g. "Pressure").
    label_name : str
        Legend prefix (e.g. "P" or "T").
    y_unit : str
        Unit string for the y-axis (e.g. "Pa").
    show : bool
        Whether to call plt.show().
    """
    fig, ax = plt.subplots()
    for key in y_dict:
        ax.plot(x_data, y_dict[key], label=f"{label_name} = {key}")

    ax.set_title(f"{y_name} Vs Time")
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel(f"{y_name} [{y_unit}]")
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.grid(True, which="minor")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def exp_stats_pressure(clean_data, confidence=0.95):
    """Compute mean, std, SEM, and confidence interval for each pressure channel
    over the detected test window.

    Parameters
    ----------
    clean_data : dict
        Output from exp_cleaner().
    confidence : float
        Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    dict : {channel: {'mean', 'std', 'sem', 'CI_95', 'N'}}
    """
    i_start, i_end, _, _ = detect_test_window(clean_data)

    stats_out = {}
    for ch, signal in clean_data['P'].items():
        signal = signal[i_start:i_end + 1]
        n = len(signal)
        mean = np.mean(signal)
        std = np.std(signal, ddof=1)
        sem = std / np.sqrt(n)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        ci = t_crit * sem
        stats_out[ch] = {'mean': mean, 'std': std, 'sem': sem, 'CI_95': ci, 'N': n}

    return stats_out


def slice_extractor_wall(dirc):
    """Extract wall surface data (x, y, P, T) from a Tecplot CFD solution file
    by slicing at z=0.

    Parameters
    ----------
    dirc : str or Path
        Path to the Tecplot binary file (e.g. mcfd_tec.bin).

    Returns
    -------
    x_wall, y_wall, p_wall, T_wall : np.array
        Sorted wall arrays.
    """
    import tecplot as tp
    from tecplot.constant import PlotType, SliceSource, ExtractMode

    expected_title = Path(dirc).stem

    try:
        current_title = tp.active_frame().dataset.title
        already_loaded = (current_title == expected_title)
    except Exception:
        already_loaded = False

    if not already_loaded:
        dataset = tp.data.load_tecplot(dirc)
    else:
        dataset = tp.active_frame().dataset

    frame = tp.active_frame()
    frame.plot_type = PlotType.Cartesian3D

    extracted_zones = tp.data.extract.extract_slice(
        origin=(0, 0, 0),
        normal=(0, 0, 1),
        source=SliceSource.SurfaceZones,
        dataset=dataset,
        zones=dataset.zone('Section'),
        mode=ExtractMode.OneZonePerConnectedRegion)

    wall_zone = min(extracted_zones,
                    key=lambda z: z.values('Y').as_numpy_array().mean())

    x_wall = wall_zone.values('X').as_numpy_array()
    y_wall = wall_zone.values('Y').as_numpy_array()
    p_wall = wall_zone.values('P').as_numpy_array()
    T_wall = wall_zone.values('T').as_numpy_array()

    sort_idx = np.argsort(x_wall)
    return x_wall[sort_idx], y_wall[sort_idx], p_wall[sort_idx], T_wall[sort_idx]


def process_exp_case(csv_path, tap_spacing_mm, skip_channels=0,
                     pressure_scale=1.0, pressure_offset=0.0,
                     exp_native_unit='Pa', pressure_unit='Pa',
                     plot=True):
    """Full pipeline: load CSV → clean → compute elapsed time → convert pressure
    → compute stats → return everything in a single consistent unit.

    Parameters
    ----------
    csv_path : str or Path
        Path to the experimental CSV.
    tap_spacing_mm : float
        Physical spacing between pressure taps in mm.
    skip_channels : int
        Number of leading channels to skip (e.g. 3 for flat plate stagnation ports).
    pressure_scale : float
        Multiplicative factor to convert raw sensor reading to the unit specified
        by exp_native_unit (e.g. 1e5 if raw data is in bar and exp_native_unit='Pa').
    pressure_offset : float
        Additive offset applied AFTER scaling to exp_native_unit
        (e.g. 101325 to convert gauge Pa → absolute Pa).
    exp_native_unit : str
        The unit that (raw * pressure_scale + pressure_offset) produces.
        Must be a key in PRESSURE_UNIT_FROM_PA (default 'Pa').
    pressure_unit : str
        The desired output unit for all returned pressures.
        Must be a key in PRESSURE_UNIT_FROM_PA (e.g. 'Pa', 'kPa', 'bar', 'psi', 'atm').
    plot : bool
        Whether to display time-series plots.

    Returns
    -------
    dict with keys:
        'clean_data'    : raw cleaned data from exp_cleaner
        't_elapsed'     : elapsed time array [s]
        'P_converted'   : {channel: np.array} pressure in pressure_unit
        'T_data'        : {channel: np.array} temperature data
        'stats'         : {channel: stats dict} from exp_stats_pressure (raw units)
        'channels'      : sorted list of analysis channels (after skip)
        'means'         : np.array of mean pressures in pressure_unit
        'errors'        : np.array of 95% CI values in pressure_unit
        'x_exp_mm'      : tap locations in mm
        'pressure_unit' : str, the unit of means/errors/P_converted
    """
    clean_data = exp_cleaner(csv_path)

    t_full = clean_data['meta']['FTime[s]'] + clean_data['meta']['FTime[ns]'] / 1e9
    t_elapsed = t_full - t_full[0]

    # Convert raw → exp_native_unit → Pa → pressure_unit
    native_to_pa = 1.0 / PRESSURE_UNIT_FROM_PA[exp_native_unit]

    P_converted = {}
    for key in clean_data['P']:
        val_native = clean_data['P'][key] * pressure_scale
        val_pa = val_native * native_to_pa
        P_converted[key] = _convert_pressure(val_pa, pressure_unit)

    p_unit_label = PRESSURE_UNIT_LABELS[pressure_unit]

    if plot:
        exp_plotter(t_elapsed, P_converted, "Pressure", "P", p_unit_label)
        exp_plotter(t_elapsed, clean_data['T'], "Temperature", "T", "C")

    p_stats = exp_stats_pressure(clean_data)
    channels = sorted(p_stats.keys())[skip_channels:]

    means_native = np.array([
        p_stats[ch]['mean'] * pressure_scale + pressure_offset for ch in channels
    ])
    errors_native = np.array([
        p_stats[ch]['CI_95'] * pressure_scale for ch in channels
    ])

    means_pa = means_native * native_to_pa
    errors_pa = errors_native * native_to_pa

    means = _convert_pressure(means_pa, pressure_unit)
    errors = _convert_pressure(errors_pa, pressure_unit)

    x_exp_mm = tap_spacing_mm * np.arange(len(channels))

    return {
        'clean_data': clean_data,
        't_elapsed': t_elapsed,
        'P_converted': P_converted,
        'T_data': clean_data['T'],
        'stats': p_stats,
        'channels': channels,
        'means': means,
        'errors': errors,
        'x_exp_mm': x_exp_mm,
        'pressure_unit': pressure_unit,
    }


def plot_exp_errorbar(exp_result, title="Experimental Wall Pressure",
                      xlabel="X [mm]", ylabel=None, show=True):
    """Plot error-bar chart from process_exp_case output.

    Parameters
    ----------
    exp_result : dict
        Output from process_exp_case().
    ylabel : str or None
        If None, auto-generates from the pressure_unit stored in exp_result.

    Returns
    -------
    fig, ax
    """
    if ylabel is None:
        unit = exp_result.get('pressure_unit', 'Pa')
        ylabel = f"Pressure [{PRESSURE_UNIT_LABELS.get(unit, unit)}]"

    fig, ax = plt.subplots()
    ax.errorbar(exp_result['x_exp_mm'], exp_result['means'],
                yerr=exp_result['errors'], fmt='o', capsize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def compare_exp_to_cfd(exp_result, cfd_path, x_min_cfd, x_max_cfd,
                       cfd_native_unit='Pa', cfd_pressure_offset=0.0,
                       pressure_unit=None, normalize=False,
                       title="Pressure Vs X", xlabel="X [mm]", ylabel=None,
                       show=True):
    """Load CFD wall data, trim to matching region, convert both datasets to
    the same pressure unit, and overlay with experimental data.

    Parameters
    ----------
    exp_result : dict
        Output from process_exp_case().
    cfd_path : str or Path
        Path to Tecplot binary file.
    x_min_cfd, x_max_cfd : float
        Bounds to trim the CFD x-domain (in CFD native units, e.g. meters).
    cfd_native_unit : str
        The pressure unit that the CFD solver writes (default 'Pa').
    cfd_pressure_offset : float
        Value subtracted from CFD pressure IN cfd_native_unit before unit
        conversion (e.g. back-pressure in Pa to get gauge pressure).
    pressure_unit : str or None
        Desired output unit for the comparison plot. If None, uses whatever
        unit the exp_result was processed in.
    normalize : bool
        If True, normalize both datasets by their respective maxima
        (overrides pressure_unit for the y-axis).
    title, xlabel, ylabel : str
        Plot labels. ylabel auto-generates from pressure_unit if None.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    dict with keys:
        'x_cfd', 'P_cfd'  : trimmed CFD arrays in pressure_unit
        'x_exp', 'P_exp', 'errors' : experimental arrays in pressure_unit
        'x_wall', 'y_wall', 'P_wall', 'T_wall' : raw wall data (cfd_native_unit)
        'pressure_unit'    : the unit used
        'fig', 'ax'        : matplotlib objects
    """
    if pressure_unit is None:
        pressure_unit = exp_result.get('pressure_unit', 'Pa')

    x_wall, y_wall, P_wall, T_wall = slice_extractor_wall(Path(cfd_path).as_posix())

    mask = (x_wall >= x_min_cfd) & (x_wall <= x_max_cfd)
    x_cfd = (x_wall[mask] - x_min_cfd) * 1e3

    # CFD: native → Pa → pressure_unit
    cfd_native_to_pa = 1.0 / PRESSURE_UNIT_FROM_PA[cfd_native_unit]
    P_cfd_pa = (P_wall[mask] - cfd_pressure_offset) * cfd_native_to_pa
    P_cfd = _convert_pressure(P_cfd_pa, pressure_unit)

    # Experiment: already in exp_result['pressure_unit'] → convert if different
    exp_unit = exp_result.get('pressure_unit', 'Pa')
    if exp_unit != pressure_unit:
        exp_to_pa = 1.0 / PRESSURE_UNIT_FROM_PA[exp_unit]
        y_exp = _convert_pressure(exp_result['means'] * exp_to_pa, pressure_unit)
        errors = _convert_pressure(exp_result['errors'] * exp_to_pa, pressure_unit)
    else:
        y_exp = exp_result['means'].copy()
        errors = exp_result['errors'].copy()

    x_exp = exp_result['x_exp_mm']

    if normalize:
        errors = errors / np.max(y_exp)
        P_cfd = P_cfd / np.max(P_cfd)
        y_exp = y_exp / np.max(y_exp)

    if ylabel is None:
        p_label = PRESSURE_UNIT_LABELS.get(pressure_unit, pressure_unit)
        ylabel = "Normalized P [-]" if normalize else f"P [{p_label}]"

    fig, ax = plt.subplots()
    ax.plot(x_cfd, P_cfd, label="3D CFD", linewidth=3)
    ax.errorbar(x_exp, y_exp, yerr=errors, fmt='o', capsize=4,
                label="Experiment", color="red")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()

    return {
        'x_cfd': x_cfd, 'P_cfd': P_cfd,
        'x_exp': x_exp, 'P_exp': y_exp, 'errors': errors,
        'x_wall': x_wall, 'y_wall': y_wall, 'P_wall': P_wall, 'T_wall': T_wall,
        'pressure_unit': pressure_unit,
        'fig': fig, 'ax': ax,
    }
