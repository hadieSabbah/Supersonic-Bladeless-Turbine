"""
SBTTD Sweep Study Analysis GUI
===============================
A graphical interface for processing experimental sweep studies (P0, Mach, etc.)
and comparing them against CFD solutions.

HOW GUIs WORK (crash course):
-----------------------------
A GUI is built from 3 concepts:

1. WIDGETS: Visual elements (buttons, text fields, dropdowns, labels).
   Each widget is an object you create and place on a window.

2. LAYOUT: How widgets are arranged. We use a "grid" layout — think of the
   window as a spreadsheet. Each widget goes in row X, column Y.

3. CALLBACKS: Functions that run when the user interacts with a widget.
   E.g., clicking "Browse" calls a function that opens a file dialog.

The pattern is always:
    widget = SomeWidget(parent, options...)   # create it
    widget.grid(row=R, column=C)              # place it
    widget.configure(command=my_function)     # connect a callback

ARCHITECTURE:
-------------
This file is structured as a single class (SweepStudyApp) that:
    - __init__: builds all the widgets
    - Methods: handle user actions (browse files, run analysis, etc.)

The class inherits from customtkinter.CTk, which IS the main window.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.experimental import (
    set_publication_style,
    process_exp_case,
    plot_exp_errorbar,
    compare_exp_to_cfd,
    PRESSURE_UNIT_FROM_PA,
)


# =============================================================================
# APPEARANCE CONFIGURATION
# customtkinter lets you set a global theme. "Dark" or "Light" mode,
# and a color theme ("blue", "green", "dark-blue").
# =============================================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SweepStudyApp(ctk.CTk):
    """Main application window.

    TEACHING NOTE: A class that inherits from CTk IS the window.
    Everything you add to 'self' appears in that window.
    """

    def __init__(self):
        super().__init__()

        # --- Window configuration ---
        self.title("SBTTD Sweep Study Analyzer")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Store results for later use
        self.exp_result = None
        self.cfd_result = None

        # --- Build the UI ---
        self._create_sidebar()
        self._create_main_area()
        self._create_status_bar()

    # =========================================================================
    # SIDEBAR: Navigation and study info
    # =========================================================================
    def _create_sidebar(self):
        """The left panel with study type selection and global controls.

        TEACHING NOTE: A CTkFrame is a container — a box you put other
        widgets inside. It helps organize your layout into sections.
        """
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)  # push bottom items down

        # App title
        title_label = ctk.CTkLabel(
            self.sidebar, text="SBTTD Analyzer",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Study type selector
        study_label = ctk.CTkLabel(self.sidebar, text="Study Type:")
        study_label.grid(row=1, column=0, padx=20, pady=(10, 0))

        self.study_type_var = ctk.StringVar(value="Mach Study")
        self.study_type_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Mach Study", "P0 Study", "Custom"],
            variable=self.study_type_var
        )
        self.study_type_menu.grid(row=2, column=0, padx=20, pady=10)

        # Appearance mode
        appearance_label = ctk.CTkLabel(self.sidebar, text="Theme:")
        appearance_label.grid(row=11, column=0, padx=20, pady=(10, 0))

        self.appearance_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Dark", "Light", "System"],
            command=self._change_appearance
        )
        self.appearance_menu.grid(row=12, column=0, padx=20, pady=(5, 20))

    # =========================================================================
    # MAIN AREA: Tabbed interface for Experiment and CFD
    # =========================================================================
    def _create_main_area(self):
        """The center content area with tabs.

        TEACHING NOTE: CTkTabview creates a tabbed interface — like browser
        tabs. Each tab is its own frame where you place widgets independently.
        """
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Configure grid weights so main area expands
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Tabview
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Create tabs
        self.tab_exp = self.tabview.add("Experimental Data")
        self.tab_cfd = self.tabview.add("CFD Comparison")
        self.tab_results = self.tabview.add("Results")

        # Build each tab
        self._build_exp_tab()
        self._build_cfd_tab()
        self._build_results_tab()

    # =========================================================================
    # EXPERIMENTAL DATA TAB
    # =========================================================================
    def _build_exp_tab(self):
        """Build the experimental data processing interface.

        TEACHING NOTE: This demonstrates the core pattern:
            1. Create a labeled entry field
            2. Add a browse button next to it
            3. Connect the button to a file dialog callback
        """
        tab = self.tab_exp
        tab.grid_columnconfigure(1, weight=1)

        row = 0

        # --- Section: File Selection ---
        section_label = ctk.CTkLabel(
            tab, text="File Selection",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label.grid(row=row, column=0, columnspan=3, pady=(10, 5), sticky="w")
        row += 1

        # CSV path
        ctk.CTkLabel(tab, text="CSV File:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.csv_path_var = ctk.StringVar()
        self.csv_entry = ctk.CTkEntry(tab, textvariable=self.csv_path_var, width=500)
        self.csv_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(tab, text="Browse", width=80,
                      command=self._browse_csv).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # --- Section: Sensor Parameters ---
        section_label2 = ctk.CTkLabel(
            tab, text="Sensor Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label2.grid(row=row, column=0, columnspan=3, pady=(20, 5), sticky="w")
        row += 1

        # Tap spacing
        ctk.CTkLabel(tab, text="Tap Spacing [mm]:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.tap_spacing_var = ctk.StringVar(value="7.25")
        ctk.CTkEntry(tab, textvariable=self.tap_spacing_var, width=150).grid(
            row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # Skip channels
        ctk.CTkLabel(tab, text="Skip Channels:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.skip_channels_var = ctk.StringVar(value="3")
        ctk.CTkEntry(tab, textvariable=self.skip_channels_var, width=150).grid(
            row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # --- Section: Pressure Conversion ---
        section_label3 = ctk.CTkLabel(
            tab, text="Pressure Conversion",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label3.grid(row=row, column=0, columnspan=3, pady=(20, 5), sticky="w")
        row += 1

        # Pressure scale
        ctk.CTkLabel(tab, text="Pressure Scale:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.pressure_scale_var = ctk.StringVar(value="1e5")
        ctk.CTkEntry(tab, textvariable=self.pressure_scale_var, width=150).grid(
            row=row, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(tab, text="(raw * scale = native unit)").grid(
            row=row, column=2, padx=5, pady=5, sticky="w")
        row += 1

        # Pressure offset
        ctk.CTkLabel(tab, text="Pressure Offset:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.pressure_offset_var = ctk.StringVar(value="101325")
        ctk.CTkEntry(tab, textvariable=self.pressure_offset_var, width=150).grid(
            row=row, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(tab, text="(added after scale, e.g. gauge->abs)").grid(
            row=row, column=2, padx=5, pady=5, sticky="w")
        row += 1

        # Native unit
        ctk.CTkLabel(tab, text="Native Unit:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.exp_native_unit_var = ctk.StringVar(value="Pa")
        ctk.CTkOptionMenu(
            tab, values=list(PRESSURE_UNIT_FROM_PA.keys()),
            variable=self.exp_native_unit_var, width=150
        ).grid(row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # Output unit
        ctk.CTkLabel(tab, text="Output Unit:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.output_unit_var = ctk.StringVar(value="kPa")
        ctk.CTkOptionMenu(
            tab, values=list(PRESSURE_UNIT_FROM_PA.keys()),
            variable=self.output_unit_var, width=150
        ).grid(row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # --- Run Button ---
        self.run_exp_btn = ctk.CTkButton(
            tab, text="Process Experimental Data",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40, command=self._run_exp_processing
        )
        self.run_exp_btn.grid(row=row, column=0, columnspan=3, pady=20)

    # =========================================================================
    # CFD COMPARISON TAB
    # =========================================================================
    def _build_cfd_tab(self):
        """Build the CFD comparison interface."""
        tab = self.tab_cfd
        tab.grid_columnconfigure(1, weight=1)

        row = 0

        # --- Section: CFD File ---
        section_label = ctk.CTkLabel(
            tab, text="CFD Solution File",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label.grid(row=row, column=0, columnspan=3, pady=(10, 5), sticky="w")
        row += 1

        # CFD path
        ctk.CTkLabel(tab, text="Tecplot File:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.cfd_path_var = ctk.StringVar()
        self.cfd_entry = ctk.CTkEntry(tab, textvariable=self.cfd_path_var, width=500)
        self.cfd_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(tab, text="Browse", width=80,
                      command=self._browse_cfd).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # --- Section: CFD Parameters ---
        section_label2 = ctk.CTkLabel(
            tab, text="CFD Domain Trim",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label2.grid(row=row, column=0, columnspan=3, pady=(20, 5), sticky="w")
        row += 1

        # X min
        ctk.CTkLabel(tab, text="X min (CFD coords):").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.x_min_cfd_var = ctk.StringVar(value="0.0")
        ctk.CTkEntry(tab, textvariable=self.x_min_cfd_var, width=150).grid(
            row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # X max
        ctk.CTkLabel(tab, text="X max (CFD coords):").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.x_max_cfd_var = ctk.StringVar(value="0.1")
        ctk.CTkEntry(tab, textvariable=self.x_max_cfd_var, width=150).grid(
            row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # --- Section: CFD Pressure Settings ---
        section_label3 = ctk.CTkLabel(
            tab, text="CFD Pressure Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        section_label3.grid(row=row, column=0, columnspan=3, pady=(20, 5), sticky="w")
        row += 1

        # CFD native unit
        ctk.CTkLabel(tab, text="CFD Native Unit:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.cfd_native_unit_var = ctk.StringVar(value="Pa")
        ctk.CTkOptionMenu(
            tab, values=list(PRESSURE_UNIT_FROM_PA.keys()),
            variable=self.cfd_native_unit_var, width=150
        ).grid(row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # CFD pressure offset
        ctk.CTkLabel(tab, text="Pressure Offset:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.cfd_offset_var = ctk.StringVar(value="0.0")
        ctk.CTkEntry(tab, textvariable=self.cfd_offset_var, width=150).grid(
            row=row, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(tab, text="(subtracted from CFD P before conversion)").grid(
            row=row, column=2, padx=5, pady=5, sticky="w")
        row += 1

        # Comparison unit
        ctk.CTkLabel(tab, text="Comparison Unit:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.compare_unit_var = ctk.StringVar(value="Auto (match experiment)")
        compare_options = ["Auto (match experiment)"] + list(PRESSURE_UNIT_FROM_PA.keys())
        ctk.CTkOptionMenu(
            tab, values=compare_options,
            variable=self.compare_unit_var, width=200
        ).grid(row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1

        # Normalize checkbox
        self.normalize_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab, text="Normalize both datasets by their maxima",
                        variable=self.normalize_var).grid(
            row=row, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        row += 1

        # --- Run Button ---
        self.run_cfd_btn = ctk.CTkButton(
            tab, text="Compare Experiment vs CFD",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40, command=self._run_cfd_comparison
        )
        self.run_cfd_btn.grid(row=row, column=0, columnspan=3, pady=20)

    # =========================================================================
    # RESULTS TAB
    # =========================================================================
    def _build_results_tab(self):
        """Build the results display area.

        TEACHING NOTE: Matplotlib figures can be embedded directly into
        tkinter using FigureCanvasTkAgg. This is how you show plots
        inside your app instead of popping up separate windows.
        """
        tab = self.tab_results
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # Placeholder text (replaced by plots after analysis)
        self.results_placeholder = ctk.CTkLabel(
            tab,
            text="Run an analysis to see results here.\n\n"
                 "1. Go to 'Experimental Data' tab and process your CSV\n"
                 "2. Go to 'CFD Comparison' tab to overlay with CFD\n"
                 "3. Results will appear in this tab",
            font=ctk.CTkFont(size=14),
            justify="center"
        )
        self.results_placeholder.grid(row=0, column=0, pady=50)

        # Frame for embedded matplotlib (hidden until we have results)
        self.plot_frame = ctk.CTkFrame(tab)
        self.canvas = None

    # =========================================================================
    # STATUS BAR
    # =========================================================================
    def _create_status_bar(self):
        """A status bar at the bottom for feedback messages."""
        self.status_bar = ctk.CTkLabel(
            self, text="Ready", anchor="w",
            font=ctk.CTkFont(size=11)
        )
        self.status_bar.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 5))

    def _set_status(self, message):
        self.status_bar.configure(text=message)
        self.update_idletasks()

    # =========================================================================
    # CALLBACKS: These functions run when the user clicks buttons
    # =========================================================================

    def _change_appearance(self, mode):
        """Called when the user changes the theme dropdown."""
        ctk.set_appearance_mode(mode)

    def _browse_csv(self):
        """Open a file dialog for CSV selection.

        TEACHING NOTE: filedialog.askopenfilename() opens the native OS
        file picker. The 'filetypes' argument filters what files are shown.
        """
        path = filedialog.askopenfilename(
            title="Select Experimental CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.csv_path_var.set(path)

    def _browse_cfd(self):
        """Open a file dialog for Tecplot binary selection."""
        path = filedialog.askopenfilename(
            title="Select Tecplot CFD File",
            filetypes=[("Tecplot binary", "*.bin"), ("All files", "*.*")]
        )
        if path:
            self.cfd_path_var.set(path)

    def _run_exp_processing(self):
        """Process the experimental data when the user clicks 'Run'.

        TEACHING NOTE: This is where the GUI connects to your backend
        functions. The GUI collects parameters from widgets, calls the
        processing function, and displays results.
        """
        # Validate inputs
        csv_path = self.csv_path_var.get()
        if not csv_path or not Path(csv_path).exists():
            messagebox.showerror("Error", "Please select a valid CSV file.")
            return

        try:
            tap_spacing = float(self.tap_spacing_var.get())
            skip_channels = int(self.skip_channels_var.get())
            pressure_scale = float(self.pressure_scale_var.get())
            pressure_offset = float(self.pressure_offset_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid numeric input: {e}")
            return

        exp_native_unit = self.exp_native_unit_var.get()
        pressure_unit = self.output_unit_var.get()

        self._set_status("Processing experimental data...")
        self.run_exp_btn.configure(state="disabled")

        try:
            self.exp_result = process_exp_case(
                csv_path=csv_path,
                tap_spacing_mm=tap_spacing,
                skip_channels=skip_channels,
                pressure_scale=pressure_scale,
                pressure_offset=pressure_offset,
                exp_native_unit=exp_native_unit,
                pressure_unit=pressure_unit,
                plot=False,  # Don't pop up matplotlib windows
            )

            # Show results in the embedded plot
            self._display_exp_plot()
            self._set_status(
                f"Done! Processed {len(self.exp_result['channels'])} channels "
                f"in {pressure_unit}."
            )
            # Switch to results tab
            self.tabview.set("Results")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            self._set_status(f"Error: {e}")
        finally:
            self.run_exp_btn.configure(state="normal")

    def _run_cfd_comparison(self):
        """Run the CFD comparison."""
        if self.exp_result is None:
            messagebox.showwarning(
                "No Experiment Data",
                "Please process experimental data first (Experimental Data tab)."
            )
            return

        cfd_path = self.cfd_path_var.get()
        if not cfd_path or not Path(cfd_path).exists():
            messagebox.showerror("Error", "Please select a valid Tecplot file.")
            return

        try:
            x_min = float(self.x_min_cfd_var.get())
            x_max = float(self.x_max_cfd_var.get())
            cfd_offset = float(self.cfd_offset_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid numeric input: {e}")
            return

        compare_unit = self.compare_unit_var.get()
        if compare_unit == "Auto (match experiment)":
            compare_unit = None

        self._set_status("Loading CFD and comparing...")
        self.run_cfd_btn.configure(state="disabled")

        try:
            self.cfd_result = compare_exp_to_cfd(
                exp_result=self.exp_result,
                cfd_path=cfd_path,
                x_min_cfd=x_min,
                x_max_cfd=x_max,
                cfd_native_unit=self.cfd_native_unit_var.get(),
                cfd_pressure_offset=cfd_offset,
                pressure_unit=compare_unit,
                normalize=self.normalize_var.get(),
                show=False,  # Don't pop up windows
            )

            self._display_comparison_plot()
            self._set_status("Comparison complete!")
            self.tabview.set("Results")

        except Exception as e:
            messagebox.showerror("CFD Comparison Error", str(e))
            self._set_status(f"Error: {e}")
        finally:
            self.run_cfd_btn.configure(state="normal")

    # =========================================================================
    # PLOT DISPLAY: Embedding matplotlib into the GUI
    # =========================================================================

    def _display_exp_plot(self):
        """Embed the experimental error-bar plot into the Results tab.

        TEACHING NOTE: To put a matplotlib figure inside a tkinter window:
            1. Create the figure with plt.subplots() (don't call plt.show())
            2. Wrap it in FigureCanvasTkAgg(figure, master=parent_widget)
            3. Call canvas.draw() to render
            4. Call canvas.get_tk_widget().grid(...) to place it
        """
        # Clear previous content
        self._clear_results_area()

        set_publication_style(dpi=100)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(
            self.exp_result['x_exp_mm'],
            self.exp_result['means'],
            yerr=self.exp_result['errors'],
            fmt='o', capsize=4, color='#1f77b4', markersize=8
        )
        unit = self.exp_result['pressure_unit']
        ax.set_xlabel("X [mm]")
        ax.set_ylabel(f"Pressure [{unit}]")
        ax.set_title("Experimental Wall Pressure")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._embed_figure(fig)

    def _display_comparison_plot(self):
        """Embed the CFD vs experiment comparison plot."""
        self._clear_results_area()

        set_publication_style(dpi=100)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(self.cfd_result['x_cfd'], self.cfd_result['P_cfd'],
                label="3D CFD", linewidth=2.5, color='#1f77b4')
        ax.errorbar(
            self.cfd_result['x_exp'], self.cfd_result['P_exp'],
            yerr=self.cfd_result['errors'],
            fmt='o', capsize=4, label="Experiment", color='red', markersize=8
        )

        unit = self.cfd_result['pressure_unit']
        normalize = self.normalize_var.get()
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Normalized P [-]" if normalize else f"P [{unit}]")
        ax.set_title("Experiment vs CFD")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._embed_figure(fig)

    def _embed_figure(self, fig):
        """Helper: embed a matplotlib figure into the results tab."""
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _clear_results_area(self):
        """Remove previous plot/placeholder from results tab."""
        self.results_placeholder.grid_forget()
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        plt.close('all')


# =============================================================================
# ENTRY POINT
#
# TEACHING NOTE: This is how every Python GUI app starts:
#   1. Create the app object (which builds the window)
#   2. Call .mainloop() — this starts the event loop that listens for
#      user actions (clicks, typing, etc.) and keeps the window alive.
#      Your code "pauses" here until the user closes the window.
# =============================================================================
if __name__ == "__main__":
    app = SweepStudyApp()
    app.mainloop()
