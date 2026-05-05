"""
SBTTD Mach Post-Processing GUI
================================
Main application window that hosts all analysis tabs.

Launch with:
    python -m gui
"""

import customtkinter as ctk
from gui.state import AppState
from gui.tabs.tab_data_io import DataIOTab
from gui.tabs.tab_variables import VariablesTab
from gui.tabs.tab_boundary_layer import BoundaryLayerTab
from gui.tabs.tab_separation import SeparationTab
from gui.tabs.tab_wall_plots import WallPlotsTab
from gui.tabs.tab_force_power import ForcePowerTab
from gui.tabs.tab_theoretical import TheoreticalTab
from gui.tabs.tab_contours import ContoursTab

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SBTTDApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("SBTTD Mach Post-Processing")
        self.geometry("1500x900")
        self.minsize(1200, 800)

        self.app_state = AppState()

        self._build_sidebar()
        self._build_tabs()
        self._build_status_bar()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=180, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            sidebar, text="SBTTD\nAnalyzer",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(20, 10))

        ctk.CTkLabel(sidebar, text="Theme:").grid(row=11, column=0, padx=20, pady=(10, 0))
        ctk.CTkOptionMenu(
            sidebar,
            values=["Dark", "Light", "System"],
            command=lambda m: ctk.set_appearance_mode(m),
        ).grid(row=12, column=0, padx=20, pady=(5, 20))

    def _build_tabs(self):
        container = ctk.CTkFrame(self)
        container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(container)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tab_specs = [
            ("1. Data I/O",        DataIOTab),
            ("2. Variables",       VariablesTab),
            ("3. Boundary Layer",  BoundaryLayerTab),
            ("4. Separation",      SeparationTab),
            ("5. Wall Plots",      WallPlotsTab),
            ("6. Force & Power",   ForcePowerTab),
            ("7. Theory (S-E)",    TheoreticalTab),
            ("8. Contours",        ContoursTab),
        ]

        self.tabs = {}
        for label, cls in tab_specs:
            frame = self.tabview.add(label)
            tab = cls(frame, self.app_state, status_callback=self._set_status)
            tab.pack(fill="both", expand=True)
            self.tabs[label] = tab

    def _build_status_bar(self):
        self.status_bar = ctk.CTkLabel(
            self, text="Ready", anchor="w", font=ctk.CTkFont(size=11)
        )
        self.status_bar.grid(row=1, column=1, sticky="ew", padx=10, pady=(0, 5))

    def _set_status(self, message):
        self.status_bar.configure(text=message)
        self.update_idletasks()
