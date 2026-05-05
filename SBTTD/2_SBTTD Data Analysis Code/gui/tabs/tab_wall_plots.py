"""
Tab 5 --- Wall Variable Plots
Plot tau_wall, y+, pressure, pressure gradient, etc. for selected h/l or Mach.
"""

import customtkinter as ctk
import numpy as np
import re
import matplotlib.cm as cm
from tkinter import messagebox
from gui.widgets.plot_canvas import PlotCanvas


class WallPlotsTab(ctk.CTkFrame):
    def __init__(self, master, state, status_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.status = status_callback or (lambda msg: None)
        self._build_ui()

    def _build_ui(self):
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            ctrl, text="Wall Variable Plotter",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=5)

        # Variable selector
        ctk.CTkLabel(ctrl, text="Variable:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.var_choice = ctk.StringVar(value="tau_wall")
        ctk.CTkOptionMenu(
            ctrl,
            values=[
                "tau_wall", "y_plus", "P", "Px (dP/dx)", "P0",
                "mach", "T", "rho", "u", "v", "q_dot",
            ],
            variable=self.var_choice,
            width=160,
        ).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Filter by
        ctk.CTkLabel(ctrl, text="Filter by:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.filter_param = ctk.StringVar(value="h_l")
        ctk.CTkOptionMenu(
            ctrl, values=["h_l", "mach"], variable=self.filter_param, width=100,
        ).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(ctrl, text="Value:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.filter_val = ctk.CTkEntry(ctrl, width=100)
        self.filter_val.insert(0, "0.04")
        self.filter_val.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        ctk.CTkButton(ctrl, text="Plot", command=self._plot).grid(
            row=3, column=0, columnspan=2, padx=5, pady=10, sticky="w"
        )
        ctk.CTkButton(ctrl, text="Subplot All h/l", command=self._subplot_all).grid(
            row=3, column=2, columnspan=2, padx=5, pady=10, sticky="w"
        )

        self.plot_canvas = PlotCanvas(self, figsize=(9, 5))
        self.plot_canvas.pack(fill="both", expand=True, padx=10, pady=10)

    def _get_y_dict(self):
        var = self.var_choice.get()
        mapping = {
            "tau_wall": self.state.tau_wall,
            "y_plus": self.state.y_plus,
            "P": self.state.P,
            "Px (dP/dx)": self.state.Px,
            "P0": self.state.P0,
            "mach": self.state.mach,
            "T": self.state.T,
            "rho": self.state.rho,
            "u": self.state.u,
            "v": self.state.v,
            "q_dot": self.state.q_dot,
        }
        return mapping.get(var), var

    def _plot(self):
        if self.state.x is None:
            messagebox.showwarning("Missing", "Extract variables first (Tab 2).")
            return

        y_dict, y_label = self._get_y_dict()
        if y_dict is None:
            messagebox.showwarning("Missing", f"{y_label} not available. Compute it first.")
            return

        try:
            fv = float(self.filter_val.get())
        except ValueError:
            messagebox.showerror("Error", "Filter value must be a number.")
            return

        fp = self.filter_param.get()
        keys = self._filter_keys(fp, fv)
        if not keys:
            messagebox.showinfo("No data", f"No cases match {fp} = {fv}")
            return

        fig, ax = self.plot_canvas.get_fig_ax()
        ax.cla()

        n = len(keys)
        cmap_obj = cm.get_cmap("cividis", max(n, 1))

        for i, key in enumerate(keys):
            if key not in y_dict or key not in self.state.x:
                continue
            m = re.search(r"Mach_([\d.]+)", key) if fp == "h_l" else re.search(r"h_l_([\d.]+)", key)
            label = f"M={m.group(1)}" if m and fp == "h_l" else (f"h/l={m.group(1)}" if m else key)
            ax.plot(self.state.x[key], y_dict[key], color=cmap_obj(i), label=label)

        ax.set_xlabel("x [m]")
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label}  ({fp} = {fv})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        self.plot_canvas.draw()

    def _subplot_all(self):
        if self.state.x is None:
            messagebox.showwarning("Missing", "Extract variables first.")
            return

        y_dict, y_label = self._get_y_dict()
        if y_dict is None:
            messagebox.showwarning("Missing", f"{y_label} not available.")
            return

        h_l_values = sorted(self.state.h_l_values) if self.state.h_l_values else []
        if not h_l_values:
            messagebox.showwarning("Missing", "No h/l metadata found.")
            return

        import matplotlib.pyplot as plt

        n_hl = len(h_l_values)
        ncols = min(3, n_hl)
        nrows = int(np.ceil(n_hl / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes_flat = np.array(axes).flatten() if n_hl > 1 else [axes]

        for idx, hl in enumerate(h_l_values):
            ax = axes_flat[idx]
            keys = self._filter_keys("h_l", hl)
            n = len(keys)
            cmap_obj = cm.get_cmap("cividis", max(n, 1))

            for i, key in enumerate(keys):
                if key not in y_dict or key not in self.state.x:
                    continue
                m = re.search(r"Mach_([\d.]+)", key)
                label = f"M={m.group(1)}" if m else key
                ax.plot(self.state.x[key], y_dict[key], color=cmap_obj(i), label=label)

            ax.set_title(f"h/l = {hl:.2f}", fontsize=10)
            ax.set_xlabel("x [m]", fontsize=8)
            ax.set_ylabel(y_label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(n_hl, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   frameon=False, fontsize=8, title="Mach")
        fig.suptitle(f"{y_label} --- All h/l Cases", fontsize=12)
        fig.tight_layout()

        self.plot_canvas.replace_figure(fig)
        self.plot_canvas.draw()

    def _filter_keys(self, param, value):
        keys = []
        for key in self.state.x:
            if param == "h_l":
                m = re.search(r"h_l_([\d.]+)", key)
                if m and abs(float(m.group(1)) - value) < 1e-6:
                    keys.append(key)
            elif param == "mach":
                m = re.search(r"Mach_([\d.]+)", key)
                if m and abs(float(m.group(1)) - value) < 1e-6:
                    keys.append(key)
        keys.sort()
        return keys
