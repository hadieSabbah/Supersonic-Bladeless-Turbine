"""
Tab 4 --- Separation Analysis
Compute separation length, find separation/reattachment points, generate
heatmaps and contour scatter plots.
"""

import customtkinter as ctk
import threading
import numpy as np
from tkinter import messagebox
from gui.widgets.plot_canvas import PlotCanvas


class SeparationTab(ctk.CTkFrame):
    def __init__(self, master, state, status_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.status = status_callback or (lambda msg: None)
        self._build_ui()

    def _build_ui(self):
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            top, text="Separation Analysis",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=5, pady=5)

        btn_row = ctk.CTkFrame(top)
        btn_row.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(btn_row, text="Compute Separation", command=self._compute).pack(
            side="left", padx=5
        )
        ctk.CTkButton(btn_row, text="Find Geometry Extrema", command=self._find_extrema).pack(
            side="left", padx=5
        )

        plot_section = ctk.CTkFrame(self)
        plot_section.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            plot_section, text="Visualisation",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=5, pady=5)

        self.plot_var = ctk.StringVar(value="Separation Heatmap")
        ctk.CTkOptionMenu(
            plot_section,
            values=[
                "Separation Heatmap",
                "Sep Length vs h/l",
                "Sep Length (non-dim) vs h/l",
                "BL & Separation Contours",
            ],
            variable=self.plot_var,
        ).pack(side="left", padx=10, pady=5)

        ctk.CTkButton(plot_section, text="Plot", command=self._plot).pack(
            side="left", padx=5, pady=5
        )

        self.lbl = ctk.CTkLabel(self, text="Run 'Compute Separation' first.")
        self.lbl.pack(padx=10, anchor="w")

        self.plot_canvas = PlotCanvas(self, figsize=(9, 5))
        self.plot_canvas.pack(fill="both", expand=True, padx=10, pady=10)

    def _compute(self):
        if self.state.x is None or self.state.tau_wall is None:
            messagebox.showwarning("Missing", "Extract variables and compute tau_wall first (Tabs 2).")
            return

        self.status("Computing separation lengths...")

        def task():
            try:
                from utils.models import find_sepLength

                (
                    self.state.sep_length,
                    self.state.sep_length_nonDim,
                    self.state.x_sep,
                    self.state.y_sep,
                    self.state.x_attach,
                    self.state.y_attach,
                ) = find_sepLength(self.state.ds_by_case, self.state.x, self.state.tau_wall)

                n_sep = sum(
                    1 for v in self.state.sep_length.values() if v > 0
                )
                self.lbl.configure(text=f"Separation computed: {n_sep} cases with separation.")
                self.status("Separation computation complete.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status(f"Separation failed: {e}")

        threading.Thread(target=task, daemon=True).start()

    def _find_extrema(self):
        if self.state.x is None:
            messagebox.showwarning("Missing", "Extract variables first.")
            return

        try:
            from utils.models import max_min_finder

            (
                self.state.x_max,
                self.state.x_min,
                self.state.y_max,
                self.state.y_min,
            ) = max_min_finder(self.state.ds_by_case, self.state.x, self.state.y)
            self.status("Geometry extrema computed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _plot(self):
        if self.state.sep_length is None:
            messagebox.showwarning("Missing", "Run 'Compute Separation' first.")
            return

        import re
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        choice = self.plot_var.get()
        fig, ax = self.plot_canvas.get_fig_ax()
        ax.cla()

        h_l_values = sorted(self.state.h_l_values) if self.state.h_l_values else []
        mach_values = sorted(self.state.mach_values) if self.state.mach_values else []

        if choice == "Separation Heatmap":
            if not h_l_values or not mach_values:
                messagebox.showwarning("Missing", "No h/l or Mach metadata.")
                return

            matrix = np.zeros((len(h_l_values), len(mach_values)))
            for i, hl in enumerate(h_l_values):
                for j, m in enumerate(mach_values):
                    key = f"h_l_{hl:.2f}_Mach_{m:.1f}"
                    sl = self.state.sep_length.get(key, 0.0)
                    matrix[i, j] = 1.0 if sl > 0 else 0.0

            cmap_bin = ListedColormap(["#2ECC71", "#E74C3C"])
            im = ax.imshow(matrix, cmap=cmap_bin, aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(mach_values)))
            ax.set_xticklabels([f"{m}" for m in mach_values], fontsize=8)
            ax.set_yticks(range(len(h_l_values)))
            ax.set_yticklabels([f"{h:.2f}" for h in h_l_values], fontsize=8)
            ax.set_xlabel("Mach")
            ax.set_ylabel("h/l")
            ax.set_title("Flow Separation Occurrence")

            for i in range(len(h_l_values)):
                for j in range(len(mach_values)):
                    txt = "Sep" if matrix[i, j] == 1 else "No"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        elif choice in ("Sep Length vs h/l", "Sep Length (non-dim) vs h/l"):
            use_nd = "non-dim" in choice
            data = self.state.sep_length_nonDim if use_nd else self.state.sep_length
            ylabel = r"$L_{sep}/L_{domain}$" if use_nd else "Sep Length [m]"

            for m in mach_values:
                hs, vals = [], []
                for hl in h_l_values:
                    key = f"h_l_{hl:.2f}_Mach_{m:.1f}"
                    v = data.get(key, 0.0)
                    if v and v > 0:
                        hs.append(hl)
                        vals.append(v)
                if hs:
                    ax.plot(hs, vals, "o-", label=f"M={m}", markersize=5)

            ax.set_xlabel("h/l")
            ax.set_ylabel(ylabel)
            ax.set_title(choice)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        elif choice == "BL & Separation Contours":
            if self.state.delta_n_dict is None:
                messagebox.showwarning("Missing", "Compute BL thickness first (Tab 3).")
                return

            hl_arr, mach_arr, sep_vals = [], [], []
            for key in self.state.sep_length:
                hl_m = re.search(r"h_l_([\d.]+)", key)
                mach_m = re.search(r"Mach_([\d.]+)", key)
                if not hl_m or not mach_m:
                    continue
                hl_arr.append(float(hl_m.group(1)))
                mach_arr.append(float(mach_m.group(1)))
                nd = self.state.sep_length_nonDim.get(key, 0.0)
                sep_vals.append((nd or 0.0) * 100)

            if hl_arr:
                sc = ax.scatter(hl_arr, mach_arr, c=sep_vals, cmap="cividis", s=120, edgecolors="k", linewidths=0.5)
                self.plot_canvas.fig.colorbar(sc, ax=ax, label="Separation [%]")
                ax.set_xlabel("h/l")
                ax.set_ylabel("Mach")
                ax.set_title("Separation Percentage")
                ax.grid(True, alpha=0.3)

        self.plot_canvas.draw()
