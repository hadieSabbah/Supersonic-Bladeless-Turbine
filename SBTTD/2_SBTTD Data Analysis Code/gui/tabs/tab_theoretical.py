"""
Tab 7 --- Theoretical Models (Shock-Expansion)
Run shock-expansion analysis, small-perturbation solver, and compare
theoretical predictions to CFD results.
"""

import customtkinter as ctk
import threading
import numpy as np
import re
from tkinter import messagebox
from gui.widgets.plot_canvas import PlotCanvas


class TheoreticalTab(ctk.CTkFrame):
    def __init__(self, master, state, status_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.status = status_callback or (lambda msg: None)
        self._build_ui()

    def _build_ui(self):
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            ctrl, text="Theoretical Models",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(ctrl, text="Model:").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.model_var = ctk.StringVar(value="Small Perturbation")
        ctk.CTkOptionMenu(
            ctrl,
            values=["Small Perturbation", "Shock-Expansion (SE)", "Combined (SP + SE)"],
            variable=self.model_var,
            width=220,
        ).grid(row=1, column=1, columnspan=2, padx=5, pady=3, sticky="w")

        btn_row = ctk.CTkFrame(ctrl)
        btn_row.grid(row=2, column=0, columnspan=4, pady=10, sticky="w")

        ctk.CTkButton(btn_row, text="Run Model", command=self._run_model).pack(
            side="left", padx=5
        )

        # Plot controls
        plot_ctrl = ctk.CTkFrame(self)
        plot_ctrl.pack(fill="x", padx=10, pady=5)

        self.plot_var = ctk.StringVar(value="Scaled Force vs h/l")
        ctk.CTkOptionMenu(
            plot_ctrl,
            values=[
                "Scaled Force vs h/l",
                "SE Scaled Force vs h/l",
                "Pressure Distribution",
            ],
            variable=self.plot_var,
        ).pack(side="left", padx=10, pady=5)

        ctk.CTkButton(plot_ctrl, text="Plot", command=self._plot).pack(
            side="left", padx=5, pady=5
        )

        self.lbl = ctk.CTkLabel(self, text="Select a model and run.")
        self.lbl.pack(padx=10, anchor="w")

        self.plot_canvas = PlotCanvas(self, figsize=(9, 5))
        self.plot_canvas.pack(fill="both", expand=True, padx=10, pady=10)

    def _run_model(self):
        if self.state.ds_by_case is None:
            messagebox.showwarning("Missing", "Load data first (Tab 1).")
            return
        if not self.state.h_l_values:
            messagebox.showwarning("Missing", "No h/l values found. Load and parse data first.")
            return

        self.status("Running theoretical model...")

        def task():
            try:
                h_l_values = np.array(sorted(self.state.h_l_values))
                model = self.model_var.get()

                if model == "Small Perturbation":
                    from utils.models import smallPertSolver
                    result = smallPertSolver(h_l_values, self.state.ds_by_case)
                    self.state.axialForceScaled = result
                    self.lbl.configure(text="Small-perturbation model complete.")

                elif model == "Shock-Expansion (SE)":
                    from utils.models import smallPertSolver_with_SE
                    result, result_se = smallPertSolver_with_SE(
                        h_l_values, self.state.ds_by_case
                    )
                    self.state.axialForceScaled = result
                    self.state.axialForceScaled_SE = result_se
                    self.lbl.configure(text="SE model complete.")

                elif model == "Combined (SP + SE)":
                    from utils.models import smallPertSolver_combined
                    result, result_se, result_comb = smallPertSolver_combined(
                        h_l_values, self.state.ds_by_case
                    )
                    self.state.axialForceScaled = result
                    self.state.axialForceScaled_SE = result_se
                    self.state.axialForceScaled_combined = result_comb
                    self.lbl.configure(text="Combined model complete.")

                self.status("Theoretical model complete.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status(f"Model failed: {e}")

        threading.Thread(target=task, daemon=True).start()

    def _plot(self):
        import matplotlib.cm as cm

        choice = self.plot_var.get()
        fig, ax = self.plot_canvas.get_fig_ax()
        ax.cla()

        h_l_values = sorted(self.state.h_l_values) if self.state.h_l_values else []
        mach_values = sorted(self.state.mach_values) if self.state.mach_values else []

        if choice in ("Scaled Force vs h/l", "SE Scaled Force vs h/l"):
            if choice == "Scaled Force vs h/l":
                data = self.state.axialForceScaled
                title = r"$F_{RANS}/F_{Small\;Pert}$ vs h/l"
            else:
                data = self.state.axialForceScaled_SE
                title = r"$F_{RANS}/F_{SE}$ vs h/l"

            if data is None:
                messagebox.showwarning("Missing", "Run the corresponding model first.")
                return

            cmap_obj = cm.get_cmap("cividis", max(len(mach_values), 1))
            for i, m in enumerate(mach_values):
                hs, vals = [], []
                for hl in h_l_values:
                    key = f"h_l_{hl:.2f}_Mach_{m:.1f}"
                    if key in data:
                        hs.append(hl)
                        vals.append(data[key])
                if hs:
                    ax.plot(hs, vals, "o-", color=cmap_obj(i), label=f"M={m}", markersize=5)

            ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.set_xlabel("h/l")
            ax.set_ylabel(r"$F_{RANS}/F_{theory}$")
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        elif choice == "Pressure Distribution":
            if self.state.x is None or self.state.P is None:
                messagebox.showwarning("Missing", "No data.")
                return

            keys = sorted(self.state.x.keys())[:6]
            cmap_obj = cm.get_cmap("tab10", max(len(keys), 1))
            for i, key in enumerate(keys):
                ax.plot(self.state.x[key], self.state.P[key],
                        color=cmap_obj(i), label=key, linewidth=0.8)

            ax.set_xlabel("x [m]")
            ax.set_ylabel("P [Pa]")
            ax.set_title("Pressure Distribution (first 6 cases)")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

        self.plot_canvas.draw()
