"""
Tab 6 --- Force & Power Computation
Compute axial force from pressure integration, torque, and power.
"""

import customtkinter as ctk
import threading
import numpy as np
import re
from tkinter import messagebox
from gui.widgets.plot_canvas import PlotCanvas


class ForcePowerTab(ctk.CTkFrame):
    def __init__(self, master, state, status_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.status = status_callback or (lambda msg: None)
        self._build_ui()

    def _build_ui(self):
        ctrl = ctk.CTkFrame(self)
        ctrl.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            ctrl, text="Force & Power Analysis",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=5)

        ctk.CTkLabel(ctrl, text="Rotor radius R [m]:").grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.entry_radius = ctk.CTkEntry(ctrl, width=100)
        self.entry_radius.insert(0, "0.05")
        self.entry_radius.grid(row=1, column=1, padx=5, pady=3, sticky="w")

        ctk.CTkLabel(ctrl, text="RPM:").grid(row=1, column=2, padx=5, pady=3, sticky="w")
        self.entry_rpm = ctk.CTkEntry(ctrl, width=100)
        self.entry_rpm.insert(0, "10000")
        self.entry_rpm.grid(row=1, column=3, padx=5, pady=3, sticky="w")

        btn_row = ctk.CTkFrame(ctrl)
        btn_row.grid(row=2, column=0, columnspan=4, pady=10, sticky="w")

        ctk.CTkButton(btn_row, text="Compute Force & Torque", command=self._compute_force).pack(
            side="left", padx=5
        )
        ctk.CTkButton(btn_row, text="Build DataFrame", command=self._build_df).pack(
            side="left", padx=5
        )

        plot_ctrl = ctk.CTkFrame(self)
        plot_ctrl.pack(fill="x", padx=10, pady=5)

        self.plot_var = ctk.StringVar(value="Axial Force vs h/l")
        ctk.CTkOptionMenu(
            plot_ctrl,
            values=[
                "Axial Force vs h/l",
                "Torque vs h/l",
                "Power vs h/l",
                "Pressure Loss vs Force",
            ],
            variable=self.plot_var,
        ).pack(side="left", padx=10, pady=5)

        ctk.CTkButton(plot_ctrl, text="Plot", command=self._plot).pack(
            side="left", padx=5, pady=5
        )

        self.lbl = ctk.CTkLabel(self, text="Compute force to see results.")
        self.lbl.pack(padx=10, anchor="w")

        self.plot_canvas = PlotCanvas(self, figsize=(9, 5))
        self.plot_canvas.pack(fill="both", expand=True, padx=10, pady=10)

    def _compute_force(self):
        if self.state.x is None or self.state.P is None:
            messagebox.showwarning("Missing", "Extract variables first (Tab 2).")
            return

        self.status("Computing axial force & torque...")

        def task():
            try:
                from utils.models import compute_force_2D

                R = float(self.entry_radius.get())
                force_dict = {}
                torque_dict = {}

                for key in self.state.x:
                    result = compute_force_2D(
                        self.state.x[key], self.state.y[key],
                        self.state.P[key], R
                    )
                    force_dict[key] = result["F_theta"]
                    torque_dict[key] = result["tau"]

                self.state.df_axial_force = force_dict
                self.state._torque_dict = torque_dict
                self.lbl.configure(text=f"Force & torque computed for {len(force_dict)} cases.")
                self.status("Force computation complete.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status(f"Force computation failed: {e}")

        threading.Thread(target=task, daemon=True).start()

    def _build_df(self):
        if self.state.tau_wall is None or self.state.x is None:
            messagebox.showwarning("Missing", "Compute tau_wall first (Tab 2).")
            return

        try:
            from utils.models import create_axial_force_dataframe

            df = create_axial_force_dataframe(self.state.tau_wall, self.state.x)
            self.state._force_df = df
            self.lbl.configure(text=f"DataFrame built: {len(df)} rows.")
            self.status("Force DataFrame created.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _plot(self):
        if self.state.df_axial_force is None:
            messagebox.showwarning("Missing", "Compute force first.")
            return

        import matplotlib.cm as cm

        choice = self.plot_var.get()
        fig, ax = self.plot_canvas.get_fig_ax()
        ax.cla()

        h_l_values = sorted(self.state.h_l_values) if self.state.h_l_values else []
        mach_values = sorted(self.state.mach_values) if self.state.mach_values else []

        if choice in ("Axial Force vs h/l", "Torque vs h/l", "Power vs h/l"):
            if choice == "Axial Force vs h/l":
                data = self.state.df_axial_force
                ylabel = "Axial Force [N]"
            elif choice == "Torque vs h/l":
                data = getattr(self.state, "_torque_dict", None)
                ylabel = r"Torque [N$\cdot$m]"
            else:
                rpm = float(self.entry_rpm.get())
                omega = 2 * np.pi * rpm / 60.0
                torque = getattr(self.state, "_torque_dict", None)
                if torque is None:
                    messagebox.showwarning("Missing", "Compute force first.")
                    return
                data = {k: v * omega for k, v in torque.items()}
                ylabel = "Power [W]"

            if data is None:
                messagebox.showwarning("Missing", "Compute force first.")
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

            ax.set_xlabel("h/l")
            ax.set_ylabel(ylabel)
            ax.set_title(choice)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        elif choice == "Pressure Loss vs Force":
            if self.state.P0_quad is None or self.state.ds_by_case_inlet is None:
                messagebox.showwarning("Missing", "Quad/inlet data not loaded.")
                return

            forces, losses = [], []
            for key in self.state.df_axial_force:
                f = self.state.df_axial_force[key]
                try:
                    p0_out = np.mean(self.state.P0_quad[key])
                    p0_in = np.mean(self.state.ds_by_case_inlet[key]["P_total"].data)
                    loss = 1.0 - p0_out / p0_in
                    forces.append(f)
                    losses.append(loss * 100)
                except Exception:
                    continue

            if forces:
                ax.scatter(forces, losses, c="steelblue", edgecolors="k", s=50)
                ax.set_xlabel("Axial Force [N]")
                ax.set_ylabel("Total Pressure Loss [%]")
                ax.set_title("Pressure Loss vs Axial Force")
                ax.grid(True, alpha=0.3)

        self.plot_canvas.draw()
