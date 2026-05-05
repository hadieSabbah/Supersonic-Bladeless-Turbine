"""
Tab 3 — Boundary Layer Analysis
Compute BL thickness via Tecplot line extraction, save/load results, plot.
"""

import customtkinter as ctk
import threading
import numpy as np
from tkinter import messagebox
from gui.widgets.dir_browser import DirBrowser
from gui.widgets.plot_canvas import PlotCanvas


class BoundaryLayerTab(ctk.CTkFrame):
    def __init__(self, master, state, status_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.status = status_callback or (lambda msg: None)
        self._build_ui()

    def _build_ui(self):
        left = ctk.CTkFrame(self)
        left.pack(side="left", fill="y", padx=10, pady=10)

        # --- Parameters ---
        ctk.CTkLabel(left, text="BL Thickness Computation",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=5)

        params = ctk.CTkFrame(left)
        params.pack(fill="x", pady=5)

        ctk.CTkLabel(params, text="Ny (num points):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.entry_ny = ctk.CTkEntry(params, width=80)
        self.entry_ny.insert(0, "1000")
        self.entry_ny.grid(row=0, column=1, padx=5, pady=2)

        ctk.CTkLabel(params, text="BL height [mm]:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.entry_blh = ctk.CTkEntry(params, width=80)
        self.entry_blh.insert(0, "3")
        self.entry_blh.grid(row=1, column=1, padx=5, pady=2)

        ctk.CTkLabel(params, text="Stride:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.entry_stride = ctk.CTkEntry(params, width=80)
        self.entry_stride.insert(0, "1")
        self.entry_stride.grid(row=2, column=1, padx=5, pady=2)

        ctk.CTkLabel(params, text="Threshold:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.entry_thresh = ctk.CTkEntry(params, width=80)
        self.entry_thresh.insert(0, "0.02")
        self.entry_thresh.grid(row=3, column=1, padx=5, pady=2)

        # File paths for tecplot cases
        self.dir_study = DirBrowser(left, label_text="Study directory:")
        self.dir_study.pack(fill="x", pady=5)

        self.btn_compute = ctk.CTkButton(left, text="Compute BL Thickness (Tecplot)",
                                         command=self._compute_bl)
        self.btn_compute.pack(pady=5, anchor="w")

        # --- Save / Load ---
        ctk.CTkLabel(left, text="Save / Load Results",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(15, 5))

        self.dir_bl_save = DirBrowser(left, label_text="BL results dir:")
        self.dir_bl_save.pack(fill="x", pady=5)

        btn_row = ctk.CTkFrame(left)
        btn_row.pack(fill="x", pady=5)
        ctk.CTkButton(btn_row, text="Save", command=self._save_bl).pack(side="left", padx=5)
        ctk.CTkButton(btn_row, text="Load Latest", command=self._load_bl).pack(side="left", padx=5)

        # --- Plot ---
        ctk.CTkButton(left, text="Plot BL Thickness", command=self._plot_bl).pack(pady=10, anchor="w")

        # --- Plot canvas (right side) ---
        self.plot = PlotCanvas(self, figsize=(7, 4))
        self.plot.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def _compute_bl(self):
        if self.state.x is None:
            messagebox.showwarning("No variables", "Extract variables first (Tab 2).")
            return
        if not self.state.tecplot_connected:
            messagebox.showwarning("Tecplot", "Connect to Tecplot first (Tab 1).")
            return

        study_dir = self.dir_study.get()
        if not study_dir:
            messagebox.showwarning("Missing", "Select the study directory.")
            return

        self.status("Computing BL thickness — this takes time...")
        self.btn_compute.configure(state="disabled")

        def task():
            try:
                import tecplot as tp
                from tecplot.constant import PlotType
                from pathlib import Path

                num_points = int(self.entry_ny.get())
                bl_h = float(self.entry_blh.get()) / 1000
                stride = int(self.entry_stride.get())
                threshold = float(self.entry_thresh.get())
                slice_cut_int = 0

                root = Path(study_dir)
                sub1 = [p for p in root.iterdir() if p.is_dir()]
                sub2 = [p for d in sub1 for p in d.iterdir() if p.is_dir()]
                file_paths = [p / "mcfd_tec.bin" for p in sub2]

                delta_n_dict = {}

                for idx, key in enumerate(self.state.ds_by_case.keys()):
                    tp.new_layout()
                    tp.data.load_tecplot(file_paths[idx].as_posix())
                    fr = tp.active_frame()
                    fr.plot_type = PlotType.Cartesian2D

                    dx_ds = np.gradient(self.state.x[key][slice_cut_int:-1])
                    dy_ds = np.gradient(self.state.y[key][slice_cut_int:-1])

                    nx, ny_arr = -dy_ds, dx_ds
                    norm = np.hypot(nx, ny_arr)
                    norm = np.where(norm < 1e-12, np.nan, norm)
                    ux, uy = nx / norm, ny_arr / norm

                    x_final = self.state.x[key][slice_cut_int:-1] + bl_h * ux
                    y_final = self.state.y[key][slice_cut_int:-1] + bl_h * uy
                    ok = np.isfinite(x_final) & np.isfinite(y_final)

                    x_start = self.state.x[key][slice_cut_int:-1][ok][::stride]
                    y_start = self.state.y[key][slice_cut_int:-1][ok][::stride]
                    x_end = x_final[ok][::stride]
                    y_end = y_final[ok][::stride]

                    dx_arr = dx_ds[ok][::stride]
                    dy_arr = dy_ds[ok][::stride]
                    ds_arr = np.sqrt(dx_arr**2 + dy_arr**2)
                    tx_arr = dx_arr / ds_arr
                    ty_arr = dy_arr / ds_arr

                    n = min(len(x_start), len(y_start), len(x_end), len(y_end))
                    x_pairs = np.column_stack((x_start[:n], x_end[:n]))
                    y_pairs = np.column_stack((y_start[:n], y_end[:n]))

                    N = x_pairs.shape[0]
                    delta_n_dict[key] = np.full(N, np.nan, dtype=float)

                    for i in range(N):
                        p0 = (float(x_pairs[i, 0]), float(y_pairs[i, 0]), 0.0)
                        p1 = (float(x_pairs[i, 1]), float(y_pairs[i, 1]), 0.0)
                        line = tp.data.extract.extract_line([p0, p1], num_points=num_points)

                        tau_x_l = line.values("Tau_x").as_numpy_array()[0]
                        tau_y_l = line.values("Tau_y").as_numpy_array()[0]
                        tau_wall_l = tau_x_l * tx_arr[i] + tau_y_l * ty_arr[i]

                        if tau_wall_l <= 0:
                            continue

                        x_BL = line.values("X").as_numpy_array()
                        y_BL = line.values("Y").as_numpy_array()
                        omega_z_BL = line.values("Vort_z").as_numpy_array()

                        d_omega = np.gradient(omega_z_BL)
                        d_omega_norm = np.abs(d_omega) / (np.abs(d_omega).max() + 1e-30)
                        start_idx = max(1, num_points // 10)
                        candidates = np.where(d_omega_norm[start_idx:] < threshold)[0]

                        if len(candidates) > 0:
                            y_index = candidates[0] + start_idx
                        else:
                            y_index = num_points - 1

                        if y_BL[0] > y_BL[-1]:
                            y_BL = y_BL[::-1]
                        y_corr = y_BL - y_BL[0]
                        delta_n_dict[key][i] = float(y_corr[y_index]) * 1000

                    self.status(f"BL done: {key}")

                self.state.delta_n_dict = delta_n_dict
                self.status("BL thickness computation complete for all cases.")
            except Exception as e:
                messagebox.showerror("BL Error", str(e))
                self.status(f"BL computation failed: {e}")
            finally:
                self.btn_compute.configure(state="normal")

        threading.Thread(target=task, daemon=True).start()

    def _save_bl(self):
        if self.state.delta_n_dict is None:
            messagebox.showwarning("No data", "Compute or load BL data first.")
            return
        save_dir = self.dir_bl_save.get()
        if not save_dir:
            messagebox.showwarning("Missing", "Select directory.")
            return

        from pathlib import Path
        from datetime import date
        out = Path(save_dir) / f"BL_results_{date.today().isoformat()}"
        out.mkdir(exist_ok=True)
        np.savez(out / "delta_n_dict.npz", **self.state.delta_n_dict)
        self.status(f"Saved to {out}")

    def _load_bl(self):
        load_dir = self.dir_bl_save.get()
        if not load_dir:
            messagebox.showwarning("Missing", "Select directory.")
            return

        from pathlib import Path
        from datetime import date
        base = Path(load_dir)
        all_dirs = sorted(base.glob("BL_results_*"))
        if not all_dirs:
            messagebox.showwarning("Not found", "No BL_results_* folders found.")
            return

        latest = max(all_dirs, key=lambda p: p.name)
        loaded = np.load(latest / "delta_n_dict.npz", allow_pickle=False)
        self.state.delta_n_dict = {k: loaded[k] for k in loaded.files}
        self.status(f"Loaded BL data from {latest.name}")

    def _plot_bl(self):
        if self.state.delta_n_dict is None:
            messagebox.showwarning("No data", "Compute or load BL data first.")
            return

        fig, ax = self.plot.get_fig_ax()
        ax.cla()

        for key, delta in self.state.delta_n_dict.items():
            valid = np.isfinite(delta)
            if self.state.x is not None and key in self.state.x:
                x_arr = self.state.x[key][:-1][:len(delta)][valid]
            else:
                x_arr = np.arange(np.sum(valid))
            ax.plot(x_arr, delta[valid], label=key)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("BL Thickness [mm]")
        ax.set_title("Boundary Layer Thickness")
        ax.legend(fontsize=7, ncol=2)
        self.plot.draw()
