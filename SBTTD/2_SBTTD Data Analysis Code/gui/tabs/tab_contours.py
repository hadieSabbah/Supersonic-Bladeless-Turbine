"""
Tab 8 --- Contour Export & Pressure-Loss Plots
"""

import customtkinter as ctk
import threading
import numpy as np
import re
from tkinter import messagebox
from gui.widgets.dir_browser import DirBrowser
from gui.widgets.plot_canvas import PlotCanvas


class ContoursTab(ctk.CTkFrame):
    def __init__(self, master, state, status_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.status = status_callback or (lambda msg: None)
        self._abort = threading.Event()
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # === Mach Contour Export ===
        sec = ctk.CTkFrame(self)
        sec.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(sec, text="Mach Contour Export (Tecplot)",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        self.dir_study = DirBrowser(sec, label_text="Study directory:")
        self.dir_study.pack(fill="x", padx=10, pady=3)
        self.dir_png = DirBrowser(sec, label_text="PNG output dir:")
        self.dir_png.pack(fill="x", padx=10, pady=3)

        row_lay = ctk.CTkFrame(sec)
        row_lay.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row_lay, text="Layout file:").pack(side="left", padx=(0, 5))
        self.entry_layout = ctk.CTkEntry(row_lay, width=400)
        self.entry_layout.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ctk.CTkButton(row_lay, text="Browse", width=70,
                       command=self._browse_layout).pack(side="left")

        # Contour legend range
        row_r = ctk.CTkFrame(sec)
        row_r.pack(fill="x", padx=10, pady=3)

        ctk.CTkLabel(row_r, text="Contour legend min:").pack(side="left", padx=(0, 5))
        self.entry_cmin = ctk.CTkEntry(row_r, width=70, placeholder_text="auto")
        self.entry_cmin.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(row_r, text="max:").pack(side="left", padx=(0, 5))
        self.entry_cmax = ctk.CTkEntry(row_r, width=70, placeholder_text="auto")
        self.entry_cmax.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(row_r, text="Levels:").pack(side="left", padx=(0, 5))
        self.entry_levels = ctk.CTkEntry(row_r, width=45)
        self.entry_levels.insert(0, "6")
        self.entry_levels.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(row_r, text="Res:").pack(side="left", padx=(0, 5))
        self.entry_res = ctk.CTkEntry(row_r, width=55)
        self.entry_res.insert(0, "4096")
        self.entry_res.pack(side="left")

        ctk.CTkLabel(
            sec,
            text="Leave min/max blank to auto-detect from inlet zone mean Mach. "
                 "Levels = number of contour entries (e.g. 6).",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(anchor="w", padx=15, pady=(0, 3))

        # Progress bar + export button
        prog_row = ctk.CTkFrame(sec)
        prog_row.pack(fill="x", padx=10, pady=5)

        self.btn_export = ctk.CTkButton(prog_row, text="Export Contours",
                                        command=self._export_contours)
        self.btn_export.pack(side="left", padx=(0, 10))

        self.progress = ctk.CTkProgressBar(prog_row, width=300)
        self.progress.set(0)
        self.progress.pack(side="left", padx=(0, 10))

        self.lbl_progress = ctk.CTkLabel(prog_row, text="")
        self.lbl_progress.pack(side="left")

        self.btn_stop = ctk.CTkButton(
            prog_row, text="Stop", width=60, state="disabled",
            fg_color="#c0392b", hover_color="#922b21",
            command=self._stop_export)
        self.btn_stop.pack(side="left", padx=(10, 0))

        # === Pressure Loss ===
        sec2 = ctk.CTkFrame(self)
        sec2.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(sec2, text="Total Pressure Loss",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        btn2 = ctk.CTkFrame(sec2)
        btn2.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(btn2, text="Plot Pressure Loss Map", command=self._plot_ploss).pack(side="left", padx=5)
        ctk.CTkButton(btn2, text="Plot Contours per h/l", command=self._plot_contours_per_hl).pack(side="left", padx=5)

        # === Contour visualisation ===
        sec3 = ctk.CTkFrame(self)
        sec3.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(sec3, text="Mach Contour Visualisation",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        row_hl = ctk.CTkFrame(sec3)
        row_hl.pack(fill="x", padx=10, pady=3)
        ctk.CTkLabel(row_hl, text="h/l value:").pack(side="left", padx=(0, 5))
        self.entry_hl = ctk.CTkEntry(row_hl, width=80)
        self.entry_hl.insert(0, "0.04")
        self.entry_hl.pack(side="left")
        ctk.CTkLabel(row_hl, text="ncols:").pack(side="left", padx=(15, 5))
        self.entry_ncols = ctk.CTkEntry(row_hl, width=50)
        self.entry_ncols.insert(0, "3")
        self.entry_ncols.pack(side="left")

        self.dir_viscous = DirBrowser(sec3, label_text="Viscous contour dir:")
        self.dir_viscous.pack(fill="x", padx=10, pady=3)
        ctk.CTkButton(sec3, text="Plot Mach Contour Grid", command=self._plot_contours_per_hl).pack(
            padx=10, pady=5, anchor="w")

        self.plot_canvas = PlotCanvas(self, figsize=(9, 4))
        self.plot_canvas.pack(fill="both", expand=True, padx=10, pady=10)

    # ------------------------------------------------------------------ helpers
    def _browse_layout(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Tecplot Layout",
            filetypes=[("Tecplot Layout", "*.lay"), ("All files", "*.*")])
        if path:
            self.entry_layout.delete(0, "end")
            self.entry_layout.insert(0, path)

    def _opt_float(self, entry):
        t = entry.get().strip()
        return float(t) if t else None

    # ------------------------------------------------------------------ export
    def _export_contours(self):
        study  = self.dir_study.get()
        png    = self.dir_png.get()
        layout = self.entry_layout.get().strip()

        if not study or not png or not layout:
            messagebox.showwarning("Missing", "Fill in study dir, PNG dir, and layout file.")
            return

        try:
            cmin = self._opt_float(self.entry_cmin)
            cmax = self._opt_float(self.entry_cmax)
            levels = int(self.entry_levels.get())
            res    = int(self.entry_res.get())
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        self._abort.clear()
        self.btn_export.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress.set(0)
        self.lbl_progress.configure(text="Starting...")
        self.status("Exporting Mach contours...")

        def on_progress(current, total, name):
            frac = current / total
            self.progress.set(frac)
            self.lbl_progress.configure(text=f"{current}/{total}  —  {name}")
            self.status(f"{current}/{total}: {name}")

        def task():
            try:
                from utils.plotting import export_mach_contours
                export_mach_contours(
                    study_dir=study, png_dest=png, layout_dest=layout,
                    res_number=res, contour_min=cmin, contour_max=cmax,
                    num_levels=levels, progress_callback=on_progress,
                    abort_event=self._abort,
                )
                if self._abort.is_set():
                    self.lbl_progress.configure(text="Stopped")
                    self.status("Export stopped by user.")
                else:
                    self.lbl_progress.configure(text="Done!")
                    self.status("Contour export complete.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status(f"Export failed: {e}")
            finally:
                self.btn_export.configure(state="normal")
                self.btn_stop.configure(state="disabled")

        threading.Thread(target=task, daemon=True).start()

    def _stop_export(self):
        self._abort.set()

    # ------------------------------------------------------------------ plots
    def _plot_ploss(self):
        if self.state.ds_by_case_quad is None or self.state.ds_by_case_inlet is None:
            messagebox.showwarning("Missing", "Load quad and inlet data first (Tab 1).")
            return

        fig, ax = self.plot_canvas.get_fig_ax()
        ax.cla()

        hl_arr, mach_arr, loss_arr = [], [], []
        for key in self.state.ds_by_case_quad:
            hl_m = re.search(r"h_l_([\d.]+)", key)
            mach_m = re.search(r"Mach_([\d.]+)", key)
            if not hl_m or not mach_m:
                continue
            try:
                p0_out = np.mean(self.state.ds_by_case_quad[key]["P_total"].data)
                p0_in  = np.mean(self.state.ds_by_case_inlet[key]["P_total"].data)
                hl_arr.append(float(hl_m.group(1)))
                mach_arr.append(float(mach_m.group(1)))
                loss_arr.append((1.0 - p0_out / p0_in) * 100)
            except Exception:
                continue

        if hl_arr:
            sc = ax.scatter(hl_arr, mach_arr, c=loss_arr, cmap="hot", s=150,
                            edgecolors="k", linewidths=0.5)
            self.plot_canvas.fig.colorbar(sc, ax=ax, label="Pressure Loss [%]")
            ax.set_xlabel("h/l"); ax.set_ylabel("Mach")
            ax.set_title("Total Pressure Loss Map")
            ax.grid(True, alpha=0.3)
        self.plot_canvas.draw()

    def _plot_contours_per_hl(self):
        try:
            from utils.plotting import plot_mach_contours_per_hl
            hl    = float(self.entry_hl.get())
            ncols = int(self.entry_ncols.get())
            viscous = self.dir_viscous.get()
            if not viscous:
                messagebox.showwarning("Missing", "Select viscous contour directory.")
                return
            plot_mach_contours_per_hl(hl_value=hl, viscous_dir=viscous, ncols=ncols, save=False)
            self.status(f"Contour grid plotted for h/l = {hl}.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
