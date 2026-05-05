"""
Tab 2 — Variable Extraction
Run variableImporterMasked on loaded data and compute tau_wall.
"""

import customtkinter as ctk
import threading
import numpy as np
from tkinter import messagebox


class VariablesTab(ctk.CTkFrame):
    def __init__(self, master, state, status_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.status = status_callback or (lambda msg: None)
        self._build_ui()

    def _build_ui(self):
        # --- Mask parameters ---
        sec = ctk.CTkFrame(self)
        sec.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(sec, text="Variable Extraction (variableImporterMasked)",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        row = ctk.CTkFrame(sec)
        row.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(row, text="min_l:").pack(side="left", padx=(0, 5))
        self.entry_min = ctk.CTkEntry(row, width=80)
        self.entry_min.insert(0, "0")
        self.entry_min.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(row, text="max_l:").pack(side="left", padx=(0, 5))
        self.entry_max = ctk.CTkEntry(row, width=80)
        self.entry_max.insert(0, "0.1")
        self.entry_max.pack(side="left")

        self.btn_extract = ctk.CTkButton(sec, text="Extract Variables",
                                         command=self._extract)
        self.btn_extract.pack(padx=10, pady=10, anchor="w")

        # --- Tau wall computation ---
        sec2 = ctk.CTkFrame(self)
        sec2.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(sec2, text="Wall Shear Stress (tau_wall)",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        self.btn_tau = ctk.CTkButton(sec2, text="Compute tau_wall",
                                     command=self._compute_tau_wall)
        self.btn_tau.pack(padx=10, pady=10, anchor="w")

        # --- Status ---
        self.lbl_status = ctk.CTkLabel(self, text="Variables not yet extracted.")
        self.lbl_status.pack(padx=10, pady=10, anchor="w")

    def _extract(self):
        if not self.state.data_loaded:
            messagebox.showwarning("No data", "Load data in the Data I/O tab first.")
            return

        try:
            min_l = float(self.entry_min.get())
            max_l = float(self.entry_max.get())
        except ValueError:
            messagebox.showerror("Invalid input", "min_l and max_l must be numbers.")
            return

        self.status("Extracting variables...")
        self.btn_extract.configure(state="disabled")

        def task():
            try:
                from utils.parameterComputation import variableImporterMasked

                # Wall data (masked)
                (self.state.y_plus, self.state.tau_x, self.state.tau_y,
                 self.state.tau_separation, self.state.tau_separation_idx,
                 self.state.x, self.state.y, self.state.T,
                 self.state.P, self.state.Px, self.state.Py,
                 self.state.P0, self.state.rho, self.state.mach,
                 self.state.omega_z, self.state.u, self.state.v,
                 self.state.q_dot) = variableImporterMasked(
                    self.state.ds_by_case, min_l, max_l)

                # Quadrant data (unmasked)
                (_, _, _, _, _,
                 self.state.x_quad, self.state.y_quad, _,
                 self.state.P_quad, _, _, self.state.P0_quad,
                 self.state.rho_quad, self.state.mach_quad,
                 _, _, _, _) = variableImporterMasked(
                    self.state.ds_by_case_quad, 0, 0, mask_input=False)

                # Inlet data (unmasked)
                (_, _, _, _, _,
                 _, _, _,
                 self.state.P_inlet, _, _, _,
                 _, self.state.mach_inlet,
                 _, self.state.u_inlet, _, _) = variableImporterMasked(
                    self.state.ds_by_case_inlet, 0, 0, mask_input=False)

                self.state.variables_extracted = True
                self.lbl_status.configure(
                    text=f"Variables extracted for {len(self.state.x)} cases.")
                self.status("Variable extraction complete.")
            except Exception as e:
                messagebox.showerror("Extraction Error", str(e))
                self.status(f"Extraction failed: {e}")
            finally:
                self.btn_extract.configure(state="normal")

        threading.Thread(target=task, daemon=True).start()

    def _compute_tau_wall(self):
        if self.state.x is None or self.state.tau_x is None:
            messagebox.showwarning("No variables", "Extract variables first.")
            return

        tau_wall = {}
        for key in self.state.x:
            dx = np.gradient(self.state.x[key])
            dy = np.gradient(self.state.y[key])
            ds = np.sqrt(dx**2 + dy**2)
            tx = dx / ds
            ty = dy / ds
            tau_wall[key] = self.state.tau_x[key] * tx + self.state.tau_y[key] * ty

        self.state.tau_wall = tau_wall
        self.lbl_status.configure(text="tau_wall computed for all cases.")
        self.status("tau_wall computed.")
