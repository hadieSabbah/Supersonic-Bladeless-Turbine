"""
Tab 1 — Data Import / Export
Connect to Tecplot, run bigImport, or load previously saved data via runLoader.
"""

import customtkinter as ctk
import threading
from tkinter import messagebox
from gui.widgets.dir_browser import DirBrowser


class DataIOTab(ctk.CTkFrame):
    def __init__(self, master, state, status_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.status = status_callback or (lambda msg: None)

        self._build_ui()

    def _build_ui(self):
        # --- Section: Tecplot Connection ---
        sec_tp = ctk.CTkFrame(self)
        sec_tp.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(sec_tp, text="Tecplot Connection",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        self.btn_connect = ctk.CTkButton(sec_tp, text="Connect to Tecplot",
                                         command=self._connect_tecplot)
        self.btn_connect.pack(padx=10, pady=5, anchor="w")

        self.lbl_tp_status = ctk.CTkLabel(sec_tp, text="Status: Not connected")
        self.lbl_tp_status.pack(padx=10, anchor="w")

        # --- Section: Import from Tecplot (bigImport) ---
        sec_import = ctk.CTkFrame(self)
        sec_import.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(sec_import, text="Import Raw Data (bigImport)",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        self.dir_raw = DirBrowser(sec_import, label_text="Base directory:")
        self.dir_raw.pack(fill="x", padx=10, pady=5)

        row_fn = ctk.CTkFrame(sec_import)
        row_fn.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row_fn, text="File name:").pack(side="left", padx=(0, 5))
        self.entry_filename = ctk.CTkEntry(row_fn, width=200)
        self.entry_filename.insert(0, "mcfd_tec.bin")
        self.entry_filename.pack(side="left")

        self.btn_import = ctk.CTkButton(sec_import, text="Run bigImport",
                                        command=self._run_big_import)
        self.btn_import.pack(padx=10, pady=5, anchor="w")

        # --- Section: Load Saved Data (runLoader) ---
        sec_load = ctk.CTkFrame(self)
        sec_load.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(sec_load, text="Load Saved Data (runLoader)",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        self.dir_load = DirBrowser(sec_load, label_text="Saved data directory:")
        self.dir_load.pack(fill="x", padx=10, pady=5)

        self.btn_load = ctk.CTkButton(sec_load, text="Load Data",
                                      command=self._run_loader)
        self.btn_load.pack(padx=10, pady=5, anchor="w")

        # --- Section: Save Data (runSaver) ---
        sec_save = ctk.CTkFrame(self)
        sec_save.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(sec_save, text="Save Current Data (runSaver)",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=5, pady=5)

        self.btn_save = ctk.CTkButton(sec_save, text="Save Data",
                                      command=self._run_saver)
        self.btn_save.pack(padx=10, pady=5, anchor="w")

        # --- Info label ---
        self.lbl_info = ctk.CTkLabel(self, text="No data loaded yet.")
        self.lbl_info.pack(padx=10, pady=10, anchor="w")

    def _connect_tecplot(self):
        def task():
            try:
                import tecplot as tp
                tp.session.connect()
                self.state.tecplot_connected = True
                self.lbl_tp_status.configure(text="Status: Connected")
                self.status("Tecplot connected successfully.")
            except Exception as e:
                self.lbl_tp_status.configure(text=f"Status: Failed — {e}")
                self.status(f"Tecplot connection failed: {e}")

        threading.Thread(target=task, daemon=True).start()

    def _run_big_import(self):
        base = self.dir_raw.get()
        fname = self.entry_filename.get().strip()
        if not base:
            messagebox.showwarning("Missing input", "Select a base directory first.")
            return
        if not fname:
            messagebox.showwarning("Missing input", "Enter a file name.")
            return

        self.status("Running bigImport — this may take a while...")
        self.btn_import.configure(state="disabled")

        def task():
            try:
                from pathlib import Path
                from utils.dataload_util import bigImport
                ds, ds_quad, ds_inlet = bigImport(Path(base), fname)
                self.state.ds_by_case = ds
                self.state.ds_by_case_quad = ds_quad
                self.state.ds_by_case_inlet = ds_inlet
                self.state.data_loaded = True
                self.state.parse_case_metadata()
                n = len(ds)
                self.lbl_info.configure(text=f"Loaded {n} cases from bigImport.")
                self.status(f"bigImport complete — {n} cases loaded.")
            except Exception as e:
                messagebox.showerror("Import Error", str(e))
                self.status(f"bigImport failed: {e}")
            finally:
                self.btn_import.configure(state="normal")

        threading.Thread(target=task, daemon=True).start()

    def _run_loader(self):
        load_dir = self.dir_load.get()
        if not load_dir:
            messagebox.showwarning("Missing input", "Select the saved data directory.")
            return

        self.status("Loading saved data...")
        self.btn_load.configure(state="disabled")

        def task():
            try:
                from utils.dataload_util import runLoader
                ds, ds_quad, ds_inlet = runLoader(load_dir_dic=load_dir)
                self.state.ds_by_case = ds
                self.state.ds_by_case_quad = ds_quad
                self.state.ds_by_case_inlet = ds_inlet
                self.state.data_loaded = True
                self.state.parse_case_metadata()
                n = len(ds)
                self.lbl_info.configure(text=f"Loaded {n} cases from saved data.")
                self.status(f"Data loaded — {n} cases.")
            except Exception as e:
                messagebox.showerror("Load Error", str(e))
                self.status(f"Load failed: {e}")
            finally:
                self.btn_load.configure(state="normal")

        threading.Thread(target=task, daemon=True).start()

    def _run_saver(self):
        if self.state.ds_by_case is None:
            messagebox.showwarning("No data", "Import or load data first.")
            return

        def task():
            try:
                from utils.dataload_util import runSaver
                runSaver(self.state.ds_by_case, self.state.ds_by_case_quad,
                         self.state.ds_by_case_inlet)
                self.status("Data saved successfully.")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
                self.status(f"Save failed: {e}")

        threading.Thread(target=task, daemon=True).start()
