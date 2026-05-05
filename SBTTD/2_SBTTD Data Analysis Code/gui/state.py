"""
Central application state shared across all tabs.
Every tab reads from and writes to this single object.
"""


class AppState:
    def __init__(self):
        # --- Raw data (from bigImport or runLoader) ---
        self.ds_by_case = None
        self.ds_by_case_quad = None
        self.ds_by_case_inlet = None

        # --- Extracted wall variables (from variableImporterMasked) ---
        self.y_plus = None
        self.tau_x = None
        self.tau_y = None
        self.tau_separation = None
        self.tau_separation_idx = None
        self.x = None
        self.y = None
        self.T = None
        self.P = None
        self.Px = None
        self.Py = None
        self.P0 = None
        self.rho = None
        self.mach = None
        self.omega_z = None
        self.u = None
        self.v = None
        self.q_dot = None

        # --- Quadrant variables ---
        self.x_quad = None
        self.y_quad = None
        self.P_quad = None
        self.rho_quad = None
        self.mach_quad = None
        self.P0_quad = None

        # --- Inlet variables ---
        self.P_inlet = None
        self.mach_inlet = None
        self.u_inlet = None

        # --- Computed: wall shear stress ---
        self.tau_wall = None

        # --- Boundary layer results ---
        self.delta_n_dict = None

        # --- Separation results ---
        self.sep_length = None
        self.sep_length_nonDim = None
        self.x_sep = None
        self.y_sep = None
        self.x_attach = None
        self.y_attach = None
        self.x_max = None
        self.x_min = None
        self.y_max = None
        self.y_min = None

        # --- Force results ---
        self.df_axial_force = None
        self.first_shock_pressures = None

        # --- Theoretical model results ---
        self.axialForceScaled = None
        self.axialForceScaled_SE = None
        self.axialForceScaled_combined = None

        # --- Case metadata ---
        self.case_keys = None
        self.h_l_values = None
        self.mach_values = None
        self.cases_by_hl = None

        # --- Status flags ---
        self.data_loaded = False
        self.variables_extracted = False
        self.tecplot_connected = False

    def parse_case_metadata(self):
        """Extract h/l values, Mach values, and groupings from case keys."""
        import re
        if self.ds_by_case is None:
            return

        self.case_keys = list(self.ds_by_case.keys())
        self.cases_by_hl = {}
        h_l_set = set()
        mach_set = set()

        for key in self.case_keys:
            hl_match = re.match(r"(h_l_(?:\d+\.\d+|[a-zA-Z]+))", key)
            if hl_match:
                hl = hl_match.group(1)
                self.cases_by_hl.setdefault(hl, []).append(key)

            m = re.search(r"Mach_([0-9]*\.?[0-9]+)", key)
            if m:
                mach_set.add(float(m.group(1)))

            hl_num = re.search(r"h_l_([0-9]*\.?[0-9]+)", key)
            if hl_num:
                h_l_set.add(float(hl_num.group(1)))

        self.h_l_values = sorted(h_l_set)
        self.mach_values = sorted(mach_set)
