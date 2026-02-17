# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 23:55:22 2026

@author: hhsabbah
"""

"""
SBTTD Data Architecture — Overview Diagram
============================================
Streamlined view emphasizing:
  1. PyTecplot automation pipeline
  2. Data structure (how variables are stored)
  3. Code complexity breadth
"""

import graphviz

def create_overview():
    dot = graphviz.Digraph('SBTTD_Overview', format='png', engine='dot')

    # =========================================================================
    # GLOBAL
    # =========================================================================
    dot.attr(
        rankdir='TB',
        bgcolor='#0D1117',
        fontname='Helvetica Neue',
        pad='0.6',
        nodesep='0.5',
        ranksep='0.9',
        dpi='200',
        label=(
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            '<TR><TD><FONT POINT-SIZE="30" COLOR="#58A6FF"><B>SBTTD Data Architecture</B></FONT></TD></TR>'
            '<TR><TD><FONT POINT-SIZE="13" COLOR="#8B949E">'
            'Supersonic Bladeless Turbine — Automated CFD Post-Processing Pipeline'
            '</FONT></TD></TR>'
            '<TR><TD><FONT POINT-SIZE="11" COLOR="#6E7681">'
            'HS  |  BEFAST Lab  |  NC State Aerospace Engineering'
            '</FONT></TD></TR>'
            '</TABLE>>'
        ),
        labelloc='t',
    )

    # -- palette --
    BLUE = '#1F6FEB';    BLUE_L = '#58A6FF'
    GREEN = '#238636';   GREEN_L = '#3FB950'
    ORANGE = '#D29922';  ORANGE_L = '#E3B341'
    RED = '#DA3633';     RED_L = '#F85149'
    PURPLE = '#8957E5';  PURPLE_L = '#BC8CFF'
    TEAL = '#1B7C83';    TEAL_L = '#39D2C0'
    BG = '#161B22';      BORDER = '#30363D'
    TXT = '#C9D1D9';     DIM = '#8B949E'

    # -- helpers --
    def box(title, rows, tc, bc):
        hdr = (f'<TR><TD COLSPAN="2" BGCOLOR="{tc}" ALIGN="CENTER">'
               f'<B><FONT COLOR="#FFFFFF" POINT-SIZE="12">{title}</FONT></B></TD></TR>')
        body = ''
        for k, v in rows:
            body += (f'<TR><TD ALIGN="LEFT" BGCOLOR="{BG}">'
                     f'<FONT COLOR="{BLUE_L}" POINT-SIZE="10">{k}</FONT></TD>'
                     f'<TD ALIGN="LEFT" BGCOLOR="{BG}">'
                     f'<FONT COLOR="{DIM}" POINT-SIZE="10">{v}</FONT></TD></TR>')
        return (f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5" '
                f'COLOR="{bc}" BGCOLOR="{BG}">{hdr}{body}</TABLE>>')

    def badge(text, color):
        return (f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="6">'
                f'<TR><TD BGCOLOR="{color}" STYLE="ROUNDED">'
                f'<FONT COLOR="#FFFFFF" POINT-SIZE="11"><B>  {text}  </B></FONT>'
                f'</TD></TR></TABLE>>')

    def section(text, color):
        return f'<<FONT POINT-SIZE="15" COLOR="{color}"><B>{text}</B></FONT>>'

    e = dict(color=BORDER, penwidth='1.6', arrowsize='0.8')
    e_hi = dict(color=BLUE_L, penwidth='2.2', arrowsize='0.9', style='bold')
    e_dim = dict(color='#21262D', penwidth='1.2', arrowsize='0.7', style='dashed')

    # =====================================================================
    # 1  RAW FILES
    # =====================================================================
    with dot.subgraph(name='cluster_raw') as c:
        c.attr(label=section('RAW DATA', ORANGE_L), style='dashed',
               color=ORANGE, penwidth='1.5')

        c.node('cfd_bin', box('CFD++ Binaries', [
            ('File', 'mcfd_tec.bin  (Tecplot binary)'),
            ('Per case', '~48 parametric cases'),
            ('Naming', 'h_l_{amp}_Mach_{M}'),
        ], '#6E4400', ORANGE), shape='plaintext')

        c.node('other_raw', box('Other Sources', [
            ('Residuals', 'mcfd.rhsgi, minfo1_e1/e2/e3'),
            ('Experimental', 'Schlieren .png images'),
        ], '#6E4400', ORANGE), shape='plaintext')

    # =====================================================================
    # 2  PYTECPLOT AUTOMATION  (detailed)
    # =====================================================================
    with dot.subgraph(name='cluster_pytecplot') as c:
        c.attr(label=section('PYTECPLOT AUTOMATION PIPELINE', BLUE_L),
               style='dashed', color=BLUE, penwidth='1.5')

        c.node('discover', box('① File Discovery', [
            ('base_dir', 'Path to parametric study root'),
            ('Walk', 'base_dir / h_l_* / case_* / mcfd_tec.bin'),
            ('Result', 'List[Path] of all case binaries'),
        ], '#0D2240', BLUE), shape='plaintext')

        c.node('load', box('② tp.data.load_tecplot()', [
            ('Action', 'Load binary into Tecplot engine'),
            ('Per file', 'tp.new_layout() then load'),
            ('Auto-detect', 'All variables in the dataset'),
        ], '#0D2240', BLUE), shape='plaintext')

        c.node('zones', box('③ Zone Extraction', [
            ('test.zone("Section")', '→ Wall surface data'),
            ('test.zone("QUADRILATERAL_cells")', '→ Full 2D flow field'),
            ('test.zone("Inlet")', '→ Inlet boundary data'),
        ], '#0D2240', BLUE), shape='plaintext')

        c.node('numpy_extract', box('④ NumPy Extraction', [
            ('Loop', 'for var in var_names:'),
            ('Call', 'zone.values(var).as_numpy_array()'),
            ('Result', '{var_name: np.ndarray} per zone'),
        ], '#0D2240', BLUE), shape='plaintext')

        c.node('xr_convert', box('⑤ xr.Dataset Conversion', [
            ('Function', 'dict_to_ds_1d(data_dict)'),
            ('Transform', 'xr.Dataset({k: (("n",), v)})'),
            ('Key by', 'folder name → case identifier'),
        ], '#0D2240', BLUE), shape='plaintext')

    dot.edge('discover', 'load', **e_hi)
    dot.edge('load', 'zones', **e_hi)
    dot.edge('zones', 'numpy_extract', **e_hi)
    dot.edge('numpy_extract', 'xr_convert', **e_hi)

    # =====================================================================
    # 3  DATA STRUCTURE  (detailed)
    # =====================================================================
    with dot.subgraph(name='cluster_structure') as c:
        c.attr(label=section('DATA STRUCTURE', GREEN_L), style='dashed',
               color=GREEN, penwidth='1.5')

        c.node('ds_wall', box('ds_by_case', [
            ('Type', 'Dict[ str, xr.Dataset ]'),
            ('Key example', '"h_l_0.04_Mach_2.5"'),
            ('Dataset dim', '"n"  (1D along wall)'),
            ('# variables', '~20 per case'),
        ], '#0B3D0B', GREEN), shape='plaintext')

        c.node('ds_quad', box('ds_by_case_quad', [
            ('Type', 'Dict[ str, xr.Dataset ]'),
            ('Zone', 'Full 2D quadrilateral field'),
            ('Use', 'BL detection, shock capture'),
        ], '#0B3D0B', GREEN), shape='plaintext')

        c.node('ds_inlet', box('ds_by_case_inlet', [
            ('Type', 'Dict[ str, xr.Dataset ]'),
            ('Zone', 'Inlet boundary'),
            ('Use', 'Freestream conditions, Re'),
        ], '#0B3D0B', GREEN), shape='plaintext')

        # Access pattern detail
        c.node('access', box('Variable Access Pattern', [
            ('Dataset', 'ds_by_case["h_l_0.04_Mach_2.5"]'),
            ('DataArray', '...["P"]  →  xr.DataArray'),
            ('NumPy', '...["P"].data  →  np.ndarray'),
            ('Flow vars', 'X, Y, U, V, P, T, R, M, P_x, P_y'),
            ('Wall vars', 'P_total, Tau_x, Tau_y, Y_plus'),
            ('Other vars', 'Vort_z, Qdot, Mutur'),
        ], '#0B3D0B', GREEN), shape='plaintext')

        # Extracted variable dicts
        c.node('var_dicts', box('Extracted Variable Dicts', [
            ('Type', 'Dict[ str, np.ndarray ]'),
            ('Same keys', '"h_l_0.04_Mach_2.5"'),
            ('Examples', 'x{}, y{}, tau_x{}, P{}, mach{}, ...'),
            ('Masking', 'Spatial filter on [x_min, x_max]'),
            ('Count', '18 separate dictionaries'),
        ], '#2D1B69', PURPLE), shape='plaintext')

        c.node('grouping', box('Case Grouping', [
            ('cases_by_hl', 'Dict[ str, List[str] ]'),
            ('Key', '"h_l_0.04"'),
            ('Value', '["..._Mach_1.5", "..._Mach_2.0", ...]'),
        ], '#2D1B69', PURPLE), shape='plaintext')

        c.node('pickle', box('Persistence (pickle)', [
            ('Save', 'ds_by_case → .pkl  (date-stamped)'),
            ('Load', 'Auto-detect latest saved run'),
            ('Skip', 'Re-processing on subsequent runs'),
        ], '#0D2240', BLUE), shape='plaintext')

    # =====================================================================
    # 4  ANALYSIS MODULES  (summary-level, showing breadth)
    # =====================================================================
    with dot.subgraph(name='cluster_analysis') as c:
        c.attr(label=section('ANALYSIS MODULES  (6 domains, ~3500 lines)', RED_L),
               style='dashed', color=RED, penwidth='1.5')

        c.node('mod_sep', badge('Separation Detection', '#5C0A0A'), shape='plaintext')
        c.node('mod_bl', badge('Boundary Layer Analysis', '#5C0A0A'), shape='plaintext')
        c.node('mod_force', badge('Force / Torque Integration', '#5C0A0A'), shape='plaintext')
        c.node('mod_shock', badge('Shock Detection (Schlieren)', '#5C0A0A'), shape='plaintext')
        c.node('mod_re', badge('Reynolds Number Computation', '#5C0A0A'), shape='plaintext')
        c.node('mod_conv', badge('Convergence / Residuals', '#5C0A0A'), shape='plaintext')

    with dot.subgraph(name='cluster_theory') as c:
        c.attr(label=section('THEORETICAL MODELS  (3 methods)', TEAL_L),
               style='dashed', color=TEAL, penwidth='1.5')

        c.node('th_sp', badge('Small Perturbation Theory  (SymPy)', '#0A3D40'), shape='plaintext')
        c.node('th_se', badge('Shock-Expansion Theory', '#0A3D40'), shape='plaintext')
        c.node('th_comb', badge('Combined SPT + SE  (weighted blend)', '#0A3D40'), shape='plaintext')

    # =====================================================================
    # 5  OUTPUTS
    # =====================================================================
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label=section('OUTPUTS', ORANGE_L), style='dashed',
               color=ORANGE, penwidth='1.5')

        c.node('out_plots', badge('Matplotlib Visualizations', '#6E4400'), shape='plaintext')
        c.node('out_csv', badge('pandas DataFrames → .csv', '#6E4400'), shape='plaintext')
        c.node('out_tables', badge('Pivot Tables (RANS vs Theory)', '#6E4400'), shape='plaintext')

    # =====================================================================
    # EDGES
    # =====================================================================

    # Raw → PyTecplot
    dot.edge('cfd_bin', 'discover', **e_hi)
    dot.edge('other_raw', 'mod_conv', **e_dim)
    dot.edge('other_raw', 'mod_shock', **e_dim)

    # PyTecplot → Structure
    dot.edge('xr_convert', 'ds_wall', label='  wall zone', fontcolor=DIM, fontsize='9', **e_hi)
    dot.edge('xr_convert', 'ds_quad', label='  quad zone', fontcolor=DIM, fontsize='9', **e)
    dot.edge('xr_convert', 'ds_inlet', label='  inlet zone', fontcolor=DIM, fontsize='9', **e)

    # Structure internal
    dot.edge('ds_wall', 'access', **e)
    dot.edge('access', 'var_dicts', label='  variableImporterMasked()', fontcolor=DIM, fontsize='9', **e_hi)
    dot.edge('ds_wall', 'grouping', label='  regex parse keys', fontcolor=DIM, fontsize='9', **e)
    dot.edge('ds_wall', 'pickle', **e_dim)
    dot.edge('ds_quad', 'pickle', **e_dim)
    dot.edge('ds_inlet', 'pickle', **e_dim)

    # Structure → Analysis
    dot.edge('var_dicts', 'mod_sep', **e)
    dot.edge('var_dicts', 'mod_bl', **e)
    dot.edge('var_dicts', 'mod_force', **e)
    dot.edge('var_dicts', 'mod_re', **e)
    dot.edge('ds_quad', 'mod_bl', **e_dim)
    dot.edge('ds_inlet', 'mod_re', **e_dim)

    # Structure → Theory
    dot.edge('var_dicts', 'th_sp', **e)
    dot.edge('var_dicts', 'th_se', **e)
    dot.edge('th_sp', 'th_comb', **e_dim)
    dot.edge('th_se', 'th_comb', **e_dim)

    # Analysis/Theory → Outputs
    dot.edge('mod_sep', 'out_plots', **e_dim)
    dot.edge('mod_force', 'out_csv', **e_dim)
    dot.edge('th_comb', 'out_csv', **e_dim)
    dot.edge('out_csv', 'out_tables', **e_dim)
    dot.edge('mod_bl', 'out_plots', **e_dim)

    # =====================================================================
    # LEGEND
    # =====================================================================
    dot.node('legend', (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3" CELLPADDING="3" BGCOLOR="#0D1117">'
        f'<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT COLOR="{TXT}" POINT-SIZE="12"><B>TYPE LEGEND</B></FONT></TD></TR>'
        f'<TR><TD BGCOLOR="{BLUE}" WIDTH="14" HEIGHT="10"></TD>'
        f'<TD ALIGN="LEFT"><FONT COLOR="{TXT}" POINT-SIZE="9">PyTecplot automation pipeline</FONT></TD></TR>'
        f'<TR><TD BGCOLOR="{GREEN}" WIDTH="14" HEIGHT="10"></TD>'
        f'<TD ALIGN="LEFT"><FONT COLOR="{TXT}" POINT-SIZE="9">Dict[str, xr.Dataset] containers</FONT></TD></TR>'
        f'<TR><TD BGCOLOR="{PURPLE}" WIDTH="14" HEIGHT="10"></TD>'
        f'<TD ALIGN="LEFT"><FONT COLOR="{TXT}" POINT-SIZE="9">Dict[str, np.ndarray] variables</FONT></TD></TR>'
        f'<TR><TD BGCOLOR="{RED}" WIDTH="14" HEIGHT="10"></TD>'
        f'<TD ALIGN="LEFT"><FONT COLOR="{TXT}" POINT-SIZE="9">Analysis modules (~3500 LOC)</FONT></TD></TR>'
        f'<TR><TD BGCOLOR="{TEAL}" WIDTH="14" HEIGHT="10"></TD>'
        f'<TD ALIGN="LEFT"><FONT COLOR="{TXT}" POINT-SIZE="9">Theoretical models (SymPy + NumPy)</FONT></TD></TR>'
        f'<TR><TD BGCOLOR="{ORANGE}" WIDTH="14" HEIGHT="10"></TD>'
        f'<TD ALIGN="LEFT"><FONT COLOR="{TXT}" POINT-SIZE="9">I/O and exports</FONT></TD></TR>'
        '</TABLE>>'
    ), shape='plaintext')

    return dot


if __name__ == '__main__':
    d = create_overview()
    out = d.render(r'C:\Users\hhsabbah\Pictures\6_Data Structure Code\graph', cleanup= True)
    print(f"Saved: {out}")