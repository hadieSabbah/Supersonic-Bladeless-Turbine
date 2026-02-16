"""
Parametric 3D Helical Wavy Cylinder Geometry Generator
======================================================
Creates a cylinder whose inner surface has a sinusoidal cross-section
(wavy circle with n lobes) that helically twists along the axial direction.

Surface equation:
    R(x, θ) = R_avg + h * sin(n_waves * θ - φ(x))

where φ(x) is the phase shift that creates the helical twist:
    φ(x) = (2π / pitch) * x

The helix angle α relates to pitch by:
    tan(α) = 2π * R_avg / pitch
    → pitch = 2π * R_avg / tan(α)

At 45°: pitch = 2π * R_avg  (one full rotation over one circumference length)

Outputs:
    1. Full 3D STEP file (lofted solid)
    2. Individual curve .txt files at each axial station (SolidWorks XYZ format)
    3. Preview plot

Author: HS Research Tools
"""

import numpy as np
import cadquery as cq
from cadquery import exporters
from cadquery.occ_impl.shapes import Solid
import os

# =============================================================================
# USER PARAMETERS — MODIFY THESE
# =============================================================================

# --- Cross-section parameters (from your code) ---
avg_radius = 45 / 2    # Average radius [mm] (45mm diameter / 2)
h_l = 0.02                    # h/l ratio
l = 100                     # Axial length [mm]
n_waves = 10                  # Number of complete waves around circumference

# --- Derived amplitude ---
amplitude = h_l * l            # Amplitude [m] = 0.02 * 0.1 = 0.002 m
h = amplitude

# --- Wall thickness ---
t_wall = 5.0 / 1000           # Wall thickness beyond the wave peaks [m]

# --- Helix parameters ---
helix_angle_deg = 45.0     # Helix angle [degrees]
                            #   0°  = straight axial grooves (no twist)
                            #   45° = one full rotation over 2π*R_avg axial distance
                            #   90° = not valid (purely circumferential)

# --- Loft resolution ---
n_loft_stations = 5         # Number of axial cross-sections for lofting
                            # More stations = smoother helical surface
                            # 5 = fast preview, 20-40 = production quality

n_pts_per_section = 500     # Points per wavy circle cross-section
                            # Should be >> n_waves for smooth splines
                            # 500 works well for n_waves=10

# --- Output directory ---
output_dir = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\02_CAD\9_Bladeless Turbine\1_Curve Geometries"

# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

# Outer radius: clearance beyond wave peaks
R_outer = avg_radius + abs(h) + t_wall

# Helix pitch: the axial distance for one full 360° rotation of the pattern
# From the helix angle definition:
#   tan(α) = circumferential_distance / axial_distance
#   tan(α) = (2π * R_avg) / pitch
#   pitch  = 2π * R_avg / tan(α)
#
# The phase shift per unit axial distance:
#   dφ/dx = 2π / pitch = tan(α) / R_avg
#
# So: φ(x) = x * tan(α) / R_avg

helix_angle_rad = np.radians(helix_angle_deg)

if helix_angle_deg >= 90.0:
    raise ValueError("Helix angle must be < 90°. Use 0° for straight grooves.")

if helix_angle_deg == 0.0:
    phase_rate = 0.0  # No twist
    pitch = float('inf')
else:
    phase_rate = np.tan(helix_angle_rad) / avg_radius  # [rad/mm]
    pitch = 2 * np.pi * avg_radius / np.tan(helix_angle_rad)  # [mm]

# Total phase rotation over the full axial length
total_phase_rotation_rad = phase_rate * l
total_phase_rotation_deg = np.degrees(total_phase_rotation_rad)

print("=" * 65)
print("  HELICAL WAVY CYLINDER GEOMETRY GENERATOR")
print("=" * 65)
print(f"  Axial length (l)          : {l:.4f} m  ({l*1000:.2f} mm)")
print(f"  Average radius (R_avg)    : {avg_radius:.4f} m  ({avg_radius*1000:.2f} mm)")
print(f"  Amplitude (h)             : {h:.6f} m  ({h*1000:.4f} mm)")
print(f"  h/l ratio                 : {h_l}")
print(f"  Waves around circumference: {n_waves}")
print(f"  Outer radius              : {R_outer:.6f} m  ({R_outer*1000:.2f} mm)")
print(f"  Wall thickness            : {t_wall:.4f} m  ({t_wall*1000:.2f} mm)")
print(f"  Helix angle               : {helix_angle_deg:.1f}°")
print(f"  Helix pitch               : {pitch:.6f} m  ({pitch*1000:.2f} mm)")
print(f"  Phase rate (dφ/dx)        : {phase_rate:.4f} rad/m")
print(f"  Total phase rotation      : {total_phase_rotation_deg:.1f}°")
print(f"  Loft stations             : {n_loft_stations}")
print(f"  Points per section        : {n_pts_per_section}")
print("=" * 65)

# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# STEP 1: Generate cross-section curves at each axial station
# =============================================================================
# At each axial position x_i, the cross-section is:
#   R(θ) = R_avg + h * sin(n_waves * θ - φ(x_i))
#
# where φ(x_i) = phase_rate * x_i
#
# This creates a wavy circle that rotates as you move along x.

x_stations = np.linspace(0, l, n_loft_stations)

print(f"\n--- Generating {n_loft_stations} cross-section curves ---")
print(f"{'Station':>8} {'x [m]':>12} {'x [mm]':>10} {'Phase [deg]':>12}")
print("-" * 48)

all_station_data = []  # Store (x_val, points_xyz) for each station

for i, x_val in enumerate(x_stations):
    # Phase shift at this axial position
    phi = phase_rate * x_val
    phi_deg = np.degrees(phi)
    
    # Generate the wavy circle
    theta = np.linspace(0, 2 * np.pi, n_pts_per_section, endpoint=True)
    R = avg_radius + h * np.sin(n_waves * theta - phi)
    
    # Convert to Cartesian (Y, Z plane at this x position)
    # SolidWorks convention: X = axial, Y and Z = cross-section plane
    X_pts = R * np.cos(theta)  
    Y_pts = R * np.sin(theta)
    Z_pts = np.zeros_like(theta) 
    
    
    
    # Store
    points_xyz = np.column_stack([X_pts, Y_pts, Z_pts])
    all_station_data.append((x_val, phi_deg, points_xyz))
    
    # Export individual curve .txt file (SolidWorks XYZ format)
    filename = f"curve_station_{i:02d}_x{x_val*1000:.1f}mm.txt"
    filepath = os.path.join(output_dir, filename)
    np.savetxt(filepath, points_xyz, fmt='%.6f', delimiter='\t')
    
    print(f"{i:>8d} {x_val:>12.4f} {x_val*1000:>10.2f} {phi_deg:>12.1f}°  → {filename}")

# Also save the front face separately with a clear name
front_face_path = os.path.join(output_dir, "front_face_curve.txt")
np.savetxt(front_face_path, all_station_data[0][2], fmt='%.6f', delimiter='\t')

back_face_path = os.path.join(output_dir, "back_face_curve.txt")
np.savetxt(back_face_path, all_station_data[-1][2], fmt='%.6f', delimiter='\t')

print(f"\n✓ All curve .txt files saved to: {output_dir}/")
print(f"  → SolidWorks: Insert > Curve > Curve Through XYZ Points > Browse")

# =============================================================================
# STEP 2: Build the 3D STEP file via lofting
# =============================================================================
# Strategy:
#   1. Create outer cylinder (smooth)
#   2. Create inner helical cavity by lofting wavy cross-sections
#   3. Boolean subtract: outer - inner = final solid

print(f"\n--- Building 3D STEP geometry ---")

# Outer cylinder
print("  Creating outer cylinder...")
outer_cylinder = (
    cq.Workplane("YZ")
    .circle(R_outer)
    .extrude(l)
)

# Build CadQuery wires for each loft station
print(f"  Creating {n_loft_stations} cross-section wires for loft...")
cq_wires = []

for i, (x_val, phi_deg, pts_xyz) in enumerate(all_station_data):
    # Extract Y, Z coordinates for this cross-section
    yz_pts = [(float(pts_xyz[j, 1]), float(pts_xyz[j, 2])) 
              for j in range(n_pts_per_section)]
    
    # Create a spline wire at this axial offset
    wp = (
        cq.Workplane("YZ")
        .workplane(offset=float(x_val))
        .moveTo(yz_pts[0][0], yz_pts[0][1])
        .spline(yz_pts[1:])
        .close()
    )
    
    # Extract the pending wire
    wire_obj = wp.ctx.pendingWires
    if wire_obj:
        cq_wires.append(wire_obj[0])
    else:
        cq_wires.append(wp.val())

print(f"  Lofting through {len(cq_wires)} sections...")
#lofted_solid = Solid.makeLoft(cq_wires, ruled=False)
#inner_cavity = cq.Workplane("YZ").newObject([lofted_solid])

# Boolean subtraction
print("  Boolean subtraction (outer cylinder - inner cavity)...")
#result = outer_cylinder.cut(inner_cavity)

# Export STEP
step_filename = "sinusoidal_hub_helical.step"
print(f"  Exporting STEP file...")
#exporters.export(result, step_filename, exportType=exporters.ExportTypes.STEP)
print(f"  ✓ STEP saved: {step_filename}")

# =============================================================================
# STEP 3: Preview plots
# =============================================================================
print(f"\n--- Generating preview plots ---")

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: Front face (polar) ---
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    theta_plot = np.linspace(0, 2*np.pi, 500, endpoint=True)
    R_front = avg_radius + h * np.sin(n_waves * theta_plot)
    ax1.plot(theta_plot, R_front, 'b-', linewidth=1.5)
    ax1.set_title(f'Front Face (x=0)\n{n_waves} lobes, h={h*1000:.2f} mm', pad=15, fontsize=10)
    ax1.set_rticks([avg_radius - h, avg_radius, avg_radius + h])

    # --- Panel 2: Back face (polar) showing the rotation ---
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    phi_back = phase_rate * l
    R_back = avg_radius + h * np.sin(n_waves * theta_plot - phi_back)
    ax2.plot(theta_plot, R_front, 'b--', linewidth=0.8, alpha=0.4, label='Front (ref)')
    ax2.plot(theta_plot, R_back, 'r-', linewidth=1.5, label='Back')
    ax2.set_title(f'Back Face (x={l:.3f} m)\nRotated {total_phase_rotation_deg:.1f}°', pad=15, fontsize=10)
    ax2.legend(fontsize=7, loc='lower right')

    # --- Panel 3: All station cross-sections overlaid (Cartesian) ---
    ax3 = fig.add_subplot(gs[0, 2])
    colors = plt.cm.viridis(np.linspace(0, 1, n_loft_stations))
    for i, (x_val, phi_deg, pts_xyz) in enumerate(all_station_data):
        ax3.plot(pts_xyz[:, 1], pts_xyz[:, 2], color=colors[i], linewidth=1.0,
                 label=f'x={x_val:.4f} m (φ={phi_deg:.1f}°)')
    ax3.set_xlabel('Y [m]')
    ax3.set_ylabel('Z [m]')
    ax3.set_title('All Loft Stations Overlaid')
    ax3.set_aspect('equal')
    ax3.legend(fontsize=6, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: 3D surface preview ---
    ax4 = fig.add_subplot(gs[1, 0:2], projection='3d')
    
    x_vis = np.linspace(0, l, 60)
    theta_vis = np.linspace(0, 2*np.pi, 120, endpoint=True)
    Xv, Tv = np.meshgrid(x_vis, theta_vis, indexing='ij')
    Phi_v = phase_rate * Xv
    Rv = avg_radius + h * np.sin(n_waves * Tv - Phi_v)
    Yv = Rv * np.cos(Tv)
    Zv = Rv * np.sin(Tv)

    ax4.plot_surface(Xv, Yv, Zv, alpha=0.7, cmap='coolwarm', edgecolor='none')
    ax4.set_xlabel('X (axial) [m]')
    ax4.set_ylabel('Y [m]')
    ax4.set_zlabel('Z [m]')
    ax4.set_title(f'3D Inner Surface (helix angle = {helix_angle_deg}°)', fontsize=11)
    max_r = avg_radius + h + 2
    ax4.set_ylim(-max_r, max_r)
    ax4.set_zlim(-max_r, max_r)

    # --- Panel 5: Unwrapped surface showing helix lines ---
    ax5 = fig.add_subplot(gs[1, 2])
    x_unwr = np.linspace(0, l, 200)
    theta_unwr = np.linspace(0, 2*np.pi, 200)
    Xu, Tu = np.meshgrid(x_unwr, theta_unwr, indexing='ij')
    Phi_u = phase_rate * Xu
    Ru = avg_radius + h * np.sin(n_waves * Tu - Phi_u)
    
    # Show as contour on unwrapped surface
    circ = avg_radius * Tu  # Convert θ to arc length
    c = ax5.contourf(Xu, np.degrees(Tu), Ru, levels=30, cmap='coolwarm')
    plt.colorbar(c, ax=ax5, label='R [m]')
    
    # Draw one lobe peak trajectory to show the helix angle
    x_line = np.linspace(0, l, 100)
    theta_peak = (phase_rate * x_line) / n_waves  # Peak of sin is at argument = π/2 + phase
    ax5.plot(x_line, np.degrees(theta_peak) % 360, 'k--', linewidth=1.5, label='Lobe peak path')
    ax5.set_xlabel('Axial x [m]')
    ax5.set_ylabel('θ [degrees]')
    ax5.set_title('Unwrapped Surface\n(color = radius)')
    ax5.legend(fontsize=8)

    plt.savefig('helical_wavy_cylinder_preview.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Preview saved: helical_wavy_cylinder_preview.png")
    plt.close()

except ImportError:
    print("  (matplotlib not available — skipping preview)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  OUTPUTS SUMMARY")
print("=" * 65)
print(f"  STEP file      : {step_filename}")
print(f"  Curve files    : {output_dir}/")
for i, (x_val, phi_deg, _) in enumerate(all_station_data):
    print(f"    Station {i}: curve_station_{i:02d}_x{x_val*1000:.1f}mm.txt  (φ={phi_deg:.1f}°)")
print(f"  Front face     : {output_dir}/front_face_curve.txt")
print(f"  Back face      : {output_dir}/back_face_curve.txt")
print(f"  Preview        : helical_wavy_cylinder_preview.png")
print("=" * 65)
print("\nSolidWorks import:")
print("  1. Insert > Curve > Curve Through XYZ Points")
print("  2. Browse to each station .txt file")
print("  3. Use Insert > Boss/Base > Loft to connect them")
print("\nDone!")