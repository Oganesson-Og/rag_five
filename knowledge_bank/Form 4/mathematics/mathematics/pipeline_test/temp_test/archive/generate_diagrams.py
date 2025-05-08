import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle, Polygon, Rectangle, PathPatch
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms # Added import for transforms
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plots
from itertools import product, combinations # Required for 3D cube plotting


# Import matplotlib_venn conditionally (Included from snippet 2)
try:
    from matplotlib_venn import venn2, venn3, venn3_circles # Added venn2 for potential use
    venn_installed = True
except ImportError:
    venn_installed = False
    print("Warning: matplotlib_venn not installed. Venn diagrams requiring it cannot be generated.")


# Ensure the output directory exists
output_dir = "math_diagrams"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Diagram Generation Functions ---

def save_plot(fig, filename):
    """Helper function to save the plot."""
    filepath = os.path.join(output_dir, filename + ".png")
    fig.savefig(filepath)
    plt.close(fig)
    # print(f"Saved: {filepath}") # Optional: print confirmation


# --- Helper Function ---
def save_plot(fig, filename):
    """Helper function to save the plot and close the figure."""
    filepath = os.path.join(output_dir, filename + ".png")
    try:
        # Set white background explicitly before saving
        fig.patch.set_facecolor('white')
        ax = fig.gca() # Get current axes
        # Check if it's a 3D plot
        if isinstance(ax, Axes3D):
            ax.set_facecolor('white')
            # For 3D, make panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # Make grid lines lighter
            ax.xaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 1)
            ax.yaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 1)
            ax.zaxis._axinfo["grid"]['color'] = (0.8, 0.8, 0.8, 1)
        else:
            ax.set_facecolor('white')

        fig.savefig(filepath, dpi=150, facecolor=fig.get_facecolor()) # Use slightly higher DPI
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
    plt.close(fig)
    # print(f"Saved: {filepath}") # Optional: print confirmation


#<editor-fold desc="Requested Functions">

def form2_variation_direct_variation_graph():
    """Generates graph for direct variation y=kx."""
    fig, ax = plt.subplots(figsize=(5, 4))
    # Assuming white background is default, otherwise use fig.patch.set_facecolor('white')
    x_direct = np.linspace(0, 5, 100)
    k_direct = 1.5 # Example slope
    y_direct = k_direct * x_direct
    ax.plot(x_direct, y_direct, label=f'y = kx') # Simplified label
    ax.set_title('Direct Variation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle=':')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.set_xlim(0, 5.1)
    ax.set_ylim(0, k_direct*5 + 1)
    plt.tight_layout()
    save_plot(fig, "form2_variation_direct_variation_graph")

def form1_travel_graphs_car_grid():
    """Generates blank grid for car journey problem."""
    fig, ax = plt.subplots(figsize=(6, 5))
    # fig.patch.set_facecolor('white') # Set background if needed

    # Labels and Title
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance (km)")
    ax.set_title("Car Journey Grid")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(np.arange(0, 4.5, 0.5))
    ax.set_yticks(np.arange(0, 151, 25))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 150)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.tight_layout()
    save_plot(fig, "form1_travel_graphs_car_grid") # Note: Renamed from F2 in original JSON

def form1_sets_venn_music_example():
    """Generates Venn diagram for the music preference example."""
    fig, ax = plt.subplots(figsize=(6, 4))
    # fig.patch.set_facecolor('white') # Set background if needed

    # Universal set rectangle
    ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.95, 0.9, 'U=30', ha='right', va='top', fontsize=12)

    # Circles for sets P and R
    center_p = (0.4, 0.5)
    center_r = (0.6, 0.5)
    radius = 0.3
    circle_p = Circle(center_p, radius, fill=False, edgecolor='blue', linewidth=1.5)
    circle_r = Circle(center_r, radius, fill=False, edgecolor='red', linewidth=1.5)
    ax.add_patch(circle_p)
    ax.add_patch(circle_r)

    # Labels for sets
    ax.text(0.25, 0.75, 'P (Pop)', ha='center', va='center', fontsize=12, color='blue')
    ax.text(0.75, 0.75, 'R (Rock)', ha='center', va='center', fontsize=12, color='red')

    # Place counts in regions
    ax.text(0.5, 0.5, '8', ha='center', va='center', fontsize=12)      # Intersection P ∩ R
    ax.text(0.3, 0.5, '7', ha='center', va='center', fontsize=12)     # P only
    ax.text(0.7, 0.5, '12', ha='center', va='center', fontsize=12)     # R only
    ax.text(0.5, 0.15, '3', ha='center', va='center', fontsize=12)    # Neither

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Music Preferences (n=30)")
    plt.tight_layout()
    save_plot(fig, "form1_sets_venn_music_example") # Note: Renamed from F2 in original JSON

def form1_graphs_plot_points_example2():
     """Plots points for y=x-2 example."""
     fig, ax = plt.subplots(figsize=(5, 5))
     # fig.patch.set_facecolor('white') # Set background if needed
     points = [(-1, -3), (0, -2), (1, -1), (2, 0), (3, 1)]

     x_coords = [p[0] for p in points]
     y_coords = [p[1] for p in points]

     # Set limits and ticks
     ax.set_xlim(-2, 4)
     ax.set_ylim(-4, 2)
     ax.set_xticks(np.arange(-2, 5))
     ax.set_yticks(np.arange(-4, 3))
     ax.grid(True, linestyle='--', alpha=0.6)

     # Move axes to center
     ax.spines['left'].set_position('zero')
     ax.spines['bottom'].set_position('zero')
     ax.spines['right'].set_color('none')
     ax.spines['top'].set_color('none')

     # Plot points
     ax.plot(x_coords, y_coords, 'ro', markersize=7)
     for i, (x, y) in enumerate(points):
         ax.text(x+0.1, y+0.1, f'({x},{y})', fontsize=9)

     # Add axis labels
     ax.set_xlabel("x", loc='right')
     ax.set_ylabel("y", loc='top', rotation=0, labelpad=-10)

     ax.set_aspect('equal', adjustable='box')
     ax.set_title("Plotting Points for y=x-2")
     plt.tight_layout()
     save_plot(fig, "form1_graphs_plot_points_example2") # Note: Renamed from F2

def form1_graphs_line_graph_y_eq_x_m2():
     """Draws graph for y=x-2 example."""
     fig, ax = plt.subplots(figsize=(5, 5))
     # fig.patch.set_facecolor('white') # Set background if needed
     points = [(-1, -3), (0, -2), (1, -1), (2, 0), (3, 1)]

     x_coords = [p[0] for p in points]
     y_coords = [p[1] for p in points]

     # Set limits and ticks
     ax.set_xlim(-2, 4)
     ax.set_ylim(-4, 2)
     ax.set_xticks(np.arange(-2, 5))
     ax.set_yticks(np.arange(-4, 3))
     ax.grid(True, linestyle='--', alpha=0.6)

     # Move axes to center
     ax.spines['left'].set_position('zero')
     ax.spines['bottom'].set_position('zero')
     ax.spines['right'].set_color('none')
     ax.spines['top'].set_color('none')

     # Plot points and line
     ax.plot(x_coords, y_coords, 'ro', markersize=7)
     ax.plot(x_coords, y_coords, 'b-', linewidth=1.5, label='y = x - 2')

     # Add axis labels
     ax.set_xlabel("x", loc='right')
     ax.set_ylabel("y", loc='top', rotation=0, labelpad=-10)
     ax.legend(loc='lower right', fontsize=9)

     ax.set_aspect('equal', adjustable='box')
     ax.set_title("Graph of y = x - 2")
     plt.tight_layout()
     save_plot(fig, "form1_graphs_line_graph_y_eq_x_m2") # Note: Renamed from F2

def form1_graphs_line_graph_y_eq_4_m_x():
     """Draws graph for y=4-x example."""
     fig, ax = plt.subplots(figsize=(5, 6)) # Taller to show y=5
     # fig.patch.set_facecolor('white') # Set background if needed
     points = [(-1, 5), (0, 4), (1, 3), (2, 2)]

     x_coords = [p[0] for p in points]
     y_coords = [p[1] for p in points]

     # Set limits and ticks
     ax.set_xlim(-2, 3)
     ax.set_ylim(0, 6)
     ax.set_xticks(np.arange(-2, 4))
     ax.set_yticks(np.arange(0, 7))
     ax.grid(True, linestyle='--', alpha=0.6)

     # Move axes
     ax.spines['left'].set_position('zero')
     ax.spines['bottom'].set_position('zero')
     ax.spines['right'].set_color('none')
     ax.spines['top'].set_color('none')

     # Plot points and line
     ax.plot(x_coords, y_coords, 'go', markersize=7)
     # Extend line
     x_line = np.array([-1.5, 2.5])
     y_line = 4 - x_line
     ax.plot(x_line, y_line, 'g-', linewidth=1.5, label='y = 4 - x')


     # Add axis labels
     ax.set_xlabel("x", loc='right')
     ax.set_ylabel("y", loc='top', rotation=0, labelpad=-10)
     ax.legend(loc='upper right')

     # ax.set_aspect('equal', adjustable='box') # y range larger, 'auto' might be better
     ax.set_title("Graph of y = 4 - x")
     plt.tight_layout()
     save_plot(fig, "form1_graphs_line_graph_y_eq_4_m_x") # Note: Renamed from F2

def form3_geometry_bearing_240_deg():
    """Illustrates bearing 240 degrees."""
    fig, ax = plt.subplots(figsize=(4, 4.5))
    fig.patch.set_facecolor('white')
    O = (0, 0)
    angle_deg = 240
    length = 1

    # North line
    ax.plot([O[0], O[0]], [O[1], O[1]+1.2], 'k--')
    ax.plot(O[0], O[1]+1.2, '^k')
    ax.text(O[0], O[1]+1.3, 'N', ha='center', va='bottom')

    # Bearing Line
    plot_angle_rad = np.deg2rad(90 - angle_deg) # Angle from +x axis ACW
    P = (O[0] + length * np.cos(plot_angle_rad), O[1] + length * np.sin(plot_angle_rad))
    ax.plot([O[0], P[0]], [O[1], P[1]], 'r-', linewidth=1.5)
    ax.plot(P[0], P[1], 'ro')
    ax.text(P[0]-0.1, P[1]-0.1, 'P', ha='right', va='top')


    # Angle Arc (Large clockwise arc)
    arc = Arc(O, 0.8, 0.8, angle=90, theta1=-angle_deg, theta2=0, color='red', linewidth=1)
    ax.add_patch(arc)

    # Angle Label (place carefully for large angles)
    # Midpoint angle (clockwise from North) is angle_deg/2
    # Convert to angle from +x axis for text placement
    label_angle_plot_rad = np.deg2rad(90 - angle_deg*0.6) # Place label partway along arc
    ax.text(0.6*np.cos(label_angle_plot_rad), 0.6*np.sin(label_angle_plot_rad),
            f'{angle_deg:03d}°', color='red', fontsize=9, ha='left', va='bottom') # Adjust placement


    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.4)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Bearing 240°")
    plt.tight_layout()
    save_plot(fig, "form3_geometry_bearing_240_deg")

#</editor-fold>
def form1_real_numbers_number_line():
    """Generates Form 1 Real Numbers Number Line."""
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_ticks_position('bottom')

    # Set limits and ticks
    limit = 5.5
    ax.set_xlim(-limit, limit)
    ticks = np.arange(-5, 6)
    ax.set_xticks(ticks)

    # Labels for specific points
    ax.set_xticklabels(['' if t not in [-5, 0, 5] else str(t) for t in ticks])
    for t in [-5, 0, 5]:
        ax.text(t, -0.15, str(t), ha='center', va='top', fontsize=10)

    # Arrowheads
    ax.plot(limit, 0, ">k", markersize=10, clip_on=False)
    ax.plot(-limit, 0, "<k", markersize=10, clip_on=False)

    # Thicker line for the axis itself
    ax.spines['bottom'].set_linewidth(1.5)

    # Tick marks
    for tick in ticks:
        ax.plot([tick, tick], [-0.05, 0.05], color='black', linewidth=1.5)

    ax.set_title("Horizontal Number Line (-5 to +5)", fontsize=10, y=0.8)
    plt.tight_layout()
    save_plot(fig, "form1_real_numbers_number_line")

def form1_real_numbers_submarine_movement():
    """Generates Form 1 Submarine Movement Diagram."""
    fig, ax = plt.subplots(figsize=(2, 7))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0)) # Sea level
    ax.spines['left'].set_position('zero')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_ticks_position('left')

    # Set limits and ticks
    ax.set_ylim(-110, 60)
    ticks = np.arange(-100, 51, 25)
    ax.set_yticks(ticks)
    ax.tick_params(axis='y', labelsize=9)

    # Label sea level
    ax.text(0.1, 5, 'Sea Level (0m)', ha='left', va='center', fontsize=9)

    # Define points
    start_depth = -50
    intermediate_depth = start_depth + 25
    final_depth = intermediate_depth - 40

    # Plot points
    ax.plot(0, start_depth, 'bo', markersize=8, label='Start (-50m)')
    ax.plot(0, intermediate_depth, 'go', markersize=8, label=f'Intermediate ({intermediate_depth}m)')
    ax.plot(0, final_depth, 'ro', markersize=8, label=f'End ({final_depth}m)')

    # Add labels next to points
    ax.text(0.1, start_depth, 'Start (-50m)', ha='left', va='center', fontsize=9)
    ax.text(0.1, intermediate_depth, f'After rising 25m ({intermediate_depth}m)', ha='left', va='center', fontsize=9)
    ax.text(0.1, final_depth, f'Final Depth ({final_depth}m)', ha='left', va='center', fontsize=9)

    # Draw arrows for movement
    ax.arrow(0, start_depth, 0, 25, head_width=0.2, head_length=5, fc='blue', ec='blue', length_includes_head=True)
    ax.arrow(0, intermediate_depth, 0, -40, head_width=0.2, head_length=5, fc='red', ec='red', length_includes_head=True)

    ax.set_title("Submarine Depth", fontsize=10)
    plt.tight_layout(rect=[0.1, 0, 1, 1]) # Adjust layout to prevent label cutoff
    save_plot(fig, "form1_real_numbers_submarine_movement")

def form1_scales_representations():
    """Generates different representations of scales."""
    fig, axs = plt.subplots(3, 1, figsize=(6, 4.5)) # Increased height

    # Plotting area setup for text
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off') # Turn off axes for text plots

    # 1. Ratio Scale Text
    axs[0].text(0.5, 0.5, 'Ratio Scale: 1 : 100,000', ha='center', va='center', fontsize=12)

    # 2. Representative Fraction Text
    axs[1].text(0.5, 0.5, 'Representative Fraction (R.F.): 1/100,000', ha='center', va='center', fontsize=12)

    # 3. Linear Scale
    ax_bar = fig.add_axes([0.1, 0.05, 0.8, 0.2]) # Manually position the axes for the bar scale

    # Bar properties
    bar_height = 0.1
    bar_start = 0
    segment_length = 1 # Arbitrary unit length on plot for 1 km
    main_segments = 3
    sub_divisions = 5 # for the left part (100m increments mean 1000m/100m=10, or use 5 for 200m)

    # Draw main bar segments
    for i in range(main_segments):
        ax_bar.add_patch(Rectangle((i * segment_length, 0), segment_length, bar_height,
                                 edgecolor='black', facecolor='white' if i % 2 == 0 else 'black'))
        ax_bar.text((i + 1) * segment_length, -0.05, f'{i+1} km', ha='center', va='top')

    # Draw subdivided left segment
    sub_segment_length = segment_length / sub_divisions
    for i in range(sub_divisions):
         ax_bar.add_patch(Rectangle((- (i + 1) * sub_segment_length, 0), sub_segment_length, bar_height,
                                  edgecolor='black', facecolor='black' if i % 2 == 0 else 'white'))

    # Add labels for the bar scale
    ax_bar.text(0, -0.05, '0', ha='center', va='top')
    # Optional: Label the end of the left segment
    ax_bar.text(-segment_length, -0.05, f'{sub_divisions * 1000 / sub_divisions} m', ha='center', va='top') # Label shows 1000m


    # Customize bar scale axes
    ax_bar.set_xlim(-segment_length - 0.2*segment_length, main_segments * segment_length + 0.2*segment_length)
    ax_bar.set_ylim(-0.1, bar_height + 0.1)
    ax_bar.axis('off')
    ax_bar.set_title("Linear Scale Example", fontsize=12, y=0.6)

    fig.suptitle("Scale Representations", fontsize=14)
    plt.subplots_adjust(hspace=0.4) # Add space between text elements
    save_plot(fig, "form1_scales_representations")

def form1_sets_venn_diagram_two_sets():
    """Generates a simple two-set Venn diagram structure."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Universal set rectangle
    ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.95, 0.9, 'U', ha='right', va='top', fontsize=14)

    # Circles for sets A and B
    circle_a = Circle((0.4, 0.5), 0.3, fill=False, edgecolor='blue', linewidth=1.5)
    circle_b = Circle((0.6, 0.5), 0.3, fill=False, edgecolor='red', linewidth=1.5)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    # Labels for sets
    ax.text(0.25, 0.75, 'A', ha='center', va='center', fontsize=14, color='blue')
    ax.text(0.75, 0.75, 'B', ha='center', va='center', fontsize=14, color='red')

    # Labels for regions (optional text, positioning is tricky)
    ax.text(0.5, 0.5, 'A ∩ B', ha='center', va='center', fontsize=10) # Intersection
    ax.text(0.3, 0.5, 'A only', ha='center', va='center', fontsize=10)
    ax.text(0.7, 0.5, 'B only', ha='center', va='center', fontsize=10)
    ax.text(0.5, 0.2, '(A ∪ B)\'', ha='center', va='center', fontsize=10) # Neither

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Venn Diagram (2 Sets)")
    plt.tight_layout()
    save_plot(fig, "form1_sets_venn_diagram_two_sets")

def form1_sets_venn_example_1_to_7():
    """Generates Venn diagram for the specific example U={1..7}, A={1,2,3}, B={3,4,5}."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Universal set rectangle
    ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.95, 0.9, 'U', ha='right', va='top', fontsize=14)

    # Circles for sets A and B
    center_a = (0.4, 0.5)
    center_b = (0.6, 0.5)
    radius = 0.3
    circle_a = Circle(center_a, radius, fill=False, edgecolor='blue', linewidth=1.5)
    circle_b = Circle(center_b, radius, fill=False, edgecolor='red', linewidth=1.5)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    # Labels for sets
    ax.text(0.25, 0.75, 'A', ha='center', va='center', fontsize=14, color='blue')
    ax.text(0.75, 0.75, 'B', ha='center', va='center', fontsize=14, color='red')

    # Place elements in regions
    ax.text(0.5, 0.5, '3', ha='center', va='center', fontsize=12)      # Intersection
    ax.text(0.3, 0.55, '1', ha='center', va='center', fontsize=12)     # A only
    ax.text(0.3, 0.45, '2', ha='center', va='center', fontsize=12)     # A only
    ax.text(0.7, 0.55, '4', ha='center', va='center', fontsize=12)     # B only
    ax.text(0.7, 0.45, '5', ha='center', va='center', fontsize=12)     # B only
    ax.text(0.15, 0.15, '6', ha='center', va='center', fontsize=12)    # Neither
    ax.text(0.85, 0.15, '7', ha='center', va='center', fontsize=12)    # Neither

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Venn Diagram Example: U={1..7}, A={1,2,3}, B={3,4,5}")
    plt.tight_layout()
    save_plot(fig, "form1_sets_venn_example_1_to_7")

def form1_sets_venn_shade_intersection():
    """Generates Venn diagram for U={a..g}, A={a,b,c,d}, B={c,d,e,f} shading A intersect B."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Universal set rectangle
    rect = Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.95, 0.9, 'U', ha='right', va='top', fontsize=14)

    # Circles for sets A and B
    center_a = (0.4, 0.5)
    center_b = (0.6, 0.5)
    radius = 0.3
    circle_a = Circle(center_a, radius, fill=False, edgecolor='blue', linewidth=1.5, label='A')
    circle_b = Circle(center_b, radius, fill=False, edgecolor='red', linewidth=1.5, label='B')
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    # Labels for sets
    ax.text(0.25, 0.75, 'A', ha='center', va='center', fontsize=14, color='blue')
    ax.text(0.75, 0.75, 'B', ha='center', va='center', fontsize=14, color='red')

    # Place elements in regions
    ax.text(0.5, 0.55, 'c', ha='center', va='center', fontsize=12)     # Intersection
    ax.text(0.5, 0.45, 'd', ha='center', va='center', fontsize=12)     # Intersection
    ax.text(0.3, 0.55, 'a', ha='center', va='center', fontsize=12)     # A only
    ax.text(0.3, 0.45, 'b', ha='center', va='center', fontsize=12)     # A only
    ax.text(0.7, 0.55, 'e', ha='center', va='center', fontsize=12)     # B only
    ax.text(0.7, 0.45, 'f', ha='center', va='center', fontsize=12)     # B only
    ax.text(0.15, 0.15, 'g', ha='center', va='center', fontsize=12)    # Neither

    # Shade the intersection A ∩ B
    # Create paths for circles
    path_a = Path.circle(center_a, radius)
    path_b = Path.circle(center_b, radius)

    # Use PathPatch intersection (requires some effort) or approximate shading
    # Simpler: Add a shaded circle/ellipse in the intersection area
    intersection_patch = Circle((0.5, 0.5), 0.15, color='lightgray', alpha=0.7, zorder=0) # Adjust size/position
    ax.add_patch(intersection_patch) # This is an approximation

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Venn Diagram: A ∩ B Shaded")
    plt.tight_layout()
    save_plot(fig, "form1_sets_venn_shade_intersection")

def form1_measures_thermometers():
    """Generates Celsius and Fahrenheit thermometer illustrations."""
    fig, axs = plt.subplots(1, 2, figsize=(4, 6))

    # Celsius Thermometer
    ax_c = axs[0]
    c_temps = np.linspace(-10, 40, 6) # Major ticks
    c_values = np.linspace(-10, 100, 12) # Full range for calculations if needed
    ax_c.bar(0, 25 - (-10), bottom=-10, width=0.5, color='red', align='center') # Bar from -10 up to 25
    ax_c.set_yticks(c_temps)
    ax_c.set_ylim(-15, 45)
    ax_c.set_xticks([])
    ax_c.set_title('Celsius (°C)')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.spines['bottom'].set_visible(False)
    ax_c.yaxis.set_ticks_position('left')
    ax_c.axhline(0, color='blue', linestyle='--', linewidth=1, label='Freezing (0°C)') # Freezing line
    ax_c.text(0.3, 25, '25°C', va='center', ha='left')

    # Fahrenheit Thermometer
    ax_f = axs[1]
    # Calculate corresponding F temps for C ticks
    f_temps_major = (c_temps * 9/5) + 32
    f_current = (25 * 9/5) + 32 # 77
    f_min_display = (-10 * 9/5) + 32 # 14
    f_max_display = (40 * 9/5) + 32 # 104
    f_freezing = 32
    f_boiling = 212

    ax_f.bar(0, f_current - f_min_display, bottom=f_min_display, width=0.5, color='red', align='center')
    ax_f.set_yticks(np.linspace(round(f_min_display), round(f_max_display), 6)) # Approx ticks
    ax_f.set_ylim(f_min_display - 5, f_max_display + 5)
    ax_f.set_xticks([])
    ax_f.set_title('Fahrenheit (°F)')
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.spines['bottom'].set_visible(False)
    ax_f.yaxis.set_ticks_position('left')
    ax_f.axhline(f_freezing, color='blue', linestyle='--', linewidth=1, label='Freezing (32°F)')
    ax_f.text(0.3, f_current, '77°F', va='center', ha='left')

    plt.tight_layout()
    save_plot(fig, "form1_measures_thermometers")

def form1_measures_capacity_containers():
    """Generates measuring jug and bottle illustrations."""
    fig, axs = plt.subplots(1, 2, figsize=(5, 5))

    # Measuring Jug
    ax_jug = axs[0]
    jug_width = 0.6
    jug_base = 0.1
    jug_top = 0.9
    jug_x = 0.5

    # Jug outline
    ax_jug.plot([jug_x - jug_width/2, jug_x - jug_width/2], [jug_base, jug_top], 'k-') # Left side
    ax_jug.plot([jug_x + jug_width/2, jug_x + jug_width/2], [jug_base, jug_top], 'k-') # Right side
    ax_jug.plot([jug_x - jug_width/2, jug_x + jug_width/2], [jug_base, jug_base], 'k-') # Bottom

    # Markings and liquid level (approximate y positions based on 0-1 scale)
    markings = {250: 0.3, 500: 0.5, 750: 0.7, 1000: 0.9}
    liquid_level_val = 750
    liquid_level_y = markings[liquid_level_val]

    # Draw liquid
    ax_jug.add_patch(Rectangle((jug_x - jug_width/2, jug_base), jug_width, liquid_level_y - jug_base,
                             facecolor='lightblue', edgecolor='none'))
    # Liquid surface
    ax_jug.plot([jug_x - jug_width/2, jug_x + jug_width/2], [liquid_level_y, liquid_level_y], 'blue')

    # Draw markings
    for val, y_pos in markings.items():
        ax_jug.plot([jug_x - jug_width/2 - 0.05, jug_x - jug_width/2], [y_pos, y_pos], 'k-')
        ax_jug.text(jug_x - jug_width/2 - 0.1, y_pos, f'{val} mL', ha='right', va='center', fontsize=9)
        if val == 1000:
             ax_jug.text(jug_x - jug_width/2 - 0.1, y_pos+0.02, '(1 L)', ha='right', va='bottom', fontsize=8)


    ax_jug.set_xlim(0, 1)
    ax_jug.set_ylim(0, 1)
    ax_jug.axis('off')
    ax_jug.set_title("Measuring Jug")

    # Bottle
    ax_bottle = axs[1]
    bottle_width = 0.4
    bottle_base = 0.1
    bottle_body_height = 0.6
    bottle_neck_height = 0.2
    bottle_neck_width = 0.15
    bottle_x = 0.5

    # Bottle outline
    ax_bottle.plot([bottle_x - bottle_width/2, bottle_x - bottle_width/2], [bottle_base, bottle_base + bottle_body_height], 'k-') # Left body
    ax_bottle.plot([bottle_x + bottle_width/2, bottle_x + bottle_width/2], [bottle_base, bottle_base + bottle_body_height], 'k-') # Right body
    ax_bottle.plot([bottle_x - bottle_width/2, bottle_x + bottle_width/2], [bottle_base, bottle_base], 'k-') # Bottom
    # Shoulder
    ax_bottle.plot([bottle_x - bottle_width/2, bottle_x - bottle_neck_width/2], [bottle_base + bottle_body_height, bottle_base + bottle_body_height + bottle_neck_height*0.3], 'k-')
    ax_bottle.plot([bottle_x + bottle_width/2, bottle_x + bottle_neck_width/2], [bottle_base + bottle_body_height, bottle_base + bottle_body_height + bottle_neck_height*0.3], 'k-')
    # Neck
    ax_bottle.plot([bottle_x - bottle_neck_width/2, bottle_x - bottle_neck_width/2], [bottle_base + bottle_body_height+ bottle_neck_height*0.3, bottle_base + bottle_body_height + bottle_neck_height], 'k-')
    ax_bottle.plot([bottle_x + bottle_neck_width/2, bottle_x + bottle_neck_width/2], [bottle_base + bottle_body_height+ bottle_neck_height*0.3, bottle_base + bottle_body_height + bottle_neck_height], 'k-')
    # Top
    ax_bottle.plot([bottle_x - bottle_neck_width/2, bottle_x + bottle_neck_width/2], [bottle_base + bottle_body_height + bottle_neck_height, bottle_base + bottle_body_height + bottle_neck_height], 'k-')


    # Label
    ax_bottle.text(bottle_x, bottle_base + bottle_body_height / 2, '1 L\n(1000 mL)', ha='center', va='center', fontsize=10)

    ax_bottle.set_xlim(0, 1)
    ax_bottle.set_ylim(0, 1)
    ax_bottle.axis('off')
    ax_bottle.set_title("1 Liter Bottle")

    plt.tight_layout()
    save_plot(fig, "form1_measures_capacity_containers")

def form1_mensuration_perimeter_formulas():
    """Generates diagrams for perimeter formulas."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    # Square
    s = 1 # unit size for drawing
    axs[0].add_patch(Rectangle((0.1, 0.1), s, s, fill=False, edgecolor='black', linewidth=1.5))
    axs[0].text(0.1 + s/2, 0.05, 's', ha='center', va='top', fontsize=12)
    axs[0].text(0.5, 0.9, 'P = 4s', ha='center', va='bottom', fontsize=12)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].axis('off')
    axs[0].set_title("Square Perimeter")

    # Rectangle
    l, w = 1.5, 0.8
    axs[1].add_patch(Rectangle((0.1, 0.1), l, w, fill=False, edgecolor='black', linewidth=1.5))
    axs[1].text(0.1 + l/2, 0.05, 'l (length)', ha='center', va='top', fontsize=12)
    axs[1].text(0.05, 0.1 + w/2, 'w\n(width)', ha='right', va='center', fontsize=12)
    axs[1].text(0.5 * (0.1+0.1+l) , 0.1 + w + 0.1 , 'P = 2(l + w)', ha='center', va='bottom', fontsize=12)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].axis('off')
    axs[1].set_title("Rectangle Perimeter")

    # Triangle
    verts = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.7)]
    axs[2].add_patch(Polygon(verts, fill=False, edgecolor='black', linewidth=1.5))
    axs[2].text(0.5, 0.05, 'a', ha='center', va='top', fontsize=12) # Base
    axs[2].text(0.3, 0.4, 'c', ha='right', va='center', fontsize=12) # Left side
    axs[2].text(0.7, 0.4, 'b', ha='left', va='center', fontsize=12) # Right side
    axs[2].text(0.5, 0.9, 'P = a + b + c', ha='center', va='bottom', fontsize=12)
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].axis('off')
    axs[2].set_title("Triangle Perimeter")

    plt.tight_layout()
    save_plot(fig, "form1_mensuration_perimeter_formulas")

def form1_mensuration_area_formulas():
    """Generates diagrams for area formulas."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5)) # Slightly taller for triangle height

    # Square
    s = 1
    axs[0].add_patch(Rectangle((0.1, 0.1), s, s, fill=False, edgecolor='black', linewidth=1.5))
    # Optional grid lines
    axs[0].grid(True, which='both', axis='both', linestyle=':', linewidth=0.5)
    axs[0].set_xticks(np.linspace(0.1, 0.1+s, 5)) # Example grid
    axs[0].set_yticks(np.linspace(0.1, 0.1+s, 5)) # Example grid
    axs[0].tick_params(labelbottom=False, labelleft=False)

    axs[0].text(0.1 + s/2, 0.05, 's', ha='center', va='top', fontsize=12)
    axs[0].text(0.5, 0.9, 'A = s²', ha='center', va='bottom', fontsize=12)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlim(0, s + 0.2)
    axs[0].set_ylim(0, s + 0.2)
    axs[0].axis('off')
    axs[0].set_title("Square Area")

    # Rectangle
    l, w = 1.5, 0.8
    axs[1].add_patch(Rectangle((0.1, 0.1), l, w, fill=False, edgecolor='black', linewidth=1.5))
    # Optional grid lines
    axs[1].grid(True, which='both', axis='both', linestyle=':', linewidth=0.5)
    axs[1].set_xticks(np.linspace(0.1, 0.1+l, 7)) # Example grid
    axs[1].set_yticks(np.linspace(0.1, 0.1+w, 5)) # Example grid
    axs[1].tick_params(labelbottom=False, labelleft=False)

    axs[1].text(0.1 + l/2, 0.05, 'l', ha='center', va='top', fontsize=12)
    axs[1].text(0.05, 0.1 + w/2, 'w', ha='right', va='center', fontsize=12)
    axs[1].text(0.5 * (0.1+0.1+l) , 0.1 + w + 0.1 , 'A = l × w', ha='center', va='bottom', fontsize=12)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlim(0, l + 0.2)
    axs[1].set_ylim(0, w + 0.2)
    axs[1].axis('off')
    axs[1].set_title("Rectangle Area")

    # Triangle
    base_start = 0.1
    base_end = 0.9
    base = base_end - base_start
    height = 0.6
    top_x = 0.4 # Vertex x-coordinate
    verts = [(base_start, 0.1), (base_end, 0.1), (top_x, 0.1 + height)]

    axs[2].add_patch(Polygon(verts, fill=False, edgecolor='black', linewidth=1.5))
    # Base label
    axs[2].text((base_start + base_end)/2, 0.05, 'b', ha='center', va='top', fontsize=12)
    # Height line and label
    axs[2].plot([top_x, top_x], [0.1, 0.1 + height], 'k--') # Dashed height line
    axs[2].plot([top_x-0.02, top_x], [0.1, 0.1], 'k-') # Small indication of right angle
    axs[2].plot([top_x, top_x], [0.1, 0.1+0.02], 'k-')
    axs[2].text(top_x + 0.03, 0.1 + height/2, 'h', ha='left', va='center', fontsize=12)
    # Area formula
    axs[2].text(0.5, 0.9, 'A = ½bh', ha='center', va='bottom', fontsize=12)

    axs[2].set_aspect('equal', adjustable='box')
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[2].axis('off')
    axs[2].set_title("Triangle Area")

    plt.tight_layout()
    save_plot(fig, "form1_mensuration_area_formulas")

def form1_mensuration_square_area_problem():
    """Generates diagram for square area problem."""
    fig, ax = plt.subplots(figsize=(4, 4))
    s_val = 7 # Since area is 49
    ax.add_patch(Rectangle((0.1, 0.1), s_val, s_val, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.1 + s_val/2, 0.1 + s_val/2, 'Area = 49 m²', ha='center', va='center', fontsize=12)
    ax.text(0.1 + s_val/2, 0.05, 's = ?', ha='center', va='top', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, s_val + 0.2)
    ax.set_ylim(0, s_val + 0.2)
    ax.axis('off')
    ax.set_title("Square Area Problem")
    plt.tight_layout()
    save_plot(fig, "form1_mensuration_square_area_problem")

def form1_graphs_cartesian_plane():
    """Generates the standard Cartesian plane with quadrants."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits and ticks
    limit = 5.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ticks = np.arange(-5, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Add arrowheads
    ax.plot(limit, 0, ">k", markersize=10, clip_on=False)
    ax.plot(0, limit, "^k", markersize=10, clip_on=False)

    # Labels
    ax.set_xlabel("x", loc='right', fontsize=12)
    ax.set_ylabel("y", loc='top', rotation=0, fontsize=12)
    ax.text(0.1, -0.4, '(0,0)', ha='left', va='top', fontsize=10)

    # Quadrant Labels
    q_offset = limit * 0.7
    ax.text(q_offset, q_offset, 'I', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(-q_offset, q_offset, 'II', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(-q_offset, -q_offset, 'III', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(q_offset, -q_offset, 'IV', ha='center', va='center', fontsize=14, weight='bold')

    # Line thickness
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    save_plot(fig, "form1_graphs_cartesian_plane")

def form1_graphs_cartesian_plane_example():
    """Generates a Cartesian plane from -4 to 4, 1cm=1unit assumed."""
    fig, ax = plt.subplots(figsize=(5, 5)) # Adjust size as needed

    # Set limits and ticks (assuming 1 unit = 1 data unit)
    limit = 4.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ticks = np.arange(-4, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Add arrowheads
    ax.plot(limit, 0, ">k", markersize=8, clip_on=False)
    ax.plot(0, limit, "^k", markersize=8, clip_on=False)

    # Labels
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0)
    ax.text(0.1, -0.3, '(0,0)', ha='left', va='top')

    # Line thickness
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Set aspect ratio to equal to simulate 1cm=1unit visually
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    save_plot(fig, "form1_graphs_cartesian_plane_example")

def form1_graphs_plot_point_a32():
    """Plots point A(3,2) with projection lines."""
    fig, ax = plt.subplots(figsize=(5, 5))
    x, y = 3, 2

    # Set limits and ticks
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_xticks(np.arange(-1, 6))
    ax.set_yticks(np.arange(-1, 6))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Plot point
    ax.plot(x, y, 'ro', markersize=8, label='A(3, 2)')

    # Add projection lines
    ax.plot([x, x], [0, y], 'r--') # Vertical line to x-axis
    ax.plot([0, x], [y, y], 'r--') # Horizontal line to y-axis

    # Add labels
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0)
    ax.text(x + 0.1, y + 0.1, 'A(3, 2)', fontsize=10, color='red')

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    save_plot(fig, "form1_graphs_plot_point_a32")

def form1_graphs_plot_points_bcd():
    """Plots points B(-2,4), C(-3,-1), D(4,-3)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    points = {'B': (-2, 4), 'C': (-3, -1), 'D': (4, -3)}
    colors = {'B': 'blue', 'C': 'green', 'D': 'purple'}

    # Set limits and ticks
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xticks(np.arange(-5, 6))
    ax.set_yticks(np.arange(-5, 6))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Plot points and labels
    for label, (x, y) in points.items():
        ax.plot(x, y, marker='o', color=colors[label], markersize=8, label=f'{label}({x}, {y})')
        ax.text(x + 0.2, y + 0.2, f'{label}({x}, {y})', fontsize=10, color=colors[label])

    # Add axis labels
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Plotting Points B, C, D")
    plt.tight_layout()
    save_plot(fig, "form1_graphs_plot_points_bcd")

def form1_graphs_identify_points_pqr():
    """Plots points P(2,3), Q(-3,1), R(1,-2) for identification."""
    fig, ax = plt.subplots(figsize=(5, 5))
    points = {'P': (2, 3), 'Q': (-3, 1), 'R': (1, -2)}

    # Set limits and ticks
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xticks(np.arange(-4, 5))
    ax.set_yticks(np.arange(-4, 5))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Plot points and labels (only letter)
    for label, (x, y) in points.items():
        ax.plot(x, y, 'ko', markersize=8)
        ax.text(x + 0.2, y + 0.2, label, fontsize=12, weight='bold')

    # Add axis labels
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Identify Coordinates of P, Q, R")
    plt.tight_layout()
    save_plot(fig, "form1_graphs_identify_points_pqr")

def form1_graphs_cartesian_grid_m5_to_5():
    """Generates a blank Cartesian grid from -5 to 5."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set limits and ticks
    limit = 5.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ticks = np.arange(-5, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Add arrowheads
    ax.plot(limit, 0, ">k", markersize=8, clip_on=False)
    ax.plot(0, limit, "^k", markersize=8, clip_on=False)

    # Labels
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Cartesian Plane (-5 to 5)")
    plt.tight_layout()
    save_plot(fig, "form1_graphs_cartesian_grid_m5_to_5")

def form1_graphs_rectangle_find_d():
    """Plots three vertices of a rectangle A(1,2), B(4,2), C(4,4)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    points = {'A': (1, 2), 'B': (4, 2), 'C': (4, 4)}

    # Set limits and ticks
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(0, 6))
    ax.set_yticks(np.arange(0, 6))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot points and labels
    for label, (x, y) in points.items():
        ax.plot(x, y, 'ko', markersize=8)
        ax.text(x + 0.1, y + 0.1, f'{label}({x}, {y})', fontsize=10)

    # Draw known sides
    ax.plot([points['A'][0], points['B'][0]], [points['A'][1], points['B'][1]], 'k-') # AB
    ax.plot([points['B'][0], points['C'][0]], [points['B'][1], points['C'][1]], 'k-') # BC

    # Add axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Find Vertex D of Rectangle ABCD")
    plt.tight_layout()
    save_plot(fig, "form1_graphs_rectangle_find_d")

def form1_travel_graphs_distance_time_sample():
    """Generates the sample distance-time graph."""
    fig, ax = plt.subplots(figsize=(6, 4))
    time = [0, 2, 3, 5]
    distance = [0, 80, 80, 0]

    ax.plot(time, distance, 'b-o', linewidth=1.5, markersize=5)

    # Annotate points
    ax.text(0, 5, 'O(0,0)', ha='left', va='bottom')
    ax.text(2, 85, 'A(2h, 80km)', ha='center', va='bottom')
    ax.text(3, 85, 'B(3h, 80km)', ha='center', va='bottom')
    ax.text(5, 5, 'C(5h, 0km)', ha='right', va='bottom')

    # Labels and Title
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance from Start (km)")
    ax.set_title("Sample Distance-Time Graph")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-10, 100)

    plt.tight_layout()
    save_plot(fig, "form1_travel_graphs_distance_time_sample")



def form1_geometry_types_of_lines():
     """Generates diagrams illustrating types of lines."""
     fig, axs = plt.subplots(3, 2, figsize=(8, 10)) # Arrange in 3 rows, 2 columns
     axs_flat = axs.flatten()
     fig.patch.set_facecolor('white')

     # 1. Parallel Lines
     ax = axs_flat[0]
     ax.plot([0, 3], [1, 1], 'k-', lw=1.5)
     ax.plot([0, 3], [0, 0], 'k-', lw=1.5)
     ax.text(1.5, 1.05, '>', ha='center', va='bottom', fontsize=12)
     ax.text(1.5, 0.05, '>', ha='center', va='bottom', fontsize=12)
     ax.set_title("Parallel Lines")
     ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 1.5)

     # 2. Intersecting Lines
     ax = axs_flat[1]
     ax.plot([0, 2], [0, 2], 'k-', lw=1.5)
     ax.plot([0, 2], [2, 0], 'k-', lw=1.5)
     ax.set_title("Intersecting Lines")
     ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5)

     # 3. Perpendicular Lines
     ax = axs_flat[2]
     ax.plot([0, 2], [1, 1], 'k-', lw=1.5)
     ax.plot([1, 1], [0, 2], 'k-', lw=1.5)
     ax.plot([1, 1.1], [1, 1], 'k-'); ax.plot([1, 1], [1, 1.1], 'k-') # Right angle symbol
     ax.set_title("Perpendicular Lines")
     ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5)

     # 4. Line Segment
     ax = axs_flat[3]
     ax.plot([0, 2], [0.5, 0.5], 'k-', lw=1.5)
     ax.plot(0, 0.5, 'ko', markersize=6) # Endpoint A dot
     ax.plot(2, 0.5, 'ko', markersize=6) # Endpoint B dot
     ax.text(-0.1, 0.5, 'A', ha='right', va='center')
     ax.text(2.1, 0.5, 'B', ha='left', va='center')
     ax.set_title("Line Segment AB")
     ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 1)

     # 5. Ray
     ax = axs_flat[4]
     ax.plot(0, 0.5, 'ko', markersize=6) # Endpoint C dot
     ax.plot([0, 3], [0.5, 0.5], 'k-', lw=1.5) # Line
     ax.plot(3, 0.5, '>k', markersize=8, clip_on=False) # Arrowhead
     ax.text(-0.1, 0.5, 'C', ha='right', va='center')
     ax.text(1.5, 0.6, 'D', ha='center', va='bottom') # Point D on the ray
     ax.plot(1.5, 0.5, 'kx', markersize=5) # Mark point D
     ax.set_title("Ray CD")
     ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-0.5, 3.5); ax.set_ylim(0, 1)

     # Hide the last empty subplot
     axs_flat[5].axis('off')

     plt.tight_layout(pad=2.0) # Add padding between plots
     save_plot(fig, "form1_geometry_types_of_lines")


def form1_travel_graphs_walk_rest_return():
     """Generates distance-time graph for walk/rest/return (Renamed from F2)."""
     fig, ax = plt.subplots(figsize=(5, 5))
     fig.patch.set_facecolor('white')
     # Points: (0,0), (1,4), (1.5,4), (3,0)
     time = [0, 1, 1.5, 3]
     distance = [0, 4, 4, 0]

     ax.plot(time, distance, 'b-o', linewidth=1.5, markersize=5)

     # Labels and Title
     ax.set_xlabel("Time (hours)")
     ax.set_ylabel("Distance from home (km)")
     ax.set_title("Walk, Rest, Return Journey")
     ax.grid(True, linestyle=':', alpha=0.7)
     ax.set_xticks(np.arange(0, 3.5, 0.5))
     ax.set_yticks(np.arange(0, 5.5, 1))
     ax.set_xlim(-0.1, 3.1)
     ax.set_ylim(-0.2, 5)
     ax.spines['left'].set_position('zero')
     ax.spines['bottom'].set_position('zero')
     ax.spines['right'].set_color('none')
     ax.spines['top'].set_color('none')

     # Annotate key points
     ax.text(0-0.05, 0-0.1, '(0,0)', ha='right', va='top', fontsize=8)
     ax.text(1, 4+0.1, '(1h, 4km)', ha='center', va='bottom', fontsize=8)
     ax.text(1.5, 4+0.1, '(1.5h, 4km)', ha='center', va='bottom', fontsize=8)
     ax.text(3+0.05, 0-0.1, '(3h, 0km)', ha='left', va='top', fontsize=8)


     plt.tight_layout()
     save_plot(fig, "form1_travel_graphs_walk_rest_return") # Corrected filename

def form1_travel_graphs_example_journey1():
    """Generates the distance-time graph for worked example 1."""
    fig, ax = plt.subplots(figsize=(6, 5))
    time = [0, 2, 3, 4]
    distance = [0, 100, 100, 120]

    ax.plot(time, distance, 'b-o', linewidth=1.5, markersize=5)

    # Labels and Title
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance (km)")
    ax.set_title("Car's Journey (Distance-Time)")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 131, 20))
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, 130)

    plt.tight_layout()
    save_plot(fig, "form1_travel_graphs_example_journey1")

def form1_travel_graphs_cyclist_grid():
    """Generates a blank grid for the cyclist problem."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Labels and Title
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance from home (km)")
    ax.set_title("Cyclist Journey (Distance-Time)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(np.arange(0, 5.5, 0.5))
    ax.set_yticks(np.arange(0, 41, 5))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 40)

    plt.tight_layout()
    save_plot(fig, "form1_travel_graphs_cyclist_grid")

def form1_transformation_translate_triangle_pqr():
    """Generates diagram for translation of triangle PQR."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Original vertices
    P = np.array([1, 1])
    Q = np.array([4, 1])
    R = np.array([1, 3])
    obj_triangle = np.array([P, Q, R, P]) # Close the triangle

    # Translation vector
    trans_vec = np.array([2, -3])

    # Image vertices
    P_prime = P + trans_vec
    Q_prime = Q + trans_vec
    R_prime = R + trans_vec
    img_triangle = np.array([P_prime, Q_prime, R_prime, P_prime]) # Close the triangle

    # Plot object
    ax.plot(obj_triangle[:, 0], obj_triangle[:, 1], 'b-o', label='Object PQR')
    ax.text(P[0], P[1]+0.1, 'P(1,1)', color='blue')
    ax.text(Q[0], Q[1]+0.1, 'Q(4,1)', color='blue')
    ax.text(R[0], R[1]+0.1, 'R(1,3)', color='blue')

    # Plot image
    ax.plot(img_triangle[:, 0], img_triangle[:, 1], 'r--o', label="Image P'Q'R'")
    ax.text(P_prime[0], P_prime[1]-0.3, "P'(3,-2)", color='red')
    ax.text(Q_prime[0], Q_prime[1]-0.3, "Q'(6,-2)", color='red')
    ax.text(R_prime[0], R_prime[1]+0.1, "R'(3,0)", color='red')

    # Optional: Draw translation arrows
    ax.arrow(P[0], P[1], trans_vec[0], trans_vec[1], head_width=0.2, head_length=0.3, fc='gray', ec='gray', linestyle=':')
    ax.arrow(Q[0], Q[1], trans_vec[0], trans_vec[1], head_width=0.2, head_length=0.3, fc='gray', ec='gray', linestyle=':')
    ax.arrow(R[0], R[1], trans_vec[0], trans_vec[1], head_width=0.2, head_length=0.3, fc='gray', ec='gray', linestyle=':')

    # Setup plot
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Translation of Triangle PQR by <2, -3>")
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-3, 5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    plt.tight_layout()
    save_plot(fig, "form1_transformation_translate_triangle_pqr")

def form1_transformation_describe_translation_ab():
    """Generates two shapes A and B for describing translation."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Shape A vertices (Square)
    A_verts = np.array([[1,1], [2,1], [2,2], [1,2], [1,1]])
    # Shape B vertices (Translated Square)
    B_verts = np.array([[4,3], [5,3], [5,4], [4,4], [4,3]])

    # Plot shapes
    ax.plot(A_verts[:, 0], A_verts[:, 1], 'b-o', label='Shape A')
    ax.text(1.5, 1.5, 'A', ha='center', va='center', fontsize=14, color='blue')
    ax.plot(B_verts[:, 0], B_verts[:, 1], 'r--o', label='Shape B')
    ax.text(4.5, 3.5, 'B', ha='center', va='center', fontsize=14, color='red')

    # Setup plot
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Describe Translation from A to B")
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 7, 1))
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 6.5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    plt.tight_layout()
    save_plot(fig, "form1_transformation_describe_translation_ab")

# --- Form 2 ---

def form2_real_numbers_sqrt_prime_factorization():
    """Illustrates sqrt(144) using prime factors."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    text_content = (
        r"$\sqrt{144}$ Calculation:" + "\n\n" +
        r"1. Prime Factorization: $144 = 2 \times 2 \times 2 \times 2 \times 3 \times 3$" + "\n\n" +
        r"2. Group Pairs: $144 = (2 \times 2) \times (2 \times 2) \times (3 \times 3) = 2^2 \times 2^2 \times 3^2$" + "\n\n" +
        r"3. Take one factor from each pair: $2 \times 2 \times 3$" + "\n\n" +
        r"4. Result: $= 12$" + "\n\n" +
        r"Therefore: $\sqrt{144} = 12$"
    )
    ax.text(0.5, 0.5, text_content, ha='center', va='center', fontsize=12)
    ax.set_title("Square Root via Prime Factorization")
    plt.tight_layout()
    save_plot(fig, "form2_real_numbers_sqrt_prime_factorization")

def form2_real_numbers_cbrt_prime_factorization():
    """Illustrates cbrt(216) using prime factors."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    text_content = (
        r"$\sqrt[3]{216}$ Calculation:" + "\n\n" +
        r"1. Prime Factorization: $216 = 2 \times 2 \times 2 \times 3 \times 3 \times 3$" + "\n\n" +
        r"2. Group Triplets: $216 = (2 \times 2 \times 2) \times (3 \times 3 \times 3) = 2^3 \times 3^3$" + "\n\n" +
        r"3. Take one factor from each triplet: $2 \times 3$" + "\n\n" +
        r"4. Result: $= 6$" + "\n\n" +
        r"Therefore: $\sqrt[3]{216} = 6$"
    )
    ax.text(0.5, 0.5, text_content, ha='center', va='center', fontsize=12)
    ax.set_title("Cube Root via Prime Factorization")
    plt.tight_layout()
    save_plot(fig, "form2_real_numbers_cbrt_prime_factorization")

def form2_proportion_direct_vs_inverse_graphs():
    """Generates graphs for direct and inverse proportion."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Direct Proportion
    x_direct = np.linspace(0, 5, 100)
    k_direct = 1.5 # Example slope
    y_direct = k_direct * x_direct
    axs[0].plot(x_direct, y_direct, label=f'y = {k_direct}x')
    axs[0].set_title('Direct Proportion (y = kx)')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].grid(True, linestyle=':')
    axs[0].axhline(0, color='black', linewidth=0.5)
    axs[0].axvline(0, color='black', linewidth=0.5)
    axs[0].legend()
    axs[0].set_xlim(0, 5)
    axs[0].set_ylim(0, k_direct*5 + 1)

    # Inverse Proportion
    x_inverse = np.linspace(0.5, 5, 100) # Avoid x=0
    k_inverse = 5 # Example constant
    y_inverse = k_inverse / x_inverse
    axs[1].plot(x_inverse, y_inverse, label=f'y = {k_inverse}/x')
    axs[1].set_title('Inverse Proportion (y = k/x)')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].grid(True, linestyle=':')
    axs[1].axhline(0, color='black', linewidth=0.5)
    axs[1].axvline(0, color='black', linewidth=0.5)
    axs[1].legend()
    axs[1].set_xlim(0, 5)
    axs[1].set_ylim(0, k_inverse/0.5 + 1)


    plt.tight_layout()
    save_plot(fig, "form2_proportion_direct_vs_inverse_graphs")

def form2_scales_garden_drawing():
    """Draws the scaled rectangle for the garden example."""
    fig, ax = plt.subplots(figsize=(4, 3))
    map_length = 6
    map_width = 4

    ax.add_patch(Rectangle((0.1, 0.1), map_length, map_width, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.1 + map_length/2, 0.05, '6 cm', ha='center', va='top', fontsize=10)
    ax.text(0.05, 0.1 + map_width/2, '4 cm', ha='right', va='center', fontsize=10)
    ax.text(0.1 + map_length/2, 0.1 + map_width + 0.1, 'Scale 1 : 500', ha='center', va='bottom', fontsize=10)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, map_length + 0.2)
    ax.set_ylim(0, map_width + 0.3)
    ax.axis('off')
    ax.set_title("Scaled Garden Plan")
    plt.tight_layout()
    save_plot(fig, "form2_scales_garden_drawing")

def form2_sets_venn_two_sets_regions():
    """Generates labeled regions for a two-set Venn diagram."""
    # This is identical to form1_sets_venn_diagram_two_sets
    form1_sets_venn_diagram_two_sets()
    # Rename the file if a specific form 2 name is desired
    os.rename(os.path.join(output_dir, "form1_sets_venn_diagram_two_sets.png"),
              os.path.join(output_dir, "form2_sets_venn_two_sets_regions.png"))

def form2_sets_venn_example_counts():
    """Generates Venn diagram with counts for U={1..8}, A={1,2,3,4}, B={3,4,5,6}."""
    # This is identical to form1_sets_venn_example_1_to_7
    form1_sets_venn_example_1_to_7()
    # Rename the file if a specific form 2 name is desired
    os.rename(os.path.join(output_dir, "form1_sets_venn_example_1_to_7.png"),
              os.path.join(output_dir, "form2_sets_venn_example_counts.png"))

def form2_sets_venn_music_example():
    """Generates Venn diagram for the music preference example."""
    # This is identical to form1_sets_venn_music_example
    form1_sets_venn_music_example()
     # Rename the file if a specific form 2 name is desired
    os.rename(os.path.join(output_dir, "form1_sets_venn_music_example.png"),
              os.path.join(output_dir, "form2_sets_venn_music_example.png"))

def form2_measures_area_conversion_m2_cm2():
    """Illustrates m^2 to cm^2 conversion."""
    fig, ax = plt.subplots(figsize=(5, 5))
    s = 1 # Represents 1m

    ax.add_patch(Rectangle((0.1, 0.1), s, s, fill=False, edgecolor='black', linewidth=2))
    ax.text(0.1 + s/2, 0.1 + s/2, 'Area = 1 m²', ha='center', va='center', fontsize=12)
    ax.text(0.1 + s/2, 0.05, '1 m', ha='center', va='top', fontsize=10)
    ax.text(0.05, 0.1 + s/2, '1 m', ha='right', va='center', fontsize=10)

    # Indicate cm conversion
    ax.text(0.1 + s/2, 1.15, '(100 cm)', ha='center', va='bottom', fontsize=9)
    ax.text(-0.05, 0.1 + s/2, '(100 cm)', ha='right', va='center', rotation=90, fontsize=9)

    # Grid lines (suggestive)
    n_lines = 10
    for i in range(1, n_lines):
        ax.plot([0.1 + i*s/n_lines, 0.1 + i*s/n_lines], [0.1, 0.1+s], 'gray', linestyle=':', linewidth=0.5)
        ax.plot([0.1, 0.1+s], [0.1 + i*s/n_lines, 0.1 + i*s/n_lines], 'gray', linestyle=':', linewidth=0.5)

    # Formula text
    ax.text(0.5, -0.05, '1 m² = 100 cm × 100 cm = 10,000 cm²', ha='center', va='top', fontsize=10)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, s + 0.2)
    ax.set_ylim(-0.1, s + 0.3)
    ax.axis('off')
    ax.set_title("Area Conversion: m² to cm²")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_plot(fig, "form2_measures_area_conversion_m2_cm2")

def form2_measures_volume_conversion_m3_cm3():
    """Illustrates m^3 to cm^3 conversion."""
    from itertools import product, combinations # Moved import here
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Simple cube outline (edges)
    r = [0, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="k")

    # Labels
    ax.text(0.5, -0.1, 0, '1 m', zdir='x', ha='center')
    ax.text(1.1, 0.5, 0, '1 m', zdir='y', ha='left')
    ax.text(1.1, 1, 0.5, '1 m', zdir='z', va='center')
    ax.text(0.5, 0.5, 1.2, 'Volume = 1 m³', zdir='z', ha='center')

    # Formula text below
    fig.text(0.5, 0.1, '1 m³ = 100cm × 100cm × 100cm = 1,000,000 cm³',
             ha='center', va='center', fontsize=10)

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,1]) # Aspect ratio is 1:1:1
    ax.axis('off')
    ax.set_title("Volume Conversion: m³ to cm³")
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    save_plot(fig, "form2_measures_volume_conversion_m3_cm3")

def form2_measures_floor_area_problem():
    """Diagram for floor area problem."""
    fig, ax = plt.subplots(figsize=(4, 3))
    l, w = 5, 4
    ax.add_patch(Rectangle((0.1, 0.1), l, w, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.1 + l/2, 0.05, '5 m', ha='center', va='top', fontsize=10)
    ax.text(0.05, 0.1 + w/2, '4 m', ha='right', va='center', fontsize=10)
    ax.text(0.1 + l/2, 0.1 + w/2, 'Area = ? cm²', ha='center', va='center', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, l + 0.2)
    ax.set_ylim(0, w + 0.2)
    ax.axis('off')
    ax.set_title("Floor Area Calculation")
    plt.tight_layout()
    save_plot(fig, "form2_measures_floor_area_problem")

def form2_mensuration_volume_cuboid_cube():
    """Generates diagrams for cuboid and cube volume."""
    fig = plt.figure(figsize=(8, 4))

    # Cuboid
    ax1 = fig.add_subplot(121, projection='3d')
    l, w, h = 3, 2, 1.5
    pos = [0,0,0]
    size = [l, w, h]
    # Vertices of the cuboid
    verts = [
        (pos[0], pos[1], pos[2]),
        (pos[0]+size[0], pos[1], pos[2]),
        (pos[0]+size[0], pos[1]+size[1], pos[2]),
        (pos[0], pos[1]+size[1], pos[2]),
        (pos[0], pos[1], pos[2]+size[2]),
        (pos[0]+size[0], pos[1], pos[2]+size[2]),
        (pos[0]+size[0], pos[1]+size[1], pos[2]+size[2]),
        (pos[0], pos[1]+size[1], pos[2]+size[2])
    ]
    # Edges
    edges = [
        [verts[0], verts[1]], [verts[1], verts[2]], [verts[2], verts[3]], [verts[3], verts[0]], # bottom
        [verts[4], verts[5]], [verts[5], verts[6]], [verts[6], verts[7]], [verts[7], verts[4]], # top
        [verts[0], verts[4]], [verts[1], verts[5]], [verts[2], verts[6]], [verts[3], verts[7]]  # sides
    ]
    for edge in edges:
        ax1.plot3D(*zip(*edge), color="k")

    ax1.text(l/2, -0.5, 0, 'l', zdir='x', ha='center')
    ax1.text(l+0.1, w/2, 0, 'w', zdir='y', va='center')
    ax1.text(l+0.1, w, h/2, 'h', zdir='z', ha='left')
    ax1.text(l/2, w/2, h*1.3, 'V = lwh', zdir='z', ha='center')
    ax1.set_box_aspect([l,w,h]) # Proportional aspect
    ax1.axis('off')
    ax1.set_title("Cuboid Volume")

     # Cube
    ax2 = fig.add_subplot(122, projection='3d')
    s = 2 # side length
    pos = [0,0,0]
    size = [s, s, s]
    verts = [
        (pos[0], pos[1], pos[2]), (pos[0]+size[0], pos[1], pos[2]), (pos[0]+size[0], pos[1]+size[1], pos[2]), (pos[0], pos[1]+size[1], pos[2]),
        (pos[0], pos[1], pos[2]+size[2]), (pos[0]+size[0], pos[1], pos[2]+size[2]), (pos[0]+size[0], pos[1]+size[1], pos[2]+size[2]), (pos[0], pos[1]+size[1], pos[2]+size[2])
    ]
    edges = [
        [verts[0], verts[1]], [verts[1], verts[2]], [verts[2], verts[3]], [verts[3], verts[0]], # bottom
        [verts[4], verts[5]], [verts[5], verts[6]], [verts[6], verts[7]], [verts[7], verts[4]], # top
        [verts[0], verts[4]], [verts[1], verts[5]], [verts[2], verts[6]], [verts[3], verts[7]]  # sides
    ]
    for edge in edges:
        ax2.plot3D(*zip(*edge), color="k")

    ax2.text(s/2, -0.3, 0, 's', zdir='x', ha='center')
    # ax2.text(s+0.1, s/2, 0, 's', zdir='y', va='center') # Optional second label
    # ax2.text(s+0.1, s, s/2, 's', zdir='z', ha='left') # Optional third label
    ax2.text(s/2, s/2, s*1.3, 'V = s³', zdir='z', ha='center')
    ax2.set_box_aspect([1,1,1]) # Equal aspect
    ax2.axis('off')
    ax2.set_title("Cube Volume")

    plt.tight_layout()
    # Need these imports
    from mpl_toolkits.mplot3d import Axes3D
    from itertools import product, combinations # if not already imported
    save_plot(fig, "form2_mensuration_volume_cuboid_cube")

def form2_graphs_cartesian_plane_scales():
    """Generates Cartesian plane with specific scales 2cm:1unit(x), 1cm:1unit(y)."""
    # This is identical to form1_graphs_cartesian_plane_example, but assumes figure size reflects scale
    # Visual representation depends on actual output size (e.g., on screen or printed paper)
    # Matplotlib uses data units, not cm. We simulate the *appearance* if printed/viewed at a certain DPI.
    # Let's adjust figsize to reflect the different scales visually (make x-range appear wider)
    fig, ax = plt.subplots(figsize=(8, 5)) # Wider figure

    # Set limits and ticks
    x_limit, y_limit = 2.5, 5.5
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit) # Adjusted y-limit to keep origin central
    x_ticks = np.arange(-2, 3)
    y_ticks = np.arange(-5, 6) # Corrected range from description
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Add arrowheads
    ax.plot(x_limit, 0, ">k", markersize=8, clip_on=False)
    ax.plot(0, y_limit, "^k", markersize=8, clip_on=False)

    # Labels
    ax.set_xlabel("x (Scale: 2cm = 1 unit)", loc='right')
    ax.set_ylabel("y (Scale: 1cm = 1 unit)", loc='top', rotation=0, labelpad=-10)
    ax.text(0.1, -0.3, '(0,0)', ha='left', va='top')

    # Line thickness
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Attempt to adjust aspect ratio to reflect scales (might be imperfect)
    # Data aspect ratio: (1 unit y / 1 cm) / (1 unit x / 2 cm) = 2
    ax.set_aspect(2, adjustable='box')

    plt.title("Cartesian Plane with Scales (x: 2cm/unit, y: 1cm/unit)")
    plt.tight_layout()
    save_plot(fig, "form2_graphs_cartesian_plane_scales")

def form2_graphs_plot_points_example2():
     """Plots points for y=x-2 example."""
     # This is identical to form1_graphs_plot_points_example2
     form1_graphs_plot_points_example2()
     os.rename(os.path.join(output_dir, "form1_graphs_plot_points_example2.png"),
              os.path.join(output_dir, "form2_graphs_plot_points_example2.png"))


def form2_graphs_line_graph_y_eq_x_m2():
     """Draws graph for y=x-2 example."""
     # This is identical to form1_graphs_line_graph_y_eq_x_m2
     form1_graphs_line_graph_y_eq_x_m2()
     os.rename(os.path.join(output_dir, "form1_graphs_line_graph_y_eq_x_m2.png"),
              os.path.join(output_dir, "form2_graphs_line_graph_y_eq_x_m2.png"))

def form2_graphs_line_graph_y_eq_4_m_x():
     """Draws graph for y=4-x example."""
     # This is identical to form1_graphs_line_graph_y_eq_4_m_x
     form1_graphs_line_graph_y_eq_4_m_x()
     os.rename(os.path.join(output_dir, "form1_graphs_line_graph_y_eq_4_m_x.png"),
              os.path.join(output_dir, "form2_graphs_line_graph_y_eq_4_m_x.png"))

def form2_travel_graphs_example_journey2():
    """Generates distance-time graph for WE1."""
    fig, ax = plt.subplots(figsize=(6, 4))
    time = [0, 2, 3, 5]
    distance = [0, 60, 60, 0]

    ax.plot(time, distance, 'b-o', linewidth=1.5, markersize=5)

    # Labels and Title
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance (km)")
    ax.set_title("Journey Graph (Distance-Time)")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 81, 20))
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 85)

    plt.tight_layout()
    save_plot(fig, "form2_travel_graphs_example_journey2")

def form2_travel_graphs_walk_rest_return():
     """Generates distance-time graph for walk/rest/return."""
     # This is identical to form1_travel_graphs_walk_rest_return
     form1_travel_graphs_walk_rest_return()
     os.rename(os.path.join(output_dir, "form1_travel_graphs_walk_rest_return.png"),
              os.path.join(output_dir, "form2_travel_graphs_walk_rest_return.png"))

def form2_travel_graphs_car_grid():
    """Generates blank grid for car journey problem."""
     # This is identical to form1_travel_graphs_car_grid
    form1_travel_graphs_car_grid()
    os.rename(os.path.join(output_dir, "form1_travel_graphs_car_grid.png"),
              os.path.join(output_dir, "form2_travel_graphs_car_grid.png"))

def form2_variation_direct_variation_graph():
    """Generates graph for direct variation y=kx."""
    # This is identical to form2_variation_direct_variation_graph in the other file
    # Assuming k=1.5 for visual example
    fig, ax = plt.subplots(figsize=(5, 4))
    x_direct = np.linspace(0, 5, 100)
    k_direct = 1.5 # Example slope
    y_direct = k_direct * x_direct
    ax.plot(x_direct, y_direct, label=f'y = kx (k={k_direct})')
    ax.set_title('Direct Variation')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle=':')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, k_direct*5 + 1)
    plt.tight_layout()
    save_plot(fig, "form2_variation_direct_variation_graph")

def form2_geometry_types_of_lines():
     """Generates diagrams for types of lines."""
     # This is identical to form1_geometry_types_of_lines
     form1_geometry_types_of_lines()
     os.rename(os.path.join(output_dir, "form1_geometry_types_of_lines.png"),
              os.path.join(output_dir, "form2_geometry_types_of_lines.png"))

def form2_geometry_parallel_lines_transversal():
    """Generates parallel lines cut by a transversal with numbered angles."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Parallel lines
    ax.plot([-1, 4], [2, 2], 'k-', label='L1') # Top line
    ax.plot([-1, 4], [0, 0], 'k-', label='L2') # Bottom line
    # Parallel marks
    ax.plot([1.4, 1.6], [2.1, 1.9], 'k-')
    ax.plot([1.6, 1.8], [1.9, 2.1], 'k-')
    ax.plot([1.4, 1.6], [0.1, -0.1], 'k-')
    ax.plot([1.6, 1.8], [-0.1, 0.1], 'k-')


    # Transversal line
    x_trans = [-0.5, 3.5]
    y_trans = [-0.8, 2.8] # Calculated for a slope
    ax.plot(x_trans, y_trans, 'k-', label='T (Transversal)')

    # Intersection points (approximate for drawing)
    int1_x, int1_y = 1.5, 2 # Intersection with L1
    int2_x, int2_y = 0.5, 0 # Intersection with L2

    # Angle numbering (approximate positions)
    offset = 0.2
    axs = 11 # Angle label size
    ax.text(int1_x - offset, int1_y + offset, '1', ha='right', va='bottom', fontsize=axs)
    ax.text(int1_x + offset, int1_y + offset, '2', ha='left', va='bottom', fontsize=axs)
    ax.text(int1_x - offset, int1_y - offset, '3', ha='right', va='top', fontsize=axs)
    ax.text(int1_x + offset, int1_y - offset, '4', ha='left', va='top', fontsize=axs)

    ax.text(int2_x - offset, int2_y + offset, '5', ha='right', va='bottom', fontsize=axs)
    ax.text(int2_x + offset, int2_y + offset, '6', ha='left', va='bottom', fontsize=axs)
    ax.text(int2_x - offset, int2_y - offset, '7', ha='right', va='top', fontsize=axs)
    ax.text(int2_x + offset, int2_y - offset, '8', ha='left', va='top', fontsize=axs)

    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-1, 3.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Parallel Lines and Transversal Angles")
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_plot(fig, "form2_geometry_parallel_lines_transversal")

def form2_geometry_parallel_lines_corr_angle():
    """Diagram for corresponding angles example."""
    fig, ax = plt.subplots(figsize=(5, 4))
    # Parallel lines
    ax.plot([0, 5], [2, 2], 'k-') # AB
    ax.plot([0, 5], [0, 0], 'k-') # CD
    ax.text(-0.1, 2, 'A', ha='right', va='center')
    ax.text(5.1, 2, 'B', ha='left', va='center')
    ax.text(-0.1, 0, 'C', ha='right', va='center')
    ax.text(5.1, 0, 'D', ha='left', va='center')
    # Parallel marks
    ax.plot([2.4, 2.6], [2.1, 1.9], 'k-')
    ax.plot([2.6, 2.8], [1.9, 2.1], 'k-')
    ax.plot([2.4, 2.6], [0.1, -0.1], 'k-')
    ax.plot([2.6, 2.8], [-0.1, 0.1], 'k-')
    # Transversal
    ax.plot([1, 4], [-1, 3], 'k-') # EF
    ax.text(0.8, -1.2, 'E', ha='center', va='top')
    ax.text(4.2, 3.2, 'F', ha='center', va='bottom')
    # Intersection points G and H (calculated)
    G = (2.5, 2)
    H = (1.75, 0)
    ax.text(G[0]-0.1, G[1]+0.1, 'G', ha='right', va='bottom')
    ax.text(H[0]-0.1, H[1]-0.1, 'H', ha='right', va='top')
    # Angles
    ax.text(G[0]-0.3, G[1]+0.2, '60°', ha='right', va='bottom', color='blue') # EGA
    ax.text(H[0]-0.3, H[1]+0.2, 'x', ha='right', va='bottom', color='red') # GHC

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')
    ax.set_title("Corresponding Angles")
    plt.tight_layout()
    save_plot(fig, "form2_geometry_parallel_lines_corr_angle")

def form2_geometry_parallel_lines_alt_int_angle():
    """Diagram for alternate interior angles example."""
    fig, ax = plt.subplots(figsize=(5, 4))
    # Parallel lines
    ax.plot([0, 5], [2, 2], 'k-') # AB
    ax.plot([0, 5], [0, 0], 'k-') # CD
    ax.text(-0.1, 2, 'A', ha='right', va='center')
    ax.text(5.1, 2, 'B', ha='left', va='center')
    ax.text(-0.1, 0, 'C', ha='right', va='center')
    ax.text(5.1, 0, 'D', ha='left', va='center')
    # Parallel marks
    ax.plot([2.4, 2.6], [2.1, 1.9], 'k-'); ax.plot([2.6, 2.8], [1.9, 2.1], 'k-')
    ax.plot([2.4, 2.6], [0.1, -0.1], 'k-'); ax.plot([2.6, 2.8], [-0.1, 0.1], 'k-')
    # Transversal
    ax.plot([1, 4], [-1, 3], 'k-') # EF
    ax.text(0.8, -1.2, 'E', ha='center', va='top')
    ax.text(4.2, 3.2, 'F', ha='center', va='bottom')
    # Intersection points G and H
    G = (2.5, 2)
    H = (1.75, 0)
    ax.text(G[0]-0.1, G[1]+0.1, 'G', ha='right', va='bottom')
    ax.text(H[0]-0.1, H[1]-0.1, 'H', ha='right', va='top')
    # Angles
    ax.text(G[0]-0.2, G[1]-0.3, '110°', ha='right', va='top', color='blue') # AGH
    ax.text(H[0]+0.2, H[1]+0.2, 'y', ha='left', va='bottom', color='red') # GHD

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')
    ax.set_title("Alternate Interior Angles")
    plt.tight_layout()
    save_plot(fig, "form2_geometry_parallel_lines_alt_int_angle")

def form2_geometry_parallel_lines_co_int_angle():
    """Diagram for co-interior angles example."""
    fig, ax = plt.subplots(figsize=(5, 4))
    # Parallel lines
    ax.plot([0, 5], [2, 2], 'k-') # AB
    ax.plot([0, 5], [0, 0], 'k-') # CD
    ax.text(-0.1, 2, 'A', ha='right', va='center')
    ax.text(5.1, 2, 'B', ha='left', va='center')
    ax.text(-0.1, 0, 'C', ha='right', va='center')
    ax.text(5.1, 0, 'D', ha='left', va='center')
    # Parallel marks
    ax.plot([2.4, 2.6], [2.1, 1.9], 'k-'); ax.plot([2.6, 2.8], [1.9, 2.1], 'k-')
    ax.plot([2.4, 2.6], [0.1, -0.1], 'k-'); ax.plot([2.6, 2.8], [-0.1, 0.1], 'k-')
    # Transversal
    ax.plot([1, 4], [-1, 3], 'k-') # EF
    ax.text(0.8, -1.2, 'E', ha='center', va='top')
    ax.text(4.2, 3.2, 'F', ha='center', va='bottom')
    # Intersection points G and H
    G = (2.5, 2)
    H = (1.75, 0)
    ax.text(G[0]-0.1, G[1]+0.1, 'G', ha='right', va='bottom')
    ax.text(H[0]-0.1, H[1]-0.1, 'H', ha='right', va='top')
    # Angles
    ax.text(G[0]+0.2, G[1]-0.3, '75°', ha='left', va='top', color='blue') # BGH
    ax.text(H[0]-0.2, H[1]+0.2, 'z', ha='right', va='bottom', color='red') # GHC

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')
    ax.set_title("Co-interior Angles")
    plt.tight_layout()
    save_plot(fig, "form2_geometry_parallel_lines_co_int_angle")

def form2_geometry_parallel_lines_algebra_example():
    """Diagram for alternate interior angles algebra problem."""
    fig, ax = plt.subplots(figsize=(5, 4))
     # Parallel lines
    ax.plot([0, 5], [2, 2], 'k-')
    ax.plot([0, 5], [0, 0], 'k-')
     # Parallel marks
    ax.plot([2.4, 2.6], [2.1, 1.9], 'k-'); ax.plot([2.6, 2.8], [1.9, 2.1], 'k-')
    ax.plot([2.4, 2.6], [0.1, -0.1], 'k-'); ax.plot([2.6, 2.8], [-0.1, 0.1], 'k-')
    # Transversal
    ax.plot([1, 4], [-1, 3], 'k-')
    # Intersection points G and H (approx)
    G = (2.5, 2)
    H = (1.75, 0)
    # Angles
    ax.text(G[0]+0.3, G[1]-0.3, '(3x - 20)°', ha='left', va='top', color='blue') # Right interior top
    ax.text(H[0]-0.3, H[1]+0.3, '(2x + 10)°', ha='right', va='bottom', color='red') # Left interior bottom

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')
    ax.set_title("Alternate Interior Angles (Algebra)")
    plt.tight_layout()
    save_plot(fig, "form2_geometry_parallel_lines_algebra_example")

def form2_geometry_parallel_lines_mod_question():
    """Diagram for moderate question (parallel vertical lines)."""
    fig, ax = plt.subplots(figsize=(4, 5))
    # Parallel lines
    ax.plot([0, 0], [-1, 4], 'k-', label='m') # Left line
    ax.plot([2, 2], [-1, 4], 'k-', label='n') # Right line
    # Parallel marks (vertical style)
    ax.plot([-0.1, 0.1], [1.4, 1.6], 'k-'); ax.plot([-0.1, 0.1], [1.6, 1.8], 'k-')
    ax.plot([1.9, 2.1], [1.4, 1.6], 'k-'); ax.plot([1.9, 2.1], [1.6, 1.8], 'k-')

    # Transversal line
    ax.plot([-1, 3], [1, 3], 'k-') # Slanted up left-to-right

    # Intersection points (calculated)
    int_m_y = 1.5 # y = 0.5x + 1.5 => y=1.5 at x=0
    int_n_y = 2.5 # y = 0.5x + 1.5 => y=2.5 at x=2
    int_m = (0, 1.5)
    int_n = (2, 2.5)

    # Angle labels (approximate positions)
    offset=0.2
    axs=11
    ax.text(int_m[0] + offset, int_m[1] + offset, '70°', ha='left', va='bottom', fontsize=axs, color='blue') # Top-right at m
    ax.text(int_n[0] - offset, int_n[1] + offset, 'a', ha='right', va='bottom', fontsize=axs, color='red') # Top-left at n
    ax.text(int_n[0] - offset, int_n[1] - offset, 'b', ha='right', va='top', fontsize=axs, color='red') # Bottom-left at n
    ax.text(int_n[0] + offset, int_n[1] - offset, 'c', ha='left', va='top', fontsize=axs, color='red') # Bottom-right at n

    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-1.5, 4.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Find angles a, b, c")
    ax.legend(loc='lower right')
    plt.tight_layout()
    save_plot(fig, "form2_geometry_parallel_lines_mod_question")

def form2_geometry_parallel_lines_chall_question():
    """Diagram for challenging question AB || DE."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Points
    A = (0, 1)
    B = (3, 1)
    C = (4, 0) # Example position
    D = (3, -1)
    E = (0, -1)

    # Lines AB and DE
    ax.plot([A[0]-1, B[0]+1], [A[1], B[1]], 'k-', label='Line AB')
    ax.plot([E[0]-1, D[0]+1], [E[1], D[1]], 'k-', label='Line DE')
    # Parallel marks
    ax.plot([1.4, 1.6], [1.1, 0.9], 'k-'); ax.plot([1.6, 1.8], [0.9, 1.1], 'k-')
    ax.plot([1.4, 1.6], [-0.9, -1.1], 'k-'); ax.plot([1.6, 1.8], [-1.1, -0.9], 'k-')

    # Lines BC and DC
    ax.plot([B[0], C[0]], [B[1], C[1]], 'k-')
    ax.plot([D[0], C[0]], [D[1], C[1]], 'k-')

    # Labels
    ax.text(A[0], A[1]+0.1, 'A', ha='center', va='bottom')
    ax.text(B[0], B[1]+0.1, 'B', ha='center', va='bottom')
    ax.text(C[0]+0.1, C[1], 'C', ha='left', va='center')
    ax.text(D[0], D[1]-0.1, 'D', ha='center', va='top')
    ax.text(E[0], E[1]-0.1, 'E', ha='center', va='top')

    # Angles
    # Angle ABC requires extending AB leftwards. Line from (-1, 1) to B(3, 1). Angle with BC.
    # Let's mark the given supplementary angle. Extend AB left to F(-1,1). Angle FBC = 130.
    ax.text(B[0]-0.5, B[1]-0.2, '130°', ha='right', va='top', color='blue')
    # Angle CDE requires extending DE rightwards. Line from D(3,-1) to G(5,-1). Angle CDG = 140.
    ax.text(D[0]+0.5, D[1]+0.2, '140°', ha='left', va='bottom', color='blue')
    # Angle BCD
    ax.text(C[0]-0.2, C[1]+0.1, 'x', ha='right', va='bottom', color='red')

    ax.set_xlim(-1.5, 5.5)
    ax.set_ylim(-2, 2)
    ax.axis('off')
    ax.set_title("Find angle x (AB || DE)")

    plt.tight_layout()
    save_plot(fig, "form2_geometry_parallel_lines_chall_question")

def form2_geometry_compass_rose():
    """Generates a compass rose."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    ax.set_aspect('equal')

    # Outer circle (optional)
    # ax.add_patch(Circle((0,0), 1.1, fill=False, edgecolor='grey', linestyle='--'))

    # Main directions
    ax.plot([0, 0], [0, 1], 'k-', linewidth=2) # N
    ax.plot([0, 0], [0, -1], 'k-', linewidth=1.5) # S
    ax.plot([0, 1], [0, 0], 'k-', linewidth=1.5) # E
    ax.plot([0, -1], [0, 0], 'k-', linewidth=1.5) # W

    # Intercardinal directions (thinner lines)
    diag = 1 / np.sqrt(2) # length for 45 deg
    ax.plot([0, diag], [0, diag], 'k-', linewidth=1) # NE
    ax.plot([0, diag], [0, -diag], 'k-', linewidth=1) # SE
    ax.plot([0, -diag], [0, -diag], 'k-', linewidth=1) # SW
    ax.plot([0, -diag], [0, diag], 'k-', linewidth=1) # NW

    # Arrowheads (optional, can make cluttered)
    ax.plot(0, 1, "^k", markersize=10)

    # Labels
    offset = 1.1
    ax.text(0, offset, 'N', ha='center', va='bottom', fontsize=14, weight='bold')
    ax.text(0, -offset, 'S', ha='center', va='top', fontsize=12)
    ax.text(offset, 0, 'E', ha='left', va='center', fontsize=12)
    ax.text(-offset, 0, 'W', ha='right', va='center', fontsize=12)

    ioffset = 0.9
    ax.text(ioffset*diag, ioffset*diag, 'NE', ha='left', va='bottom', fontsize=10)
    ax.text(ioffset*diag, -ioffset*diag, 'SE', ha='left', va='top', fontsize=10)
    ax.text(-ioffset*diag, -ioffset*diag, 'SW', ha='right', va='top', fontsize=10)
    ax.text(-ioffset*diag, ioffset*diag, 'NW', ha='right', va='bottom', fontsize=10)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    ax.set_title("Compass Points")
    save_plot(fig, "form2_geometry_compass_rose")

def form2_geometry_compass_bearing_examples():
    """Generates examples of compass bearings."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    fig.patch.set_facecolor('white')

    def plot_bearing(ax, angle_deg, from_dir, towards_dir, label):
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        ax.text(0, 1.1, 'N', ha='center', va='bottom')
        ax.text(0, -1.1, 'S', ha='center', va='top')
        ax.text(1.1, 0, 'E', ha='left', va='center')
        ax.text(-1.1, 0, 'W', ha='right', va='center')

        if from_dir == 'N':
            start_angle = 90 # Angle from positive x-axis
            if towards_dir == 'E':
                angle_rad = np.deg2rad(start_angle - angle_deg)
                theta1 = 90 - angle_deg
                theta2 = 90
            else: # W
                angle_rad = np.deg2rad(start_angle + angle_deg)
                theta1 = 90
                theta2 = 90 + angle_deg
        else: # S
            start_angle = -90
            if towards_dir == 'E':
                 angle_rad = np.deg2rad(start_angle + angle_deg)
                 theta1 = -90 + angle_deg
                 theta2 = -90 # Incorrect range for Arc
                 theta1 = 270
                 theta2 = 270+angle_deg

            else: # W
                 angle_rad = np.deg2rad(start_angle - angle_deg)
                 theta1 = 270-angle_deg
                 theta2 = 270

        end_x = np.cos(angle_rad)
        end_y = np.sin(angle_rad)
        ax.plot([0, end_x], [0, end_y], 'r-', linewidth=1.5)
        ax.plot(end_x, end_y, 'ro')

        # Arc for angle
        arc_radius = 0.4
        if from_dir == 'N' and towards_dir == 'E':
             arc = Arc((0,0), 2*arc_radius, 2*arc_radius, angle=0, theta1=90-angle_deg, theta2=90, color='red')
        elif from_dir == 'N' and towards_dir == 'W':
             arc = Arc((0,0), 2*arc_radius, 2*arc_radius, angle=0, theta1=90, theta2=90+angle_deg, color='red')
        elif from_dir == 'S' and towards_dir == 'E':
             arc = Arc((0,0), 2*arc_radius, 2*arc_radius, angle=0, theta1=270, theta2=270+angle_deg, color='red')
        elif from_dir == 'S' and towards_dir == 'W':
             arc = Arc((0,0), 2*arc_radius, 2*arc_radius, angle=0, theta1=270-angle_deg, theta2=270, color='red')
        ax.add_patch(arc)

        # Angle label - position needs adjustment based on quadrant
        angle_label_rad = np.deg2rad(np.mean([theta1, theta2]) if from_dir=='N' else (270 + (angle_deg/2 if towards_dir=='E' else -angle_deg/2))) # Approximate midpoint
        ax.text(arc_radius*1.2 * np.cos(angle_label_rad), arc_radius*1.2 * np.sin(angle_label_rad), f'{angle_deg}°', color='red', ha='center', va='center', fontsize=9)


        ax.text(end_x * 1.1, end_y * 1.1, label, color='r', ha='center', va='center')

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(label)

    plot_bearing(axs[0], 30, 'N', 'E', 'N 30° E')
    plot_bearing(axs[1], 45, 'S', 'W', 'S 45° W')
    plot_bearing(axs[2], 70, 'N', 'W', 'N 70° W')

    plt.tight_layout()
    save_plot(fig, "form2_geometry_compass_bearing_examples")

def form2_geometry_three_figure_bearing_examples():
    """Generates examples of three-figure bearings."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    fig.patch.set_facecolor('white')

    def plot_3fig_bearing(ax, angle_deg, label):
        ax.plot([0, 0], [0, 1], 'k-', linewidth=1.5) # North line
        ax.plot(0, 1, "^k", markersize=8)
        ax.text(0, 1.1, 'N', ha='center', va='bottom')

        # Direction line
        # Angle for trig functions is measured anti-clockwise from positive x-axis
        # Angle for plotting is measured clockwise from positive y-axis (North)
        plot_angle_rad = np.deg2rad(90 - angle_deg)
        end_x = np.cos(plot_angle_rad)
        end_y = np.sin(plot_angle_rad)
        ax.plot([0, end_x], [0, end_y], 'r-', linewidth=1.5)
        ax.plot(end_x, end_y, 'ro')

        # Arc
        arc_radius = 0.6
        arc = Arc((0,0), 2*arc_radius, 2*arc_radius, angle=90, theta1=-angle_deg, theta2=0, color='red', linewidth=1)
        ax.add_patch(arc)

        # Angle Label
        label_angle_rad = np.deg2rad(90 - angle_deg/2) # Midpoint angle from x-axis
        ax.text(arc_radius * 0.7 * np.cos(label_angle_rad),
                arc_radius * 0.7 * np.sin(label_angle_rad),
                f'{angle_deg:03d}°', color='red', rotation = -angle_deg/2 + (180 if angle_deg>180 else 0) , ha='center', va='center', fontsize=9) # Simple rotation

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1.3)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(f'Bearing {angle_deg:03d}°')

    plot_3fig_bearing(axs[0], 60, 'Bearing 060°')
    plot_3fig_bearing(axs[1], 135, 'Bearing 135°')
    plot_3fig_bearing(axs[2], 310, 'Bearing 310°')

    plt.tight_layout()
    save_plot(fig, "form2_geometry_three_figure_bearing_examples")

def form2_geometry_bearing_back_bearing_example():
    """Diagram illustrating bearing and back bearing."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    A = (1, 1)
    B = (3, 4) # Example points

    # North lines
    ax.plot([A[0], A[0]], [A[1], A[1]+2], 'k--', label='North at A')
    ax.plot(A[0], A[1]+2, '^k')
    ax.text(A[0], A[1]+2.1, 'N_A', ha='center', va='bottom')

    ax.plot([B[0], B[0]], [B[1], B[1]+2], 'k--', label='North at B')
    ax.plot(B[0], B[1]+2, '^k')
    ax.text(B[0], B[1]+2.1, 'N_B', ha='center', va='bottom')

    # Line segment AB
    ax.plot([A[0], B[0]], [A[1], B[1]], 'r-', marker='o')
    ax.text(A[0]-0.1, A[1]-0.1, 'A', ha='right', va='top')
    ax.text(B[0]+0.1, B[1]+0.1, 'B', ha='left', va='bottom')

    # Calculate angles (approximate from coords)
    vec_ab = np.array(B) - np.array(A)
    # Angle from positive y-axis (North) clockwise
    angle_ab_rad = np.arctan2(vec_ab[0], vec_ab[1]) # atan2(x,y) gives angle from +y axis
    angle_ab_deg = np.rad2deg(angle_ab_rad)
    if angle_ab_deg < 0: angle_ab_deg += 360 # Ensure positive angle
    bearing_ab = angle_ab_deg # Bearing of B from A

    vec_ba = -vec_ab
    angle_ba_rad = np.arctan2(vec_ba[0], vec_ba[1])
    angle_ba_deg = np.rad2deg(angle_ba_rad)
    if angle_ba_deg < 0: angle_ba_deg += 360
    bearing_ba = angle_ba_deg # Bearing of A from B (back bearing)

    # Display angles using Arc
    arc_a = Arc(A, 1.0, 1.0, angle=90, theta1=-bearing_ab, theta2=0, color='red')
    ax.add_patch(arc_a)
    ax.text(A[0]+0.6*np.sin(np.deg2rad(bearing_ab/2)),
            A[1]+0.6*np.cos(np.deg2rad(bearing_ab/2)),
            f'θ = {bearing_ab:.0f}°', color='red', fontsize=10)

    arc_b = Arc(B, 1.0, 1.0, angle=90, theta1=-bearing_ba, theta2=0, color='blue')
    ax.add_patch(arc_b)
    ax.text(B[0]+0.7*np.sin(np.deg2rad(bearing_ba/2)),
            B[1]+0.7*np.cos(np.deg2rad(bearing_ba/2)),
             f'{bearing_ba:.0f}°\n= θ+180°' if bearing_ab < 180 else f'{bearing_ba:.0f}°\n= θ-180°',
             color='blue', fontsize=10)


    ax.set_xlim(0, 5)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Bearing (θ) and Back Bearing")

    plt.tight_layout()
    save_plot(fig, "form2_geometry_bearing_back_bearing_example")

def form2_geometry_bearing_angle_between_points():
    """Diagram for finding angle BAC from bearings."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    A = (0, 0)

    # North line at A
    ax.plot([A[0], A[0]], [A[1], A[1]+1.5], 'k--')
    ax.plot(A[0], A[1]+1.5, '^k')
    ax.text(A[0], A[1]+1.6, 'N', ha='center', va='bottom')

    # Bearings
    bearing_b = 120
    bearing_c = 210
    angle_bac = bearing_c - bearing_b # 90

    # Lines AB and AC (length arbitrary)
    len_ab = 1.2
    len_ac = 1.0
    angle_b_rad = np.deg2rad(90 - bearing_b)
    angle_c_rad = np.deg2rad(90 - bearing_c)
    B = (A[0] + len_ab * np.cos(angle_b_rad), A[1] + len_ab * np.sin(angle_b_rad))
    C = (A[0] + len_ac * np.cos(angle_c_rad), A[1] + len_ac * np.sin(angle_c_rad))

    ax.plot([A[0], B[0]], [A[1], B[1]], 'r-', marker='o')
    ax.plot([A[0], C[0]], [A[1], C[1]], 'b-', marker='o')

    # Labels
    ax.text(A[0]-0.1, A[1]-0.1, 'A', ha='right', va='top')
    ax.text(B[0]+0.1, B[1], 'B', ha='left', va='center')
    ax.text(C[0], C[1]-0.1, 'C', ha='center', va='top')

    # Angle arcs for bearings
    arc_b = Arc(A, 1.0, 1.0, angle=90, theta1=-bearing_b, theta2=0, color='red', linestyle=':')
    ax.add_patch(arc_b)
    ax.text(0.6*np.cos(np.deg2rad(90-bearing_b*0.6)), 0.6*np.sin(np.deg2rad(90-bearing_b*0.6)), f'{bearing_b}°', color='red', fontsize=9)

    arc_c = Arc(A, 0.8, 0.8, angle=90, theta1=-bearing_c, theta2=0, color='blue', linestyle=':')
    ax.add_patch(arc_c)
    ax.text(0.5*np.cos(np.deg2rad(90-bearing_c*0.7)), 0.5*np.sin(np.deg2rad(90-bearing_c*0.7)), f'{bearing_c}°', color='blue', fontsize=9)

    # Angle BAC
    arc_bac = Arc(A, 0.4, 0.4, angle=0, theta1=np.rad2deg(angle_c_rad), theta2=np.rad2deg(angle_b_rad), color='black')
    ax.add_patch(arc_bac)
    ax.text(0.3*np.cos(np.deg2rad(90-(bearing_b+angle_bac/2))), 0.3*np.sin(np.deg2rad(90-(bearing_b+angle_bac/2))), f'{angle_bac}°', color='black', fontsize=9)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.7)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Angle Between Bearings")

    plt.tight_layout()
    save_plot(fig, "form2_geometry_bearing_angle_between_points")

def form2_geometry_bearing_compass_s40e():
    """Diagram showing S40E compass bearing."""
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    Q = (0, 0)

    # Axes
    ax.plot([-1, 1], [0, 0], 'grey', linestyle='--') # E-W
    ax.plot([0, 0], [-1, 1], 'grey', linestyle='--') # N-S
    ax.text(0, 1.1, 'N', ha='center', va='bottom')
    ax.text(0, -1.1, 'S', ha='center', va='top')
    ax.text(1.1, 0, 'E', ha='left', va='center')
    ax.text(-1.1, 0, 'W', ha='right', va='center')

    # Bearing Line
    angle_deg = 40
    plot_angle_rad = np.deg2rad(270 + angle_deg) # Angle from +x axis ACW
    end_x = np.cos(plot_angle_rad)
    end_y = np.sin(plot_angle_rad)
    ax.plot([Q[0], end_x], [Q[1], end_y], 'r-', linewidth=1.5)
    ax.plot(end_x, end_y, 'ro')
    ax.text(end_x*1.1, end_y*1.1, 'P', color='r', ha='center', va='center')

    # Angle arc from South
    arc = Arc(Q, 0.6, 0.6, angle=0, theta1=270, theta2=270+angle_deg, color='red')
    ax.add_patch(arc)
    label_angle_rad = np.deg2rad(270 + angle_deg/2)
    ax.text(0.4*np.cos(label_angle_rad), 0.4*np.sin(label_angle_rad), f'{angle_deg}°', color='red', fontsize=9)

    ax.text(0.7*end_x, 0.7*end_y + 0.1, 'S 40° E', color='r', ha='center', va='bottom', fontsize=10)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Locating P on bearing S 40° E from Q")

    plt.tight_layout()
    save_plot(fig, "form2_geometry_bearing_compass_s40e")

def form2_geometry_triangle_types():
    """Generates diagrams of triangle types."""
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs_flat = axs.flatten()
    fig.patch.set_facecolor('white')

    # Equilateral
    ax = axs_flat[0]
    s = 1
    h = s * np.sqrt(3)/2
    verts = [(0, 0), (s, 0), (s/2, h)]
    ax.add_patch(Polygon(verts, fill=False, edgecolor='k', lw=1.5))
    # Side marks
    ax.plot([s*0.45, s*0.55], [-0.02, -0.02], 'k-', lw=1) # Base
    ax.plot([s*0.2, s*0.23], [h*0.45, h*0.55], 'k-', lw=1, transform=ax.transData + mtransforms.Affine2D().rotate_deg_around(s*0.25, h*0.5, -60)) # Left
    ax.plot([s*0.77, s*0.8], [h*0.55, h*0.45], 'k-', lw=1, transform=ax.transData + mtransforms.Affine2D().rotate_deg_around(s*0.75, h*0.5, 60)) # Right
    # Angle marks
    ax.add_patch(Arc(verts[0], 0.2, 0.2, angle=0, theta1=0, theta2=60))
    ax.add_patch(Arc(verts[1], 0.2, 0.2, angle=0, theta1=120, theta2=180))
    ax.add_patch(Arc(verts[2], 0.2, 0.2, angle=0, theta1=240, theta2=300))
    ax.text(s/2, h+0.05, 'All 60°', ha='center', va='bottom', fontsize=9)
    ax.set_title("Equilateral")
    ax.axis('off'); ax.set_aspect('equal')

    # Isosceles
    ax = axs_flat[1]
    base = 1.2
    side = 1
    h = np.sqrt(side**2 - (base/2)**2)
    verts = [(0, 0), (base, 0), (base/2, h)]
    ax.add_patch(Polygon(verts, fill=False, edgecolor='k', lw=1.5))
    # Side marks
    ax.plot([base*0.2, base*0.23], [h*0.45, h*0.55], 'k-', lw=1, transform=ax.transData + mtransforms.Affine2D().rotate_deg_around(base*0.25, h*0.5, -70)) # Left approx
    ax.plot([base*0.77, base*0.8], [h*0.55, h*0.45], 'k-', lw=1, transform=ax.transData + mtransforms.Affine2D().rotate_deg_around(base*0.75, h*0.5, 70)) # Right approx
    # Angle marks
    angle_base = np.rad2deg(np.arctan(h/(base/2)))
    ax.add_patch(Arc(verts[0], 0.2, 0.2, angle=0, theta1=0, theta2=angle_base))
    ax.add_patch(Arc(verts[1], 0.2, 0.2, angle=0, theta1=180-angle_base, theta2=180))
    ax.set_title("Isosceles")
    ax.axis('off'); ax.set_aspect('equal')

     # Scalene
    ax = axs_flat[2]
    verts = [(0, 0), (1.3, 0.1), (0.4, 0.8)]
    ax.add_patch(Polygon(verts, fill=False, edgecolor='k', lw=1.5))
    ax.set_title("Scalene")
    ax.axis('off'); ax.set_aspect('equal')

    # Right-angled
    ax = axs_flat[3]
    verts = [(0, 0), (1.2, 0), (0, 0.9)]
    ax.add_patch(Polygon(verts, fill=False, edgecolor='k', lw=1.5))
    # Right angle mark
    ax.plot([0.05, 0.05], [0, 0.05], 'k-'); ax.plot([0, 0.05], [0.05, 0.05], 'k-')
    ax.set_title("Right-angled")
    ax.axis('off'); ax.set_aspect('equal')


    plt.tight_layout()
    save_plot(fig, "form2_geometry_triangle_types")

def form2_geometry_quadrilateral_types():
    """Generates diagrams of common quadrilaterals."""
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    axs_flat = axs.flatten()
    fig.patch.set_facecolor('white')
    s=1 # Base size

    # Square
    ax = axs_flat[0]
    ax.add_patch(Rectangle((0.1, 0.1), s, s, fill=False, edgecolor='k', lw=1.5))
    ax.text(0.1+s/2, 0.05, '=', ha='center', va='top'); ax.text(0.05, 0.1+s/2, '=', ha='right', va='center') # Side marks (simple)
    ax.plot([0.1, 0.15],[0.1,0.15],'k'); ax.plot([0.15, 0.15],[0.1,0.15],'k') # Right angle
    ax.set_title("Square")
    ax.axis('off'); ax.set_aspect('equal')

    # Rectangle
    ax = axs_flat[1]
    l, w = 1.5, 0.8
    ax.add_patch(Rectangle((0.1, 0.1), l, w, fill=False, edgecolor='k', lw=1.5))
    ax.text(0.1+l/2, 0.05, '=', ha='center', va='top'); ax.text(0.1+l/2, 0.1+w+0.05, '=', ha='center', va='bottom')
    ax.text(0.05, 0.1+w/2, '||', ha='right', va='center'); ax.text(0.1+l+0.05, 0.1+w/2, '||', ha='left', va='center')
    ax.plot([0.1, 0.15],[0.1,0.15],'k'); ax.plot([0.15, 0.15],[0.1,0.15],'k') # Right angle
    ax.set_title("Rectangle")
    ax.axis('off'); ax.set_aspect('equal')

    # Parallelogram
    ax = axs_flat[2]
    shear_factor = 0.3
    verts = [(0.1, 0.1), (0.1+s, 0.1), (0.1+s+shear_factor, 0.1+s), (0.1+shear_factor, 0.1+s)]
    ax.add_patch(Polygon(verts, fill=False, edgecolor='k', lw=1.5))
    ax.text(0.1+s/2, 0.05, '=', ha='center', va='top')
    ax.text(0.1+s/2+shear_factor, 0.1+s+0.05, '=', ha='center', va='bottom')
    ax.text(0.1+shear_factor/2 - 0.05, 0.1+s/2, '||', ha='right', va='center')
    ax.text(0.1+s+shear_factor/2+0.05, 0.1+s/2, '||', ha='left', va='center')
    ax.text(0.1+s*0.4, 0.1+0.05, '>', ha='center', va='top', fontsize=10) # Parallel marks
    ax.text(0.1+s*0.4+shear_factor, 0.1+s-0.05, '>', ha='center', va='bottom', fontsize=10)
    ax.set_title("Parallelogram")
    ax.axis('off'); ax.set_aspect('equal')

    # Rhombus
    ax = axs_flat[3]
    d1, d2 = 1.5, 1.0
    verts = [(0, d2/2), (d1/2, 0), (0, -d2/2), (-d1/2, 0)]
    ax.add_patch(Polygon(verts, fill=False, edgecolor='k', lw=1.5))
    # Diagonal marks for perp bisect
    ax.plot([0,0],[-d2/2, d2/2],'k:')
    ax.plot([-d1/2, d1/2],[0,0],'k:')
    ax.plot([0.05, 0.05],[0,0.05],'k'); ax.plot([0, 0.05],[0.05,0.05],'k')
    # Side marks
    ax.text(d1*0.2, -d2*0.2, '=', color='k', ha='center', va='center', rotation=35)
    ax.text(-d1*0.2, -d2*0.2, '=', color='k', ha='center', va='center', rotation=-35)
    ax.text(d1*0.2, d2*0.2, '=', color='k', ha='center', va='center', rotation=-35)
    ax.text(-d1*0.2, d2*0.2, '=', color='k', ha='center', va='center', rotation=35)
    ax.set_title("Rhombus")
    ax.axis('off'); ax.set_aspect('equal')

    # Trapezium
    ax = axs_flat[4]
    b1, b2, h = 1.5, 1.0, 0.8
    offset = 0.2
    verts = [(0, 0), (b1, 0), (offset+b2, h), (offset, h)]
    ax.add_patch(Polygon(verts, fill=False, edgecolor='k', lw=1.5))
    ax.text(b1/2, -0.05, '>', ha='center', va='top', fontsize=10) # Parallel marks
    ax.text(offset+b2/2, h+0.05, '>', ha='center', va='bottom', fontsize=10)
    ax.set_title("Trapezium")
    ax.axis('off'); ax.set_aspect('equal')

    # Kite
    ax = axs_flat[5]
    d1, d2_up, d2_down = 1.0, 0.8, 1.2 # Vertical d1, horizontal d2 split
    verts = [(0, d2_up), (d1/2, 0), (0, -d2_down), (-d1/2, 0)]
    ax.add_patch(Polygon(verts, fill=False, edgecolor='k', lw=1.5))
    # Diagonal marks
    ax.plot([0,0],[-d2_down, d2_up],'k:')
    ax.plot([-d1/2, d1/2],[0,0],'k:')
    ax.plot([0.05, 0.05],[0,0.05],'k'); ax.plot([0, 0.05],[0.05,0.05],'k')
    # Side marks
    ax.text(d1*0.2, d2_up*0.3, '=', color='k'); ax.text(-d1*0.2, d2_up*0.3, '=', color='k')
    ax.text(d1*0.2, -d2_down*0.3, '||', color='k'); ax.text(-d1*0.2, -d2_down*0.3, '||', color='k')
    ax.set_title("Kite")
    ax.axis('off'); ax.set_aspect('equal')

    plt.tight_layout(pad=0.5)
    save_plot(fig, "form2_geometry_quadrilateral_types")


# --- Form 3 ---

def form3_sets_venn_three_sets_regions():
    """Generates a 3-set Venn diagram structure with numbered regions."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor('white')

    # Universal set rectangle
    ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.95, 0.9, 'U', ha='right', va='top', fontsize=14)

    # Circles
    center_a = (0.40, 0.60)
    center_b = (0.60, 0.60)
    center_c = (0.50, 0.35)
    radius = 0.30

    circle_a = Circle(center_a, radius, fill=False, edgecolor='blue', linewidth=1.5)
    circle_b = Circle(center_b, radius, fill=False, edgecolor='red', linewidth=1.5)
    circle_c = Circle(center_c, radius, fill=False, edgecolor='green', linewidth=1.5)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    ax.add_patch(circle_c)

    # Set Labels
    ax.text(center_a[0] - radius*0.8, center_a[1] + radius*0.8, 'A', color='blue', fontsize=14)
    ax.text(center_b[0] + radius*0.8, center_b[1] + radius*0.8, 'B', color='red', fontsize=14)
    ax.text(center_c[0], center_c[1] - radius*1.1, 'C', color='green', fontsize=14)

    # Region Labels (approximate centers) - Using numbers 1-8
    ax.text(0.50, 0.50, '7', ha='center', va='center') # A intersect B intersect C
    ax.text(0.50, 0.65, '4', ha='center', va='center') # A and B only
    ax.text(0.35, 0.45, '5', ha='center', va='center') # A and C only
    ax.text(0.65, 0.45, '6', ha='center', va='center') # B and C only
    ax.text(0.25, 0.65, '1', ha='center', va='center') # A only
    ax.text(0.75, 0.65, '2', ha='center', va='center') # B only
    ax.text(0.50, 0.20, '3', ha='center', va='center') # C only
    ax.text(0.10, 0.10, '8', ha='center', va='center') # Neither

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Venn Diagram (3 Sets with Regions)")
    plt.tight_layout()
    save_plot(fig, "form3_sets_venn_three_sets_regions")

def form3_sets_venn_elements_example():
    """Generates Venn for U={1..10}, A={1,2,3,4,5}, B={2,4,6,8}, C={3,5,7,9}."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor('white')

    # Universal set rectangle
    ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.95, 0.9, 'U={1..10}', ha='right', va='top', fontsize=10)

    # Circles
    center_a = (0.40, 0.60)
    center_b = (0.60, 0.60)
    center_c = (0.50, 0.35)
    radius = 0.30

    circle_a = Circle(center_a, radius, fill=False, edgecolor='blue', linewidth=1.5)
    circle_b = Circle(center_b, radius, fill=False, edgecolor='red', linewidth=1.5)
    circle_c = Circle(center_c, radius, fill=False, edgecolor='green', linewidth=1.5)
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    ax.add_patch(circle_c)

    # Set Labels
    ax.text(center_a[0] - radius*0.8, center_a[1] + radius*0.8, 'A', color='blue', fontsize=14)
    ax.text(center_b[0] + radius*0.8, center_b[1] + radius*0.8, 'B', color='red', fontsize=14)
    ax.text(center_c[0], center_c[1] - radius*1.1, 'C', color='green', fontsize=14)

    # Region Counts / Elements
    ax.text(0.50, 0.50, '', ha='center', va='center') # A intersect B intersect C (empty) -> count 0
    ax.text(0.50, 0.65, '2, 4', ha='center', va='center', fontsize=9) # A and B only (count 2)
    ax.text(0.35, 0.45, '3, 5', ha='center', va='center', fontsize=9) # A and C only (count 2)
    ax.text(0.65, 0.45, '', ha='center', va='center') # B and C only (empty) -> count 0
    ax.text(0.25, 0.65, '1', ha='center', va='center', fontsize=9) # A only (count 1)
    ax.text(0.75, 0.65, '6, 8', ha='center', va='center', fontsize=9) # B only (count 2)
    ax.text(0.50, 0.20, '7, 9', ha='center', va='center', fontsize=9) # C only (count 2)
    ax.text(0.10, 0.10, '10', ha='center', va='center', fontsize=9) # Neither (count 1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Venn Example: A={1-5}, B={Evens}, C={Odds 3-9}")
    plt.tight_layout()
    save_plot(fig, "form3_sets_venn_elements_example")

def form3_sets_venn_p_c_b_problem():
    """Generates Venn diagram for Physics/Chem/Bio problem."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor('white')

    # Universal set rectangle
    ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(0.95, 0.9, 'U=40', ha='right', va='top', fontsize=12)

    # Circles
    center_p = (0.40, 0.60)
    center_c = (0.60, 0.60)
    center_b = (0.50, 0.35)
    radius = 0.30

    circle_p = Circle(center_p, radius, fill=False, edgecolor='blue', linewidth=1.5)
    circle_c = Circle(center_c, radius, fill=False, edgecolor='red', linewidth=1.5)
    circle_b = Circle(center_b, radius, fill=False, edgecolor='green', linewidth=1.5)
    ax.add_patch(circle_p)
    ax.add_patch(circle_c)
    ax.add_patch(circle_b)

    # Set Labels
    ax.text(center_p[0] - radius*0.8, center_p[1] + radius*0.8, 'P(20)', color='blue', fontsize=12)
    ax.text(center_c[0] + radius*0.8, center_c[1] + radius*0.8, 'C(25)', color='red', fontsize=12)
    ax.text(center_b[0], center_b[1] - radius*1.1, 'B(18)', color='green', fontsize=12)

    # Region Counts
    ax.text(0.50, 0.50, '5', ha='center', va='center', fontweight='bold') # P∩C∩B
    ax.text(0.50, 0.65, '5', ha='center', va='center') # P and C only
    ax.text(0.35, 0.45, '4', ha='center', va='center') # P and B only
    ax.text(0.65, 0.45, '7', ha='center', va='center') # C and B only
    ax.text(0.25, 0.65, '6', ha='center', va='center') # P only
    ax.text(0.75, 0.65, '8', ha='center', va='center') # C only
    ax.text(0.50, 0.20, '2', ha='center', va='center') # B only
    ax.text(0.10, 0.10, '3', ha='center', va='center', fontweight='bold') # Neither

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Subject Choices (Physics, Chemistry, Biology)")
    plt.tight_layout()
    save_plot(fig, "form3_sets_venn_p_c_b_problem")

def form3_finance_compound_vs_simple_interest():
    """Generates graph comparing simple and compound interest."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')

    # Parameters
    P = 1000 # Principal
    R = 10   # Rate (%)
    years = np.arange(0, 11, 1) # Time in years

    # Calculations
    simple_amount = P * (1 + (R / 100) * years)
    compound_amount = P * (1 + R / 100)**years

    # Plotting
    ax.plot(years, simple_amount, 'b--', label='Simple Interest (A = P(1+Rt/100))')
    ax.plot(years, compound_amount, 'r-', label='Compound Interest (A = P(1+R/100)^t)')

    # Formatting
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Amount ($)")
    ax.set_title("Simple vs. Compound Interest (P=$1000, R=10%)")
    ax.grid(True, linestyle=':')
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom=P*0.95) # Start y-axis near P
    # Format y-axis as currency
    formatter = mticker.FormatStrFormatter('$%.0f')
    ax.yaxis.set_major_formatter(formatter)


    plt.tight_layout()
    save_plot(fig, "form3_finance_compound_vs_simple_interest")

def form3_mensuration_combined_shape_perimeter_area():
    """Generates diagram of rectangle with semi-circle."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('white')

    rect_w, rect_h = 6, 4
    semi_d = rect_h # Attached to shorter side
    semi_r = semi_d / 2

    # Rectangle
    rect_verts = [(0, 0), (rect_w, 0), (rect_w, rect_h), (0, rect_h)]
    ax.add_patch(Polygon(rect_verts, fill=False, edgecolor='k', lw=1.5))

    # Semi-circle attached to right side (x=rect_w)
    center_semi = (rect_w, rect_h / 2)
    semi_circle = Arc(center_semi, semi_d, semi_d, angle=0, theta1=-90, theta2=90, color='k', lw=1.5)
    ax.add_patch(semi_circle)

    # Dimensions
    ax.text(rect_w/2, -0.2, f'{rect_w} cm', ha='center', va='top')
    ax.text(-0.2, rect_h/2, f'{rect_h} cm', ha='right', va='center')

    # Dashed lines for calculation hint
    ax.plot([rect_w, rect_w], [0, rect_h], 'k:') # Join line
    ax.text(rect_w/2, rect_h/2, 'Area (Rect)', ha='center', va='center', fontsize=8, color='grey')
    ax.text(rect_w + semi_r*0.7, rect_h/2, 'Area\n(Semi-C)', ha='center', va='center', fontsize=8, color='grey')
    ax.text(rect_w/2, rect_h+0.2, 'Perimeter = Sum of outer boundary', ha='center', va='bottom', fontsize=8, color='blue')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, rect_w + semi_r + 0.5)
    ax.set_ylim(-0.5, rect_h + 0.5)
    ax.axis('off')
    ax.set_title("Combined Shape (Rectangle + Semi-circle)")
    plt.tight_layout()
    save_plot(fig, "form3_mensuration_combined_shape_perimeter_area")

def form3_mensuration_prism_pyramid_volume():
    """Generates diagrams for prism and pyramid volume."""
    fig = plt.figure(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')

    # Triangular Prism
    ax1 = fig.add_subplot(121, projection='3d', proj_type='ortho')
    # Base triangle vertices (example)
    base_verts = [(0,0,0), (2,0,0), (1,1.73,0)] # Equilateral base side 2
    height = 3 # Prism height
    top_verts = [(v[0], v[1], height) for v in base_verts]
    # Edges
    edges = [
        [base_verts[0], base_verts[1]], [base_verts[1], base_verts[2]], [base_verts[2], base_verts[0]], # Base
        [top_verts[0], top_verts[1]], [top_verts[1], top_verts[2]], [top_verts[2], top_verts[0]], # Top
        [base_verts[0], top_verts[0]], [base_verts[1], top_verts[1]], [base_verts[2], top_verts[2]] # Sides
    ]
    for edge in edges:
        ax1.plot3D(*zip(*edge), color="k")

    # Labels
    ax1.text(1, 0.8, 0, 'Base (Area A)', zdir='z', ha='center', va='top', fontsize=9)
    ax1.plot([1, 1], [0.8, 0.8], [0, height], 'r--') # Height line
    ax1.text(1.1, 0.8, height/2, 'h', zdir='y', color='red', fontsize=10)
    ax1.text(1, 1, height*1.1, 'V = A × h', zdir='z', ha='center')

    ax1.set_box_aspect((2,2,height)) # Approximate aspect
    ax1.axis('off')
    ax1.set_title("Triangular Prism Volume")

    # Square Pyramid
    ax2 = fig.add_subplot(122, projection='3d', proj_type='ortho')
    base_side = 2
    height = 2.5
    apex = (base_side/2, base_side/2, height)
    base_verts = [(0,0,0), (base_side,0,0), (base_side,base_side,0), (0,base_side,0)]
    # Edges
    edges = [
        [base_verts[0], base_verts[1]], [base_verts[1], base_verts[2]], [base_verts[2], base_verts[3]], [base_verts[3], base_verts[0]], # Base
        [base_verts[0], apex], [base_verts[1], apex], [base_verts[2], apex], [base_verts[3], apex] # Slanted edges
    ]
    for edge in edges:
        ax2.plot3D(*zip(*edge), color="k")

    # Height line
    ax2.plot([base_side/2, apex[0]], [base_side/2, apex[1]], [0, apex[2]], 'r--')
    ax2.text(apex[0]+0.1, apex[1], height/2, 'h', zdir='y', color='red', fontsize=10)
    ax2.text(base_side/2, base_side/2, -0.2, 'Base (Area A)', zdir='z', ha='center', va='top', fontsize=9)
    ax2.text(base_side/2, base_side/2, height*1.1, 'V = (1/3) × A × h', zdir='z', ha='center')

    ax2.set_box_aspect((base_side, base_side, height)) # Approximate aspect
    ax2.axis('off')
    ax2.set_title("Square Pyramid Volume")

    plt.tight_layout()
    save_plot(fig, "form3_mensuration_prism_pyramid_volume")

def form3_mensuration_cone_volume():
    """Generates diagram for cone volume."""
    fig = plt.figure(figsize=(4, 5))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    # Parameters
    radius = 1.5
    height = 3.0
    apex = (0, 0, height)
    n_points = 100

    # Base circle
    theta = np.linspace(0, 2*np.pi, n_points)
    xb = radius * np.cos(theta)
    yb = radius * np.sin(theta)
    zb = np.zeros(n_points)
    ax.plot(xb, yb, zb, color='k')

    # Lines from apex to base edge
    ax.plot([apex[0], xb[0]], [apex[1], yb[0]], [apex[2], zb[0]], color='k')
    ax.plot([apex[0], xb[n_points//2]], [apex[1], yb[n_points//2]], [apex[2], zb[n_points//2]], color='k')

    # Radius and Height lines
    ax.plot([0, xb[0]], [0, yb[0]], [0, 0], 'k:') # Radius line
    ax.plot([0, apex[0]], [0, apex[1]], [0, apex[2]], 'r--') # Height line

    # Labels
    ax.text(xb[0]/2, yb[0]/2, 0, 'r', zdir='z', ha='center', va='top', fontsize=10)
    ax.text(0.1, 0, height/2, 'h', zdir='y', color='red', fontsize=10)
    ax.text(0, 0, height*1.1, 'V = (1/3)πr²h', zdir='z', ha='center', fontsize=11)

    # Make axes invisible but maintain aspect ratio
    max_dim = max(radius*2, height) * 1.1
    ax.set_xlim(-max_dim/2, max_dim/2)
    ax.set_ylim(-max_dim/2, max_dim/2)
    ax.set_zlim(0, max_dim)
    ax.set_box_aspect((radius*2, radius*2, height))
    ax.axis('off')
    ax.set_title("Cone Volume")

    plt.tight_layout()
    save_plot(fig, "form3_mensuration_cone_volume")

def form3_mensuration_sphere_volume():
    """Generates diagram for sphere volume."""
    fig = plt.figure(figsize=(4, 4.5)) # Slightly taller for formula
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    # Parameters
    radius = 1.5
    n_points = 50

    # Sphere surface using parametric equations
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='lightblue', alpha=0.6, edgecolor='k', linewidth=0.2)

    # Radius line (example)
    ax.plot([0, radius], [0, 0], [0, 0], 'r-', lw=1.5)
    ax.text(radius/2, 0.1, 0, 'r', zdir='z', color='red', fontsize=10)

     # Formula text below
    fig.text(0.5, 0.1, 'V = (4/3)πr³', ha='center', va='center', fontsize=11)


    # Make axes invisible but maintain aspect ratio
    lim = radius * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1,1,1))
    ax.axis('off')
    ax.set_title("Sphere Volume")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    save_plot(fig, "form3_mensuration_sphere_volume")

def form3_mensuration_surface_area_nets():
    """Generates nets for cylinder and cone surface area."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor('white')

    # Cylinder Net
    ax1 = axs[0]
    r, h = 1, 2.5 # Example dimensions
    circ = 2 * np.pi * r

    # Circles
    ax1.add_patch(Circle((circ/2, r*2+h/2), r, fill=False, edgecolor='blue'))
    ax1.add_patch(Circle((circ/2, -r*1+h/2), r, fill=False, edgecolor='blue'))

    # Rectangle (Curved Surface)
    ax1.add_patch(Rectangle((0, h/2), circ, h, fill=False, edgecolor='red'))

    # Labels
    ax1.text(circ/2, h*1.2+r*2.1, '2πr', ha='center', va='bottom', fontsize=9)
    ax1.text(circ+0.1, h, 'h', ha='left', va='center', fontsize=9)
    ax1.text(circ/2, r*2+h/2, 'πr²', ha='center', va='center', fontsize=9, color='blue') # Top area
    ax1.text(circ/2, -r*1+h/2, 'πr²', ha='center', va='center', fontsize=9, color='blue') # Bottom area
    ax1.text(circ/2, h, '2πrh', ha='center', va='center', fontsize=9, color='red') # Curved area
    ax1.text(circ/2, -r*1.8, 'SA = 2πr² + 2πrh', ha='center', va='center', fontsize=10)


    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim(-0.5, circ+0.5)
    ax1.set_ylim(-r*2, h+r*2.5)
    ax1.axis('off')
    ax1.set_title("Cylinder Net & Surface Area")

    # Cone Net
    ax2 = axs[1]
    r, h = 1, 2 # Example dimensions
    l = np.sqrt(r**2 + h**2) # Slant height

    # Base Circle
    ax2.add_patch(Circle((l*0.8, -r*1.2), r, fill=False, edgecolor='blue'))
    ax2.text(l*0.8, -r*1.2, 'πr²', ha='center', va='center', fontsize=9, color='blue')

    # Sector (Curved Surface)
    # Angle of sector in radians: theta = Arc Length / Radius = (2*pi*r) / l
    sector_angle_rad = (2 * np.pi * r) / l
    sector_angle_deg = np.rad2deg(sector_angle_rad)

    # Position sector (this is approximate positioning)
    start_angle = 90 - sector_angle_deg / 2
    end_angle = 90 + sector_angle_deg / 2
    sector = Arc((0, 0), 2*l, 2*l, angle=0, theta1=start_angle, theta2=end_angle, edgecolor='red', fill=False)
    ax2.add_patch(sector)
    ax2.plot([0, l*np.cos(np.deg2rad(start_angle))], [0, l*np.sin(np.deg2rad(start_angle))], 'r-')
    ax2.plot([0, l*np.cos(np.deg2rad(end_angle))], [0, l*np.sin(np.deg2rad(end_angle))], 'r-')

    # Labels
    ax2.text(0, l*0.5, 'πrl', ha='center', va='center', fontsize=9, color='red') # Curved area
    ax2.plot([0, l*np.cos(np.deg2rad(start_angle))], [0, l*np.sin(np.deg2rad(start_angle))], 'r:')
    ax2.text(l*np.cos(np.deg2rad(start_angle))*0.6, l*np.sin(np.deg2rad(start_angle))*0.6, 'l', color='red', fontsize=9) # Slant height

    ax2.text(l*0.4, -r*2.2, 'SA = πr² + πrl', ha='center', va='center', fontsize=10)


    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlim(-l*1.1, l*1.1)
    ax2.set_ylim(-r*2.5, l*1.1)
    ax2.axis('off')
    ax2.set_title("Cone Net & Surface Area")

    plt.tight_layout(pad=1)
    save_plot(fig, "form3_mensuration_surface_area_nets")

def form3_graphs_line_graph_y_eq_6_m_2x():
    """Draws graph of y=6-2x using intercepts."""
    fig, ax = plt.subplots(figsize=(5, 6))
    fig.patch.set_facecolor('white')

    x_intercept = 3
    y_intercept = 6

    # Set limits and ticks based on intercepts
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 7)
    ax.set_xticks(np.arange(-1, 5))
    ax.set_yticks(np.arange(-1, 8))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Plot intercepts
    ax.plot(x_intercept, 0, 'ro', markersize=7, label=f'({x_intercept}, 0)')
    ax.plot(0, y_intercept, 'bo', markersize=7, label=f'(0, {y_intercept})')

    # Plot line through intercepts
    x_vals = np.array(ax.get_xlim())
    y_vals = 6 - 2*x_vals
    ax.plot(x_vals, y_vals, 'g-', linewidth=1.5, label='y = 6 - 2x')


    # Add axis labels
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0, labelpad=-10)
    ax.legend(loc='upper right', fontsize=9)

    # ax.set_aspect('equal', adjustable='box') # Can distort view if ranges differ much
    ax.set_title("Graph of y = 6 - 2x")
    plt.tight_layout()
    save_plot(fig, "form3_graphs_line_graph_y_eq_6_m_2x")

def form3_graphs_quadratic_graph_table():
    """Draws graph of y = x^2 - 2x - 3."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('white')

    x = np.array([-2, -1, 0, 1, 2, 3, 4])
    y = x**2 - 2*x - 3
    points = list(zip(x, y))

    # Set limits and ticks
    ax.set_xlim(-3, 5)
    ax.set_ylim(-5, 6)
    ax.set_xticks(np.arange(-3, 6))
    ax.set_yticks(np.arange(-5, 7))
    ax.grid(True, linestyle='--', alpha=0.6)

    # Move axes to center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Plot points and smooth curve
    ax.plot(x, y, 'ro', markersize=5, label='Calculated Points')
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = x_smooth**2 - 2*x_smooth - 3
    ax.plot(x_smooth, y_smooth, 'b-', linewidth=1.5, label='y = x² - 2x - 3')

    # Add axis labels
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0, labelpad=-10)
    ax.legend(loc='upper center', fontsize=9)

    ax.set_title("Graph of y = x² - 2x - 3")
    plt.tight_layout()
    save_plot(fig, "form3_graphs_quadratic_graph_table")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def form3_graphs_quadratic_sketch_intercepts():
    """Sketch of y = x^2 - 4x + 3 using intercepts."""
    # coefficients
    a, b, c = 1, -4, 3

    # set up figure
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')

    # intercepts
    x_intercepts = [1, 3]
    y_intercept = c  # 3

    # vertex calculation
    vertex_x = -b / (2 * a)       # Calculation: -(-4) / (2 * 1) = 2
    vertex_y = a*vertex_x**2 + b*vertex_x + c  # Calculation: 1*(2**2) - 4*2 + 3 = -1
    vertex = (vertex_x, vertex_y)

    # plot setup
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-2, 6)
    ax.grid(True, linestyle=':', alpha=0.5)

    # move axes to center
    for spine in ('right','top'):
        ax.spines[spine].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # plot points and curve
    ax.plot(x_intercepts, [0, 0], 'ro', markersize=7, label='x‑intercepts')
    ax.plot(0, y_intercept,   'bo', markersize=7, label='y‑intercept')
    ax.plot(*vertex,          'go', markersize=7, label='vertex')

    x_smooth = np.linspace(0, 4, 200)
    y_smooth = a*x_smooth**2 + b*x_smooth + c
    ax.plot(x_smooth, y_smooth, 'k-', linewidth=1.5)

    # annotate
    ax.text(1, -0.3, '(1,0)', ha='center', va='top')
    ax.text(3, -0.3, '(3,0)', ha='center', va='top')
    ax.text(-0.1, 3, '(0,3)', ha='right', va='center')
    ax.text(vertex_x, vertex_y-0.3,
            f'V({vertex_x:.0f},{vertex_y:.0f})',
            ha='center', va='top')

    # labels & title
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0, labelpad=-10)
    ax.set_title("Sketch of y = x² - 4x + 3")
    # ax.legend(loc='upper center', fontsize=8)

    plt.tight_layout()
    save_plot(fig, "form3_graphs_quadratic_sketch_intercepts")


def form3_graphs_quadratic_solve_y_eq_2():
    """Uses graph of y=x^2-2x-3 to solve y=2."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('white')

    x = np.array([-2, -1, 0, 1, 2, 3, 4])
    y = x**2 - 2*x - 3

    # Plot parabola
    x_smooth = np.linspace(x.min()-0.5, x.max()+0.5, 300)
    y_smooth = x_smooth**2 - 2*x_smooth - 3
    ax.plot(x_smooth, y_smooth, 'b-', linewidth=1.5, label='y = x² - 2x - 3')
    # ax.plot(x, y, 'ro', markersize=5) # Optional: show original points

    # Plot line y=2
    ax.axhline(2, color='orange', linestyle='--', label='y = 2')

    # Find intersections (approximate graphically)
    # Solve x^2 - 2x - 3 = 2 => x^2 - 2x - 5 = 0
    # Quadratic formula: x = [-b ± sqrt(b^2 - 4ac)] / 2a
    x_sol1 = (2 - np.sqrt((-2)**2 - 4*1*(-5))) / (2*1) # Calculation: (2 - np.sqrt(24)) / 2 = 1 - np.sqrt(6) ≈ -1.45
    x_sol2 = (2 + np.sqrt((-2)**2 - 4*1*(-5))) / (2*1) # Calculation: (2 + np.sqrt(24)) / 2 = 1 + np.sqrt(6) ≈ 3.45
    ax.plot([x_sol1, x_sol2], [2, 2], 'go', markersize=7) # Mark intersections

    # Draw lines to x-axis
    ax.plot([x_sol1, x_sol1], [0, 2], 'g:')
    ax.plot([x_sol2, x_sol2], [0, 2], 'g:')

    # Label estimated solutions
    ax.text(x_sol1, -0.5, f'x ≈ {x_sol1:.1f}', ha='center', va='top', color='green')
    ax.text(x_sol2, -0.5, f'x ≈ {x_sol2:.1f}', ha='center', va='top', color='green')

    # Setup plot
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(-5, 6)
    ax.set_xticks(np.arange(-2, 5))
    ax.set_yticks(np.arange(-5, 7))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xlabel("x", loc='right')
    ax.set_ylabel("y", loc='top', rotation=0, labelpad=-10)
    ax.legend(loc='upper center', fontsize=9)
    ax.set_title("Solving x² - 2x - 3 = 2 Graphically")

    plt.tight_layout()
    save_plot(fig, "form3_graphs_quadratic_solve_y_eq_2")

def form3_travel_graphs_journey_distance_time():
    """Draws distance-time graph for journey in WE1."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    time = [0, 2, 3, 6]
    distance = [0, 120, 120, 0]

    ax.plot(time, distance, 'b-o', linewidth=1.5, markersize=5)

    # Annotate points
    ax.text(0 - 0.1, 0 - 5, 'O(0h, 0km)', ha='right', va='top', fontsize=9)
    ax.text(2, 120 + 3, '(2h, 120km)', ha='center', va='bottom', fontsize=9)
    ax.text(3, 120 + 3, '(3h, 120km)', ha='center', va='bottom', fontsize=9)
    ax.text(6 + 0.1, 0 - 5, '(6h, 0km)', ha='left', va='top', fontsize=9)


    # Labels and Title
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance from Start (km)")
    ax.set_title("Journey with Stop (Distance-Time)")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 141, 20))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-10, 140)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.tight_layout()
    save_plot(fig, "form3_travel_graphs_journey_distance_time")

def form3_travel_graphs_speed_time_accel_const():
    """Draws speed-time graph for WE2 (accel then const speed)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    time = [0, 5, 15]
    speed = [0, 10, 10]

    ax.plot(time, speed, 'b-o', linewidth=1.5, markersize=5)

    # Labels and Title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed-Time Graph Example")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xticks(np.arange(0, 16, 1))
    ax.set_yticks(np.arange(0, 13, 2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 12)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.tight_layout()
    save_plot(fig, "form3_travel_graphs_speed_time_accel_const")

def form3_travel_graphs_speed_time_area_triangle():
    """Highlights triangular area under speed-time graph (WE3)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    time = [0, 5, 15]
    speed = [0, 10, 10]

    ax.plot(time, speed, 'b-o', linewidth=1.5, markersize=5)

    # Shade the triangle area (0-5s)
    ax.fill_between([0, 5], [0, 10], color='lightblue', alpha=0.5, label='Area = Distance (0-5s)')

    # Labels and Title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Distance as Area under Speed-Time Graph")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xticks(np.arange(0, 16, 1))
    ax.set_yticks(np.arange(0, 13, 2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 12)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_plot(fig, "form3_travel_graphs_speed_time_area_triangle")

def form3_travel_graphs_speed_time_total_area():
    """Highlights total area under speed-time graph (WE4)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    time = [0, 5, 15]
    speed = [0, 10, 10]

    ax.plot(time, speed, 'b-o', linewidth=1.5, markersize=5)

    # Shade the total area (0-15s)
    ax.fill_between([0, 5], [0, 10], color='lightblue', alpha=0.5)
    ax.fill_between([5, 15], [10, 10], color='lightcoral', alpha=0.5)
    # Add a combined label (tricky to place, maybe in title or text)
    ax.text(7.5, 5, 'Total Area =\nTotal Distance', ha='center', va='center', fontsize=10)


    # Labels and Title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Total Distance as Total Area under Speed-Time Graph")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xticks(np.arange(0, 16, 1))
    ax.set_yticks(np.arange(0, 13, 2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 12)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.tight_layout()
    save_plot(fig, "form3_travel_graphs_speed_time_total_area")

def form3_variation_partial_variation_graph():
    """Generates graph for partial variation y=kx+c."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')

    # Example parameters
    k = 1.5 # Slope
    c = 2   # y-intercept

    x_vals = np.linspace(0, 5, 100)
    y_vals = k * x_vals + c

    # Plotting
    ax.plot(x_vals, y_vals, 'b-', label=f'y = {k}x + {c}')
    ax.plot(0, c, 'ro', label=f'y-intercept (0, {c})') # Mark intercept

    # Formatting
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Partial Variation (y = kx + c)")
    ax.grid(True, linestyle=':')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.set_xlim(0, 5.1)
    ax.set_ylim(0, k*5 + c + 1)

    plt.tight_layout()
    save_plot(fig, "form3_variation_partial_variation_graph")

def form3_geometry_angles_elevation():
    """Diagram illustrating Angle of Elevation."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('white')

    O = (0, 0) # Observer
    T = (3, 1.5) # Top of object

    # Horizontal line
    ax.plot([O[0], T[0]+0.5], [O[1], O[1]], 'k--', label='Horizontal')
    # Line of sight
    ax.plot([O[0], T[0]], [O[1], T[1]], 'r-', label='Line of Sight')
    ax.plot(O[0], O[1], 'ko', markersize=5)
    ax.text(O[0]-0.1, O[1]-0.1, 'O', ha='right', va='top')
    ax.plot(T[0], T[1], 'bo', markersize=5)
    ax.text(T[0]+0.1, T[1], 'T', ha='left', va='center')

    # Angle Arc
    angle_deg = np.rad2deg(np.arctan2(T[1]-O[1], T[0]-O[0]))
    arc = Arc(O, 1.0, 1.0, angle=0, theta1=0, theta2=angle_deg, color='red')
    ax.add_patch(arc)
    ax.text(0.6*np.cos(np.deg2rad(angle_deg/2)), 0.6*np.sin(np.deg2rad(angle_deg/2)),
            'Angle of Elevation', color='red', ha='left', va='bottom', fontsize=9, rotation=angle_deg/2)

    ax.set_xlim(-0.5, T[0]+1)
    ax.set_ylim(-0.5, T[1]+0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    # ax.set_title("Angle of Elevation")

    plt.tight_layout()
    save_plot(fig, "form3_geometry_angles_elevation")

def form3_geometry_angles_depression():
    """Diagram illustrating Angle of Depression."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('white')

    O = (0, 2) # Observer (high up)
    B = (3, 0) # Bottom object

    # Horizontal line
    ax.plot([O[0], B[0]+0.5], [O[1], O[1]], 'k--', label='Horizontal')
    # Line of sight
    ax.plot([O[0], B[0]], [O[1], B[1]], 'r-', label='Line of Sight')
    ax.plot(O[0], O[1], 'ko', markersize=5)
    ax.text(O[0]-0.1, O[1]+0.1, 'O', ha='right', va='bottom')
    ax.plot(B[0], B[1], 'bo', markersize=5)
    ax.text(B[0], B[1]-0.1, 'B', ha='center', va='top')

    # Angle Arc
    angle_rad = np.arctan2(B[1]-O[1], B[0]-O[0]) # Angle relative to positive x-axis
    angle_deg = np.rad2deg(angle_rad) # This will be negative
    depression_angle_deg = -angle_deg # Make positive for labeling

    arc = Arc(O, 1.0, 1.0, angle=0, theta1=angle_deg, theta2=0, color='red')
    ax.add_patch(arc)
    ax.text(0.6*np.cos(np.deg2rad(angle_deg/2)), O[1] + 0.6*np.sin(np.deg2rad(angle_deg/2)),
            'Angle of Depression', color='red', ha='left', va='top', fontsize=9, rotation=angle_deg/2)

    ax.set_xlim(-0.5, B[0]+1)
    ax.set_ylim(B[1]-0.5, O[1]+0.5)
    # ax.set_aspect('equal', adjustable='box') # Can make it look squashed
    ax.axis('off')
    # ax.set_title("Angle of Depression")

    plt.tight_layout()
    save_plot(fig, "form3_geometry_angles_depression")

def form3_geometry_angles_elevation_depression_relation():
    """Diagram illustrating alpha = beta relationship."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('white')
    A = (1, 1)
    B = (4, 3)

    # Horizontal lines (Parallel)
    ax.plot([A[0]-1, B[0]+1], [A[1], A[1]], 'k--', label='Horizontal at A')
    ax.plot([A[0]-1, B[0]+1], [B[1], B[1]], 'k--', label='Horizontal at B')
    ax.text(2.5, 1.05, '>', ha='center', va='bottom', fontsize=12) # Parallel marks
    ax.text(2.5, 3.05, '>', ha='center', va='bottom', fontsize=12)

    # Line of sight AB
    ax.plot([A[0], B[0]], [A[1], B[1]], 'r-')
    ax.plot(A[0], A[1], 'ko'); ax.text(A[0]-0.1, A[1]-0.1, 'A', ha='right', va='top')
    ax.plot(B[0], B[1], 'ko'); ax.text(B[0]+0.1, B[1]+0.1, 'B', ha='left', va='bottom')

    # Angles
    vec_ab = np.array(B) - np.array(A)
    angle_rad = np.arctan2(vec_ab[1], vec_ab[0])
    angle_deg = np.rad2deg(angle_rad)

    # Elevation alpha at A
    arc_a = Arc(A, 0.8, 0.8, angle=0, theta1=0, theta2=angle_deg, color='blue')
    ax.add_patch(arc_a)
    ax.text(A[0]+0.5*np.cos(np.deg2rad(angle_deg/2)), A[1]+0.5*np.sin(np.deg2rad(angle_deg/2)),
            'α', color='blue', fontsize=12)

    # Depression beta at B
    arc_b = Arc(B, 0.8, 0.8, angle=0, theta1=180, theta2=180+angle_deg, color='green')
    ax.add_patch(arc_b)
    ax.text(B[0]+0.5*np.cos(np.deg2rad(180+angle_deg/2)), B[1]+0.5*np.sin(np.deg2rad(180+angle_deg/2)),
            'β', color='green', fontsize=12)

    ax.text(1.5, 2.5, 'α = β\n(Alternate Interior Angles)', ha='center', va='center', fontsize=10)


    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Angle of Elevation = Angle of Depression")

    plt.tight_layout()
    save_plot(fig, "form3_geometry_angles_elevation_depression_relation")

def form3_geometry_angles_elevation_tower_problem():
    """Diagram for tower angle of elevation problem."""
    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_facecolor('white')
    base = 50
    angle = 30
    height = base * np.tan(np.deg2rad(angle))

    # Triangle vertices
    Q = (0, 0) # Observer
    P = (base, 0) # Base of tower
    T = (base, height) # Top of tower

    # Draw triangle
    ax.plot([Q[0], P[0], T[0], Q[0]], [Q[1], P[1], T[1], Q[1]], 'k-')
    # Right angle
    ax.plot([P[0]-1, P[0]], [0,0], 'k-'); ax.plot([P[0], P[0]], [0,1], 'k-')

    # Labels
    ax.text(Q[0]-2, Q[1], 'Q', ha='right', va='center')
    ax.text(P[0]+2, P[1], 'P', ha='left', va='center')
    ax.text(T[0]+2, T[1], 'T', ha='left', va='center')
    ax.text(base/2, -2, '50 m', ha='center', va='top')
    ax.text(base+2, height/2, 'h', ha='left', va='center')

    # Angle
    arc = Arc(Q, 10, 10, angle=0, theta1=0, theta2=angle, color='red')
    ax.add_patch(arc)
    ax.text(8, 3, f'{angle}°', color='red')

    ax.set_xlim(-5, base+10)
    ax.set_ylim(-5, height+5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Tower Elevation Problem")
    plt.tight_layout()
    save_plot(fig, "form3_geometry_angles_elevation_tower_problem")

def form3_geometry_angles_depression_cliff_problem():
    """Diagram for cliff angle of depression problem."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')

    height = 80
    angle_dep = 40
    angle_alt = angle_dep # Alternate interior angle
    dist = height / np.tan(np.deg2rad(angle_alt))

    # Points
    T = (0, height) # Top of cliff
    F = (0, 0) # Foot of cliff
    B = (dist, 0) # Boat

    # Triangle TFB
    ax.plot([T[0], F[0], B[0], T[0]], [T[1], F[1], B[1], T[1]], 'k-')
    # Right angle at F
    ax.plot([0, 2], [0,0], 'k-'); ax.plot([0, 0], [0,2], 'k-')

    # Horizontal line at T
    ax.plot([T[0], B[0]+10], [T[1], T[1]], 'k--')

    # Labels
    ax.text(T[0]-2, T[1], 'T', ha='right', va='center')
    ax.text(F[0]-2, F[1], 'F', ha='right', va='center')
    ax.text(B[0]+2, B[1], 'B', ha='left', va='center')
    ax.text(-4, height/2, '80 m', ha='right', va='center')
    ax.text(dist/2, -4, 'd', ha='center', va='top')

    # Angle of Depression at T
    arc_dep = Arc(T, 15, 15, angle=0, theta1=180-angle_dep, theta2=180, color='red')
    ax.add_patch(arc_dep)
    ax.text(T[0]+10*np.cos(np.deg2rad(180-angle_dep/2)), T[1]+10*np.sin(np.deg2rad(180-angle_dep/2)),
            f'{angle_dep}°', color='red', ha='left', va='bottom', fontsize=9)

    # Alternate angle at B
    arc_alt = Arc(B, 15, 15, angle=0, theta1=180-angle_alt, theta2=180, color='blue')
    ax.add_patch(arc_alt)
    ax.text(B[0]-10*np.cos(np.deg2rad(angle_alt/2)), B[1]+10*np.sin(np.deg2rad(angle_alt/2)),
            f'{angle_alt}°', color='blue', ha='right', va='bottom', fontsize=9)


    ax.set_xlim(-10, dist+15)
    ax.set_ylim(-10, height+10)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Cliff Depression Problem")
    plt.tight_layout()
    save_plot(fig, "form3_geometry_angles_depression_cliff_problem")

def form3_geometry_angles_elevation_construct():
    """Constructs diagram for angle of elevation 25 deg."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('white')
    O = (0, 0)
    angle = 25
    length = 3

    # Horizontal
    ax.plot([O[0], O[0]+length+0.5], [O[1], O[1]], 'k--')
    # Line of sight
    angle_rad = np.deg2rad(angle)
    B = (O[0] + length * np.cos(angle_rad), O[1] + length * np.sin(angle_rad))
    ax.plot([O[0], B[0]], [O[1], B[1]], 'r-')
    ax.plot(O[0], O[1], 'ko'); ax.text(O[0]-0.1, O[1]-0.1, 'O', ha='right', va='top')
    ax.plot(B[0], B[1], 'bo'); ax.text(B[0]+0.1, B[1], 'B', ha='left', va='center')

    # Arc
    arc = Arc(O, 1, 1, angle=0, theta1=0, theta2=angle, color='red')
    ax.add_patch(arc)
    ax.text(0.6*np.cos(np.deg2rad(angle/2)), 0.6*np.sin(np.deg2rad(angle/2)), f'{angle}°', color='red', fontsize=9)

    ax.set_xlim(-0.5, length+1)
    ax.set_ylim(-0.5, length*np.sin(angle_rad)+0.5)
    ax.axis('off')
    ax.set_title("Angle of Elevation = 25°")
    plt.tight_layout()
    save_plot(fig, "form3_geometry_angles_elevation_construct")

def form3_geometry_angles_depression_construct():
    """Constructs diagram for angle of depression 35 deg."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('white')
    P = (0, 1.5) # Person high up
    angle = 35
    length = 3

    # Horizontal
    ax.plot([P[0], P[0]+length+0.5], [P[1], P[1]], 'k--')
    # Line of sight
    angle_rad = np.deg2rad(-angle) # Angle downwards from +x
    C = (P[0] + length * np.cos(angle_rad), P[1] + length * np.sin(angle_rad))
    ax.plot([P[0], C[0]], [P[1], C[1]], 'r-')
    ax.plot(P[0], P[1], 'ko'); ax.text(P[0]-0.1, P[1]+0.1, 'P', ha='right', va='bottom')
    ax.plot(C[0], C[1], 'bo'); ax.text(C[0], C[1]-0.1, 'C', ha='center', va='top')

    # Arc
    arc = Arc(P, 1, 1, angle=0, theta1=-angle, theta2=0, color='red')
    ax.add_patch(arc)
    ax.text(0.6*np.cos(np.deg2rad(-angle/2)), P[1]+0.6*np.sin(np.deg2rad(-angle/2)), f'{angle}°', color='red', fontsize=9, ha='left', va='top')

    ax.set_xlim(-0.5, length+1)
    ax.set_ylim(C[1]-0.5, P[1]+0.5)
    ax.axis('off')
    ax.set_title("Angle of Depression = 35°")
    plt.tight_layout()
    save_plot(fig, "form3_geometry_angles_depression_construct")

def form3_geometry_angles_elevation_scale_drawing():
    """Scale drawing diagram for tower problem."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')
    # Scale: 1cm = 10m
    map_dist = 5 # Represents 50m
    angle = 30

    # Points
    A = (0, 0) # Observer
    B = (map_dist, 0) # Base
    # Calculate C based on angle
    height_map = map_dist * np.tan(np.deg2rad(angle))
    C = (map_dist, height_map)

    # Draw triangle
    ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-', lw=1.5)
    # Right angle
    ax.plot([B[0]-0.2, B[0]], [0,0], 'k-'); ax.plot([B[0], B[0]], [0,0.2], 'k-')

    # Labels
    ax.text(A[0]-0.2, A[1], 'A', ha='right', va='center')
    ax.text(B[0]+0.1, B[1], 'B', ha='left', va='center')
    ax.text(C[0]+0.1, C[1], 'C', ha='left', va='center')
    ax.text(map_dist/2, -0.2, '5 cm (Represents 50 m)', ha='center', va='top', fontsize=9)
    ax.text(map_dist+0.1, height_map/2, 'BC = ? cm', ha='left', va='center', fontsize=9)

    # Angle
    arc = Arc(A, 1, 1, angle=0, theta1=0, theta2=angle, color='red')
    ax.add_patch(arc)
    ax.text(0.8, 0.3, f'{angle}°', color='red', fontsize=9)

    ax.set_xlim(-0.5, map_dist + 1)
    ax.set_ylim(-0.5, height_map + 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Scale Drawing (1cm : 10m)")
    plt.tight_layout()
    save_plot(fig, "form3_geometry_angles_elevation_scale_drawing")

def form3_geometry_bearing_back_bearing():
    """Diagram illustrating bearing and back bearing (WE4, F2 Bearing)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    S = (1, 1) # Ship
    # L position based on bearing 070 from S
    bearing_sl = 70
    dist = 3
    angle_rad = np.deg2rad(90 - bearing_sl)
    L = (S[0] + dist * np.cos(angle_rad), S[1] + dist * np.sin(angle_rad))

    # North lines
    ax.plot([S[0], S[0]], [S[1], S[1]+1.5], 'k--')
    ax.plot(S[0], S[1]+1.5, '^k'); ax.text(S[0], S[1]+1.6, 'N', ha='center', va='bottom')
    ax.plot([L[0], L[0]], [L[1], L[1]+1.5], 'k--')
    ax.plot(L[0], L[1]+1.5, '^k'); ax.text(L[0], L[1]+1.6, 'N', ha='center', va='bottom')

    # Line SL
    ax.plot([S[0], L[0]], [S[1], L[1]], 'r-', marker='o')
    ax.text(S[0]-0.1, S[1]-0.1, 'S', ha='right', va='top')
    ax.text(L[0]+0.1, L[1]+0.1, 'L', ha='left', va='bottom')

    # Bearing of L from S (070)
    arc_s = Arc(S, 0.8, 0.8, angle=90, theta1=-bearing_sl, theta2=0, color='red')
    ax.add_patch(arc_s)
    label_angle_s_rad = np.deg2rad(90 - bearing_sl / 2)
    ax.text(S[0]+0.5*np.cos(label_angle_s_rad), S[1]+0.5*np.sin(label_angle_s_rad),
            f'{bearing_sl:03d}°', color='red', fontsize=9)

    # Back Bearing of S from L (070+180=250)
    bearing_ls = bearing_sl + 180
    arc_l = Arc(L, 1.0, 1.0, angle=90, theta1=-bearing_ls, theta2=0, color='blue')
    ax.add_patch(arc_l)
    label_angle_l_rad = np.deg2rad(90 - (bearing_ls-180)*0.8 -180) # Complex positioning
    ax.text(L[0]+0.7*np.cos(label_angle_l_rad), L[1]+0.7*np.sin(label_angle_l_rad),
            f'{bearing_ls:03d}°', color='blue', fontsize=9, ha='right', va='top')


    ax.set_xlim(0, L[0]+1)
    ax.set_ylim(0, L[1]+2)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("Bearing and Back Bearing")
    plt.tight_layout()
    save_plot(fig, "form3_geometry_bearing_back_bearing") # Assuming Form 3

def form3_geometry_bearing_angle_between_paths():
    """Diagram for angle between two paths WE3 (F2 Bearing)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')

    A = (0, 0)
    len_ab = 3
    bearing_ab = 60
    angle_ab_rad = np.deg2rad(90 - bearing_ab)
    B = (A[0] + len_ab * np.cos(angle_ab_rad), A[1] + len_ab * np.sin(angle_ab_rad))

    len_bc = 2.4
    bearing_bc = 150
    angle_bc_rad = np.deg2rad(90 - bearing_bc)
    C = (B[0] + len_bc * np.cos(angle_bc_rad), B[1] + len_bc * np.sin(angle_bc_rad))

    # Paths
    ax.plot([A[0], B[0]], [A[1], B[1]], 'b-o', label='Path 1 (060°)')
    ax.plot([B[0], C[0]], [B[1], C[1]], 'r-o', label='Path 2 (150°)')

    # North lines
    ax.plot([A[0], A[0]], [A[1], A[1]+1], 'k:')
    ax.plot(A[0], A[1]+1, '^k')
    ax.plot([B[0], B[0]], [B[1], B[1]+1], 'k:')
    ax.plot(B[0], B[1]+1, '^k')

    # Extend AB to D
    vec_ab = np.array(B) - np.array(A)
    D = B + vec_ab * 0.3
    ax.plot([B[0], D[0]], [B[1], D[1]], 'b:')
    ax.text(D[0], D[1], 'D', ha='left', va='bottom', fontsize=9)

    # Mark angles at B
    # Angle NBD = 60
    arc_nbd = Arc(B, 0.5, 0.5, angle=90, theta1=-60, theta2=0, color='grey')
    ax.add_patch(arc_nbd)
    ax.text(B[0]+0.1, B[1]+0.5, '60°', color='grey', fontsize=8)
    # Angle NBC = 150
    arc_nbc = Arc(B, 0.7, 0.7, angle=90, theta1=-150, theta2=0, color='purple')
    ax.add_patch(arc_nbc)
    ax.text(B[0]-0.4, B[1]+0.5, '150°', color='purple', fontsize=8)

    # Angle between paths (DBC)
    angle_turn = bearing_bc - bearing_ab # This is not correct for inner angle
    angle_abc = 180 - (bearing_bc - bearing_ab) # Angle based on parallel lines logic
    angle_abc_correct = 180-(150-60) # From steps
    arc_abc = Arc(B, 0.3, 0.3, angle=np.rad2deg(angle_ab_rad)+180, theta1=0, theta2=angle_abc_correct, color='black', lw=1.5)
    ax.add_patch(arc_abc)
    ax.text(B[0]+0.2, B[1]-0.2, f'{angle_abc_correct}°', ha='left', va='top', fontsize=10)


    ax.set_xlabel("Eastings")
    ax.set_ylabel("Northings")
    ax.set_title("Angle Between Paths")
    ax.grid(True, linestyle=':')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_plot(fig, "form3_geometry_bearing_angle_between_paths") # Assuming Form 3

def form3_geometry_bearing_compass_s40e():
    """Diagram showing S40E compass bearing (WE4, F2 Bearing)."""
    # This is identical to form2_geometry_bearing_compass_s40e
    form2_geometry_bearing_compass_s40e()
    try:
        os.rename(os.path.join(output_dir, "form2_geometry_bearing_compass_s40e.png"),
                  os.path.join(output_dir, "form3_geometry_bearing_compass_s40e.png"))
    except FileNotFoundError:
        print("Skipping rename for form3_geometry_bearing_compass_s40e.png - source not found.")

def form3_geometry_polygons_triangle_types():
    """Generates diagrams of triangle types."""
    # This is identical to form2_geometry_triangle_types
    form2_geometry_triangle_types()
    try:
        os.rename(os.path.join(output_dir, "form2_geometry_triangle_types.png"),
                  os.path.join(output_dir, "form3_geometry_polygons_triangle_types.png"))
    except FileNotFoundError:
         print("Skipping rename for form3_geometry_polygons_triangle_types.png - source not found.")

def form3_geometry_polygons_quadrilateral_types():
    """Generates diagrams of common quadrilaterals."""
    # This is identical to form2_geometry_quadrilateral_types
    form2_geometry_quadrilateral_types()
    try:
        os.rename(os.path.join(output_dir, "form2_geometry_quadrilateral_types.png"),
                  os.path.join(output_dir, "form3_geometry_polygons_quadrilateral_types.png"))
    except FileNotFoundError:
        print("Skipping rename for form3_geometry_polygons_quadrilateral_types.png - source not found.")


def form3_geometry_similarity_rectangles():
    """Diagrams of two similar rectangles for WE1."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.patch.set_facecolor('white')

    # Rectangle A
    wA, lA = 4, 6
    axs[0].add_patch(Rectangle((0.1, 0.1), lA, wA, fill=False, edgecolor='blue', lw=1.5))
    axs[0].text(0.1 + lA/2, 0.05, f'{lA} cm', ha='center', va='top')
    axs[0].text(0.05, 0.1 + wA/2, f'{wA} cm', ha='right', va='center')
    axs[0].set_title("Rectangle A")
    axs[0].set_aspect('equal'); axs[0].axis('off'); axs[0].set_xlim(0, lA+0.2); axs[0].set_ylim(0, wA+0.2)

    # Rectangle B
    wB, lB = 6, 9
    axs[1].add_patch(Rectangle((0.1, 0.1), lB, wB, fill=False, edgecolor='red', lw=1.5))
    axs[1].text(0.1 + lB/2, 0.05, f'{lB} cm', ha='center', va='top')
    axs[1].text(0.05, 0.1 + wB/2, f'{wB} cm', ha='right', va='center')
    axs[1].set_title("Rectangle B")
    axs[1].set_aspect('equal'); axs[1].axis('off'); axs[1].set_xlim(0, lB+0.2); axs[1].set_ylim(0, wB+0.2)

    fig.suptitle("Similar Rectangles?")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig, "form3_geometry_similarity_rectangles")

def form3_geometry_congruency_sss():
    """Diagrams for SSS congruency example."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.patch.set_facecolor('white')

    # Triangle ABC
    ax1 = axs[0]
    A1, B1, C1 = (0,0), (6,0), (2.5, 4) # Example coords giving roughly sides 6, ~4.7, ~4.7 -> need 5,6,7
    # Use actual lengths to place points if possible, or just label
    # Let AB=6, AC=5, BC=7 (using cosine rule to find angle A first approx)
    cosA_val = 0.2
    A_ang = np.rad2deg(np.arccos(cosA_val)) # approx 78.5 deg
    A1, B1 = (0,0), (6,0)
    C1 = (5 * np.cos(np.deg2rad(A_ang)), 5 * np.sin(np.deg2rad(A_ang)))
    ax1.add_patch(Polygon([A1, B1, C1], fill=False, edgecolor='b', lw=1.5))
    ax1.text(3, -0.2, 'AB = 6 cm', ha='center', va='top')
    ax1.text(np.mean([A1[0],C1[0]]), np.mean([A1[1],C1[1]])+0.1, 'AC = 5 cm', ha='center', va='bottom', rotation=A_ang)
    ax1.text(np.mean([B1[0],C1[0]]), np.mean([B1[1],C1[1]])+0.1, 'BC = 7 cm', ha='center', va='bottom', rotation=np.rad2deg(np.arctan2(C1[1]-B1[1], C1[0]-B1[0])) )
    ax1.set_title("Triangle ABC")
    ax1.axis('off'); ax1.set_aspect('equal'); ax1.set_xlim(-1, 7); ax1.set_ylim(-1, C1[1]+1)

    # Triangle PQR (congruent, maybe rotated)
    ax2 = axs[1]
    P2, Q2 = (0,0), (6,0)
    C2 = (5 * np.cos(np.deg2rad(-A_ang)), 5 * np.sin(np.deg2rad(-A_ang))) # Flipped
    verts2 = [P2, Q2, C2]
    ax2.add_patch(Polygon(verts2, fill=False, edgecolor='r', lw=1.5))
    ax2.text(3, -0.2, 'PQ = 6 cm', ha='center', va='top')
    ax2.text(np.mean([P2[0],C2[0]]), np.mean([P2[1],C2[1]])-0.1, 'PR = 5 cm', ha='center', va='top', rotation=-A_ang)
    ax2.text(np.mean([Q2[0],C2[0]]), np.mean([Q2[1],C2[1]])-0.1, 'QR = 7 cm', ha='center', va='top', rotation=np.rad2deg(np.arctan2(C2[1]-Q2[1], C2[0]-Q2[0])))
    ax2.set_title("Triangle PQR")
    ax2.axis('off'); ax2.set_aspect('equal'); ax2.set_xlim(-1, 7); ax2.set_ylim(C2[1]-1, 1)

    fig.suptitle("Congruency by SSS")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_plot(fig, "form3_geometry_congruency_sss")
# ... (Add similar functions for SAS congruency) ...
# ... Add function for form3_geometry_similarity_pst_pqr ...

def form3_construction_sss():
    """Diagram for SSS triangle construction."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')
    PQ = 7; PR = 5; QR = 6
    P, Q = (0,0), (PQ, 0)

    # Draw base PQ
    ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-o', lw=1.5)
    ax.text(P[0]-0.1, P[1]-0.1, 'P', ha='right', va='top')
    ax.text(Q[0]+0.1, Q[1]-0.1, 'Q', ha='left', va='top')
    ax.text(PQ/2, -0.2, f'{PQ} cm', ha='center', va='top')

    # Construction arcs
    arc_p = Arc(P, 2*PR, 2*PR, angle=0, theta1=10, theta2=80, linestyle='--', color='blue')
    arc_q = Arc(Q, 2*QR, 2*QR, angle=0, theta1=100, theta2=170, linestyle='--', color='red')
    ax.add_patch(arc_p)
    ax.add_patch(arc_q)

    # Find intersection R (approximate graphically or calculate)
    # Using cosine rule on P: cosP = (7^2 + 5^2 - 6^2) / (2*7*5) = (49+25-36)/70 = 38/70
    angle_p_rad = np.arccos(38/70)
    R = (PR * np.cos(angle_p_rad), PR * np.sin(angle_p_rad))
    ax.plot(R[0], R[1], 'ko', markersize=5)
    ax.text(R[0], R[1]+0.1, 'R', ha='center', va='bottom')


    # Draw sides PR and QR
    ax.plot([P[0], R[0]], [P[1], R[1]], 'k-')
    ax.plot([Q[0], R[0]], [Q[1], R[1]], 'k-')

    ax.set_xlim(-1, PQ+1)
    ax.set_ylim(-1, R[1]+1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("SSS Triangle Construction (PQ=7, PR=5, QR=6)")
    plt.tight_layout()
    save_plot(fig, "form3_construction_sss")

def form3_construction_sas():
    """Diagram for SAS triangle construction."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')
    AB = 6; AC = 4; angleA = 60
    A, B = (0,0), (AB, 0)

    # Draw base AB
    ax.plot([A[0], B[0]], [A[1], B[1]], 'k-o', lw=1.5)
    ax.text(A[0]-0.1, A[1]-0.1, 'A', ha='right', va='top')
    ax.text(B[0]+0.1, B[1]-0.1, 'B', ha='left', va='top')
    ax.text(AB/2, -0.2, f'{AB} cm', ha='center', va='top')

    # Construct 60 deg angle at A (schematic arcs)
    arc_base = Arc(A, 1, 1, angle=0, theta1=0, theta2=80, linestyle='--', color='grey')
    ax.add_patch(arc_base)
    ax.plot([A[0]+1, A[0]+1*np.cos(np.deg2rad(60))], [A[1], A[1]+1*np.sin(np.deg2rad(60))], 'grey', linestyle='--')
    angle_line_rad = np.deg2rad(angleA)
    X = (A[0] + (AC+1)*np.cos(angle_line_rad), A[1] + (AC+1)*np.sin(angle_line_rad))
    ax.plot([A[0], X[0]], [A[1], X[1]], 'k:') # Arm AX

    # Mark angle
    arc_a = Arc(A, 0.8, 0.8, angle=0, theta1=0, theta2=angleA, color='red')
    ax.add_patch(arc_a)
    ax.text(0.5*np.cos(np.deg2rad(angleA/2)), 0.5*np.sin(np.deg2rad(angleA/2)), f'{angleA}°', color='red', fontsize=9)

    # Mark C on AX at distance AC
    C = (A[0] + AC*np.cos(angle_line_rad), A[1] + AC*np.sin(angle_line_rad))
    ax.plot(C[0], C[1], 'ko', markersize=5)
    ax.text(C[0]-0.1, C[1]+0.1, 'C', ha='right', va='bottom')
    ax.text(np.mean([A[0],C[0]])+0.1, np.mean([A[1],C[1]]), f'{AC} cm', ha='left', va='bottom', fontsize=9)

    # Join BC
    ax.plot([B[0], C[0]], [B[1], C[1]], 'k-')

    ax.set_xlim(-1, AB+1)
    ax.set_ylim(-1, C[1]+1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("SAS Triangle Construction (AB=6, AC=4, ∠A=60°)")
    plt.tight_layout()
    save_plot(fig, "form3_construction_sas")

def form3_construction_asa():
    """Diagram for ASA triangle construction."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white')
    XY = 8; angleX = 45; angleY = 60
    X, Y = (0,0), (XY, 0)

    # Draw base XY
    ax.plot([X[0], Y[0]], [X[1], Y[1]], 'k-o', lw=1.5)
    ax.text(X[0]-0.1, X[1]-0.1, 'X', ha='right', va='top')
    ax.text(Y[0]+0.1, Y[1]-0.1, 'Y', ha='left', va='top')
    ax.text(XY/2, -0.2, f'{XY} cm', ha='center', va='top')

    # Construct angles (schematic arcs)
    # Angle X = 45
    angleX_rad = np.deg2rad(angleX)
    armX_len = 6 # Arbitrary length for drawing
    W = (X[0] + armX_len*np.cos(angleX_rad), X[1] + armX_len*np.sin(angleX_rad))
    ax.plot([X[0], W[0]], [X[1], W[1]], 'k:')
    arc_x = Arc(X, 1, 1, angle=0, theta1=0, theta2=angleX, color='red')
    ax.add_patch(arc_x)
    ax.text(0.7*np.cos(np.deg2rad(angleX/2)), 0.7*np.sin(np.deg2rad(angleX/2)), f'{angleX}°', color='red', fontsize=9)

    # Angle Y = 60 (measured from YX inwards, so angle is 180-60=120 from +x axis)
    angleY_rad = np.deg2rad(180 - angleY)
    armY_len = 5
    V = (Y[0] + armY_len*np.cos(angleY_rad), Y[1] + armY_len*np.sin(angleY_rad))
    ax.plot([Y[0], V[0]], [Y[1], V[1]], 'k:')
    arc_y = Arc(Y, 1, 1, angle=0, theta1=180-angleY, theta2=180, color='blue')
    ax.add_patch(arc_y)
    ax.text(Y[0]+0.7*np.cos(np.deg2rad(180-angleY/2)), Y[1]+0.7*np.sin(np.deg2rad(180-angleY/2)), f'{angleY}°', color='blue', fontsize=9)

    # Find intersection Z (solve line equations)
    line1_m = np.tan(angleX_rad)
    line2_m = np.tan(angleY_rad)
    line1_c = 0
    line2_c = Y[1] - line2_m * Y[0]
    # line1: y = line1_m * x
    # line2: y = line2_m * x + line2_c
    # line1_m * Zx = line2_m * Zx + line2_c
    Zx = line2_c / (line1_m - line2_m)
    Zy = line1_m * Zx
    Z = (Zx, Zy)
    ax.plot(Z[0], Z[1], 'ko', markersize=5)
    ax.text(Z[0], Z[1]+0.1, 'Z', ha='center', va='bottom')

    # Draw final triangle
    ax.plot([X[0], Z[0]], [X[1], Z[1]], 'k-')
    ax.plot([Y[0], Z[0]], [Y[1], Z[1]], 'k-')


    ax.set_xlim(-1, XY+1)
    ax.set_ylim(-1, Z[1]+1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title("ASA Triangle Construction (XY=8, ∠X=45°, ∠Y=60°)")
    plt.tight_layout()
    save_plot(fig, "form3_construction_asa")


# --- Helper Function to Create Output Directory ---
def create_output_directory(dir_name="math_diagrams"):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

OUTPUT_DIR = create_output_directory()
# --- Diagram Generation Class ---

class DiagramGenerator:
    """Generates mathematical diagrams based on descriptions."""

    def __init__(self, output_dir=OUTPUT_DIR):
        """Initializes the generator with an output directory."""
        self.output_dir = output_dir
        # Store data needed for context between diagrams
        self.context_data = {}

    def _save_plot(self, fig, filename):
        """Saves the figure to the specified directory and closes it."""
        filepath = os.path.join(self.output_dir, filename)
        try:
            # Use tight_layout to prevent labels overlapping plot area
            fig.tight_layout(rect=[0, 0.03, 1, 0.95] if fig._suptitle else None) # Adjust if suptitle exists
            fig.savefig(filepath)
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
        finally:
            plt.close(fig) # Close the figure to free memory

    # --- Helper for Angle Arcs ---
    def _add_angle_arc(self, ax, center, radius, theta1_deg, theta2_deg, color='k', linestyle='--', label=None, label_pos_factor=1.2, **kwargs):
        """Adds a consistent angle arc and label to an axes."""
        arc = patches.Arc(center, radius*2, radius*2, angle=0,
                          theta1=min(theta1_deg, theta2_deg),
                          theta2=max(theta1_deg, theta2_deg),
                          color=color, ls=linestyle, **kwargs)
        ax.add_patch(arc)
        if label:
            # Calculate midpoint angle in radians for label placement
            mid_angle_rad = np.deg2rad((theta1_deg + theta2_deg) / 2)
            label_x = center[0] + label_pos_factor * radius * np.cos(mid_angle_rad)
            label_y = center[1] + label_pos_factor * radius * np.sin(mid_angle_rad)
            ax.text(label_x, label_y, label, color=color, ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.1', fc='white', ec='none', alpha=0.6))


    # --- Form 2: Algebra - Inequalities ---

    def cartesian_y_gt_x_plus_1(self):
        """Generates Cartesian plane for y > x + 1."""
        fig, ax = plt.subplots(figsize=(6, 6))
        x_vals = np.linspace(-4, 4, 400)
        y_boundary = x_vals + 1

        ax.plot(x_vals, y_boundary, 'b--', label='y = x + 1')
        # Shade UNWANTED region (y <= x + 1)
        ax.fill_between(x_vals, -4, y_boundary, where=y_boundary >= -4, color='gray', alpha=0.3, label='Unwanted Region (y ≤ x + 1)', interpolate=True)

        ax.text(-2, 3, 'Wanted Region\n(y > x + 1)', fontsize=10, ha='center')
        ax.plot(0, 0, 'ro', markersize=6, label='Test Point (0,0)')
        ax.text(0.1, -0.5, '(0,0)', color='red', fontsize=9)
        ax.text(-3.5, -2, 'Test: 0 > 0 + 1 (False)\n-> Shade other side', color='red', fontsize=9)

        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.spines['left'].set_position('zero'); ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left')
        ax.set_xlabel('x', loc='right'); ax.set_ylabel('y', loc='top', rotation=0)
        ax.set_title('Inequality: y > x + 1')
        ax.grid(True, linestyle=':'); ax.legend(loc='lower right'); ax.set_aspect('equal', adjustable='box')
        self._save_plot(fig, "cartesian_y_gt_x_plus_1.png")

    def number_line_x_le_neg1(self):
        """Generates number line for x <= -1."""
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axhline(0.5, color='black', lw=1)
        ax.yaxis.set_visible(False)
        for spine in ['left', 'right', 'top']: ax.spines[spine].set_visible(False)
        ax.set_xlim(-4, 2); ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(-4, 3)) # Extend ticks slightly
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.spines['bottom'].set_position(('data', 0.5))
        ax.plot(-1, 0.5, 'ko', markersize=8, markerfacecolor='black', label='x = -1 (included)')
        ax.arrow(-1, 0.5, -2.8, 0, head_width=0.1, head_length=0.3, fc='black', ec='black', lw=2, length_includes_head=True, overhang=0.1)
        ax.set_title('Number Line: x ≤ -1'); plt.tight_layout()
        self._save_plot(fig, "number_line_x_le_neg1.png")

    def cartesian_x_ge_2(self):
        """Generates Cartesian plane for x >= 2."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axvline(2, color='blue', linestyle='-', label='x = 2', lw=2)
        # Shade WANTED region (x >= 2)
        ax.axvspan(2, 4, alpha=0.3, color='lightblue', label='Shaded Region (x ≥ 2)')
        ax.set_xlim(-3, 4); ax.set_ylim(-3, 3) # Extend xlim to show shading better
        ax.spines['left'].set_position('zero'); ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left')
        ax.set_xlabel('x', loc='right'); ax.set_ylabel('y', loc='top', rotation=0)
        ax.set_title('Inequality: x ≥ 2')
        ax.grid(True, linestyle=':'); ax.legend(loc='lower right'); ax.set_aspect('equal', adjustable='box')
        self._save_plot(fig, "cartesian_x_ge_2.png")

    def cartesian_y_lt_3(self):
        """Generates Cartesian plane for y < 3."""
        fig, ax = plt.subplots(figsize=(6, 6)) # Adjusted size back
        x = np.linspace(-2, 5, 100)
        ax.axhline(3, color='blue', linestyle='--', label='y = 3', lw=2)
        # Shade WANTED region (y < 3)
        ax.fill_between(x, -2, 3, color='lightblue', alpha=0.5, label='Shaded Region (y < 3)')
        ax.set_xlim(-2, 5); ax.set_ylim(-2, 5)
        ax.spines['left'].set_position('zero'); ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left')
        ax.set_xlabel('x', loc='right'); ax.set_ylabel('y', loc='top', rotation=0)
        ax.set_title('Inequality: y < 3')
        ax.grid(True, linestyle=':'); ax.legend(loc='lower right')
        # ax.set_aspect('equal', adjustable='box') # Keep commented unless specifically desired
        self._save_plot(fig, "cartesian_y_lt_3.png")

    # --- Form 2: Geometry - Points, Lines, and Angles ---

    def parallel_lines_l1_l2(self):
        """Generates diagram of two parallel lines."""
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([-1.2, 1.2], [0.8, 0.8], 'k-', lw=2) # Extend lines slightly
        ax.text(0, 0.8, r'$\blacktriangleright$', ha='center', va='center', fontsize=14)
        ax.text(-1.4, 0.8, 'L1', va='center')
        ax.plot([-1.2, 1.2], [0.2, 0.2], 'k-', lw=2)
        ax.text(0, 0.2, r'$\blacktriangleright$', ha='center', va='center', fontsize=14)
        ax.text(-1.4, 0.2, 'L2', va='center')
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 1)
        ax.axis('off'); ax.set_title('Parallel Lines')
        self._save_plot(fig, "parallel_lines_l1_l2.png")

    def parallel_transversal_angles_1_8(self):
        """Generates parallel lines cut by a transversal with angles labeled."""
        fig, ax = plt.subplots(figsize=(5, 5))
        line_lim = 2.2
        ax.plot([-line_lim, line_lim], [1, 1], 'k-', lw=1.5, label='L1')
        ax.plot([-line_lim, line_lim], [-1, -1], 'k-', lw=1.5, label='L2')
        ax.text(0, 1, r'$\blacktriangleright$', ha='center', va='center', fontsize=12)
        ax.text(0, -1, r'$\blacktriangleright$', ha='center', va='center', fontsize=12)
        ax.text(-line_lim - 0.1, 1, 'L1'); ax.text(-line_lim - 0.1, -1, 'L2')

        slope = 1.5
        x_trans = np.array([-line_lim/abs(slope), line_lim/abs(slope)]) * 0.9 # Adjust transversal length
        y_trans = slope * x_trans
        ax.plot(x_trans, y_trans, 'k-', lw=1.5, label='T')
        ax.text(x_trans[1] + 0.1, y_trans[1] + 0.1, 'T')

        x_int1 = 1 / slope; y_int1 = 1
        x_int2 = -1 / slope; y_int2 = -1

        offset = 0.35 # Increased offset slightly
        angle_positions = {
            1: (x_int1 - offset, y_int1 + offset), 2: (x_int1 + offset, y_int1 + offset),
            3: (x_int1 - offset, y_int1 - offset), 4: (x_int1 + offset, y_int1 - offset),
            5: (x_int2 - offset, y_int2 + offset), 6: (x_int2 + offset, y_int2 + offset),
            7: (x_int2 - offset, y_int2 - offset), 8: (x_int2 + offset, y_int2 - offset),
        }
        for angle, pos in angle_positions.items():
            ax.text(pos[0], pos[1], str(angle), ha='center', va='center', fontsize=10, color='blue',
                    bbox=dict(boxstyle='circle,pad=0.1', fc='white', ec='none', alpha=0.8))

        ax.set_xlim(-line_lim-0.3, line_lim+0.3); ax.set_ylim(-line_lim*0.7, line_lim*0.7) # Adjust ylim
        ax.axis('off'); ax.set_title('Parallel Lines and Transversal'); ax.set_aspect('equal', adjustable='box')
        self._save_plot(fig, "parallel_transversal_angles_1_8.png")

    def _draw_parallel_lines_setup(self, ax, slope=1.5, labels=True):
        """Helper to draw standard parallel lines and transversal."""
        line_lim = 2.2
        ax.plot([-line_lim, line_lim], [1, 1], 'k-', lw=1.5)
        ax.plot([-line_lim, line_lim], [-1, -1], 'k-', lw=1.5)
        ax.text(0, 1, r'$\blacktriangleright$', ha='center', va='center')
        ax.text(0, -1, r'$\blacktriangleright$', ha='center', va='center')
        if labels:
            ax.text(-line_lim - 0.1, 1, 'A'); ax.text(line_lim + 0.1, 1, 'B')
            ax.text(-line_lim - 0.1, -1, 'C'); ax.text(line_lim + 0.1, -1, 'D')

        x_trans = np.array([-line_lim/abs(slope), line_lim/abs(slope)]) * 0.9
        y_trans = slope * x_trans
        ax.plot(x_trans, y_trans, 'k-', lw=1.5)
        if labels:
            ax.text(x_trans[0] - 0.1, y_trans[0] - 0.2, 'E', va='top')
            ax.text(x_trans[1] + 0.1, y_trans[1] + 0.2, 'F', va='bottom')

        x_int1 = 1 / slope; y_int1 = 1 # G
        x_int2 = -1 / slope; y_int2 = -1 # H
        if labels:
            ax.text(x_int1, y_int1 + 0.15, 'G', ha='center', va='bottom')
            ax.text(x_int2, y_int2 - 0.15, 'H', ha='center', va='top')

        ax.set_xlim(-line_lim-0.3, line_lim+0.3); ax.set_ylim(-line_lim*0.7, line_lim*0.7)
        ax.axis('off'); ax.set_aspect('equal', adjustable='box')
        return x_int1, y_int1, x_int2, y_int2

    def parallel_corresponding_60deg(self):
        """Generates diagram for corresponding angles example (EGA=60)."""
        fig, ax = plt.subplots(figsize=(5, 5))
        slope = 1.5
        x_int1, y_int1, x_int2, y_int2 = self._draw_parallel_lines_setup(ax, slope)
        # Angles: EGA (Top-left at G) = 60, GHC (Bottom-left at H) = x (corresponding)
        ax.text(x_int1 - 0.3, y_int1 + 0.2, '60°', color='red', ha='right') # EGA
        ax.text(x_int2 - 0.3, y_int2 - 0.2, 'x = 60°', color='blue', ha='right') # GHC
        ax.set_title('Corresponding Angles (∠EGA = ∠GHC)')
        self._save_plot(fig, "parallel_corresponding_60deg.png")
        self.context_data['parallel_ex1_diagram'] = {'G': (x_int1, y_int1), 'H': (x_int2, y_int2)}

    def parallel_alternate_interior_110deg(self):
        """Generates diagram for alternate interior angles example (AGH=110)."""
        fig, ax = plt.subplots(figsize=(5, 5))
        slope = 1.5
        x_int1, y_int1, x_int2, y_int2 = self._draw_parallel_lines_setup(ax, slope)
        # Angles: AGH (Bottom-left at G) = 110, GHD (Top-right at H) = y (alt. int.)
        ax.text(x_int1 - 0.3, y_int1 - 0.25, '110°', color='red', ha='right') # AGH
        ax.text(x_int2 + 0.3, y_int2 + 0.25, 'y = 110°', color='blue', ha='left') # GHD
        ax.set_title('Alternate Interior Angles (∠AGH = ∠GHD)')
        self._save_plot(fig, "parallel_alternate_interior_110deg.png")

    def parallel_cointerior_75deg(self):
        """Generates diagram for co-interior angles example (BGH=75)."""
        fig, ax = plt.subplots(figsize=(5, 5))
        slope = 1.5
        x_int1, y_int1, x_int2, y_int2 = self._draw_parallel_lines_setup(ax, slope)
        # Angles: BGH (Bottom-right at G) = 75, GHC (Bottom-left at H) = z (co-int.)
        ax.text(x_int1 + 0.3, y_int1 - 0.25, '75°', color='red', ha='left') # BGH
        ax.text(x_int2 - 0.3, y_int2 - 0.25, 'z = 105°', color='blue', ha='right') # GHC
        ax.set_title('Co-interior Angles (∠BGH + ∠GHC = 180°)')
        self._save_plot(fig, "parallel_cointerior_75deg.png")
        self.context_data['parallel_ex3_diagram'] = {
            'coords': {'G': (x_int1, y_int1), 'H': (x_int2, y_int2)},
            'angles': {'BGH': 75, 'GHC': 105} # Store calculated GHC
        }

    def parallel_alternate_exterior_ref_ex3(self):
        """Generates diagram referencing Example 3 for alternate exterior angles (EGA/DHF)."""
        fig, ax = plt.subplots(figsize=(5, 5))
        slope = 1.5
        x_int1, y_int1, x_int2, y_int2 = self._draw_parallel_lines_setup(ax, slope)
        # Angles needed: BGH=75 (from Ex 3 context), DHF=? (alt. ext. to EGA), EGA=? (vert. opp. BGH)
        ax.text(x_int1 + 0.3, y_int1 - 0.25, '75°', color='gray', ha='left', style='italic', fontsize=9) # BGH (context)
        ax.text(x_int1 - 0.3, y_int1 + 0.25, '75°', color='red', ha='right') # EGA (vert. opp. BGH)
        ax.text(x_int2 + 0.3, y_int2 - 0.25, '75°', color='blue', ha='left') # DHF (alt. ext. to EGA)
        ax.set_title('Alternate Exterior Angles (∠EGA = ∠DHF = 75°)')
        self._save_plot(fig, "parallel_alternate_exterior_ref_ex3.png")

    def parallel_alternate_interior_algebra(self):
        """Generates diagram for alternate interior angles with algebraic expressions."""
        fig, ax = plt.subplots(figsize=(5, 5))
        slope = 1.5
        x_int1, y_int1, x_int2, y_int2 = self._draw_parallel_lines_setup(ax, slope, labels=False) # No A,B,C,D labels needed
        # Angles: Bottom-Left Interior at top = (2x+10), Top-Right Interior at bottom = (3x-20)
        ax.text(x_int1 - 0.4, y_int1 - 0.3, '(2x + 10)°', color='red', ha='right', fontsize=9)
        ax.text(x_int2 + 0.4, y_int2 + 0.3, '(3x - 20)°', color='blue', ha='left', fontsize=9)
        ax.set_title('Alternate Interior Angles (Algebra)')
        self._save_plot(fig, "parallel_alternate_interior_algebra.png")

    def parallel_vertical_transversal_q2(self):
        """Generates diagram for Question 2 (vertical parallel lines)."""
        fig, ax = plt.subplots(figsize=(5, 5))
        line_lim = 2.2
        ax.plot([-1, -1], [-line_lim, line_lim], 'k-', lw=1.5, label='m')
        ax.plot([1, 1], [-line_lim, line_lim], 'k-', lw=1.5, label='n')
        ax.text(-1, 0, '↓', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(1, 0, '↓', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(-1, line_lim + 0.1, 'm', ha='center')
        ax.text(1, line_lim + 0.1, 'n', ha='center')

        slope = 0.8
        x_trans = np.array([-line_lim/slope*0.8, line_lim/slope*0.8]) # Adjust transversal length/angle
        y_trans = slope * x_trans
        ax.plot(x_trans, y_trans, 'k-', lw=1.5)

        x_int1 = -1; y_int1 = slope * x_int1 # Intersection with m
        x_int2 = 1; y_int2 = slope * x_int2 # Intersection with n

        ax.text(x_int1 + 0.3, y_int1 + 0.3, '70°', color='red', ha='left') # Top-right at m
        ax.text(x_int2 - 0.3, y_int2 + 0.3, 'a', color='blue', ha='right') # Top-left at n
        ax.text(x_int2 - 0.3, y_int2 - 0.3, 'b', color='blue', ha='right') # Bottom-left at n
        ax.text(x_int2 + 0.3, y_int2 - 0.3, 'c', color='blue', ha='left') # Bottom-right at n

        ax.set_xlim(-line_lim-0.3, line_lim+0.3); ax.set_ylim(-line_lim-0.3, line_lim+0.3)
        ax.axis('off'); ax.set_title('Parallel Lines Question'); ax.set_aspect('equal', adjustable='box')
        self._save_plot(fig, "parallel_vertical_transversal_q2.png")

    def parallel_lines_angle_between_q3(self):
        """Generates diagram for Question 3 (angle between path segments)."""
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([-1, 3], [1, 1], 'k-', lw=1.5); ax.text(-1.2, 1, 'A'); ax.text(3.2, 1, 'B') # Line AB
        ax.plot([-1, 3], [-1, -1], 'k-', lw=1.5); ax.text(-1.2, -1, 'D'); ax.text(3.2, -1, 'E') # Line DE (Parallel)
        ax.text(1, 1, r'$\blacktriangleright$', ha='center', va='center')
        ax.text(1, -1, r'$\blacktriangleright$', ha='center', va='center')

        B_coord = (3, 1); D_coord = (-1, -1) # Using segment ends for clarity
        C = (1.5, 0) # Adjusted C position slightly
        ax.plot(C[0], C[1], 'ko'); ax.text(C[0], C[1] + 0.1, 'C')

        ax.plot([B_coord[0], C[0]], [B_coord[1], C[1]], 'k-') # BC
        ax.plot([D_coord[0], C[0]], [D_coord[1], C[1]], 'k-') # DC

        # Angles given (often reflex or exterior to path)
        # Angle "at B" is 130 deg relative to extension of AB
        # Angle "at D" is 140 deg relative to extension of ED
        # We need angle BCD = x

        # Show hint line and calculate x using alternate interior angles
        ax.plot([-1.5, 3.5], [C[1], C[1]], 'g--', lw=1, label='Parallel Line through C')
        # Angle below hint line, left of CD = Alt. Int. to angle beside D = 180-140 = 40
        # Angle below hint line, right of CB = Alt. Int. to angle beside B = 180-130 = 50
        # So x = 40 + 50 = 90
        ax.text(C[0] - 0.3, C[1] - 0.15, '40°', color='gray', fontsize=8, ha='right')
        ax.text(C[0] + 0.3, C[1] - 0.15, '50°', color='gray', fontsize=8, ha='left')

        # Mark angle x = BCD
        vec_CB = np.array(B_coord) - np.array(C)
        vec_CD = np.array(D_coord) - np.array(C)
        angle_CB_rad = np.arctan2(vec_CB[1], vec_CB[0])
        angle_CD_rad = np.arctan2(vec_CD[1], vec_CD[0])
        self._add_angle_arc(ax, C, 0.5, np.degrees(angle_CD_rad), np.degrees(angle_CB_rad), color='blue', linestyle='-', label='x=90°', label_pos_factor=0.8)

        ax.text(B_coord[0] - 0.7, 1.15, '130°', color='red', ha='center') # Angle near B
        ax.text(D_coord[0] + 0.7, -1.15, '140°', color='red', ha='center', va='top') # Angle near D

        ax.set_xlim(-1.5, 3.5); ax.set_ylim(-1.5, 1.5)
        ax.axis('off'); ax.set_title('Angle Between Path Segments (Q3)')
        ax.legend(loc='lower right', fontsize=8)
        self._save_plot(fig, "parallel_lines_angle_between_q3.png")

    # --- Form 2: Geometry - Bearing ---

    def compass_rose_8_points(self):
        """Generates an 8-point compass rose."""
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1) # North=0, Clockwise

        points = {'N': 0, 'NE': 45, 'E': 90, 'SE': 135, 'S': 180, 'SW': 225, 'W': 270, 'NW': 315}
        angles_rad = {name: np.deg2rad(angle) for name, angle in points.items()}

        line_length = 1.0
        for name, angle_rad in angles_rad.items():
            is_cardinal = points[name] % 90 == 0
            ax.plot([angle_rad, angle_rad], [0, line_length], color='black', lw=1.5 if is_cardinal else 1)
            if is_cardinal: # Add arrows for N, E, S, W
                 ax.arrow(angle_rad, line_length*0.8, 0, line_length*0.2, head_width=0.15, head_length=0.15, fc='k', ec='k', length_includes_head=True, zorder=5)
            ax.text(angle_rad, line_length + 0.2, name, ha='center', va='center', fontsize=12, fontweight='bold' if is_cardinal else 'normal')

        ax.set_yticks([]); ax.set_xticks([]) # Remove default ticks/labels
        ax.spines['polar'].set_visible(False); ax.set_title('Compass Rose', va='bottom', y=1.05)
        ax.set_ylim(0, line_length + 0.3) # Ensure labels fit
        self._save_plot(fig, "compass_rose_8_points.png")

    def _draw_bearing_axes(self, ax):
        """Helper to draw N, E, S, W axes lines."""
        lim = 1.2
        ax.axhline(0, color='grey', lw=0.5, zorder=0); ax.axvline(0, color='grey', lw=0.5, zorder=0)
        ax.text(0, lim, 'N', ha='center', va='bottom', fontsize=10)
        ax.text(0, -lim, 'S', ha='center', va='top', fontsize=10)
        ax.text(lim, 0, 'E', ha='left', va='center', fontsize=10)
        ax.text(-lim, 0, 'W', ha='right', va='center', fontsize=10)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off')

    def compass_bearing_examples(self):
        """Generates diagrams for compass bearing examples."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        fig.suptitle('Compass Bearing Examples', fontsize=14)

        bearings_info = {
            'N 30° E': {'ref': 'N', 'angle': 30, 'dir': 'E'},
            'S 45° W': {'ref': 'S', 'angle': 45, 'dir': 'W'},
            'N 70° W': {'ref': 'N', 'angle': 70, 'dir': 'W'}
        }

        for i, (title, info) in enumerate(bearings_info.items()):
            ax = axes[i]
            self._draw_bearing_axes(ax)

            # Calculate plot angle (anti-clockwise from positive x-axis)
            if info['ref'] == 'N':
                plot_angle_deg = 90 - info['angle'] if info['dir'] == 'E' else 90 + info['angle']
            else: # info['ref'] == 'S'
                plot_angle_deg = 270 + info['angle'] if info['dir'] == 'E' else 270 - info['angle']

            plot_angle_rad = np.deg2rad(plot_angle_deg)
            ax.plot([0, np.cos(plot_angle_rad)], [0, np.sin(plot_angle_rad)], 'r-', lw=2)
            ax.arrow(0, 0, np.cos(plot_angle_rad)*0.95, np.sin(plot_angle_rad)*0.95, head_width=0.08, head_length=0.1, fc='r', ec='r')

            # Angle Arc and Label from reference N or S
            ref_angle_deg = 90 if info['ref'] == 'N' else 270
            self._add_angle_arc(ax, (0,0), 0.4, ref_angle_deg, plot_angle_deg, color='k', label=f"{info['angle']}°", label_pos_factor=1.3)
            ax.set_title(title)

        self._save_plot(fig, "compass_bearing_examples.png")

    def three_figure_bearing_examples(self):
        """Generates diagrams for three-figure bearing examples."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        fig.suptitle('Three-Figure Bearing Examples', fontsize=14)

        bearings_deg = {'060°': 60, '135°': 135, '310°': 310}

        for i, (title, bearing_deg) in enumerate(bearings_deg.items()):
            ax = axes[i]
            self._draw_bearing_axes(ax) # Use helper for consistency
            # Remove unwanted direction labels (E, S, W), keeping only N
            texts_to_remove = []
            for t in ax.texts:
                if t.get_text() in ['E', 'S', 'W']:
                    texts_to_remove.append(t)
            for t in texts_to_remove:
                t.remove()

            # Convert bearing (clockwise from N) to plot angle (anti-clockwise from E)
            plot_angle_rad = np.deg2rad(90 - bearing_deg)
            ax.plot([0, np.cos(plot_angle_rad)], [0, np.sin(plot_angle_rad)], 'r-', lw=2)
            ax.arrow(0, 0, np.cos(plot_angle_rad)*0.95, np.sin(plot_angle_rad)*0.95, head_width=0.08, head_length=0.1, fc='r', ec='r')

            # Angle Arc (clockwise from North)
            north_angle_deg = 90
            bearing_plot_angle_deg = 90 - bearing_deg
            self._add_angle_arc(ax, (0,0), 0.5, north_angle_deg, bearing_plot_angle_deg, color='k', label=title, label_pos_factor=1.3)
            # ax.set_title(title) # Title already shown in angle label

        self._save_plot(fig, "three_figure_bearing_examples.png")

    def bearing_back_bearing_70_250(self):
        """Generates diagram relating bearing (L from S = 070) and back bearing (S from L = 250)."""
        fig, ax = plt.subplots(figsize=(6, 6))
        S = (0, 0); L_bearing_deg = 70; dist = 3
        L_plot_rad = np.deg2rad(90 - L_bearing_deg)
        L = (dist * np.cos(L_plot_rad), dist * np.sin(L_plot_rad))

        ax.plot(S[0], S[1], 'bo', ms=5); ax.text(S[0]-0.1, S[1]-0.2, 'S (Ship)')
        ax.plot(L[0], L[1], 'ro', ms=5); ax.text(L[0]+0.1, L[1]+0.1, 'L (Lighthouse)')
        ax.plot([S[0], L[0]], [S[1], L[1]], 'k--')

        # North lines & Angles
        north_angle_deg = 90
        ax.plot([S[0], S[0]], [S[1], S[1]+dist*0.5], 'b-', lw=1); ax.text(S[0], S[1]+dist*0.5 + 0.1, 'N', color='blue')
        self._add_angle_arc(ax, S, 0.6, north_angle_deg, 90 - L_bearing_deg, color='blue', label=f'{L_bearing_deg:03}°', label_pos_factor=1.3)

        ax.plot([L[0], L[0]], [L[1], L[1]+dist*0.5], 'r-', lw=1); ax.text(L[0], L[1]+dist*0.5 + 0.1, 'N', color='red')
        S_back_bearing_deg = L_bearing_deg + 180 # = 250
        self._add_angle_arc(ax, L, 0.6, north_angle_deg, 90 - S_back_bearing_deg, color='red', label=f'{S_back_bearing_deg:03}°', label_pos_factor=1.3)

        ax.set_xlim(-1, dist + 1); ax.set_ylim(-1, dist + 1)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off')
        ax.set_title('Bearing (L from S) vs Back Bearing (S from L)')
        self._save_plot(fig, "bearing_back_bearing_70_250.png")

    def bearing_angle_between_paths_120_210(self):
        """Generates diagram for angle between two bearings (120, 210) from A."""
        fig, ax = plt.subplots(figsize=(6, 6))
        A = (0, 0); bearing_B_deg = 120; bearing_C_deg = 210; dist = 3
        plot_B_rad = np.deg2rad(90 - bearing_B_deg); plot_C_rad = np.deg2rad(90 - bearing_C_deg)
        B = (dist * np.cos(plot_B_rad), dist * np.sin(plot_B_rad))
        C = (dist * 0.8 * np.cos(plot_C_rad), dist * 0.8 * np.sin(plot_C_rad)) # Make C closer

        ax.plot(A[0], A[1], 'ko'); ax.text(A[0]-0.1, A[1]-0.2, 'A')
        ax.plot(B[0], B[1], 'bo'); ax.text(B[0]+0.1, B[1], 'B')
        ax.plot(C[0], C[1], 'ro'); ax.text(C[0]-0.2, C[1], 'C')
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b-'); ax.plot([A[0], C[0]], [A[1], C[1]], 'r-')

        # North line & Bearing Arcs
        ax.plot([A[0], A[0]], [A[1], A[1]+dist*0.5], 'k-', lw=1); ax.text(A[0], A[1]+dist*0.5 + 0.1, 'N')
        north_angle_deg = 90
        self._add_angle_arc(ax, A, 1.0, north_angle_deg, 90 - bearing_B_deg, color='blue', label=f'{bearing_B_deg:03}°', label_pos_factor=1.2)
        self._add_angle_arc(ax, A, 1.3, north_angle_deg, 90 - bearing_C_deg, color='red', label=f'{bearing_C_deg:03}°', label_pos_factor=1.15)

        # Angle BAC
        angle_BAC_deg = bearing_C_deg - bearing_B_deg
        self._add_angle_arc(ax, A, 0.6, 90 - bearing_B_deg, 90 - bearing_C_deg, color='green', linestyle='-', label=f'{angle_BAC_deg}°', label_pos_factor=1.2)

        # Adjust limits based on points
        all_x = [A[0], B[0], C[0]]; all_y = [A[1], B[1], C[1]]
        ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
        ax.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off'); ax.set_title('Angle Between Bearings from A')
        self._save_plot(fig, "bearing_angle_between_paths_120_210.png")

    # --- Form 2: Geometry - Polygons (Triangles/Quadrilaterals) ---
    # Placeholder methods - these would need actual drawing logic
    def triangle_types(self):
        """Placeholder: Generates diagrams of different triangle types."""
        print("Placeholder function called: triangle_types")
        # Example: Draw equilateral, isosceles, scalene, right-angled
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        fig.suptitle("Triangle Types (Placeholder)")
        types = ["Equilateral", "Isosceles", "Scalene", "Right-Angled"]
        for i, title in enumerate(types):
            ax[i].text(0.5, 0.5, title, ha='center', va='center')
            ax[i].set_xticks([]); ax[i].set_yticks([])
            ax[i].set_title(title)
        self._save_plot(fig, "triangle_types_placeholder.png")

    def quadrilateral_types(self):
        """Placeholder: Generates diagrams of different quadrilateral types."""
        print("Placeholder function called: quadrilateral_types")
        fig, ax = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle("Quadrilateral Types (Placeholder)")
        types = ["Square", "Rectangle", "Parallelogram", "Rhombus", "Trapezium", "Kite"]
        ax = ax.ravel()
        for i, title in enumerate(types):
             ax[i].text(0.5, 0.5, title, ha='center', va='center')
             ax[i].set_xticks([]); ax[i].set_yticks([])
             ax[i].set_title(title)
        if len(types) < len(ax): # Turn off unused axes
            for i in range(len(types), len(ax)): ax[i].axis('off')
        self._save_plot(fig, "quadrilateral_types_placeholder.png")

    # --- Form 2: Geometry - Congruence and Similarity ---
    # Placeholder methods
    def similar_triangles_abc_pqr(self):
        """Placeholder: Generates diagram of similar triangles ABC and PQR."""
        print("Placeholder function called: similar_triangles_abc_pqr")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Similar Triangles\nABC ~ PQR\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "similar_triangles_abc_pqr.png")

    def congruent_triangles_def_stu(self):
        """Placeholder: Generates diagram of congruent triangles DEF and STU."""
        print("Placeholder function called: congruent_triangles_def_stu")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Congruent Triangles\nDEF ≅ STU\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "congruent_triangles_def_stu.png")

    def similar_rectangles_4x6_6x9(self):
        """Placeholder: Generates diagram of similar rectangles."""
        print("Placeholder function called: similar_rectangles_4x6_6x9")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Similar Rectangles\n4x6 ~ 6x9\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "similar_rectangles_4x6_6x9.png")

    def congruent_triangles_sss(self):
        """Placeholder: Illustrates SSS congruence."""
        print("Placeholder function called: congruent_triangles_sss")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "SSS Congruence\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "congruent_triangles_sss.png")

    def congruent_triangles_sas(self):
        """Placeholder: Illustrates SAS congruence."""
        print("Placeholder function called: congruent_triangles_sas")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "SAS Congruence\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "congruent_triangles_sas.png")

    def similar_triangles_lmn_rst_lengths(self):
        """Placeholder: Similar triangles with lengths for ratio calculation."""
        print("Placeholder function called: similar_triangles_lmn_rst_lengths")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Similar Triangles\nLengths for Ratios\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "similar_triangles_lmn_rst_lengths.png")

    def similar_triangles_nested_pst_pqr(self):
        """Placeholder: Nested similar triangles."""
        print("Placeholder function called: similar_triangles_nested_pst_pqr")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Nested Similar Triangles\nPST ~ PQR\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "similar_triangles_nested_pst_pqr.png")

    # --- Form 2: Geometry - Constructions (Placeholders) ---
    def construction_perp_bisector(self):
        """Placeholder: Construction of a perpendicular bisector."""
        print("Placeholder function called: construction_perp_bisector")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Construction:\nPerpendicular Bisector\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "construction_perp_bisector.png")

    def construction_angle_bisector(self):
        """Placeholder: Construction of an angle bisector."""
        print("Placeholder function called: construction_angle_bisector")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Construction:\nAngle Bisector\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "construction_angle_bisector.png")

    def construction_perp_bisector_8cm(self):
        """Placeholder: Construction of perpendicular bisector of 8cm line."""
        print("Placeholder function called: construction_perp_bisector_8cm")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Construction:\nPerpendicular Bisector\nof 8cm Line\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "construction_perp_bisector_8cm.png")

    def construction_60_degrees(self):
        """Placeholder: Construction of a 60 degree angle."""
        print("Placeholder function called: construction_60_degrees")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Construction:\n60 Degree Angle\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "construction_60_degrees.png")

    def angle_approx_80_deg(self):
        """Placeholder: Drawing an approximate 80 degree angle."""
        print("Placeholder function called: angle_approx_80_deg")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Approximate 80° Angle\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "angle_approx_80_deg.png")

    def construction_bisect_angle_80(self):
        """Placeholder: Construction bisecting an 80 degree angle."""
        print("Placeholder function called: construction_bisect_angle_80")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Construction:\nBisect ~80° Angle\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "construction_bisect_angle_80.png")

    def construction_90_degrees_on_line(self):
        """Placeholder: Construction of 90 degrees at a point on a line."""
        print("Placeholder function called: construction_90_degrees_on_line")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Construction:\n90° Angle on Line\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "construction_90_degrees_on_line.png")

    def construction_30_degrees(self):
        """Placeholder: Construction of a 30 degree angle (bisect 60)."""
        print("Placeholder function called: construction_30_degrees")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Construction:\n30 Degree Angle\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "construction_30_degrees.png")

    # --- Form 2: Geometry - Symmetry (Placeholders/Simple Examples) ---
    def lines_of_symmetry_examples(self):
        """Placeholder: Examples of lines of symmetry."""
        print("Placeholder function called: lines_of_symmetry_examples")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Lines of Symmetry Examples\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "lines_of_symmetry_examples.png")

    def symmetry_square_4lines(self):
        """Placeholder: Square with 4 lines of symmetry."""
        print("Placeholder function called: symmetry_square_4lines")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Square: 4 Lines\nof Symmetry\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "symmetry_square_4lines.png")

    def symmetry_rectangle_2lines(self):
        """Placeholder: Rectangle with 2 lines of symmetry."""
        print("Placeholder function called: symmetry_rectangle_2lines")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Rectangle: 2 Lines\nof Symmetry\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "symmetry_rectangle_2lines.png")

    def symmetry_letter_h(self):
        """Placeholder: Letter H with lines of symmetry."""
        print("Placeholder function called: symmetry_letter_h")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Letter H: 2 Lines\nof Symmetry\n(Placeholder)", ha='center', va='center', fontsize=20, family='monospace')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "symmetry_letter_h.png")

    def symmetry_isosceles_triangle(self):
        """Placeholder: Isosceles triangle with 1 line of symmetry."""
        print("Placeholder function called: symmetry_isosceles_triangle")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Isosceles Triangle:\n1 Line of Symmetry\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "symmetry_isosceles_triangle.png")

    def symmetry_regular_pentagon(self):
        """Placeholder: Regular pentagon with 5 lines of symmetry."""
        print("Placeholder function called: symmetry_regular_pentagon")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Regular Pentagon:\n5 Lines of Symmetry\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "symmetry_regular_pentagon.png")

    # --- Form 2: Statistics - Charts ---
    def barchart_favorite_fruits(self):
        """Generates a bar chart for favorite fruits."""
        print("Generating: barchart_favorite_fruits")
        fruits = ['Apple', 'Banana', 'Orange', 'Mango', 'Grapes']
        counts = [25, 35, 20, 40, 30]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(fruits, counts, color=['red', 'yellow', 'orange', 'gold', 'purple'])
        ax.set_ylabel('Number of Students')
        ax.set_title('Favorite Fruits')
        ax.bar_label(bars, label_type='edge')
        ax.yaxis.set_major_locator(MultipleLocator(5)) # Ticks every 5 units
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        self._save_plot(fig, "barchart_favorite_fruits.png")

    def piechart_item_types(self):
        """Generates a pie chart for item types in a shop."""
        print("Generating: piechart_item_types")
        types = ['Books', 'Stationery', 'Toys', 'Gifts']
        counts = [45, 30, 15, 10]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        explode = (0, 0.1, 0, 0) # Explode 'Stationery' slice
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts, explode=explode, labels=types, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.axis('equal') # Equal aspect ratio ensures circle.
        ax.set_title('Item Types in Shop')
        self._save_plot(fig, "piechart_item_types.png")

    def barchart_pets_owned(self):
        """Generates a bar chart for pets owned by students."""
        print("Generating: barchart_pets_owned")
        pets = ['Dog', 'Cat', 'Fish', 'Bird', 'None']
        numbers = [12, 15, 8, 5, 10]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(pets, numbers, color='lightblue')
        ax.set_xlabel('Type of Pet')
        ax.set_ylabel('Number of Students')
        ax.set_title('Pets Owned by Students')
        ax.bar_label(bars)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        self._save_plot(fig, "barchart_pets_owned.png")

    def piechart_pets_owned(self):
        """Generates a pie chart for pets owned by students."""
        print("Generating: piechart_pets_owned")
        pets = ['Dog', 'Cat', 'Fish', 'Bird', 'None']
        numbers = [12, 15, 8, 5, 10]
        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(numbers, labels=pets, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        ax.axis('equal')
        ax.set_title('Distribution of Pets Owned')
        # Improve text contrast if needed
        # for autotext in autotexts:
        #     autotext.set_color('white')
        #     autotext.set_fontweight('bold')
        self._save_plot(fig, "piechart_pets_owned.png")

    # --- Form 2: Vectors (Basic Concepts) ---
    # Placeholders or simple examples
    def vector_notation_examples(self):
        """Placeholder: Illustrates different vector notations."""
        print("Placeholder function called: vector_notation_examples")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Vector Notations:\n-> AB\n-> u\n-> (3, 4)\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "vector_notation_examples.png")

    def vector_equal_ab_cd(self):
        """Placeholder: Diagram showing equal vectors AB and CD."""
        print("Placeholder function called: vector_equal_ab_cd")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Equal Vectors\nAB = CD\n(Same magnitude & direction)\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "vector_equal_ab_cd.png")

    def vector_negative_a_neg_a(self):
        """Placeholder: Diagram showing vector a and its negative -a."""
        print("Placeholder function called: vector_negative_a_neg_a")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Negative Vector\na and -a\n(Same magnitude, opp. direction)\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "vector_negative_a_neg_a.png")

    def vector_draw_neg3_1_from_2_2(self):
        """Draws vector (-3, 1) starting from point (2, 2)."""
        print("Generating: vector_draw_neg3_1_from_2_2")
        fig, ax = plt.subplots(figsize=(6, 6))
        start = np.array([2, 2])
        vec = np.array([-3, 1])
        end = start + vec
        ax.quiver(start[0], start[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color='r', label='v = (-3, 1)')
        ax.plot(start[0], start[1], 'bo', label='Start (2, 2)')
        ax.plot(end[0], end[1], 'ro', markerfacecolor='none', label=f'End ({end[0]}, {end[1]})') # Mark end point
        ax.set_xlim(-2, 4); ax.set_ylim(0, 5)
        ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Vector (-3, 1) from (2, 2)')
        ax.grid(True, linestyle=':'); ax.legend(); ax.set_aspect('equal', adjustable='box')
        self._save_plot(fig, "vector_draw_-3_1_from_2_2.png") # Corrected filename

    def vector_identify_equal_grid(self):
        """Placeholder: Grid with vectors to identify equal ones."""
        print("Placeholder function called: vector_identify_equal_grid")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Identify Equal Vectors\non Grid\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "vector_identify_equal_grid.png")

    def vector_addition_laws(self):
        """Placeholder: Illustrates triangle and parallelogram laws of vector addition."""
        print("Placeholder function called: vector_addition_laws")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Vector Addition Laws\n(Triangle & Parallelogram)\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "vector_addition_laws.png")

    def vector_subtraction_laws(self):
        """Placeholder: Illustrates vector subtraction a - b."""
        print("Placeholder function called: vector_subtraction_laws")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Vector Subtraction\na - b = a + (-b)\n(Placeholder)", ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        self._save_plot(fig, "vector_subtraction_laws.png")

    # --- Form 3: Real Numbers ---
    # Diagrams are less common here, maybe simple number line or Venn for sets

    # --- Form 3: Financial Maths ---
    def simple_vs_compound_interest(self):
        """Compares simple and compound interest growth over time."""
        print("Generating: simple_vs_compound_interest")
        principal = 1000
        rate = 0.05 # 5%
        years = np.arange(0, 11) # 0 to 10 years

        simple_interest = principal * (1 + rate * years)
        compound_interest = principal * (1 + rate) ** years

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(years, simple_interest, 'b-o', label=f'Simple Interest (Rate={rate*100}%)')
        ax.plot(years, compound_interest, 'r-s', label=f'Compound Interest (Rate={rate*100}%)')

        ax.set_xlabel('Years')
        ax.set_ylabel('Total Amount ($)')
        ax.set_title(f'Simple vs Compound Interest (Principal=${principal})')
        ax.legend()
        ax.grid(True, linestyle=':')
        # Format y-axis as currency
        formatter = FuncFormatter(lambda y, p: f'${y:,.0f}')
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(MultipleLocator(1)) # Ensure integer years are ticked

        self._save_plot(fig, "simple_vs_compound_interest.png")

    # --- Form 3: Geometry - Area and Volume ---
    def combined_shape_rect_semicircle(self):
        """Draws a combined shape: rectangle with a semicircle on top."""
        print("Generating: combined_shape_rect_semicircle")
        fig, ax = plt.subplots(figsize=(5, 6))
        rect_width = 4
        rect_height = 3
        radius = rect_width / 2

        # Rectangle
        rectangle = patches.Rectangle((0, 0), rect_width, rect_height, linewidth=1, edgecolor='k', facecolor='lightblue')
        ax.add_patch(rectangle)
        ax.text(rect_width/2, rect_height/2, "Rectangle", ha='center', va='center')

        # Semicircle
        center_x = rect_width / 2
        center_y = rect_height
        semicircle = patches.Arc((center_x, center_y), rect_width, 2 * radius, angle=0, theta1=0, theta2=180, linewidth=1, edgecolor='k', facecolor='lightcoral')
        ax.add_patch(semicircle)
        ax.text(center_x, center_y + radius*0.5, "Semicircle", ha='center', va='center')

        # Dimensions
        ax.plot([0, rect_width], [-0.2, -0.2], 'k-') # Plot the line
        ax.plot([0, rect_width], [-0.2, -0.2], 'k|', markersize=10) # Plot the end markers
        ax.text(rect_width/2, -0.4, f'Width = {rect_width}', ha='center', va='top')
        ax.plot([-0.2, -0.2], [0, rect_height], 'k-') # Plot the line
        ax.plot([-0.2, -0.2], [0, rect_height], 'k|', markersize=10) # Plot the end markers
        ax.text(-0.4, rect_height/2, f'Height = {rect_height}', ha='left', va='center', rotation=90)
        ax.plot([center_x, center_x], [center_y, center_y+radius], 'r--')
        ax.text(center_x+0.1, center_y+radius/2, f'Radius = {radius}', va='center', color='r')

        ax.set_xlim(-1, rect_width + 1)
        ax.set_ylim(-1, rect_height + radius + 1)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title('Combined Shape: Rectangle + Semicircle')
        self._save_plot(fig, "combined_shape_rect_semicircle.png")

    def cylinder_volume_formula(self):
        """Illustrates a cylinder with its volume formula."""
        print("Generating: cylinder_volume_formula")
        fig = plt.figure(figsize=(6, 7))
        ax = fig.add_subplot(111, projection='3d')

        radius = 1.5
        height = 4

        # Cylinder Body
        u = np.linspace(0, 2 * np.pi, 50) # Angle
        h = np.linspace(0, height, 20) # Height points
        x = radius * np.outer(np.cos(u), np.ones_like(h))
        y = radius * np.outer(np.sin(u), np.ones_like(h))
        z = np.outer(np.ones_like(u), h)
        ax.plot_surface(x, y, z, alpha=0.3, color='lightblue', rstride=1, cstride=1, edgecolor='k', linewidth=0.5)

        # Top and Bottom Circles
        u_circle = np.linspace(0, 2 * np.pi, 50)
        x_circle = radius * np.cos(u_circle)
        y_circle = radius * np.sin(u_circle)
        ax.plot(x_circle, y_circle, 0, color='k', lw=1) # Bottom
        ax.plot(x_circle, y_circle, height, color='k', lw=1) # Top
        # Fill circles (optional, makes it look solid)
        ax.add_collection3d(plt.fill_between(x_circle, y_circle, 0, color='lightblue', alpha=0.3))
        ax.add_collection3d(plt.fill_between(x_circle, y_circle, height, color='lightblue', alpha=0.3))


        # Annotations (radius and height) - harder in 3D
        ax.plot([0, radius], [0, 0], [0, 0], 'r-', lw=1.5) # Radius line on base
        ax.text(radius/2, 0.1, 0, 'r', color='r', size=12)
        ax.plot([0, 0], [0, 0], [0, height], 'g-', lw=1.5) # Height line
        ax.text(0.1, 0.1, height/2, 'h', color='g', size=12)

        # Formula
        ax.text(radius*1.5, 0, height*0.8, r'$V = \pi r^2 h$', size=14, color='blue', zdir='y')

        # Adjust view
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Cylinder Volume')
        ax.view_init(elev=20., azim=30) # Adjust viewing angle
        ax.set_box_aspect([1, 1, height/radius/1.5]) # Adjust aspect ratio
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]) # Cleaner look
        ax.axis('off') # Even cleaner

        self._save_plot(fig, "cylinder_volume_formula.png")

    def combined_shape_rect_minus_semicircle(self):
        """Draws a shape: rectangle with a semicircle removed."""
        print("Generating: combined_shape_rect_minus_semicircle")
        fig, ax = plt.subplots(figsize=(6, 5))
        rect_width = 5
        rect_height = 3
        radius = rect_width / 2 # Semicircle cut from top edge

        # Outer Rectangle boundary (we only fill the part we want)
        ax.plot([0, rect_width, rect_width, 0, 0], [0, 0, rect_height, rect_height, 0], 'k-', lw=0.5, alpha=0.5) # Outline

        # Area to fill: Need to define the boundary including the arc
        x_arc = np.linspace(0, rect_width, 100)
        y_arc = rect_height - np.sqrt(radius**2 - (x_arc - radius)**2) # Equation of bottom half of circle shifted up

        x_fill = np.concatenate(([0], x_arc, [rect_width, 0]))
        y_fill = np.concatenate(([0], y_arc, [0, 0]))
        ax.fill(x_fill, y_fill, color='lightblue', alpha=0.7)

        # Indicate removed part
        center_x = rect_width / 2
        center_y = rect_height
        semicircle_removed = patches.Arc((center_x, center_y), rect_width, 2 * radius, angle=0, theta1=180, theta2=360, linewidth=1, edgecolor='r', linestyle='--', facecolor='none', label='Removed')
        ax.add_patch(semicircle_removed)

        ax.text(rect_width/2, rect_height/2, "Remaining Area", ha='center', va='center')

        ax.set_xlim(-1, rect_width + 1)
        ax.set_ylim(-1, rect_height + 1)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off')
        ax.set_title('Combined Shape: Rectangle - Semicircle')
        ax.legend(loc='upper right')
        self._save_plot(fig, "combined_shape_rect_minus_semicircle.png")

    def combined_shape_square_plus_triangle(self):
        """Draws a combined shape: square with an equilateral triangle on top."""
        print("Generating: combined_shape_square_plus_triangle")
        fig, ax = plt.subplots(figsize=(5, 7))
        side = 4
        tri_height = side * np.sqrt(3) / 2

        # Square
        square = patches.Rectangle((0, 0), side, side, linewidth=1, edgecolor='k', facecolor='lightgreen')
        ax.add_patch(square)
        ax.text(side/2, side/2, "Square", ha='center', va='center')

        # Equilateral Triangle
        tri_points = [(0, side), (side, side), (side/2, side + tri_height)]
        triangle = patches.Polygon(tri_points, linewidth=1, edgecolor='k', facecolor='lightcoral')
        ax.add_patch(triangle)
        ax.text(side/2, side + tri_height/3, "Equilateral\nTriangle", ha='center', va='center')

        # Dimensions
        ax.plot([0, side], [-0.2, -0.2], 'k-') # Plot the line
        ax.plot([0, side], [-0.2, -0.2], 'k|', markersize=10) # Plot the end markers
        ax.text(side/2, -0.4, f'Side = {side}', ha='center', va='top') # Text for width dimension

        ax.plot([-0.2, -0.2], [0, side], 'k-') # Plot the line
        ax.plot([-0.2, -0.2], [0, side], 'k|', markersize=10) # Plot the end markers
        ax.text(-0.4, side/2, f'Side = {side}', ha='left', va='center', rotation=90) # Text for height dimension

        ax.set_xlim(-1, side + 1)
        ax.set_ylim(-1, side + tri_height + 1)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off')
        ax.set_title('Combined Shape: Square + Triangle')
        self._save_plot(fig, "combined_shape_square_plus_triangle.png")

    # --- Form 3: Graphs ---
    def _setup_cartesian_axes(self, ax, xlim, ylim, title, xlabel='x', ylabel='y'):
        """Helper to set up standard Cartesian axes."""
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.spines['left'].set_position('zero'); ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left')
        ax.set_xlabel(xlabel, loc='right'); ax.set_ylabel(ylabel, loc='top', rotation=0)
        ax.set_title(title); ax.grid(True, linestyle=':')
        # Add arrows to axes
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)


    def graph_linear_y_2x_minus_4(self):
        """Graphs the linear function y = 2x - 4."""
        print("Generating: graph_linear_y_2x_minus_4")
        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.linspace(-2, 5, 100)
        y = 2*x - 4
        self._setup_cartesian_axes(ax, (-2, 5), (-6, 7), 'Linear Graph: y = 2x - 4')
        ax.plot(x, y, 'b-', label='y = 2x - 4')
        # Intercepts
        x_intercept = 2; y_intercept = -4
        ax.plot(x_intercept, 0, 'ro', label=f'x-int ({x_intercept}, 0)')
        ax.plot(0, y_intercept, 'go', label=f'y-int (0, {y_intercept})')
        ax.legend()
        self._save_plot(fig, "graph_linear_y_2x_minus_4.png")

    def graph_quadratic_y_x_squared(self):
        """Graphs the basic quadratic function y = x^2."""
        print("Generating: graph_quadratic_y_x_squared")
        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.linspace(-3, 3, 100)
        y = x**2
        self._setup_cartesian_axes(ax, (-3, 3), (-1, 9), 'Quadratic Graph: y = x²')
        ax.plot(x, y, 'r-', label='y = x²')
        ax.plot(0, 0, 'ko', label='Vertex (0, 0)')
        ax.legend()
        self._save_plot(fig, "graph_quadratic_y_x_squared.png")

    def graph_linear_y_6_minus_2x(self):
        """Graphs the linear function y = 6 - 2x."""
        print("Generating: graph_linear_y_6_minus_2x")
        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.linspace(-1, 5, 100)
        y = 6 - 2*x
        self._setup_cartesian_axes(ax, (-1, 5), (-2, 8), 'Linear Graph: y = 6 - 2x')
        ax.plot(x, y, 'g-', label='y = 6 - 2x')
        x_intercept = 3; y_intercept = 6
        ax.plot(x_intercept, 0, 'ro', label=f'x-int ({x_intercept}, 0)')
        ax.plot(0, y_intercept, 'bo', label=f'y-int (0, {y_intercept})')
        ax.legend()
        self._save_plot(fig, "graph_linear_y_6_minus_2x.png")

    def graph_quadratic_y_x2_minus_2x_minus_3(self):
        """Graphs the quadratic function y = x^2 - 2x - 3."""
        print("Generating: graph_quadratic_y_x2_minus_2x_minus_3")
        fig, ax = plt.subplots(figsize=(7, 7))
        x = np.linspace(-3, 5, 400)
        y = x**2 - 2*x - 3
        self._setup_cartesian_axes(ax, (-3, 5), (-5, 15), 'Quadratic Graph: y = x² - 2x - 3')
        ax.plot(x, y, 'm-', label='y = x² - 2x - 3')
        # Find intercepts and vertex
        x_intercept1 = -1; x_intercept2 = 3
        y_intercept = -3
        vertex_x = -(-2) / (2*1) # -b/2a
        vertex_y = vertex_x**2 - 2*vertex_x - 3
        ax.plot([x_intercept1, x_intercept2], [0, 0], 'ro', label=f'x-int ({x_intercept1},0), ({x_intercept2},0)')
        ax.plot(0, y_intercept, 'bo', label=f'y-int (0, {y_intercept})')
        ax.plot(vertex_x, vertex_y, 'ko', label=f'Vertex ({vertex_x:.1f}, {vertex_y:.1f})')
        ax.legend()
        self._save_plot(fig, "graph_quadratic_y_x2_minus_2x_minus_3.png")

    def graph_sketch_quadratic_y_x2_minus_4x_plus_3(self):
        """Sketches y = x^2 - 4x + 3, showing intercepts and vertex."""
        print("Generating: graph_sketch_quadratic_y_x2_minus_4x_plus_3")
        fig, ax = plt.subplots(figsize=(6, 6))
        # Calculate key points
        # y = (x-1)(x-3) -> x-int: 1, 3
        # y-int (x=0): 3
        # Vertex x = -(-4)/(2*1) = 2 -> y = 2^2 - 4*2 + 3 = 4 - 8 + 3 = -1
        x_intercepts = [1, 3]; y_intercept = 3; vertex = (2, -1)

        # Generate plot points around key features
        x = np.linspace(min(x_intercepts) - 1, max(x_intercepts) + 1, 100)
        y = x**2 - 4*x + 3

        self._setup_cartesian_axes(ax, (0, 4), (-2, 5), 'Sketch: y = x² - 4x + 3')
        ax.plot(x, y, 'c-', label='y = x² - 4x + 3')
        ax.plot(x_intercepts, [0, 0], 'ro', label=f'x-int ({x_intercepts[0]},0), ({x_intercepts[1]},0)')
        ax.plot(0, y_intercept, 'bo', label=f'y-int (0, {y_intercept})')
        ax.plot(vertex[0], vertex[1], 'ko', label=f'Vertex ({vertex[0]}, {vertex[1]})')
        ax.legend()
        self._save_plot(fig, "graph_sketch_quadratic_y_x2_minus_4x_plus_3.png")

    def graph_solve_quadratic_y_equals_2(self):
        """Solves x^2 - 2x - 3 = 2 graphically."""
        print("Generating: graph_solve_quadratic_y_equals_2")
        fig, ax = plt.subplots(figsize=(7, 7))
        x = np.linspace(-3, 5, 400)
        y_quad = x**2 - 2*x - 3
        y_line = np.full_like(x, 2) # Horizontal line y=2
        self._setup_cartesian_axes(ax, (-3, 5), (-5, 15), 'Solve x² - 2x - 3 = 2 Graphically')
        ax.plot(x, y_quad, 'm-', label='y = x² - 2x - 3')
        ax.plot(x, y_line, 'k--', label='y = 2')
        # Solve x^2 - 2x - 5 = 0 for intersection points
        # x = [2 +/- sqrt(4 - 4*1*(-5))] / 2 = [2 +/- sqrt(24)] / 2 = 1 +/- sqrt(6)
        x_sol1 = 1 - np.sqrt(6)
        x_sol2 = 1 + np.sqrt(6)
        ax.plot([x_sol1, x_sol2], [2, 2], 'ro', markersize=7, label=f'Intersections at x ≈ {x_sol1:.2f}, {x_sol2:.2f}')
        ax.vlines([x_sol1, x_sol2], ymin=-5, ymax=2, color='r', linestyle=':', lw=1) # Indicate solutions on x-axis
        ax.legend()
        self._save_plot(fig, "graph_solve_quadratic_y_equals_2.png")

    # --- Form 3: Travel Graphs ---
    def travel_graphs_dist_time_vs_speed_time(self):
        """Compares typical Distance-Time and Speed-Time graphs."""
        print("Generating: travel_graphs_dist_time_vs_speed_time")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Travel Graph Concepts")

        # Distance-Time Example (Constant Speed, Stop, Constant Speed)
        time1 = [0, 2, 4, 7]; dist1 = [0, 40, 40, 100]
        ax1.plot(time1, dist1, 'b-o')
        ax1.set_xlabel("Time (hours)"); ax1.set_ylabel("Distance (km)")
        ax1.set_title("Distance-Time Graph"); ax1.grid(True)
        ax1.set_xticks(time1); ax1.set_yticks(np.arange(0, 110, 20))
        ax1.text(1, 20, "Constant Speed", ha='center')
        ax1.text(3, 42, "Stopped", ha='center')
        ax1.text(5.5, 70, "Constant Speed", ha='center')

        # Speed-Time Example (Accelerate, Constant Speed, Decelerate)
        time2 = [0, 2, 5, 7]; speed2 = [0, 30, 30, 0]
        ax2.plot(time2, speed2, 'r-s')
        ax2.set_xlabel("Time (seconds)"); ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Speed-Time Graph"); ax2.grid(True)
        ax2.set_xticks(time2); ax2.set_yticks(np.arange(0, 35, 5))
        ax2.text(1, 15, "Acceleration", ha='center')
        ax2.fill_between([0,7],[0,0],color='red', alpha=0.1) # Optional shade under graph
        ax2.text(3.5, 31, "Constant Speed", ha='center')
        ax2.text(6, 15, "Deceleration", ha='center')

        self._save_plot(fig, "travel_graphs_dist_time_vs_speed_time.png")

    def speed_time_graph_area_trapezium(self):
        """Illustrates distance as area under a speed-time graph (trapezium)."""
        print("Generating: speed_time_graph_area_trapezium")
        fig, ax = plt.subplots(figsize=(7, 5))
        time = [0, 2, 6, 8] # Example times for phases
        speed = [0, 20, 20, 0] # Corresponding speeds
        ax.plot(time, speed, 'b-o')
        ax.fill_between(time, speed, color='lightblue', alpha=0.5, label='Area = Distance Traveled')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
        ax.set_title("Distance from Speed-Time Graph (Area)")
        ax.set_xlim(0, 9); ax.set_ylim(0, 25)
        ax.grid(True, linestyle=':')
        # Annotate trapezium shape
        ax.text(4, 10, "Area = ½(a+b)h\n= ½( (6-2) + (8-0) ) * 20\n= ½(4 + 8) * 20 = 120 m", ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))
        ax.legend()
        self._save_plot(fig, "speed_time_graph_area_trapezium.png")

    # --- Form 3: Variation ---
    def direct_vs_inverse_variation_graphs(self):
        """Compares graphs of Direct (y=kx) and Inverse (y=k/x) Variation."""
        print("Generating: direct_vs_inverse_variation_graphs")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Variation Graph Shapes")
        k = 2 # Example constant

        # Direct Variation
        x1 = np.linspace(0, 5, 100)
        y1 = k * x1
        self._setup_cartesian_axes(ax1, (0, 5), (0, 12), f'Direct Variation (y = {k}x)')
        ax1.plot(x1, y1, 'b-', label=f'y = {k}x')
        ax1.legend()

        # Inverse Variation (avoiding x=0)
        x2 = np.linspace(0.2, 5, 100)
        y2 = k / x2
        self._setup_cartesian_axes(ax2, (0, 5), (0, 12), f'Inverse Variation (y = {k}/x)')
        ax2.plot(x2, y2, 'r-', label=f'y = {k}/x')
        ax2.legend()

        self._save_plot(fig, "direct_vs_inverse_variation_graphs.png")

    # --- Form 3: Trigonometry (Elevation/Depression) ---
    def angle_elevation(self):
        """Diagram illustrating the angle of elevation."""
        print("Generating: angle_elevation")
        fig, ax = plt.subplots(figsize=(6, 5))
        observer = (0, 0); object_pt = (4, 3) # Observer at origin, object higher
        # Ground line
        ax.axhline(0, color='grey', lw=1)
        ax.text(2, -0.2, "Horizontal Ground", ha='center', color='grey')
        # Observer and Object
        ax.plot(observer[0], observer[1], 'bo', ms=8, label='Observer (O)')
        ax.plot(object_pt[0], object_pt[1], 'rs', ms=8, label='Object (P)')
        # Line of sight
        ax.plot([observer[0], object_pt[0]], [observer[1], object_pt[1]], 'k--', label='Line of Sight')
        # Horizontal line from observer
        ax.plot([observer[0], object_pt[0]], [observer[1], observer[1]], 'b:')
        # Angle arc
        angle_rad = np.arctan2(object_pt[1] - observer[1], object_pt[0] - observer[0])
        self._add_angle_arc(ax, observer, 1.5, 0, np.degrees(angle_rad), color='green', linestyle='-', label='Angle of\nElevation (θ)', label_pos_factor=1.0)

        ax.set_xlim(-1, 5); ax.set_ylim(-1, 4)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off'); ax.set_title('Angle of Elevation')
        ax.legend(loc='lower right')
        self._save_plot(fig, "angle_elevation.png")

    def angle_depression(self):
        """Diagram illustrating the angle of depression."""
        print("Generating: angle_depression")
        fig, ax = plt.subplots(figsize=(6, 5))
        observer = (0, 3); object_pt = (4, 0) # Observer higher, object on ground
        # Ground line (where object is)
        ax.axhline(0, color='grey', lw=1)
        ax.text(2, -0.2, "Ground", ha='center', color='grey')
        # Observer and Object
        ax.plot(observer[0], observer[1], 'bo', ms=8, label='Observer (O)')
        ax.plot(object_pt[0], object_pt[1], 'rs', ms=8, label='Object (P)')
        # Line of sight
        ax.plot([observer[0], object_pt[0]], [observer[1], object_pt[1]], 'k--', label='Line of Sight')
        # Horizontal line from observer
        ax.plot([observer[0], object_pt[0]+1], [observer[1], observer[1]], 'b:', label='Horizontal')
        # Angle arc (measured DOWN from horizontal)
        angle_rad = np.arctan2(observer[1] - object_pt[1], object_pt[0] - observer[0]) # Angle relative to horizontal line towards object
        self._add_angle_arc(ax, observer, 1.5, 180-np.degrees(angle_rad), 180, color='red', linestyle='-', label='Angle of\nDepression (α)', label_pos_factor=1.0)

        ax.set_xlim(-1, 5); ax.set_ylim(-1, 4)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off'); ax.set_title('Angle of Depression')
        ax.legend(loc='lower left')
        self._save_plot(fig, "angle_depression.png")

    def elevation_depression_relationship(self):
        """Diagram showing relationship between elevation and depression (alternate angles)."""
        print("Generating: elevation_depression_relationship")
        fig, ax = plt.subplots(figsize=(7, 6))
        A = (0, 3) # Higher point (e.g., top of cliff)
        B = (5, 0) # Lower point (e.g., boat)
        # Ground/Sea level
        ax.axhline(0, color='grey', lw=1)
        # Horizontal at A
        ax.plot([-1, 6], [A[1], A[1]], 'b:', label='Horizontal at A')
        # Points
        ax.plot(A[0], A[1], 'bo', ms=7, label='A'); ax.text(A[0]-0.2, A[1]+0.1, 'A')
        ax.plot(B[0], B[1], 'ro', ms=7, label='B'); ax.text(B[0]+0.1, B[1]-0.2, 'B')
        # Line of Sight
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k--', label='Line of Sight')

        # Angle of Depression from A to B
        angle_rad = np.arctan2(A[1] - B[1], B[0] - A[0])
        self._add_angle_arc(ax, A, 1.5, 180-np.degrees(angle_rad), 180, color='red', label='Depression (α)', label_pos_factor=0.9)

        # Angle of Elevation from B to A
        self._add_angle_arc(ax, B, 1.5, 0, np.degrees(angle_rad), color='green', label='Elevation (θ)', label_pos_factor=0.9)

        ax.text(2.5, 1.5, "α = θ (Alternate Interior Angles)", ha='center', bbox=dict(fc="wheat", alpha=0.7))

        ax.set_xlim(-1, 6); ax.set_ylim(-1, 4)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off'); ax.set_title('Elevation vs Depression')
        ax.legend(loc='upper right', fontsize=8)
        self._save_plot(fig, "elevation_depression_relationship.png")

    # --- Form 3: Bearing (Reusing Form 2 concepts) ---
    def bearing_back_bearing_relation(self):
        """Re-illustrates bearing/back bearing relationship (general)."""
        # Can reuse bearing_back_bearing_70_250 or create a more generic one
        print("Generating: bearing_back_bearing_relation (using 70/250 example)")
        self.bearing_back_bearing_70_250() # Call the existing specific example
        # Rename the output file if desired, e.g. by modifying _save_plot temporarily or copying the code
        # os.rename(os.path.join(self.output_dir,"bearing_back_bearing_70_250.png"), os.path.join(self.output_dir,"bearing_back_bearing_relation.png"))


    # --- Form 3: Geometry - Polygons ---
    def polygon_exterior_angle(self):
        """Illustrates an exterior angle of a polygon."""
        print("Generating: polygon_exterior_angle")
        fig, ax = plt.subplots(figsize=(6, 5))
        # Draw part of a polygon (e.g., pentagon vertex)
        p1 = (0, 0); p2 = (2, 0); p3 = (2.8, 1.5) # Three consecutive vertices
        ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], 'k-o', lw=2)
        # Extend one side
        extend_len = 1.5
        vec = np.array(p2) - np.array(p1)
        norm_vec = vec / np.linalg.norm(vec)
        p_ext = np.array(p2) + norm_vec * extend_len
        ax.plot([p2[0], p_ext[0]], [p2[1], p_ext[1]], 'k:') # Extended line

        # Interior Angle
        vec1 = np.array(p1) - np.array(p2)
        vec2 = np.array(p3) - np.array(p2)
        angle1_rad = np.arctan2(vec1[1], vec1[0])
        angle2_rad = np.arctan2(vec2[1], vec2[0])
        self._add_angle_arc(ax, p2, 0.6, np.degrees(angle1_rad), np.degrees(angle2_rad), color='blue', linestyle='-', label='Interior\nAngle')

        # Exterior Angle
        angle_ext_rad = np.arctan2(vec2[1], vec2[0]) # Angle of the next side relative to horizontal
        self._add_angle_arc(ax, p2, 0.8, 0, np.degrees(angle_ext_rad), color='red', linestyle='-', label='Exterior\nAngle')

        ax.text(1, -0.3, "Polygon Side")
        ax.text(p_ext[0]+0.1, p_ext[1], "Extended Side")

        ax.set_xlim(-1, p_ext[0]+0.5); ax.set_ylim(-1, p3[1]+0.5)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off'); ax.set_title('Polygon Exterior Angle')
        ax.legend(fontsize=8)
        self._save_plot(fig, "polygon_exterior_angle.png")


    # --- Form 3: Constructions (Placeholders) ---
    def construction_sss_triangle_pqr(self):
        """Placeholder: Construct triangle PQR given 3 sides."""
        print("Placeholder function called: construction_sss_triangle_pqr")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "SSS Triangle Construction\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "construction_sss_triangle_pqr.png")

    def construction_sas_triangle_abc(self):
        """Placeholder: Construct triangle ABC given Side-Angle-Side."""
        print("Placeholder function called: construction_sas_triangle_abc")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "SAS Triangle Construction\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "construction_sas_triangle_abc.png")

    def construction_asa_triangle_xyz(self):
        """Placeholder: Construct triangle XYZ given Angle-Side-Angle."""
        print("Placeholder function called: construction_asa_triangle_xyz")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "ASA Triangle Construction\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "construction_asa_triangle_xyz.png")

    def construction_rectangle_7x4(self):
        """Placeholder: Construct a 7cm by 4cm rectangle."""
        print("Placeholder function called: construction_rectangle_7x4")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Rectangle Construction (7x4)\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "construction_rectangle_7x4.png")

    def construction_parallelogram_pqrs(self):
        """Placeholder: Construct a parallelogram PQRS."""
        print("Placeholder function called: construction_parallelogram_pqrs")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Parallelogram Construction\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "construction_parallelogram_pqrs.png")

    # --- Form 3: Symmetry (Rotational) ---
    def rotational_symmetry_examples(self):
        """Placeholder: Examples of rotational symmetry."""
        print("Placeholder function called: rotational_symmetry_examples")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Rotational Symmetry Examples\n(Order, Angle)\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "rotational_symmetry_examples.png")

    def symmetry_letter_s(self):
        """Illustrates rotational symmetry of letter S."""
        print("Generating: symmetry_letter_s")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "S", ha='center', va='center', fontsize=100, family='sans-serif', fontweight='bold')
        # Indicate center of rotation
        ax.plot(0.5, 0.5, 'ro', ms=5)
        ax.text(0.5, 0.5, " Center\n(Order 2)", ha='center', va='bottom', fontsize=8, color='red')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Rotational Symmetry: Letter S (Order 2)")
        ax.axis('off')
        self._save_plot(fig, "symmetry_letter_s.png")

    # --- Form 3: Statistics (Frequency Distributions) ---
    def histogram_student_marks(self):
        """Generates a histogram of student marks."""
        print("Generating: histogram_student_marks")
        # Example data (assuming marks are grouped)
        marks_lower = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        marks_upper = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        frequencies = [2, 5, 8, 12, 15, 10, 6, 4, 2, 1]
        # Bins are the edges
        bins = marks_lower + [marks_upper[-1]]
        # We need weights for hist() if data isn't raw
        # Or use bar chart with explicit widths
        fig, ax = plt.subplots(figsize=(8, 5))
        # Calculate bin centers for labels if needed, but bar works better here
        bar_centers = [(l+u)/2 for l,u in zip(marks_lower, marks_upper)]
        bar_widths = [(u-l) for l,u in zip(marks_lower, marks_upper)]
        bars = ax.bar(marks_lower, frequencies, width=bar_widths, align='edge', edgecolor='k', color='skyblue')

        ax.set_xlabel('Mark Range')
        ax.set_ylabel('Frequency (Number of Students)')
        ax.set_title('Histogram of Student Marks')
        ax.set_xticks(bins) # Set ticks at bin edges
        ax.tick_params(axis='x', rotation=45)
        ax.bar_label(bars, label_type='edge', fontsize=8)
        ax.grid(axis='y', linestyle=':')
        self._save_plot(fig, "histogram_student_marks.png")

    def frequency_polygon_student_marks(self):
        """Generates a frequency polygon over the histogram bars."""
        print("Generating: frequency_polygon_student_marks")
        # Use same data as histogram
        marks_lower = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        marks_upper = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        frequencies = [2, 5, 8, 12, 15, 10, 6, 4, 2, 1]
        bins = marks_lower + [marks_upper[-1]]
        midpoints = [(l+u)/2 for l,u in zip(marks_lower, marks_upper)]

        # Add points at zero frequency for start and end class midpoints
        first_mid = marks_lower[0] - (marks_upper[0] - marks_lower[0])/2
        last_mid = marks_upper[-1] + (marks_upper[-1] - marks_lower[-1])/2
        plot_midpoints = [first_mid] + midpoints + [last_mid]
        plot_frequencies = [0] + frequencies + [0]

        fig, ax = plt.subplots(figsize=(8, 5))
        # Optional: Plot histogram bars lightly in background
        ax.bar(marks_lower, frequencies, width=(bins[1]-bins[0]), align='edge', edgecolor='grey', color='lightgrey', alpha=0.5)
        # Plot frequency polygon
        ax.plot(plot_midpoints, plot_frequencies, 'r-o', label='Frequency Polygon')

        ax.set_xlabel('Mark Midpoints')
        ax.set_ylabel('Frequency')
        ax.set_title('Frequency Polygon of Student Marks')
        ax.set_xticks(bins) # Keep bin edges for reference
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle=':')
        ax.legend()
        self._save_plot(fig, "frequency_polygon_student_marks.png")

    # --- Form 3: Trigonometry (Ratios) ---
    def trig_ratios_triangle(self):
        """Illustrates trig ratios (SOH CAH TOA) in a right-angled triangle."""
        print("Generating: trig_ratios_triangle")
        fig, ax = plt.subplots(figsize=(6, 5))
        A = (0, 0); B = (4, 0); C = (4, 3) # Right angle at B
        points = np.array([A, B, C])
        triangle = patches.Polygon(points, closed=True, edgecolor='k', facecolor='lightblue', alpha=0.6)
        ax.add_patch(triangle)

        # Labels
        ax.text(A[0]-0.2, A[1]-0.2, 'A (θ)')
        ax.text(B[0]-0.1, B[1]-0.4, 'B')
        ax.text(C[0]+0.1, C[1]+0.1, 'C')
        # Sides
        ax.text((A[0]+B[0])/2, (A[1]+B[1])/2 - 0.2, 'Adjacent (Adj)', ha='center', va='top')
        ax.text((B[0]+C[0])/2 + 0.1, (B[1]+C[1])/2, 'Opposite (Opp)', va='center', ha='left', rotation=90)
        ax.text((A[0]+C[0])/2 + 0.1, (A[1]+C[1])/2 + 0.1, 'Hypotenuse (Hyp)', ha='center', va='bottom', rotation=np.degrees(np.arctan2(C[1]-A[1], C[0]-A[0])))
        # Right angle symbol
        ra_size = 0.3
        ax.plot([B[0]-ra_size, B[0]-ra_size, B[0]], [B[1], B[1]+ra_size, B[1]+ra_size], 'k-', lw=1)
        # Angle theta
        self._add_angle_arc(ax, A, 0.8, 0, np.degrees(np.arctan2(C[1]-A[1], C[0]-A[0])), color='red', label='θ')

        # Formulae text
        formulae = "sin(θ) = Opp / Hyp\ncos(θ) = Adj / Hyp\ntan(θ) = Opp / Adj"
        ax.text(1, 2.5, formulae, bbox=dict(fc="wheat", alpha=0.7))

        ax.set_xlim(-1, 5); ax.set_ylim(-1, 4)
        ax.set_aspect('equal', adjustable='box'); ax.axis('off'); ax.set_title('Trigonometric Ratios (SOH CAH TOA)')
        self._save_plot(fig, "trig_ratios_triangle.png")

    def trig_ratios_obtuse_unit_circle(self):
        """Illustrates trig ratios for obtuse angles using the unit circle."""
        print("Generating: trig_ratios_obtuse_unit_circle")
        fig, ax = plt.subplots(figsize=(6, 6))
        self._setup_cartesian_axes(ax, (-1.5, 1.5), (-1.5, 1.5), 'Trig Ratios on Unit Circle')
        # Unit circle
        circle = patches.Circle((0, 0), 1, edgecolor='k', facecolor='none', linestyle='--')
        ax.add_patch(circle)

        # Example Obtuse Angle (e.g., 135 degrees)
        angle_deg = 135
        angle_rad = np.deg2rad(angle_deg)
        x_pt = np.cos(angle_rad); y_pt = np.sin(angle_rad)
        ax.plot([0, x_pt], [0, y_pt], 'r-', label=f'Angle {angle_deg}°') # Radius line
        ax.plot(x_pt, y_pt, 'ro', ms=5)
        ax.text(x_pt-0.1, y_pt+0.1, f'P({x_pt:.2f}, {y_pt:.2f})')

        # Show coordinates relate to sin/cos
        ax.plot([x_pt, x_pt], [0, y_pt], 'b:', label=f'sin({angle_deg}°) = y = {y_pt:.2f}') # Vertical line to x-axis
        ax.plot([0, x_pt], [y_pt, y_pt], 'b:') # Horizontal guide
        ax.plot([0, x_pt], [0, 0], 'g:', label=f'cos({angle_deg}°) = x = {x_pt:.2f}') # Horizontal line on x-axis

        # Angle arc
        self._add_angle_arc(ax, (0,0), 0.4, 0, angle_deg, color='red', label=f'{angle_deg}°')

        ax.legend(loc='lower left', fontsize=8)
        ax.set_aspect('equal', adjustable='box')
        self._save_plot(fig, "trig_ratios_obtuse_unit_circle.png")

    # --- Form 3: Vectors (Position Vectors) ---
    def vector_position_vectors_ab(self):
        """Illustrates position vectors OA, OB and vector AB."""
        print("Generating: vector_position_vectors_ab")
        fig, ax = plt.subplots(figsize=(6, 6))
        O = np.array([0, 0])
        A = np.array([1, 3])
        B = np.array([4, 1])
        vec_AB = B - A

        # Setup axes
        self._setup_cartesian_axes(ax, (-1, 5), (-1, 5), 'Position Vectors and Vector AB')
        # Points
        ax.plot(O[0], O[1], 'ko', label='Origin O')
        ax.plot(A[0], A[1], 'bo', label=f'A {tuple(A)}')
        ax.plot(B[0], B[1], 'ro', label=f'B {tuple(B)}')

        # Position Vectors (from Origin)
        ax.quiver(O[0], O[1], A[0], A[1], angles='xy', scale_units='xy', scale=1, color='blue', label='OA = a')
        ax.quiver(O[0], O[1], B[0], B[1], angles='xy', scale_units='xy', scale=1, color='red', label='OB = b')
        # Vector AB
        ax.quiver(A[0], A[1], vec_AB[0], vec_AB[1], angles='xy', scale_units='xy', scale=1, color='green', label='AB = b - a')

        # Labels for vectors near arrowhead
        ax.text(A[0]*0.8, A[1]*0.8, 'a', color='blue', ha='right')
        ax.text(B[0]*0.8, B[1]*0.8, 'b', color='red', ha='right')
        ax.text(A[0] + vec_AB[0]*0.5, A[1] + vec_AB[1]*0.5 + 0.1, 'b - a', color='green', ha='center')

        ax.legend(loc='upper left')
        ax.set_aspect('equal', adjustable='box')
        self._save_plot(fig, "vector_position_vectors_ab.png")

    # --- Form 3: Matrices ---
    # Diagrams not usually needed, calculations/properties are key.

    # --- Form 4: Financial Maths ---
    # Mostly calculations, complex graphs (e.g., Hire Purchase) could be added if needed.

    # --- Form 4: Area and Volume (Solids) ---
    def prism_pyramid_volume(self):
        """Placeholder: Comparing volume formulae for Prism and Pyramid."""
        print("Placeholder function called: prism_pyramid_volume")
        fig, ax = plt.subplots(1, 2, figsize=(8, 4)); fig.suptitle("Prism vs Pyramid Volume (Placeholder)")
        ax[0].text(0.5, 0.5, "Prism\nV = Base Area * Height", ha='center', va='center'); ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[1].text(0.5, 0.5, "Pyramid\nV = (1/3) * Base Area * Height", ha='center', va='center'); ax[1].set_xticks([]); ax[1].set_yticks([])
        self._save_plot(fig, "prism_pyramid_volume.png")

    def cone_volume_formula(self):
        """Placeholder: Illustrates a cone with its volume formula."""
        print("Placeholder function called: cone_volume_formula")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Cone\nV = (1/3)πr²h\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "cone_volume_formula.png")

    def sphere_volume_formula(self):
        """Placeholder: Illustrates a sphere with its volume formula."""
        print("Placeholder function called: sphere_volume_formula")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Sphere\nV = (4/3)πr³\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "sphere_volume_formula.png")

    def surface_area_nets(self):
        """Placeholder: Examples of nets for surface area (Cube, Cylinder)."""
        print("Placeholder function called: surface_area_nets")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Surface Area Nets\n(Cube, Cylinder, etc.)\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "surface_area_nets.png")

    # --- Form 4: Graphs (Cubic, Reciprocal, Exponential) ---
    def cubic_graph_shapes(self):
        """Shows typical shapes of cubic graphs (y=ax^3+...)."""
        print("Generating: cubic_graph_shapes")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Cubic Graph Shapes (y = ax³ + ...)")
        x = np.linspace(-2.5, 2.5, 100)

        # Positive a (e.g., y = x^3 - 3x)
        y1 = x**3 - 3*x
        self._setup_cartesian_axes(ax1, (-2.5, 2.5), (-5, 5), 'Positive Leading Coefficient (a > 0)')
        ax1.plot(x, y1, 'b-')

        # Negative a (e.g., y = -x^3 + 3x)
        y2 = -x**3 + 3*x
        self._setup_cartesian_axes(ax2, (-2.5, 2.5), (-5, 5), 'Negative Leading Coefficient (a < 0)')
        ax2.plot(x, y2, 'r-')

        self._save_plot(fig, "cubic_graph_shapes.png")

    def inverse_graph_shapes(self): # Renamed from reciprocal
        """Shows shape of inverse graph y = k/x."""
        print("Generating: inverse_graph_shapes")
        fig, ax = plt.subplots(figsize=(6, 6))
        k = 2
        # Plot positive and negative branches separately
        x_pos = np.linspace(0.1, 4, 100)
        y_pos = k / x_pos
        x_neg = np.linspace(-4, -0.1, 100)
        y_neg = k / x_neg

        self._setup_cartesian_axes(ax, (-4, 4), (-8, 8), f'Inverse Graph: y = {k}/x')
        ax.plot(x_pos, y_pos, 'b-', label=f'y = {k}/x (x>0)')
        ax.plot(x_neg, y_neg, 'b-', label=f'y = {k}/x (x<0)')
        ax.legend(fontsize=8)
        # Add asymptotes
        ax.axhline(0, color='grey', linestyle='--', lw=1)
        ax.axvline(0, color='grey', linestyle='--', lw=1)

        self._save_plot(fig, "inverse_graph_shapes.png")

    # Exponential graph shape could be added similarly if needed

    # --- Form 4: Circle Theorems (Placeholders/Simple Examples) ---
    def circle_theorem1_center_circumference(self):
        """Placeholder: Circle Theorem 1 (Angle at center = 2 * Angle at circumference)."""
        print("Placeholder function called: circle_theorem1_center_circumference")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Circle Theorem 1:\nAngle at Center = 2 * Angle at Circ.\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "circle_theorem1_center_circumference.png")

    def circle_theorem2_same_segment(self):
        """Placeholder: Circle Theorem 2 (Angles in same segment are equal)."""
        print("Placeholder function called: circle_theorem2_same_segment")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Circle Theorem 2:\nAngles in Same Segment Equal\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "circle_theorem2_same_segment.png")

    # ... Add placeholders or implementations for other circle theorems (3-10) ...
    def circle_theorem_placeholder(self, theorem_num):
        """Generic placeholder for unimplemented circle theorems."""
        print(f"Placeholder function called: circle_theorem_{theorem_num}")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Circle Theorem {theorem_num}\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, f"circle_theorem_{theorem_num}_placeholder.png")


    # --- Form 4: Loci and Constructions ---
    # These require more complex drawing logic (arcs, intersections) - Placeholders for now
    def locus_placeholder(self, description):
        """Generic placeholder for locus diagrams."""
        print(f"Placeholder function called: locus_{description}")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Locus: {description}\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, f"locus_{description.replace(' ', '_').lower()}_placeholder.png")

    # --- Form 4: Statistics (Cumulative Frequency / Ogive) ---
    def ogive_student_marks(self):
        """Generates a cumulative frequency curve (Ogive) for student marks."""
        print("Generating: ogive_student_marks")
        # Use data similar to histogram, calculate cumulative freq
        marks_upper = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        frequencies = [2, 5, 8, 12, 15, 10, 6, 4, 2, 1]
        cum_freq = np.cumsum(frequencies)
        # Start curve from (lower bound of first class, 0)
        plot_marks = [0] + marks_upper
        plot_cum_freq = [0] + list(cum_freq)
        total_freq = cum_freq[-1]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(plot_marks, plot_cum_freq, 'b-o')
        ax.set_xlabel('Mark (Upper Class Boundary)')
        ax.set_ylabel('Cumulative Frequency')
        ax.set_title('Ogive (Cumulative Frequency Curve) of Student Marks')
        ax.set_xticks(plot_marks)
        ax.tick_params(axis='x', rotation=45)
        ax.set_yticks(np.arange(0, total_freq + 5, 5)) # Adjust y-ticks based on total
        ax.grid(True, linestyle=':')

        # --- Illustrate Quartiles (Example) ---
        q1_pos = total_freq * 0.25; q2_pos = total_freq * 0.50; q3_pos = total_freq * 0.75
        # Interpolate to find marks corresponding to these positions
        q1_mark = np.interp(q1_pos, plot_cum_freq, plot_marks)
        q2_mark = np.interp(q2_pos, plot_cum_freq, plot_marks) # Median
        q3_mark = np.interp(q3_pos, plot_cum_freq, plot_marks)
        iqr = q3_mark - q1_mark

        ax.axhline(q1_pos, color='r', linestyle='--', lw=1, label=f'Q1 Position ({q1_pos:.1f})')
        ax.axvline(q1_mark, color='r', linestyle='--', lw=1)
        ax.text(q1_mark + 1, q1_pos * 0.9, f'Q1 ≈ {q1_mark:.1f}', color='r', ha='left')

        ax.axhline(q2_pos, color='g', linestyle='--', lw=1, label=f'Median Position ({q2_pos:.1f})')
        ax.axvline(q2_mark, color='g', linestyle='--', lw=1)
        ax.text(q2_mark + 1, q2_pos * 0.9, f'Median ≈ {q2_mark:.1f}', color='g', ha='left')

        ax.axhline(q3_pos, color='purple', linestyle='--', lw=1, label=f'Q3 Position ({q3_pos:.1f})')
        ax.axvline(q3_mark, color='purple', linestyle='--', lw=1)
        ax.text(q3_mark + 1, q3_pos * 0.9, f'Q3 ≈ {q3_mark:.1f}', color='purple', ha='left')

        ax.text(50, 5, f"IQR = Q3 - Q1 ≈ {iqr:.1f}", bbox=dict(fc="wheat"))

        ax.legend(fontsize=8)
        self._save_plot(fig, "ogive_student_marks_with_quartiles.png")


    # --- Form 4: Transformations (Matrices, Enlargement, Combined) ---
    # Placeholders
    def transformation_matrices(self):
        """Placeholder: Show matrices for reflection, rotation, etc."""
        print("Placeholder function called: transformation_matrices")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Transformation Matrices\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "transformation_matrices.png")

    def enlargement_example(self):
        """Placeholder: Enlargement transformation."""
        print("Placeholder function called: enlargement_example")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Enlargement\n(Center, Scale Factor)\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "enlargement_example.png")

    def combined_transformations(self):
        """Placeholder: Combined transformations (e.g., reflection then rotation)."""
        print("Placeholder function called: combined_transformations")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Combined Transformations\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "combined_transformations.png")

    # --- Form 4: Probability ---
    def probability_tree_diagram(self):
        """Placeholder: Probability Tree Diagram."""
        print("Placeholder function called: probability_tree_diagram")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Probability Tree Diagram\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "probability_tree_diagram.png")

    def probability_sample_space(self):
        """Placeholder: Sample Space Diagram (e.g., for two dice)."""
        print("Placeholder function called: probability_sample_space")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "Sample Space Diagram\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, "probability_sample_space.png")

    def venn_diagram_3sets(self):
        """Generates a Venn diagram for three sets (requires matplotlib_venn)."""
        print("Generating: venn_diagram_3sets")
        if not venn_installed:
             print("Skipping venn_diagram_3sets: matplotlib_venn not installed.")
             self.venn_diagram_placeholder("3 Sets") # Call placeholder instead
             return

        fig, ax = plt.subplots(figsize=(7, 7))
        # Example subset sizes (must sum to total if known)
        # Sizes of (100, 010, 110, 001, 101, 011, 111)
        # A only, B only, A&B only, C only, A&C only, B&C only, A&B&C
        subsets = (10, 8, 5, 12, 6, 7, 3)
        v = venn3(subsets=subsets, set_labels = ('Set A', 'Set B', 'Set C'), ax=ax)
        # Optional: Style circles
        # c = venn3_circles(subsets=subsets, linestyle='dashed', linewidth=1, color="grey", ax=ax)
        ax.set_title("Venn Diagram for Three Sets")
        self._save_plot(fig, "venn_diagram_3sets.png")

    def venn_diagram_placeholder(self, description):
        """Placeholder for Venn diagrams if library missing or not implemented."""
        print(f"Placeholder function called: venn_diagram_{description}")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, f"Venn Diagram\n({description})\n(Placeholder)", ha='center', va='center'); ax.set_xticks([]); ax.set_yticks([]); self._save_plot(fig, f"venn_diagram_{description.replace(' ', '_').lower()}_placeholder.png")


# --- Instantiate and Run ---
if __name__ == "__main__":
    generator = DiagramGenerator()
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    print("\n--- Generating Form 2 Diagrams ---")
    # Algebra - Inequalities
    generator.cartesian_y_gt_x_plus_1()
    generator.number_line_x_le_neg1()
    generator.cartesian_x_ge_2()
    generator.cartesian_y_lt_3()
    # Geometry - Lines & Angles
    generator.parallel_lines_l1_l2()
    generator.parallel_transversal_angles_1_8()
    generator.parallel_corresponding_60deg()
    generator.parallel_alternate_interior_110deg()
    generator.parallel_cointerior_75deg()
    generator.parallel_alternate_exterior_ref_ex3()
    generator.parallel_alternate_interior_algebra()
    generator.parallel_vertical_transversal_q2()
    generator.parallel_lines_angle_between_q3()
    # Geometry - Bearing
    generator.compass_rose_8_points()
    generator.compass_bearing_examples()
    generator.three_figure_bearing_examples()
    generator.bearing_back_bearing_70_250()
    generator.bearing_angle_between_paths_120_210()
    # Geometry - Polygons
    generator.triangle_types()              # Placeholder
    generator.quadrilateral_types()         # Placeholder
    # Geometry - Congruence & Similarity
    generator.similar_triangles_abc_pqr()   # Placeholder
    generator.congruent_triangles_def_stu() # Placeholder
    generator.similar_rectangles_4x6_6x9()  # Placeholder
    generator.congruent_triangles_sss()     # Placeholder
    generator.congruent_triangles_sas()     # Placeholder
    generator.similar_triangles_lmn_rst_lengths() # Placeholder
    generator.similar_triangles_nested_pst_pqr()  # Placeholder
    # Geometry - Constructions
    generator.construction_perp_bisector()    # Placeholder
    generator.construction_angle_bisector()   # Placeholder
    generator.construction_perp_bisector_8cm()# Placeholder
    generator.construction_60_degrees()       # Placeholder
    generator.angle_approx_80_deg()           # Placeholder (Drawing, not construction)
    generator.construction_bisect_angle_80()  # Placeholder
    generator.construction_90_degrees_on_line()# Placeholder
    generator.construction_30_degrees()       # Placeholder
    # Geometry - Symmetry
    generator.lines_of_symmetry_examples()    # Placeholder
    generator.symmetry_square_4lines()        # Placeholder
    generator.symmetry_rectangle_2lines()     # Placeholder
    generator.symmetry_letter_h()             # Placeholder (Simple text)
    generator.symmetry_isosceles_triangle()   # Placeholder
    generator.symmetry_regular_pentagon()     # Placeholder
    # Statistics - Charts
    generator.barchart_favorite_fruits()
    generator.piechart_item_types()
    generator.barchart_pets_owned()
    generator.piechart_pets_owned()
    # Vectors - Basics
    generator.vector_notation_examples()      # Placeholder
    generator.vector_equal_ab_cd()            # Placeholder
    generator.vector_negative_a_neg_a()       # Placeholder
    generator.vector_draw_neg3_1_from_2_2()   # Corrected name, Implemented
    generator.vector_identify_equal_grid()    # Placeholder
    generator.vector_addition_laws()          # Placeholder
    generator.vector_subtraction_laws()       # Placeholder

    print("\n--- Generating Form 3 Diagrams ---")
    # Financial Maths
    generator.simple_vs_compound_interest()
    # Geometry - Area & Volume
    generator.combined_shape_rect_semicircle()
    generator.cylinder_volume_formula()         # 3D Plot
    generator.combined_shape_rect_minus_semicircle()
    generator.combined_shape_square_plus_triangle()
    # Graphs
    generator.graph_linear_y_2x_minus_4()
    generator.graph_quadratic_y_x_squared()
    generator.graph_linear_y_6_minus_2x()
    generator.graph_quadratic_y_x2_minus_2x_minus_3()
    generator.graph_sketch_quadratic_y_x2_minus_4x_plus_3()
    generator.graph_solve_quadratic_y_equals_2()
    # Travel Graphs
    generator.travel_graphs_dist_time_vs_speed_time()
    generator.speed_time_graph_area_trapezium()
    # Variation
    generator.direct_vs_inverse_variation_graphs()
    # Trigonometry - Elevation/Depression
    generator.angle_elevation()
    generator.angle_depression()
    generator.elevation_depression_relationship()
    # Bearing (Relationship)
    generator.bearing_back_bearing_relation()   # Reuses F2 diagram
    # Geometry - Polygons
    generator.polygon_exterior_angle()
    # Constructions
    generator.construction_sss_triangle_pqr()   # Placeholder
    generator.construction_sas_triangle_abc()   # Placeholder
    generator.construction_asa_triangle_xyz()   # Placeholder
    generator.construction_rectangle_7x4()      # Placeholder
    generator.construction_parallelogram_pqrs() # Placeholder
    # Symmetry (Rotational)
    generator.rotational_symmetry_examples()    # Placeholder
    generator.symmetry_letter_s()               # Simple text example
    # Statistics - Frequency Distributions
    generator.histogram_student_marks()
    generator.frequency_polygon_student_marks()
    # Trigonometry - Ratios
    generator.trig_ratios_triangle()
    generator.trig_ratios_obtuse_unit_circle()
    # Vectors - Position Vectors
    generator.vector_position_vectors_ab()
    # # Transformations (Placeholders - covered in F2/F4 detail)
    # generator.translation_triangle_lmn()      # Placeholder
    # generator.reflection_standard_lines()     # Placeholder
    # generator.reflection_x_axis_triangle()    # Placeholder

    print("\n--- Generating Form 4 Diagrams ---")
    # Area and Volume (Solids)
    generator.prism_pyramid_volume()          # Placeholder
    generator.cone_volume_formula()           # Placeholder
    generator.sphere_volume_formula()         # Placeholder
    generator.surface_area_nets()             # Placeholder
    # Graphs
    generator.cubic_graph_shapes()
    generator.inverse_graph_shapes()
    # Circle Theorems
    generator.circle_theorem1_center_circumference() # Placeholder
    generator.circle_theorem2_same_segment()        # Placeholder
    # Add calls for other theorems if implemented, e.g.:
    # for i in range(3, 11): generator.circle_theorem_placeholder(i)
    # Loci
    # Add calls for specific loci if implemented, e.g.:
    # generator.locus_placeholder("Equidistant from two points")
    # Statistics (Ogive)
    generator.ogive_student_marks() # Includes Quartiles
    # Transformations (Matrices etc)
    generator.transformation_matrices()       # Placeholder
    generator.enlargement_example()           # Placeholder
    generator.combined_transformations()      # Placeholder
    # Probability
    generator.probability_tree_diagram()      # Placeholder
    generator.probability_sample_space()      # Placeholder
    generator.venn_diagram_3sets()            # Requires matplotlib_venn

    print("\n--- Diagram Generation Complete ---")






# --- Run the functions to generate diagrams ---
# (Call each function once to generate all diagrams)
print("Generating diagrams...")
form1_real_numbers_number_line()
form1_real_numbers_submarine_movement()
form1_scales_representations()
# Form 1 Sets
form1_sets_venn_diagram_two_sets()
form1_sets_venn_example_1_to_7()
form1_sets_venn_shade_intersection()
# Form 1 Measures
form1_measures_thermometers()
form1_measures_capacity_containers()
# Form 1 Mensuration
form1_mensuration_perimeter_formulas()
form1_mensuration_area_formulas()
form1_mensuration_square_area_problem()
# Form 1 Graphs
form1_graphs_cartesian_plane()
form1_graphs_cartesian_plane_example()
form1_graphs_plot_point_a32()
form1_graphs_plot_points_bcd()
form1_graphs_identify_points_pqr()
form1_graphs_cartesian_grid_m5_to_5()
form1_graphs_rectangle_find_d()

form1_geometry_types_of_lines()
form1_travel_graphs_walk_rest_return()
# Form 1 Travel Graphs
form1_travel_graphs_distance_time_sample()
form1_travel_graphs_example_journey1()
form1_travel_graphs_cyclist_grid()
# Form 1 Transformation
form1_transformation_translate_triangle_pqr()
form1_transformation_describe_translation_ab()


form1_travel_graphs_car_grid() # Renamed F2 -> F1 based on original JSON
form1_sets_venn_music_example() # Renamed F2 -> F1 based on original JSON
form1_graphs_plot_points_example2() # Renamed F2 -> F1 based on original JSON
form1_graphs_line_graph_y_eq_x_m2() # Renamed F2 -> F1 based on original JSON
form1_graphs_line_graph_y_eq_4_m_x() # Renamed F2 -> F1 based on original JSON
# Form 2 Real Numbers
form2_real_numbers_sqrt_prime_factorization()
form2_real_numbers_cbrt_prime_factorization()
# Form 2 Ratios/Rates/Proportion
form2_proportion_direct_vs_inverse_graphs()
# Form 2 Scales
form2_scales_garden_drawing()
# Form 2 Sets
form2_sets_venn_two_sets_regions()
form2_sets_venn_example_counts()
form2_sets_venn_music_example()
# Form 2 Measures
form2_measures_area_conversion_m2_cm2()
form2_measures_volume_conversion_m3_cm3()
form2_measures_floor_area_problem()
# Form 2 Mensuration
form2_mensuration_volume_cuboid_cube()
# Form 2 Graphs
form2_graphs_cartesian_plane_scales()
form2_graphs_plot_points_example2()
form2_graphs_line_graph_y_eq_x_m2()
form2_graphs_line_graph_y_eq_4_m_x()
# Form 2 Travel Graphs
form2_travel_graphs_example_journey2()
form2_travel_graphs_walk_rest_return()
form2_travel_graphs_car_grid()
# Form 2 Variation
form2_variation_direct_variation_graph()
# Form 2 Geometry
form2_geometry_parallel_lines_transversal()
form2_geometry_parallel_lines_corr_angle()
form2_geometry_parallel_lines_alt_int_angle()
form2_geometry_parallel_lines_co_int_angle()
form2_geometry_parallel_lines_algebra_example()
form2_geometry_parallel_lines_mod_question()
form2_geometry_parallel_lines_chall_question()


form2_geometry_compass_rose()
form2_geometry_compass_bearing_examples()
form2_geometry_three_figure_bearing_examples()
form2_geometry_bearing_back_bearing_example()
form2_geometry_bearing_angle_between_points()
form2_geometry_bearing_compass_s40e()
form2_geometry_triangle_types()
form2_geometry_quadrilateral_types()
# ... Add calls for other Form 2 geometry (similarity, construction, symmetry) ...
# ... Add calls for Form 3 and Form 4 diagrams here ...
form3_sets_venn_three_sets_regions()
form3_sets_venn_elements_example()
form3_sets_venn_p_c_b_problem()
form3_finance_compound_vs_simple_interest()
form3_mensuration_combined_shape_perimeter_area()
form3_mensuration_prism_pyramid_volume()
form3_mensuration_cone_volume()
form3_mensuration_sphere_volume()
form3_mensuration_surface_area_nets()
form3_graphs_line_graph_y_eq_6_m_2x()
form3_graphs_quadratic_graph_table()
form3_graphs_quadratic_sketch_intercepts()
form3_graphs_quadratic_solve_y_eq_2()
form3_travel_graphs_journey_distance_time()
form3_travel_graphs_speed_time_accel_const()
form3_travel_graphs_speed_time_area_triangle()
form3_travel_graphs_speed_time_total_area()
# form3_variation_inverse_variation_graph() # Done in F2
form3_variation_partial_variation_graph()
form3_geometry_angles_elevation()
form3_geometry_angles_depression()
form3_geometry_angles_elevation_depression_relation()
form3_geometry_angles_elevation_tower_problem()
form3_geometry_angles_depression_cliff_problem()
form3_geometry_angles_elevation_construct()
form3_geometry_angles_depression_construct()
form3_geometry_angles_elevation_scale_drawing()
form3_geometry_bearing_240_deg()
form3_geometry_bearing_back_bearing() # Example name match
form3_geometry_bearing_angle_between_paths()
form3_geometry_bearing_compass_s40e()
form3_geometry_polygons_triangle_types() # Example name match
form3_geometry_polygons_quadrilateral_types() # Example name match

form2_variation_direct_variation_graph()

form3_geometry_bearing_240_deg()

print(f"Diagrams saved in '{output_dir}' directory.")
print("\nDiagram generation complete (Placeholders may exist). Check the 'math_diagrams' folder.")
