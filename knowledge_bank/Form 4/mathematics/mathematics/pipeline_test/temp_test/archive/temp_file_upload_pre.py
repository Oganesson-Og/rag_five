import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path # Needed for Venn shading
from matplotlib.patches import PathPatch # Needed for Venn shading
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import numpy as np
import os
import json
import inspect

# --- Configuration ---
INPUT_JSON_FILE = 'temp_file_upload_pre.json'
OUTPUT_JSON_FILE = 'temp_file_upload_pre_processed.json'
OUTPUT_IMAGE_FOLDER = 'math_diagrams'

try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except Exception as e:
    print(f"Warning: Could not set default font Dejavu Sans. Using Matplotlib default. Error: {e}")


# --- Helper Function ---
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_filename_from_caller(suffix=""):
    caller_frame = inspect.currentframe().f_back
    func_name = caller_frame.f_code.co_name
    # Basic cleaning: lowercase and replace potential issues
    cleaned_name = func_name.lower().replace(' ', '_').replace('-', '_')
    filename = f"{cleaned_name}{suffix}.png"
    return filename

# --- Plotting Class ---

class MathDiagramGenerator:
    """Class to hold diagram generation methods."""

    def __init__(self):
        self.output_folder = OUTPUT_IMAGE_FOLDER
        ensure_dir(self.output_folder)
        self.context_data = {} # Store data for potential reuse if needed
        plt.rc('axes', facecolor='white')
        plt.rc('figure', facecolor='white')
        plt.rc('savefig', facecolor='white')


    def _setup_ax(self, ax, title="", xlabel="", ylabel="", xlim=None, ylim=None, aspect='auto', grid=True):
        """Helper to set up axes."""
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.set_aspect(aspect)
        ax.grid(grid, linestyle='--', alpha=0.6)
        # Draw origin lines only if 0 is within limits
        if xlim and 0 >= xlim[0] and 0 <= xlim[1]: ax.axvline(0, color='black', linewidth=0.7)
        elif not xlim: ax.axvline(0, color='black', linewidth=0.7)
        if ylim and 0 >= ylim[0] and 0 <= ylim[1]: ax.axhline(0, color='black', linewidth=0.7)
        elif not ylim: ax.axhline(0, color='black', linewidth=0.7)
        try:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=8)
        except AttributeError: pass

    def _setup_ax_no_origin(self, ax, title="", xlabel="", ylabel="", xlim=None, ylim=None, aspect='auto', grid=False):
        """Helper to set up axes without grid/ticks."""
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.set_aspect(aspect)
        ax.grid(grid)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)

    def _save_plot(self, fig, filename):
        """Saves the figure and closes it."""
        filepath = os.path.join(self.output_folder, filename)
        try:
            fig.tight_layout()
            fig.savefig(filepath, bbox_inches='tight', facecolor='white', dpi=150)
        except Exception as e:
            print(f"Error during layout/saving {filename}: {e}")
            fig.savefig(filepath, bbox_inches='tight', facecolor='white', dpi=150)
        plt.close(fig)
        print(f"Saved diagram: {filepath}")
        # Use forward slashes for JSON path consistency
        return os.path.join(OUTPUT_IMAGE_FOLDER, os.path.basename(filepath)).replace("\\", "/")


    # === Form 1 ===

    # F1: Real Numbers - Number Concepts
    def f1_rn_nc_notes_number_line(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 1.5), facecolor='white')
        self._setup_ax(ax, title="Number Line", xlabel="", ylabel="", ylim=(-0.5, 0.5), xlim=(-6, 6), grid=False)

        # Draw the line
        ax.axhline(0, color='black', linewidth=1.5)

        # Add ticks and labels
        ticks = np.arange(-5, 6)
        for tick in ticks:
            ax.plot([tick, tick], [-0.1, 0.1], color='black', lw=1.5)
        ax.text(-5, -0.3, '-5', ha='center', va='top', fontsize=10)
        ax.text(0, -0.3, '0', ha='center', va='top', fontsize=10)
        ax.text(5, -0.3, '+5', ha='center', va='top', fontsize=10)

        # Add arrowheads
        ax.arrow(-6, 0, 11.5, 0, head_width=0.15, head_length=0.3, fc='black', ec='black', length_includes_head=True, linewidth=1.5, clip_on=False)
        ax.arrow(6, 0, -11.5, 0, head_width=0.15, head_length=0.3, fc='black', ec='black', length_includes_head=True, linewidth=1.5, clip_on=False)

        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])

        return self._save_plot(fig, filename)

    def f1_rn_nc_ques_submarine_depth(self):
        filename = get_filename_from_caller()
        # Increased figure width slightly for better label spacing
        fig, ax = plt.subplots(figsize=(3.5, 8), facecolor='white')

        # Setup Axes - Keep grid=False for _setup_ax, enable y-grid manually
        # Keep ylim and title, but customize ticks/spines locally
        self._setup_ax(ax, title="Submarine Depth", xlabel="", ylabel="", ylim=(-110, 60), xlim=(-0.5, 1), grid=False) # Wider xlim for labels

        # --- Customize Axes for Vertical Number Line ---
        # Keep the central axis line
        ax.axvline(0, color='black', linewidth=1.5, zorder=1)
        # Set specific y-ticks
        ticks = np.arange(-100, 51, 25)
        ax.set_yticks(ticks)
        # Enable y-grid lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        # Hide x-axis ticks and labels
        ax.set_xticks([])
        # Customize spines: Hide top, right, bottom; keep left for labels
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True) # Keep left spine
        # Customize tick parameters
        ax.tick_params(axis='y', which='major', labelsize=9, direction='out', length=4, left=True, labelleft=True) # Show ticks and labels on the left
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # Add "Depth (m)" label manually to the left
        ax.text(-0.4, 0.5, 'Depth (m)', transform=ax.transAxes, rotation='vertical', va='center', ha='right', fontsize=9)
        # Add "Sea Level" label next to 0 tick
        ax.text(1, 0, ' 0 (Sea Level)', ha='left', va='center', fontsize=9)
        # Format y-tick labels with sign
        ax.set_yticklabels([f'{tick:+d}' if tick != 0 else '' for tick in ticks])

        # --- Plot Submarine Movements ---
        start_depth = -50
        rise = 25
        intermediate_depth = start_depth + rise
        dive = -40
        final_depth = intermediate_depth + dive
        arrow_x_offset = 0.05 # Offset arrows slightly from axis

        # Start point
        ax.plot(0, start_depth, 'bo', markersize=8, label=f'Start ({start_depth:+d}m)', zorder=5)

        # Rise arrow and label (move label to right)
        ax.arrow(arrow_x_offset -0.2, start_depth, 0, rise, head_width=0.15, head_length=5, fc='green', ec='green', length_includes_head=True, lw=1.5, zorder=4)
        ax.text(arrow_x_offset -0.8, start_depth + rise/2, f'+{rise}m', ha='left', va='center', color='green', zorder=5)

        # Intermediate point
        ax.plot(0, intermediate_depth, 'go', markersize=6, label=f'Intermediate ({intermediate_depth:+d}m)', zorder=5)

        # Dive arrow and label (move label to right)
        ax.arrow(arrow_x_offset+ 0.1, intermediate_depth, 0, dive, head_width=0.15, head_length=5, fc='red', ec='red', length_includes_head=True, lw=1.5, zorder=4)
        ax.text(arrow_x_offset + 0.2, intermediate_depth + dive/2, f'{dive}m', ha='left', va='center', color='red', zorder=5)

        # End point
        ax.plot(0, final_depth, 'ro', markersize=8, label=f'End ({final_depth:+d}m)', zorder=5)

        # Adjust legend position (bottom center is often good for vertical plots)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)

        # Adjust bottom margin to make space for legend
        plt.subplots_adjust(bottom=0.1)

        return self._save_plot(fig, filename)

    # F1: Real Numbers - Scales
    def f1_rn_sc_notes_types_of_scales(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(3, 1, figsize=(6, 4), facecolor='white')
        fig.suptitle("Types of Scales (Example: 1:100,000)", y=1.02)

        # Ratio Scale
        ax[0].text(0.5, 0.5, 'Ratio Scale: 1 : 100,000', ha='center', va='center', fontsize=12)
        ax[0].axis('off')

        # Representative Fraction
        ax[1].text(0.5, 0.5, 'Representative Fraction (R.F.): 1/100,000', ha='center', va='center', fontsize=12)
        ax[1].axis('off')

        # Linear Scale
        ax[2].set_xlim(0, 4)
        ax[2].set_ylim(-0.5, 1)
        ax[2].set_title("Linear Scale", fontsize=10)
        ax[2].axhline(0, color='black', linewidth=2) # Main bar line

        # Main segments (0 to 3 km)
        for i in range(4):
            ax[2].plot([i+1, i+1], [-0.1, 0.1], color='black', lw=1.5) # Tick marks
            ax[2].text(i+1, 0.2, f'{i} km' if i==0 else f'{i}', ha='center', va='bottom')

        # Subdivision left of 0 (meters)
        ax[2].plot([0, 0], [-0.1, 0.1], color='black', lw=1.5)
        subdivisions = np.linspace(0, 1, 6) # e.g., 0, 200m, 400m, 600m, 800m, 1km
        for i, sub in enumerate(subdivisions[1:-1]): # Plot intermediate ticks
             ax[2].plot([1-sub, 1-sub], [-0.05, 0.05], color='black', lw=1)
        # Labels for subdivisions (optional, can get crowded)
        ax[2].text(0.5, 0.2, '1000m', ha='center', va='bottom', fontsize=8)

        ax[2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return self._save_plot(fig, filename)

    # F1: Sets - Types
    def f1_se_ty_notes_venn_diagram_two_sets(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Venn Diagram (2 Sets)")

        # Universal Set (Rectangle)
        rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.text(0.95, 0.9, 'U', ha='right', va='top', fontsize=12)

        # Circles for sets A and B
        circle_a = patches.Circle((0.4, 0.5), 0.25, linewidth=1.5, edgecolor='blue', facecolor='blue', alpha=0.2)
        circle_b = patches.Circle((0.6, 0.5), 0.25, linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.2)
        ax.add_patch(circle_a)
        ax.add_patch(circle_b)

        # Labels
        ax.text(0.2, 0.75, 'A', ha='center', va='center', fontsize=14, color='blue')
        ax.text(0.8, 0.75, 'B', ha='center', va='center', fontsize=14, color='red')

        # Region Labels (approximate positions)
        ax.text(0.25, 0.5, 'A only', ha='center', va='center', fontsize=10)
        ax.text(0.75, 0.5, 'B only', ha='center', va='center', fontsize=10)
        ax.text(0.5, 0.5, 'A ∩ B', ha='center', va='center', fontsize=10)
        ax.text(0.5, 0.15, '(A ∪ B)\'', ha='center', va='center', fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('equal')
        ax.axis('off')

        return self._save_plot(fig, filename)

    def f1_se_ty_work_venn_diagram_example(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Venn Diagram Example: U={1-7}, A={1,2,3}, B={3,4,5}")

        # Universal Set
        rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.text(0.95, 0.9, 'U', ha='right', va='top', fontsize=12)

        # Circles A and B
        circle_a = patches.Circle((0.4, 0.5), 0.25, linewidth=1.5, edgecolor='blue', facecolor='none')
        circle_b = patches.Circle((0.6, 0.5), 0.25, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(circle_a)
        ax.add_patch(circle_b)
        ax.text(0.4, 0.8, 'A', ha='center', va='center', fontsize=14, color='blue')
        ax.text(0.6, 0.8, 'B', ha='center', va='center', fontsize=14, color='red')

        # Place elements
        ax.text(0.5, 0.5, '3', ha='center', va='center', fontsize=12) # Intersection
        ax.text(0.25, 0.55, '1', ha='center', va='center', fontsize=12) # A only
        ax.text(0.25, 0.45, '2', ha='center', va='center', fontsize=12) # A only
        ax.text(0.75, 0.55, '4', ha='center', va='center', fontsize=12) # B only
        ax.text(0.75, 0.45, '5', ha='center', va='center', fontsize=12) # B only
        ax.text(0.1, 0.1, '6', ha='center', va='center', fontsize=12) # Neither
        ax.text(0.9, 0.1, '7', ha='center', va='center', fontsize=12) # Neither

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('equal')
        ax.axis('off')

        return self._save_plot(fig, filename)

    def f1_se_ty_ques_venn_diagram_shaded(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Venn Diagram: Shade A ∩ B")

        # Universal Set
        rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.text(0.95, 0.9, 'U', ha='right', va='top', fontsize=12)

        # Circles A and B
        center_a = (0.4, 0.5)
        center_b = (0.6, 0.5)
        radius = 0.25
        circle_a = patches.Circle(center_a, radius, linewidth=1.5, edgecolor='blue', facecolor='none')
        circle_b = patches.Circle(center_b, radius, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(circle_a)
        ax.add_patch(circle_b)
        ax.text(0.4, 0.8, 'A', ha='center', va='center', fontsize=14, color='blue')
        ax.text(0.6, 0.8, 'B', ha='center', va='center', fontsize=14, color='red')

        # Place elements
        ax.text(0.5, 0.55, 'c', ha='center', va='center', fontsize=12) # Intersection
        ax.text(0.5, 0.45, 'd', ha='center', va='center', fontsize=12) # Intersection
        ax.text(0.25, 0.55, 'a', ha='center', va='center', fontsize=12) # A only
        ax.text(0.25, 0.45, 'b', ha='center', va='center', fontsize=12) # A only
        ax.text(0.75, 0.55, 'e', ha='center', va='center', fontsize=12) # B only
        ax.text(0.75, 0.45, 'f', ha='center', va='center', fontsize=12) # B only
        ax.text(0.1, 0.1, 'g', ha='center', va='center', fontsize=12) # Neither

        # --- Shading the intersection A ∩ B ---
        # We need to define the intersection area for shading. This is tricky with patches.
        # An alternative is to use hatching or color the elements. Let's color the background of intersection text.
        # Or, more accurately, use fill_between with clipping - complex.
        # Simple approach: add a lightly shaded patch that approximates the intersection.
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        # Create paths for the circles
        path_a = Path.circle(center_a, radius)
        path_b = Path.circle(center_b, radius)

        # Approximate intersection using boolean operations is complex in matplotlib
        # Let's just add a visual cue for the intersection zone
        intersection_approx = patches.Ellipse(((center_a[0]+center_b[0])/2, (center_a[1]+center_b[1])/2),
                                              radius*0.8, radius*1.1, angle=0,
                                              color='lightgray', alpha=0.7, zorder=0) # place behind text/lines
        ax.add_patch(intersection_approx)


        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('equal')
        ax.axis('off')

        return self._save_plot(fig, filename)

# F1: Measures - Measures
    def f1_me_me_notes_thermometers(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 6), facecolor='white')
        fig.suptitle("Thermometer Scales")

        # Define bulb_radius in the outer scope
        bulb_radius = 0.5

        def draw_thermometer(ax, label, min_val, max_val, ticks, values, current_val, freeze_val, boil_val, freeze_label, boil_label, color='red'):
            # Basic thermometer shape uses bulb_radius from outer scope
            stem_width = 0.4
            plot_min = min(min_val, freeze_val) - 2*bulb_radius
            plot_max = max(max_val, boil_val) + bulb_radius*1.5
            stem_height = plot_max - plot_min - bulb_radius # Adjusted height for plot range

            # Bulb
            bulb_y = plot_min + bulb_radius
            bulb = patches.Circle((0, bulb_y), bulb_radius, color='white', ec='black', zorder=2)
            ax.add_patch(bulb)
            # Stem outline
            stem_y_start = plot_min + 2*bulb_radius
            stem = patches.Rectangle((-stem_width/2, stem_y_start), stem_width, plot_max - stem_y_start , color='white', ec='black', zorder=2)
            ax.add_patch(stem)

             # Fill level starts from bulb top
            fill_y_start = bulb_y + bulb_radius*0.9
            fill_height = current_val - fill_y_start
            fill_stem = patches.Rectangle((-stem_width/2*0.8, fill_y_start), stem_width*0.8, fill_height, color=color, ec=None, zorder=1)
            if current_val >= fill_y_start: # Only draw stem fill if above bulb top
                 ax.add_patch(fill_stem)

            # Fill bulb
            bulb_fill = patches.Circle((0, bulb_y), bulb_radius*0.9, color=color, ec=None, zorder=1)
            ax.add_patch(bulb_fill)


            # Ticks and labels
            for val in ticks:
                 y_pos = val
                 # Only draw ticks within stem visual area
                 if y_pos >= stem_y_start:
                     ax.plot([-stem_width/2 - 0.1, -stem_width/2], [y_pos, y_pos], color='black', lw=1, zorder=3)
                     if val in values:
                         ax.text(-stem_width/2 - 0.2, y_pos, f'{val}°', ha='right', va='center', zorder=3, fontsize=8) # Smaller fontsize

            # Freeze/Boil points
            ax.plot([-stem_width/2 - 0.1, stem_width/2 + 0.1], [freeze_val, freeze_val], color='blue', lw=1, linestyle='--', zorder=3) # Line across stem
            ax.text(stem_width/2 + 0.15, freeze_val, f' {freeze_val}° {freeze_label}', ha='left', va='center', color='blue', fontsize=8, zorder=3)
            ax.plot([-stem_width/2 - 0.1, stem_width/2 + 0.1], [boil_val, boil_val], color='black', lw=1, linestyle='--', zorder=3) # Line across stem
            ax.text(stem_width/2 + 0.15, boil_val, f' {boil_val}° {boil_label}', ha='left', va='center', color='black', fontsize=8, zorder=3)


            ax.set_title(label)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(plot_min, plot_max)
            ax.axis('equal')
            ax.axis('off')

        # Celsius
        c_min, c_max = -10, 40
        c_freeze, c_boil = 0, 100
        c_ticks = np.arange(c_min, c_boil + 1, 5)
        c_values = np.arange(c_min, c_boil + 1, 10)
        c_current = 25
        draw_thermometer(ax1, 'Celsius (°C)', c_min, c_max, c_ticks, c_values, c_current, c_freeze, c_boil, 'Freezing', 'Boiling')

        # Fahrenheit
        f_min, f_max = 0, 110 # Display range
        f_freeze, f_boil = 32, 212
        f_ticks = np.arange(f_min, f_boil + 1, 10)
        f_values = np.arange(f_min, f_boil + 1, 20)
        f_current = 77 # Equivalent of 25C
        draw_thermometer(ax2, 'Fahrenheit (°F)', f_min, f_max, f_ticks, f_values, f_current, f_freeze, f_boil, 'Freezing', 'Boiling')

        # No extra text calls needed here now as freeze/boil are handled inside

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)
    
    
    def f1_me_me_notes_capacity_containers(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), facecolor='white')
        fig.suptitle("Capacity Examples")

        # Measuring Jug (ax1)
        ax1.set_title("Measuring Jug")
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 12)
        ax1.axis('off')

        # Jug outline
        jug_points = [(1, 0), (1, 10), (1.5, 11), (3.5, 11), (4, 10), (4, 0), (1, 0)]
        jug_line = patches.Polygon(jug_points, closed=True, fill=False, edgecolor='black', linewidth=1.5)
        ax1.add_patch(jug_line)

        # Handle
        center_x, center_y = 4, 5
        width, height = 1.5, 5
        theta1_deg, theta2_deg = -90, 90
        handle = patches.Arc((center_x, center_y), width, height, angle=0, theta1=theta1_deg, theta2=theta2_deg, fill=False, edgecolor='black', linewidth=1.5)
        ax1.add_patch(handle)

        # Calculate arc endpoints to draw connecting lines
        theta1_rad = np.deg2rad(theta1_deg)
        theta2_rad = np.deg2rad(theta2_deg)
        end1_x = center_x + (width / 2) * np.cos(theta1_rad)
        end1_y = center_y + (height / 2) * np.sin(theta1_rad)
        end2_x = center_x + (width / 2) * np.cos(theta2_rad)
        end2_y = center_y + (height / 2) * np.sin(theta2_rad)

        # Draw connecting lines from jug edge (x=4) to arc endpoints
        ax1.plot([4, end1_x], [end1_y, end1_y], color='black', linewidth=1.5)
        ax1.plot([4, end2_x], [end2_y, end2_y], color='black', linewidth=1.5)


        # Water level (750 mL corresponds to ~7.5 height if 1000mL is 10)
        water_level = 7.5
        water = patches.Rectangle((1, 0), 3, water_level, color='lightblue', alpha=0.7)
        ax1.add_patch(water)

        # Markings
        marks_y = [2.5, 5, 7.5, 10]
        marks_label = ['250 mL', '500 mL', '750 mL', '1000 mL (1 L)']
        for y, label in zip(marks_y, marks_label):
            ax1.plot([1, 1.3], [y, y], color='black', lw=1)
            ax1.text(1.4, y, label, ha='left', va='center', fontsize=8)

        # Bottle (ax2)
        ax2.set_title("1 Liter Bottle")
        ax2.set_xlim(0, 4)
        ax2.set_ylim(0, 10)
        ax2.axis('off')

        # Bottle outline
        bottle_points = [(1, 0), (1, 7), (1.3, 8), (1.3, 9), (1.7, 9.5), (2.3, 9.5), (2.7, 9), (2.7, 8), (3, 7), (3, 0), (1, 0)]
        bottle_line = patches.Polygon(bottle_points, closed=True, fill=False, edgecolor='black', linewidth=1.5)
        ax2.add_patch(bottle_line)
        # Cap
        cap = patches.Rectangle((1.7, 9.5), 0.6, 0.5, color='gray', edgecolor='black')
        ax2.add_patch(cap)
        # Label
        ax2.text(2, 4, '1 L\n(1000 mL)', ha='center', va='center', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)

    # F1: Measures - Mensuration
    def f1_me_ms_notes_perimeter_diagrams(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), facecolor='white')
        fig.suptitle("Perimeter Calculations")

        # Square
        self._setup_ax_no_origin(axs[0], title="Square")
        axs[0].add_patch(patches.Rectangle((0.2, 0.2), 0.6, 0.6, fill=False, edgecolor='black', lw=1.5))
        axs[0].text(0.5, 0.1, 's', ha='center', va='top', fontsize=12)
        axs[0].text(0.5, 1.1, 'P = 4s', ha='center', va='bottom', fontsize=10)
        axs[0].axis('equal')

        # Rectangle
        self._setup_ax_no_origin(axs[1], title="Rectangle")
        axs[1].add_patch(patches.Rectangle((0.1, 0.2), 0.8, 0.5, fill=False, edgecolor='black', lw=1.5))
        axs[1].text(0.5, 0.1, 'l (length)', ha='center', va='top', fontsize=12)
        axs[1].text(0.05, 0.45, 'w\n(width)', ha='right', va='center', fontsize=12)
        axs[1].text(0.5, 0.9, 'P = 2(l + w)', ha='center', va='bottom', fontsize=10)
        axs[1].axis('equal')

        # Triangle
        self._setup_ax_no_origin(axs[2], title="Triangle")
        pts = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.7]])
        axs[2].add_patch(patches.Polygon(pts, closed=True, fill=False, edgecolor='black', lw=1.5))
        axs[2].text(0.5, 0.05, 'a', ha='center', va='top', fontsize=12)
        axs[2].text(0.75, 0.45, 'b', ha='left', va='bottom', fontsize=12)
        axs[2].text(0.25, 0.45, 'c', ha='right', va='bottom', fontsize=12)
        axs[2].text(0.5, 1.1, 'P = a + b + c', ha='center', va='bottom', fontsize=10)
        axs[2].axis('equal')

        for ax in axs: ax.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)

    def f1_me_ms_notes_area_diagrams(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(1, 3, figsize=(10, 4), facecolor='white') # Slightly wider figure
        fig.suptitle("Area Calculations", y=1.0) # Adjust y slightly if needed

        # --- Square ---
        ax = axs[0]
        self._setup_ax_no_origin(ax) # Remove title argument
        sq_side = 0.6
        sq_bl = (0.2, 0.25) # Bottom-left corner, shifted up slightly
        ax.add_patch(patches.Rectangle(sq_bl, sq_side, sq_side, fill=True, facecolor='lightblue', alpha=0.3, edgecolor='black', lw=1.5))
        # Labels below and inside
        ax.text(sq_bl[0] + sq_side / 2, sq_bl[1] + sq_side / 2, r'$A = s^2$', ha='center', va='center', fontsize=11) # Formula inside
        ax.text(sq_bl[0] + sq_side / 2, sq_bl[1] - 0.08, 's', ha='center', va='top', fontsize=12) # Side label below
        ax.text(0.5, 0.05, 'Square', ha='center', va='bottom', fontsize=12) # Name below
        ax.axis('equal'); ax.axis('off')

        # --- Rectangle ---
        ax = axs[1]
        self._setup_ax_no_origin(ax) # Remove title argument
        rect_w, rect_h = 0.8, 0.5
        rect_bl = (0.1, 0.25) # Bottom-left corner, shifted up slightly
        ax.add_patch(patches.Rectangle(rect_bl, rect_w, rect_h, fill=True, facecolor='lightgreen', alpha=0.3, edgecolor='black', lw=1.5))
        # Labels below, beside and inside
        ax.text(rect_bl[0] + rect_w / 2, rect_bl[1] + rect_h / 2, r'$A = l \times w$', ha='center', va='center', fontsize=11) # Formula inside
        ax.text(rect_bl[0] + rect_w / 2, rect_bl[1] - 0.08, 'l (length)', ha='center', va='top', fontsize=12) # Length below
        ax.text(rect_bl[0] - 0.05, rect_bl[1] + rect_h / 2, 'w\n(width)', ha='right', va='center', fontsize=12) # Width beside
        ax.text(0.5, 0.05, 'Rectangle', ha='center', va='bottom', fontsize=12) # Name below
        ax.axis('equal'); ax.axis('off')

        # --- Triangle ---
        ax = axs[2]
        self._setup_ax_no_origin(ax) # Remove title argument
        pts = np.array([[0.1, 0.25], [0.9, 0.25], [0.4, 0.8]]) # Shifted up slightly
        ax.add_patch(patches.Polygon(pts, closed=True, fill=True, facecolor='lightcoral', alpha=0.3, edgecolor='black', lw=1.5))
        # Base and Height Lines/Labels
        ax.text(0.5, pts[0, 1] - 0.08, 'b (base)', ha='center', va='top', fontsize=12) # Base below
        height_start_x, height_start_y = pts[2, 0], pts[0, 1] # Point on base below vertex
        height_end_x, height_end_y = pts[2, 0], pts[2, 1]   # Top Vertex
        ax.plot([height_start_x, height_end_x], [height_start_y, height_end_y], 'k--', lw=1) # Height line
        # Right angle symbol slightly offset from line start
        ax.add_patch(patches.Rectangle((height_start_x, height_start_y), 0.04, 0.04, fill=False, edgecolor='black', lw=0.5))
        ax.text(height_end_x + 0.05, (height_start_y + height_end_y) / 2, 'h  (height)', ha='left', va='center', fontsize=12) # Height label beside
        # Formula and Name below
        ax.text(0.5, pts[0, 1] + (pts[2, 1]-pts[0, 1])/2.5 , r'$A = \frac{1}{2}bh$', ha='center', va='center', fontsize=11) # Formula inside (adjusted placement)
        ax.text(0.5, 0.05, 'Triangle', ha='center', va='bottom', fontsize=12) # Name below
        ax.axis('equal'); ax.axis('off')

        plt.subplots_adjust(top=0.85) # Adjust top margin to make space for suptitle
        # plt.tight_layout(rect=[0, 0, 1, 0.90]) # tight_layout often overrides subplots_adjust
        return self._save_plot(fig, filename)
    
    def f1_me_ms_ques_square_area(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Square Area Problem")

        side = 0.7 # Arbitrary size for drawing
        bottom_left = (0.15, 0.15)
        ax.add_patch(patches.Rectangle(bottom_left, side, side, fill=False, edgecolor='black', lw=1.5))

        # Labels
        ax.text(bottom_left[0] + side / 2, bottom_left[1] - 0.05, 's = ?', ha='center', va='top', fontsize=12)
        ax.text(bottom_left[0] + side / 2, bottom_left[1] + side / 2, 'Area = 49 m²', ha='center', va='center', fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('equal')
        ax.axis('off')

        return self._save_plot(fig, filename)

    # F1: Graphs - Functional Graphs
    def f1_gr_fg_notes_cartesian_plane(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title="Cartesian Plane", xlabel="x-axis", ylabel="y-axis", xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), aspect='equal')

        # Add ticks and labels
        ax.set_xticks(np.arange(-5, 6))
        ax.set_yticks(np.arange(-5, 6))

        # Label origin
        ax.text(0.1, 0.1, '(0,0)', ha='left', va='bottom')
        ax.plot(0, 0, 'ko') # Mark origin

        # Label Quadrants
        ax.text(3, 3, 'I', ha='center', va='center', fontsize=14, alpha=0.7)
        ax.text(-3, 3, 'II', ha='center', va='center', fontsize=14, alpha=0.7)
        ax.text(-3, -3, 'III', ha='center', va='center', fontsize=14, alpha=0.7)
        ax.text(3, -3, 'IV', ha='center', va='center', fontsize=14, alpha=0.7)

        # Add arrowheads for axes
        ax.arrow(0, -5.5, 0, 11, head_width=0.2, head_length=0.3, fc='black', ec='black', length_includes_head=True, clip_on=False)
        ax.arrow(-5.5, 0, 11, 0, head_width=0.2, head_length=0.3, fc='black', ec='black', length_includes_head=True, clip_on=False)
        ax.text(5.5, -0.3, 'x', ha='center', va='top', fontsize=12)
        ax.text(-0.3, 5.5, 'y', ha='right', va='center', fontsize=12)

        return self._save_plot(fig, filename)

    def f1_gr_fg_work_cartesian_plane_scale(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white') # Assume 1 unit = 1 cm for default figsize
        self._setup_ax(ax, title="Cartesian Plane (-4 to 4, 1cm:1unit)", xlabel="x", ylabel="y", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), aspect='equal')
        ax.set_xticks(np.arange(-4, 5))
        ax.set_yticks(np.arange(-4, 5))
        ax.plot(0, 0, 'ko') # Mark origin
        return self._save_plot(fig, filename)

    def f1_gr_fg_work_plot_point_a(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax(ax, title="Plot Point A(3, 2)", xlabel="x", ylabel="y", xlim=(-1, 5), ylim=(-1, 5), aspect='equal')
        ax.set_xticks(np.arange(-1, 6))
        ax.set_yticks(np.arange(-1, 6))

        # Plot point
        x, y = 3, 2
        ax.plot(x, y, 'ro', markersize=8) # Red dot for point A
        ax.text(x + 0.1, y + 0.1, 'A(3, 2)', fontsize=10)

        # Dashed lines to axes
        ax.plot([x, x], [0, y], 'r--')
        ax.plot([0, x], [y, y], 'r--')

        return self._save_plot(fig, filename)

    def f1_gr_fg_work_plot_points_bcd(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title="Plot Points B, C, D", xlabel="x", ylabel="y", xlim=(-5, 5), ylim=(-5, 5), aspect='equal')
        ax.set_xticks(np.arange(-5, 6))
        ax.set_yticks(np.arange(-5, 6))

        points = {'B': (-2, 4), 'C': (-3, -1), 'D': (4, -3)}
        colors = {'B': 'blue', 'C': 'green', 'D': 'purple'}

        for label, (x, y) in points.items():
            ax.plot(x, y, marker='x', color=colors[label], markersize=8, linestyle='')
            ax.text(x + 0.1, y + 0.1, f'{label}({x}, {y})', fontsize=10, color=colors[label])

        return self._save_plot(fig, filename)

    def f1_gr_fg_work_identify_points_pqr(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Identify Points P, Q, R", xlabel="x", ylabel="y", xlim=(-4, 4), ylim=(-4, 4), aspect='equal')
        ax.set_xticks(np.arange(-4, 5))
        ax.set_yticks(np.arange(-4, 5))

        points = {'P': (2, 3), 'Q': (-3, 1), 'R': (1, -2)}
        colors = {'P': 'red', 'Q': 'blue', 'R': 'green'}

        for label, (x, y) in points.items():
            ax.plot(x, y, marker='o', color=colors[label], markersize=8, linestyle='')
            ax.text(x + 0.1, y + 0.1, label, fontsize=12, color=colors[label], fontweight='bold')

        return self._save_plot(fig, filename)

    def f1_gr_fg_ques_blank_grid_neg5_5(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title="Cartesian Plane (-5 to 5)", xlabel="x", ylabel="y", xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), aspect='equal')
        ax.set_xticks(np.arange(-5, 6))
        ax.set_yticks(np.arange(-5, 6))
        ax.plot(0,0,'ko') # Mark origin

        return self._save_plot(fig, filename)

    def f1_gr_fg_ques_rectangle_vertices(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Find Vertex D of Rectangle", xlabel="x", ylabel="y", xlim=(0, 5.5), ylim=(0, 5.5), aspect='equal')
        ax.set_xticks(np.arange(0, 6))
        ax.set_yticks(np.arange(0, 6))

        points = {'A': (1, 2), 'B': (4, 2), 'C': (4, 4)}
        x_coords = [p[0] for p in points.values()]
        y_coords = [p[1] for p in points.values()]

        ax.plot(x_coords, y_coords, 'bo', markersize=8)
        for label, (x, y) in points.items():
             ax.text(x + 0.1, y + 0.1, f'{label}({x}, {y})', fontsize=10)

        # Indicate where D should be (optional)
        ax.text(1 + 0.1, 4 + 0.1, 'D?', fontsize=10, color='red')
        ax.plot([1,4], [4,4], 'r--')
        ax.plot([1,1], [2,4], 'r--')

        return self._save_plot(fig, filename)

    # F1: Graphs - Travel Graphs
    # def f1_gr_tg_notes_distance_time(self):
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    #     self._setup_ax(ax, title="Distance-Time Graph Example", xlabel="Time (hours)", ylabel="Distance from Start (km)", xlim=(0, 6), ylim=(-5, 90))
    #     ax.set_xticks(np.arange(0, 6))

    #     # Segments O-A, A-B, B-C
    #     time = [0, 2, 3, 5]
    #     dist = [0, 80, 80, 0]
    #     points = {'O': (0,0), 'A': (2,80), 'B': (3,80), 'C': (5,0)}

    #     ax.plot(time, dist, 'b-o', markersize=5)

    #     for label, (t, d) in points.items():
    #          ax.text(t, d+3 if d < 80 else d-6, f'{label}({t}h, {d}km)', fontsize=9, ha='center')

    #     # Annotations for segments
    #     ax.text(1, 40, 'Constant Speed\nAway', ha='center', va='center', color='blue', fontsize=8)
    #     ax.text(2.5, 70, 'Stationary', ha='center', va='bottom', color='blue', fontsize=8)
    #     ax.text(4, 40, 'Constant Speed\nReturn', ha='center', va='center', color='blue', fontsize=8)

    def f1_gr_tg_notes_distance_time(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        self._setup_ax(ax, title="Distance-Time Graph Example", xlabel="Time (hours)", ylabel="Distance from Start (km)", xlim=(-0.2, 5.2), ylim=(-5, 90)) # Adjusted xlim slightly
        ax.set_xticks(np.arange(0, 6))
        ax.set_yticks(np.arange(0, 91, 10)) # Added y-ticks for clarity

        # Segments O-A, A-B, B-C
        time = [0, 2, 3, 5]
        dist = [0, 80, 80, 0]
        points = {'O': (0,0), 'A': (2,80), 'B': (3,80), 'C': (5,0)}

        ax.plot(time, dist, 'b-o', markersize=5)

        # Point Labels (Adjusted positions slightly)
        ax.text(points['O'][0], points['O'][1]-2, f'O({points["O"][0]}h, {points["O"][1]}km)', fontsize=9, ha='left', va='top')
        ax.text(points['A'][0]-0.1, points['A'][1]+2, f'A({points["A"][0]}h, {points["A"][1]}km)', fontsize=9, ha='right', va='bottom')
        ax.text(points['B'][0]+0.1, points['B'][1]+2, f'B({points["B"][0]}h, {points["B"][1]}km)', fontsize=9, ha='left', va='bottom')
        ax.text(points['C'][0], points['C'][1]-2, f'C({points["C"][0]}h, {points["C"][1]}km)', fontsize=9, ha='right', va='top')


        # Annotations for segments (Repositioned)
        # Constant Speed Away (moved above line)
        ax.text(1, 45, 'Constant Speed\nAway', ha='center', va='bottom', color='blue', fontsize=8, rotation=38) # Rotate to match line slope roughly
        # Stationary (moved above line)
        ax.text(2.5, 83, 'Stationary', ha='center', va='bottom', color='blue', fontsize=8)
        # Constant Speed Return (moved above line)
        ax.text(4, 45, 'Constant Speed\nReturn', ha='center', va='bottom', color='blue', fontsize=8, rotation=-38) # Rotate to match line slope roughly


        return self._save_plot(fig, filename) 

    def f1_gr_tg_work_car_journey_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        self._setup_ax(ax, title="Car's Journey", xlabel="Time (hours)", ylabel="Distance (km)", xlim=(0, 4.5), ylim=(-5, 130))
        ax.set_xticks(np.arange(0, 5))

        time = [0, 2, 3, 4]
        dist = [0, 100, 100, 120]

        ax.plot(time, dist, 'g-o', markersize=5)

        return self._save_plot(fig, filename)

    # Re-use function for subsequent examples/questions referencing the same graph
    f1_gr_tg_work_stationary_time = f1_gr_tg_work_car_journey_graph
    f1_gr_tg_work_speed_first_2_hours = f1_gr_tg_work_car_journey_graph
    f1_gr_tg_work_total_distance = f1_gr_tg_work_car_journey_graph
    f1_gr_tg_work_average_speed = f1_gr_tg_work_car_journey_graph
    f1_gr_tg_ques_distance_t3_t4 = f1_gr_tg_work_car_journey_graph
    f1_gr_tg_ques_speed_t3_t4 = f1_gr_tg_work_car_journey_graph

    def f1_gr_tg_ques_cyclist_journey_grid(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        self._setup_ax(ax, title="Cyclist Journey Grid", xlabel="Time (hours)", ylabel="Distance from home (km)", xlim=(-0.2, 5.2), ylim=(-2, 42))
        ax.set_xticks(np.arange(0, 6))
        ax.set_yticks(np.arange(0, 41, 5))

        return self._save_plot(fig, filename)

    # F1: Geometry - Points Lines Angles
    def f1_ge_pl_notes_types_of_lines(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(3, 2, figsize=(8, 8), facecolor='white') # Adjusted grid
        fig.suptitle("Types of Lines")

        # Parallel Lines
        ax = axs[0, 0]
        self._setup_ax_no_origin(ax, title="Parallel Lines")
        ax.plot([0.1, 0.9], [0.6, 0.6], 'k-', lw=1.5)
        ax.plot([0.1, 0.9], [0.4, 0.4], 'k-', lw=1.5)
        # Arrows for parallel
        ax.plot([0.45, 0.55], [0.6, 0.6], 'k>', markersize=5)
        ax.plot([0.45, 0.55], [0.4, 0.4], 'k>', markersize=5)
        ax.text(0.5, 0.5, '//', ha='center', va='center', fontsize=12)
        ax.axis('equal'); ax.axis('off')

        # Intersecting Lines
        ax = axs[0, 1]
        self._setup_ax_no_origin(ax, title="Intersecting Lines")
        ax.plot([0.1, 0.9], [0.1, 0.9], 'k-', lw=1.5)
        ax.plot([0.1, 0.9], [0.9, 0.1], 'k-', lw=1.5)
        ax.axis('equal'); ax.axis('off')

        # Perpendicular Lines
        ax = axs[1, 0]
        self._setup_ax_no_origin(ax, title="Perpendicular Lines")
        ax.plot([0.1, 0.9], [0.5, 0.5], 'k-', lw=1.5) # Horizontal
        ax.plot([0.5, 0.5], [0.1, 0.9], 'k-', lw=1.5) # Vertical
        # Right angle symbol
        ax.add_patch(patches.Rectangle((0.5, 0.5), 0.05, 0.05, fill=False, edgecolor='black', lw=1))
        ax.text(0.55, 0.55, '⊥', ha='left', va='bottom', fontsize=12)
        ax.axis('equal'); ax.axis('off')

        # Line Segment
        ax = axs[1, 1]
        self._setup_ax_no_origin(ax, title="Line Segment")
        ax.plot([0.2, 0.8], [0.5, 0.5], 'k-', lw=1.5)
        ax.plot([0.2, 0.8], [0.5, 0.5], 'k.', markersize=8) # Endpoints
        ax.text(0.2, 0.4, 'A', ha='center', va='top')
        ax.text(0.8, 0.4, 'B', ha='center', va='top')
        ax.axis('equal'); ax.axis('off')

        # Ray
        ax = axs[2, 0]
        self._setup_ax_no_origin(ax, title="Ray")
        ax.plot(0.2, 0.5, 'k.', markersize=8) # Startpoint
        ax.arrow(0.2, 0.5, 0.7, 0, head_width=0.05, head_length=0.1, fc='black', ec='black', length_includes_head=True, lw=1.5)
        ax.text(0.2, 0.4, 'C', ha='center', va='top')
        ax.text(0.6, 0.4, 'D', ha='center', va='top') # Point it passes through
        ax.axis('equal'); ax.axis('off')

        # Hide unused subplot
        axs[2, 1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)

    def f1_ge_pl_notes_types_of_angles(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(2, 3, figsize=(9, 6), facecolor='white')
        fig.suptitle("Types of Angles")
        axs_flat = axs.flatten()

        def draw_angle(ax, angle_deg, label, symbol=None):
            # Remove the title argument from the setup
            self._setup_ax_no_origin(ax)
            rad = np.deg2rad(angle_deg)
            origin = [0.2, 0.2] # Base origin
            # Adjust origin slightly upwards for bottom row labels if needed for clearance
            if "Reflex" in label or "Full" in label or "Straight" in label:
                origin = [0.2, 0.25] # Shift drawing up slightly for these

            ax.plot(origin[0], origin[1], 'ko') # Vertex
            # First arm (horizontal)
            ax.plot([origin[0], origin[0] + 0.7], [origin[1], origin[1]], 'k-', lw=1.5)
            # Second arm
            x2 = origin[0] + 0.7 * np.cos(rad)
            y2 = origin[1] + 0.7 * np.sin(rad)
            ax.plot([origin[0], x2], [origin[1], y2], 'k-', lw=1.5)

            # Arc for angle
            arc_rad = 0.2
            angle_marker = None

            # Match label strings exactly for specific logic
            if label == "Reflex Angle (>180°, <360°)":
                angle_marker = patches.Arc(origin, arc_rad*2, arc_rad*2, angle=0, theta1=0, theta2=angle_deg, color='red', lw=1)
            elif label == "Full Angle (360°)":
                 angle_marker = patches.Circle(origin, arc_rad, color='red', fill=False, lw=1) # Use Circle for full angle
                 # Optional: Add arrow hint
                 # ax.arrow(...)
            else: # Acute, Right, Obtuse, Straight
                angle_marker = patches.Arc(origin, arc_rad*2, arc_rad*2, angle=0, theta1=0, theta2=angle_deg, color='red', lw=1)

            if angle_marker: ax.add_patch(angle_marker)

            # Right angle symbol
            if symbol == 'square':
                 ax.add_patch(patches.Rectangle((origin[0], origin[1]), 0.05, 0.05, fill=False, edgecolor='red', lw=1))

            # Add the descriptive label text below the diagram
            # Adjust vertical position based on angle type to avoid overlap
            label_y_pos = 0.05 # Default bottom position
            if "Reflex" in label:
                label_y_pos = -0.4 # Slightly higher for these two
            elif "Full" in label:
                 label_y_pos = -0.1 # Slightly higher for these two

            ax.text(0.5, label_y_pos, label, ha='center', va='bottom', fontsize=10)

            ax.set_xlim(0, 1); ax.set_ylim(0, 1) # Keep original limits
            ax.axis('equal'); ax.axis('off')

        draw_angle(axs_flat[0], 45, "Acute Angle (<90°)")
        draw_angle(axs_flat[1], 90, "Right Angle (90°)", symbol='square')
        draw_angle(axs_flat[2], 135, "Obtuse Angle (>90°, <180°)")
        draw_angle(axs_flat[3], 180, "Straight Angle (180°)")
        draw_angle(axs_flat[4], 240, "Reflex Angle (>180°, <360°)") # Angle > 180 shown externally
        draw_angle(axs_flat[5], 360, "Full Angle (360°)") # Shown as revolution

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)

    def f1_ge_pl_notes_angles_straight_line(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
        self._setup_ax_no_origin(ax, title="Angles on a Straight Line")

        origin = (0.5, 0.3)
        # Straight line
        ax.plot([0.1, 0.9], [origin[1], origin[1]], 'k-', lw=1.5)
        ax.plot(origin[0], origin[1], 'ko') # Vertex O

        # Rays dividing the upper angle
        angles = [40, 70, 70] # Example angles summing to 180
        current_angle = 0
        labels = ['a', 'b', 'c']
        colors = ['red', 'green', 'blue']

        for i, angle in enumerate(angles):
            rad1 = np.deg2rad(current_angle)
            rad2 = np.deg2rad(current_angle + angle)
            x1 = origin[0] + 0.5 * np.cos(rad1)
            y1 = origin[1] + 0.5 * np.sin(rad1)
            x2 = origin[0] + 0.5 * np.cos(rad2)
            y2 = origin[1] + 0.5 * np.sin(rad2)

            if i < len(angles): # Draw dividing ray
                 ax.plot([origin[0], x2], [origin[1], y2], 'k-', lw=1)

            # Arc and label
            arc_rad = 0.15 + i*0.03 # Stagger arcs slightly
            mid_angle = current_angle + angle / 2
            arc = patches.Arc(origin, arc_rad*2, arc_rad*2, angle=0, theta1=current_angle, theta2=current_angle+angle, color=colors[i], lw=1)
            ax.add_patch(arc)
            label_x = origin[0] + (arc_rad + 0.05) * np.cos(np.deg2rad(mid_angle))
            label_y = origin[1] + (arc_rad + 0.05) * np.sin(np.deg2rad(mid_angle))
            ax.text(label_x, label_y, labels[i], ha='center', va='center', color=colors[i], fontsize=12)

            current_angle += angle

        ax.text(0.5, 0.05, 'a + b + c = 180°', ha='center', va='top', fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_pl_notes_angles_around_point(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Angles Around a Point")

        origin = (0.5, 0.5)
        ax.plot(origin[0], origin[1], 'ko') # Point P

        # Rays and angles
        angles = [100, 80, 50, 130] # Example angles summing to 360
        current_angle = 0
        labels = ['w', 'x', 'y', 'z']
        colors = ['red', 'green', 'blue', 'purple']

        for i, angle in enumerate(angles):
            rad1 = np.deg2rad(current_angle)
            rad2 = np.deg2rad(current_angle + angle)
            x2 = origin[0] + 0.4 * np.cos(rad2)
            y2 = origin[1] + 0.4 * np.sin(rad2)

            ax.plot([origin[0], x2], [origin[1], y2], 'k-', lw=1) # Draw ray

            # Arc and label
            arc_rad = 0.15 + i*0.02
            mid_angle = current_angle + angle / 2
            arc = patches.Arc(origin, arc_rad*2, arc_rad*2, angle=0, theta1=current_angle, theta2=current_angle+angle, color=colors[i], lw=1)
            ax.add_patch(arc)
            label_x = origin[0] + (arc_rad + 0.05) * np.cos(np.deg2rad(mid_angle))
            label_y = origin[1] + (arc_rad + 0.05) * np.sin(np.deg2rad(mid_angle))
            ax.text(label_x, label_y, labels[i], ha='center', va='center', color=colors[i], fontsize=12)

            current_angle += angle

        ax.text(0.5, 0.05, 'w + x + y + z = 360°', ha='center', va='top', fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_pl_work_angles_straight_line_ex(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
        self._setup_ax_no_origin(ax, title="Find Angle x")

        origin = (0.5, 0.3)
        # Straight line AOB
        ax.plot([0.1, 0.9], [origin[1], origin[1]], 'k-', lw=1.5)
        ax.plot(origin[0], origin[1], 'ko') # Vertex O
        ax.text(0.1, origin[1]-0.05, 'A', ha='center', va='top')
        ax.text(0.9, origin[1]-0.05, 'B', ha='center', va='top')
        ax.text(origin[0], origin[1]-0.05, 'O', ha='center', va='top')

        # Ray OC
        angle_aoc = 130
        rad_c = np.deg2rad(angle_aoc)
        xc = origin[0] - 0.5 * np.cos(np.deg2rad(180 - angle_aoc)) # Adjust to draw from correct side
        yc = origin[1] + 0.5 * np.sin(np.deg2rad(180 - angle_aoc))
        ax.plot([origin[0], xc], [origin[1], yc], 'k-', lw=1)
        ax.text(xc-0.05, yc+0.05, 'C', ha='right', va='bottom')

        # Arcs and labels
        arc_aoc = patches.Arc(origin, 0.3, 0.3, angle=0, theta1=0, theta2=angle_aoc, color='red', lw=1)
        ax.add_patch(arc_aoc)
        ax.text(origin[0] + 0.2*np.cos(np.deg2rad(angle_aoc/2)), origin[1] + 0.2*np.sin(np.deg2rad(angle_aoc/2)), '130°', color='red', ha='center', va='center')

        arc_boc = patches.Arc(origin, 0.4, 0.4, angle=0, theta1=angle_aoc, theta2=180, color='blue', lw=1)
        ax.add_patch(arc_boc)
        mid_boc = (angle_aoc + 180) / 2
        ax.text(origin[0] + 0.25*np.cos(np.deg2rad(mid_boc)), origin[1] + 0.25*np.sin(np.deg2rad(mid_boc)), 'x', color='blue', ha='center', va='center')


        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_pl_work_angles_around_point_ex(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Find Angle y")

        origin = (0.5, 0.5)
        ax.plot(origin[0], origin[1], 'ko') # Point P

        # Rays and angles
        angles = [100, 140] # Given angles
        y_angle = 360 - sum(angles)
        all_angles = angles + [y_angle]
        current_angle = 0 # Start direction (e.g., East)
        labels = ['100°', '140°', 'y']
        colors = ['red', 'green', 'blue']

        for i, angle in enumerate(all_angles):
            rad2 = np.deg2rad(current_angle + angle)
            x2 = origin[0] + 0.4 * np.cos(rad2)
            y2 = origin[1] + 0.4 * np.sin(rad2)

            ax.plot([origin[0], x2], [origin[1], y2], 'k-', lw=1) # Draw ray

            # Arc and label
            arc_rad = 0.15
            mid_angle = current_angle + angle / 2
            arc = patches.Arc(origin, arc_rad*2, arc_rad*2, angle=0, theta1=current_angle, theta2=current_angle+angle, color=colors[i], lw=1)
            ax.add_patch(arc)
            label_x = origin[0] + (arc_rad + 0.05) * np.cos(np.deg2rad(mid_angle))
            label_y = origin[1] + (arc_rad + 0.05) * np.sin(np.deg2rad(mid_angle))
            ax.text(label_x, label_y, labels[i], ha='center', va='center', color=colors[i], fontsize=12)

            current_angle += angle

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_pl_work_measure_angle_ex(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor='white') # Slightly more height
        self._setup_ax_no_origin(ax, title="Angle for Measurement")

        angle_deg = 65
        origin = (0.3, 0.3) # Shifted origin for better centering
        arm_len = 0.6
        ax.plot(origin[0], origin[1], 'ko') # Vertex B
        ax.text(origin[0]+0.03, origin[1], 'B', ha='right', va='top')

        # Arm BA (horizontal left)
        A = (origin[0] - arm_len, origin[1])
        ax.plot([origin[0], A[0]], [origin[1], A[1]], 'k-', lw=1.5)
        ax.text(A[0] - 0.02, A[1], 'A', ha='right', va='center')

        # Arm BC
        angle_BC_rad = np.deg2rad(180 - angle_deg) # Angle of BC relative to positive x-axis
        x2 = origin[0] + arm_len * np.cos(angle_BC_rad)
        y2 = origin[1] + arm_len * np.sin(angle_BC_rad)
        ax.plot([origin[0], x2], [origin[1], y2], 'k-', lw=1.5)
        ax.text(x2-0.04, y2-0.01, 'C', ha='center', va='bottom')

        # Correct Arc Parameters
        arc_rad = 0.25 # Slightly larger arc
        theta1 = 180 - angle_deg # Start angle (arm BC)
        theta2 = 180           # End angle (arm BA)
        angle_marker = patches.Arc(origin, arc_rad*2, arc_rad*2, angle=0, theta1=theta1, theta2=theta2, color='red', lw=1.5)
        ax.add_patch(angle_marker)

        # Add angle label
        mid_angle_rad = np.deg2rad((theta1 + theta2) / 2)
        label_rad = arc_rad + 0.08
        label_x = origin[0] + label_rad * np.cos(mid_angle_rad)
        label_y = origin[1] + label_rad * np.sin(mid_angle_rad)
        ax.text(label_x, label_y, f'{angle_deg}' + r'$^\circ$', color='red', ha='center', va='center', fontsize=10) # Use degree symbol

        ax.set_xlim(origin[0] - arm_len - 0.1, origin[0] + arm_len * 0.6)
        ax.set_ylim(origin[1] - 0.1, origin[1] + arm_len + 0.1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_pl_ques_angles_straight_line_ex(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
        self._setup_ax_no_origin(ax, title="Solve for x")

        origin = (0.5, 0.3)
        # Straight line
        ax.plot([0.1, 0.9], [origin[1], origin[1]], 'k-', lw=1.5)
        ax.plot(origin[0], origin[1], 'ko') # Vertex O

        # Ray OC dividing the angle
        split_angle = 70 # Angle for the ray (arbitrary for diagram)
        rad_c = np.deg2rad(split_angle)
        xc = origin[0] + 0.5 * np.cos(rad_c)
        yc = origin[1] + 0.5 * np.sin(rad_c)
        ax.plot([origin[0], xc], [origin[1], yc], 'k-', lw=1)

        # Arcs and labels
        arc1 = patches.Arc(origin, 0.3, 0.3, angle=0, theta1=0, theta2=split_angle, color='red', lw=1)
        ax.add_patch(arc1)
        mid1 = split_angle / 2
        ax.text(origin[0] + 0.2*np.cos(np.deg2rad(mid1)), origin[1] + 0.2*np.sin(np.deg2rad(mid1)), '(2x)°', color='red', ha='center', va='center', fontsize=10)

        arc2 = patches.Arc(origin, 0.4, 0.4, angle=0, theta1=split_angle, theta2=180, color='blue', lw=1)
        ax.add_patch(arc2)
        mid2 = (split_angle + 180) / 2
        ax.text(origin[0] + 0.25*np.cos(np.deg2rad(mid2)), origin[1] + 0.25*np.sin(np.deg2rad(mid2)), '(x+30)°', color='blue', ha='center', va='center', fontsize=10)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_pl_ques_angles_around_point_ex(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Find Angle z")

        origin = (0.5, 0.5)
        ax.plot(origin[0], origin[1], 'ko') # Point P

        # Rays and angles
        angles = [90, 135] # Given angles
        z_angle = 360 - sum(angles)
        all_angles = angles + [z_angle]
        current_angle = 0 # Start direction (e.g., East)
        labels = ['90°', '135°', 'z']
        colors = ['red', 'green', 'blue']
        symbols = ['square', None, None]

        for i, angle in enumerate(all_angles):
            rad2 = np.deg2rad(current_angle + angle)
            x2 = origin[0] + 0.4 * np.cos(rad2)
            y2 = origin[1] + 0.4 * np.sin(rad2)

            ax.plot([origin[0], x2], [origin[1], y2], 'k-', lw=1) # Draw ray

            # Arc and label
            arc_rad = 0.15
            mid_angle = current_angle + angle / 2
            if symbols[i] == 'square':
                 arc = patches.Rectangle((origin[0], origin[1]), 0.05*np.cos(np.deg2rad(current_angle+45)), 0.05*np.sin(np.deg2rad(current_angle+45)), fill=False, edgecolor=colors[i], lw=1)
                 ax.add_patch(patches.Polygon([[origin[0], origin[1]], [origin[0]+0.1, origin[1]], [origin[0]+0.1, origin[1]+0.1], [origin[0], origin[1]+0.1]], fill=False, edgecolor=colors[i], lw=1))
            else:
                arc = patches.Arc(origin, arc_rad*2, arc_rad*2, angle=0, theta1=current_angle, theta2=current_angle+angle, color=colors[i], lw=1)
                ax.add_patch(arc)

            label_x = origin[0] + (arc_rad + 0.05) * np.cos(np.deg2rad(mid_angle))
            label_y = origin[1] + (arc_rad + 0.05) * np.sin(np.deg2rad(mid_angle))
            ax.text(label_x, label_y, labels[i], ha='center', va='center', color=colors[i], fontsize=12)

            current_angle += angle

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # F1: Geometry - Polygons Circles
    def f1_ge_pc_notes_parts_of_circle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        self._setup_ax_no_origin(ax, title="Parts of a Circle")

        center = (0.5, 0.5)
        radius = 0.4

        # Circle
        circle = patches.Circle(center, radius, fill=False, edgecolor='black', lw=1.5)
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'ko', label='Center (O)') # Center

        # Radius
        p_on_circle_r = (center[0] + radius * np.cos(np.pi/4), center[1] + radius * np.sin(np.pi/4))
        ax.plot([center[0], p_on_circle_r[0]], [center[1], p_on_circle_r[1]], 'b-', label='Radius (r)')

        # Diameter
        p1_on_circle_d = (center[0] - radius, center[1])
        p2_on_circle_d = (center[0] + radius, center[1])
        ax.plot([p1_on_circle_d[0], p2_on_circle_d[0]], [p1_on_circle_d[1], p2_on_circle_d[1]], 'g-', label='Diameter (d)')

        # Chord
        p1_on_circle_c = (center[0] + radius * np.cos(np.pi*3/4), center[1] + radius * np.sin(np.pi*3/4))
        p2_on_circle_c = (center[0] + radius * np.cos(np.pi*1/6), center[1] + radius * np.sin(np.pi*1/6))
        ax.plot([p1_on_circle_c[0], p2_on_circle_c[0]], [p1_on_circle_c[1], p2_on_circle_c[1]], 'm-', label='Chord')

        # Arc
        arc_theta1 = 200
        arc_theta2 = 250
        arc = patches.Arc(center, radius*2, radius*2, angle=0, theta1=arc_theta1, theta2=arc_theta2, color='orange', lw=2, label='Arc')
        ax.add_patch(arc)

        # Tangent
        tan_point = (center[0], center[1] - radius)
        ax.plot([tan_point[0] - 0.3, tan_point[0] + 0.3], [tan_point[1], tan_point[1]], 'r-', label='Tangent')

        # Secant
        sec_y = center[1] + radius * 0.5
        sec_x1 = center[0] - np.sqrt(radius**2 - (sec_y - center[1])**2)
        sec_x2 = center[0] + np.sqrt(radius**2 - (sec_y - center[1])**2)
        ax.plot([sec_x1 - 0.2, sec_x2 + 0.2], [sec_y, sec_y], 'c-', label='Secant') # Extend beyond circle

        # Sector
        sector_theta1 = 290
        sector_theta2 = 340
        sector_p1 = (center[0] + radius * np.cos(np.deg2rad(sector_theta1)), center[1] + radius * np.sin(np.deg2rad(sector_theta1)))
        sector_p2 = (center[0] + radius * np.cos(np.deg2rad(sector_theta2)), center[1] + radius * np.sin(np.deg2rad(sector_theta2)))
        sector = patches.Wedge(center, radius, sector_theta1, sector_theta2, facecolor='yellow', alpha=0.5, label='Sector')
        ax.add_patch(sector)
        ax.plot([center[0], sector_p1[0]], [center[1], sector_p1[1]], 'k-', lw=0.5)
        ax.plot([center[0], sector_p2[0]], [center[1], sector_p2[1]], 'k-', lw=0.5)

        # Segment
        seg_p1 = p1_on_circle_c # Use chord endpoints
        seg_p2 = p2_on_circle_c
        seg_angle1 = np.rad2deg(np.arctan2(seg_p1[1]-center[1], seg_p1[0]-center[0]))
        seg_angle2 = np.rad2deg(np.arctan2(seg_p2[1]-center[1], seg_p2[0]-center[0]))
        segment_wedge = patches.Wedge(center, radius, min(seg_angle1, seg_angle2), max(seg_angle1, seg_angle2), facecolor='lightgrey', alpha=0.6)
        ax.add_patch(segment_wedge)
        segment_triangle = patches.Polygon([center, seg_p1, seg_p2], facecolor='white', alpha=1.0) # Mask triangle part of wedge
        ax.add_patch(segment_triangle)
        ax.plot([seg_p1[0], seg_p2[0]], [seg_p1[1], seg_p2[1]], 'm-') # Redraw chord on top
        ax.text(0.3, 0.3, 'Segment', fontsize=9)


        # Add labels via legend or direct text (direct text can be messy)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.text(center[0]+0.02, center[1]+0.02, 'O', ha='left', va='bottom') # Center Label

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        plt.tight_layout(rect=[0, 0, 0.8, 1]) # Make space for legend
        return self._save_plot(fig, filename)

    def f1_ge_pc_work_circle_diameter(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Identify Line Segments")

        center = (0.5, 0.5)
        radius = 0.4

        # Circle
        circle = patches.Circle(center, radius, fill=False, edgecolor='black', lw=1.5)
        ax.add_patch(circle)
        ax.plot(center[0], center[1], 'ko') # Center O
        ax.text(center[0]+0.02, center[1]+0.02, 'O', ha='left', va='bottom')

        # Diameter AB
        p1_A = (center[0] - radius, center[1])
        p2_B = (center[0] + radius, center[1])
        ax.plot([p1_A[0], p2_B[0]], [p1_A[1], p2_B[1]], 'g-', lw=1.5)
        ax.text(p1_A[0]-0.05, p1_A[1], 'A', ha='right', va='bottom')
        ax.text(p2_B[0]+0.05, p2_B[1], 'B', ha='left', va='bottom')

        # Chord CD
        p1_C = (center[0] + radius * np.cos(np.pi*3/4), center[1] + radius * np.sin(np.pi*3/4))
        p2_D = (center[0] + radius * np.cos(np.pi*(-1/4)), center[1] + radius * np.sin(np.pi*(-1/4)))
        ax.plot([p1_C[0], p2_D[0]], [p1_C[1], p2_D[1]], 'm-', lw=1.5)
        ax.text(p1_C[0]-0.05, p1_C[1]+0.05, 'C', ha='right', va='bottom')
        ax.text(p2_D[0]+0.05, p2_D[1]-0.05, 'D', ha='left', va='bottom')


        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # F1: Geometry - Construction Loci
    def f1_ge_cl_notes_copy_line_segment(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Copying Line Segment AB onto Ray from P")

        # Original Segment AB
        A = (0.1, 0.7); B = (0.4, 0.7)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', lw=1.5)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k.', markersize=8)
        ax.text(A[0], A[1]-0.05, 'A', ha='center', va='top')
        ax.text(B[0], B[1]-0.05, 'B', ha='center', va='top')
        length_ab = B[0] - A[0]

        # Ray from P
        P = (0.1, 0.3)
        ax.plot(P[0], P[1], 'ko', markersize=5)
        ax.text(P[0], P[1]-0.05, 'P', ha='center', va='top')
        ax.arrow(P[0], P[1], 0.8, 0, head_width=0.03, head_length=0.05, fc='black', ec='black', length_includes_head=True, lw=1)

        # Compass arc
        Q = (P[0] + length_ab, P[1])
        arc = patches.Arc(P, length_ab*2, length_ab*2, angle=0, theta1=-30, theta2=30, color='red', linestyle='--', lw=1)
        ax.add_patch(arc)
        ax.plot(Q[0], Q[1], 'ro', markersize=5) # Intersection point
        ax.text(Q[0], Q[1]-0.05, 'Q', ha='center', va='top')

        # Indicate equality
        ax.text((P[0]+Q[0])/2, P[1]+0.05, 'PQ = AB', ha='center', va='bottom')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_cl_work_draw_line_segment(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 2), facecolor='white')
        self._setup_ax_no_origin(ax, title="Drawing 7cm Line Segment with Ruler")

        # Simulate ruler
        ax.add_patch(patches.Rectangle((0.5, 0.4), 8, 0.4, fill=True, facecolor='lightyellow', alpha=0.5, edgecolor='black'))
        for i in range(9): # cm marks
             ax.plot([0.5+i, 0.5+i], [0.4, 0.8], 'k-', lw=1)
             ax.text(0.5+i, 0.85, str(i), ha='center', va='bottom', fontsize=8)
        for i in range(8): # mm marks (simplified)
             for j in range(1,10):
                 ax.plot([0.5+i+j*0.1, 0.5+i+j*0.1], [0.4, 0.6 if j!=5 else 0.7], 'k-', lw=0.5)

        # Draw the line segment
        A = (0.5, 0.2); B = (7.5, 0.2)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b-', lw=2)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b|', markersize=8) # End markers
        ax.text(A[0], A[1]-0.05, 'A (0cm)', ha='left', va='top', fontsize=10)
        ax.text(B[0], B[1]-0.05, 'B (7cm)', ha='right', va='top', fontsize=10)

        ax.set_xlim(0, 9); ax.set_ylim(0, 1.2)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_cl_work_draw_angle_protractor(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Drawing 50° Angle with Protractor")

        origin = (0.2, 0.2)
        ax.plot(origin[0], origin[1], 'ko') # Vertex V
        # Ray VX
        X = (origin[0] + 0.7, origin[1])
        ax.plot([origin[0], X[0]], [origin[1], X[1]], 'k-', lw=1.5)
        ax.text(origin[0]-0.05, origin[1]-0.05, 'V', ha='right', va='top')
        ax.text(X[0]+0.02, X[1], 'X', ha='left', va='center')

        # Simulate Protractor
        prot = patches.Arc(origin, 1.0, 1.0, angle=0, theta1=0, theta2=180, color='gray', alpha=0.3, lw=1)
        ax.add_patch(prot)
        ax.plot([origin[0]-0.5, origin[0]+0.5], [origin[1], origin[1]], 'gray', linestyle='--', lw=0.5) # Base line
        # Mark 50 deg
        angle_rad = np.deg2rad(50)
        mark_x = origin[0] + 0.5 * np.cos(angle_rad)
        mark_y = origin[1] + 0.5 * np.sin(angle_rad)
        ax.plot(mark_x, mark_y, 'ro', markersize=4, label='Mark at 50°')
        ax.text(mark_x, mark_y+0.05, "50° mark (Y')", fontsize=8, color='red')

        # Final Ray VY'
        Y_prime = (origin[0] + 0.7 * np.cos(angle_rad), origin[1] + 0.7 * np.sin(angle_rad))
        ax.plot([origin[0], Y_prime[0]], [origin[1], Y_prime[1]], 'b-', lw=1.5)
        ax.text(Y_prime[0], Y_prime[1], "Y'", color='blue', ha='left', va='bottom')

        # Final Angle Arc
        arc_final = patches.Arc(origin, 0.3, 0.3, angle=0, theta1=0, theta2=50, color='blue', lw=1)
        ax.add_patch(arc_final)
        ax.text(origin[0]+0.2*np.cos(np.deg2rad(25)), origin[1]+0.2*np.sin(np.deg2rad(25)), '50°', color='blue', fontsize=10)


        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_ge_cl_work_construct_equal_segment(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Constructing PQ equal to MN")

        # Given Segment MN
        M = (0.1, 0.7); N = (0.5, 0.7)
        ax.plot([M[0], N[0]], [M[1], N[1]], 'k-', lw=1.5, label='Given MN')
        ax.plot([M[0], N[0]], [M[1], N[1]], 'k.', markersize=8)
        ax.text(M[0], M[1]-0.05, 'M', ha='center', va='top')
        ax.text(N[0], N[1]-0.05, 'N', ha='center', va='top')
        length_mn = N[0] - M[0]

        # Ray from P
        P = (0.1, 0.3)
        ax.plot(P[0], P[1], 'bo', markersize=5)
        ax.text(P[0], P[1]-0.05, 'P', ha='center', va='top')
        ax.arrow(P[0], P[1], 0.8, 0, head_width=0.03, head_length=0.05, fc='black', ec='black', length_includes_head=True, lw=1)

        # Compass arc from P
        Q = (P[0] + length_mn, P[1])
        arc = patches.Arc(P, length_mn*2, length_mn*2, angle=0, theta1=-45, theta2=45, color='red', linestyle='--', lw=1, label='Arc with radius MN')
        ax.add_patch(arc)
        ax.plot(Q[0], Q[1], 'bo', markersize=5) # Intersection point
        ax.text(Q[0], Q[1]-0.05, 'Q', ha='center', va='top')

        # Resulting segment PQ
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'b-', lw=2, label='Constructed PQ')

        ax.text((P[0]+Q[0])/2, P[1]+0.05, 'PQ = MN', ha='center', va='bottom')
        ax.legend(fontsize=8, loc='lower right')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # F1: Transformation - Translation
    def f1_tr_tr_work_draw_triangles_pqr(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title="Translation of Triangle PQR", xlabel="x", ylabel="y", xlim=(-1, 7), ylim=(-3, 4.5), aspect='equal')
        ax.set_xticks(np.arange(-1, 8))
        ax.set_yticks(np.arange(-3, 5))

        # Object Triangle PQR
        P = (1, 1); Q = (4, 1); R = (1, 3)
        obj_poly = patches.Polygon([P, Q, R], closed=True, fill=False, edgecolor='blue', linestyle='-', lw=1.5, label='Object PQR')
        ax.add_patch(obj_poly)
        ax.text(P[0], P[1]+0.1, 'P(1,1)', color='blue', fontsize=9)
        ax.text(Q[0], Q[1]+0.1, 'Q(4,1)', color='blue', fontsize=9)
        ax.text(R[0], R[1]+0.1, 'R(1,3)', color='blue', fontsize=9)

        # Image Triangle P'Q'R'
        P_img = (3, -2); Q_img = (6, -2); R_img = (3, 0)
        img_poly = patches.Polygon([P_img, Q_img, R_img], closed=True, fill=False, edgecolor='red', linestyle='--', lw=1.5, label="Image P'Q'R'")
        ax.add_patch(img_poly)
        ax.text(P_img[0], P_img[1]-0.3, "P'(3,-2)", color='red', fontsize=9)
        ax.text(Q_img[0], Q_img[1]-0.3, "Q'(6,-2)", color='red', fontsize=9)
        ax.text(R_img[0]+0.1, R_img[1], "R'(3,0)", color='red', fontsize=9)

        # Translation Vectors (optional)
        ax.arrow(P[0], P[1], P_img[0]-P[0], P_img[1]-P[1], head_width=0.2, head_length=0.2, fc='gray', ec='gray', length_includes_head=True, lw=0.5, linestyle=':')
        ax.arrow(Q[0], Q[1], Q_img[0]-Q[0], Q_img[1]-Q[1], head_width=0.2, head_length=0.2, fc='gray', ec='gray', length_includes_head=True, lw=0.5, linestyle=':')
        ax.arrow(R[0], R[1], R_img[0]-R[0], R_img[1]-R[1], head_width=0.2, head_length=0.2, fc='gray', ec='gray', length_includes_head=True, lw=0.5, linestyle=':')
        ax.text(2, -0.5, 'Vector [2, -3]', color='gray', fontsize=8)

        ax.legend()
        return self._save_plot(fig, filename)

    def f1_tr_tr_ques_describe_translation(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        self._setup_ax(ax, title="Describe Translation from A to B", xlabel="x", ylabel="y", xlim=(0, 7), ylim=(0, 6), aspect='equal')
        ax.set_xticks(np.arange(0, 8))
        ax.set_yticks(np.arange(0, 7))

        # Object A (L-shape)
        A_pts = [(1,1), (2,1), (2,2), (1,2)]
        A_poly = patches.Polygon(A_pts, closed=True, fill=True, facecolor='blue', alpha=0.5, edgecolor='blue', lw=1)
        ax.add_patch(A_poly)
        ax.text(1.5, 1.5, 'A', ha='center', va='center', color='white', fontweight='bold')

        # Image B (L-shape)
        B_pts = [(4,3), (5,3), (5,4), (4,4)]
        B_poly = patches.Polygon(B_pts, closed=True, fill=True, facecolor='red', alpha=0.5, edgecolor='red', lw=1)
        ax.add_patch(B_poly)
        ax.text(4.5, 3.5, 'B', ha='center', va='center', color='white', fontweight='bold')

        # Arrow showing translation (optional)
        ax.arrow(A_pts[0][0], A_pts[0][1], B_pts[0][0]-A_pts[0][0], B_pts[0][1]-A_pts[0][1],
                 head_width=0.2, head_length=0.3, fc='gray', ec='gray', length_includes_head=True, lw=1, linestyle='--')
        ax.text(2.5, 2.5, 'Vector?', color='gray', fontsize=9)

        return self._save_plot(fig, filename)
    

    # F1 Inequalities (Item 16)
    def f1_al_in_notes_inequality_number_lines(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(4, 1, figsize=(8, 6), facecolor='white')
        fig.suptitle("Representing Inequalities on a Number Line", y=1.0)

        def draw_ineq_line(ax, inequality, boundary, open_circle):
            ax.axhline(0, color='k', lw=1.5)
            min_lim = min(-2, boundary - 3)
            max_lim = max(4, boundary + 3)
            ax.set_xlim(min_lim - 0.5, max_lim + 0.5)
            ax.set_ylim(-0.5, 0.5)

            ticks = np.arange(int(min_lim), int(max_lim) + 1)
            for tick in ticks:
                 ax.plot([tick, tick], [-0.05, 0.05], color='k', lw=1)
                 if tick % 1 == 0: # Label integers
                     ax.text(tick, -0.15, str(tick), ha='center', va='top', fontsize=8)

            # Circle
            faceclr = 'white' if open_circle else 'black'
            circle = patches.Circle((boundary, 0), radius=0.1, facecolor=faceclr, edgecolor='black', zorder=5)
            ax.add_patch(circle)

            # Arrow
            if '>' in inequality:
                ax.arrow(boundary, 0, max_lim - boundary, 0, head_width=0.1, head_length=0.25, fc='k', ec='k', lw=1.5, length_includes_head=True, zorder=4)
            else: # <
                ax.arrow(boundary, 0, min_lim - boundary, 0, head_width=0.1, head_length=0.25, fc='k', ec='k', lw=1.5, length_includes_head=True, zorder=4)

            ax.text((min_lim+max_lim)/2, 0.3, inequality, ha='center', va='center', fontsize=10)
            ax.axis('off')

        draw_ineq_line(axs[0], "x > 2", 2, True)
        draw_ineq_line(axs[1], "x < 0", 0, True)
        draw_ineq_line(axs[2], "x ≥ -1", -1, False)
        draw_ineq_line(axs[3], "x ≤ 3", 3, False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return self._save_plot(fig, filename)

    def f1_al_in_work_ineq_p_gt_neg2(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 1.5), facecolor='white')
        ax.set_title("p > -2", fontsize=10)
        ax.axhline(0, color='k', lw=1.5)
        boundary = -2
        min_lim, max_lim = -4, 2
        ax.set_xlim(min_lim - 0.5, max_lim + 0.5)
        ax.set_ylim(-0.5, 0.5)

        ticks = np.arange(int(min_lim), int(max_lim) + 1)
        for tick in ticks:
            ax.plot([tick, tick], [-0.05, 0.05], color='k', lw=1)
            ax.text(tick, -0.15, str(tick), ha='center', va='top', fontsize=8)

        circle = patches.Circle((boundary, 0), radius=0.1, facecolor='white', edgecolor='black', zorder=5)
        ax.add_patch(circle)
        ax.arrow(boundary, 0, max_lim - boundary + 0.3, 0, head_width=0.1, head_length=0.25, fc='k', ec='k', lw=1.5, length_includes_head=True, zorder=4)

        ax.axis('off')
        return self._save_plot(fig, filename)

    def f1_al_in_work_ineq_k_lt_4(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 1.5), facecolor='white')
        ax.set_title("k < 4", fontsize=10)
        ax.axhline(0, color='k', lw=1.5)
        boundary = 4
        min_lim, max_lim = 0, 7
        ax.set_xlim(min_lim - 0.5, max_lim + 0.5)
        ax.set_ylim(-0.5, 0.5)

        ticks = np.arange(int(min_lim), int(max_lim) + 1)
        for tick in ticks:
            ax.plot([tick, tick], [-0.05, 0.05], color='k', lw=1)
            ax.text(tick, -0.15, str(tick), ha='center', va='top', fontsize=8)

        circle = patches.Circle((boundary, 0), radius=0.1, facecolor='white', edgecolor='black', zorder=5)
        ax.add_patch(circle)
        ax.arrow(boundary, 0, min_lim - boundary - 0.3, 0, head_width=0.1, head_length=0.25, fc='k', ec='k', lw=1.5, length_includes_head=True, zorder=4)

        ax.axis('off')
        return self._save_plot(fig, filename)

    # === Form 2 ===

    # F2: Real Numbers - Number Concepts
    def f2_rn_nc_notes_sqrt_prime_factorization(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Square Root using Prime Factorization (√144)")

        text = (
            "144 = 2 × 2 × 2 × 2 × 3 × 3\n\n"
            "Group pairs: (2 × 2) × (2 × 2) × (3 × 3)\n"
            "             ↓         ↓         ↓\n"
            "Take one factor:   2    ×    2    ×    3\n\n"
            "Result: 12\n\n"
            "Therefore, √144 = 12"
        )
        ax.text(0.1, 0.8, text, ha='left', va='top', fontsize=11, family='monospace')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_rn_nc_notes_cuberoot_prime_factorization(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Cube Root using Prime Factorization (³√216)")

        text = (
            "216 = 2 × 2 × 2 × 3 × 3 × 3\n\n"
            "Group triplets: (2 × 2 × 2) × (3 × 3 × 3)\n"
            "                 ↓           ↓\n"
            "Take one factor:       2      ×      3\n\n"
            "Result: 6\n\n"
            "Therefore, ³√216 = 6"
        )
        ax.text(0.1, 0.8, text, ha='left', va='top', fontsize=11, family='monospace')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        return self._save_plot(fig, filename)

    # F2: Real Numbers - Ratios Rates Proportions
    def f2_rn_rr_notes_proportion_graphs(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor='white')
        fig.suptitle("Types of Proportion Graphs")

        # Direct Proportion
        self._setup_ax(ax1, title="Direct Proportion (y = kx)", xlabel="x", ylabel="y", xlim=(0, 5), ylim=(0, 10), aspect='auto')
        x = np.array([0, 5])
        k = 1.8 # Example slope
        y = k * x
        ax1.plot(x, y, 'b-', lw=2, label='y = kx')
        ax1.text(2.5, 6, 'Straight line\nthrough origin', color='blue', ha='center')
        ax1.legend()

        # Inverse Proportion
        self._setup_ax(ax2, title="Inverse Proportion (y = k/x)", xlabel="x", ylabel="y", xlim=(0.5, 5), ylim=(0, 10), aspect='auto')
        x = np.linspace(0.5, 5, 100)
        k = 4 # Example constant
        y = k / x
        ax2.plot(x, y, 'r-', lw=2, label='y = k/x')
        ax2.text(2.5, 6, 'Curve (Hyperbola)\napproaching axes', color='red', ha='center')
        ax2.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)
    # F2 Scales (Item 28)
    def f2_rn_sc_work_scaled_rectangle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white') # Adjusted size for clarity
        self._setup_ax_no_origin(ax, title="Scale Drawing of Garden (Scale 1:500)")

        map_l = 6 # cm
        map_w = 4 # cm

        # Draw rectangle (using relative units proportional to cm)
        aspect = map_w / map_l
        rect_w_disp = 0.8
        rect_h_disp = rect_w_disp * aspect
        rect_bl = (0.1, 0.5 - rect_h_disp/2)

        ax.add_patch(patches.Rectangle(rect_bl, rect_w_disp, rect_h_disp, fill=False, ec='k', lw=1.5))

        # Labels
        ax.text(rect_bl[0] + rect_w_disp/2, rect_bl[1] - 0.05, f'{map_l} cm', ha='center', va='top')
        ax.text(rect_bl[0] - 0.05, rect_bl[1] + rect_h_disp/2, f'{map_w} cm', ha='right', va='center')
        ax.text(0.5, 0.1, 'Scale 1 : 500', ha='center', va='top', fontsize=10)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # F2 Sets (Item 29) - Remapping F1 functions
    f2_se_notes_venn_2_sets = f1_se_ty_notes_venn_diagram_two_sets
    f2_se_work_venn_2_sets_numbers = f1_se_ty_work_venn_diagram_example # Reuse F1 number example

    def f2_se_work_venn_2_sets_music(self): # Specific function for the music example
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Music Preferences (Pop/Rock, n=30)")

        # Universal Set
        rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.text(0.95, 0.9, 'U (30)', ha='right', va='top', fontsize=10)

        # Circles P and R
        center_p = (0.4, 0.5); radius = 0.3
        center_r = (0.6, 0.5)
        circle_p = patches.Circle(center_p, radius, linewidth=1.5, edgecolor='blue', facecolor='none')
        circle_r = patches.Circle(center_r, radius, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(circle_p); ax.add_patch(circle_r)
        ax.text(center_p[0], 0.85, 'P (Pop=15)', ha='center', va='center', fontsize=9, color='blue')
        ax.text(center_r[0], 0.85, 'R (Rock=20)', ha='center', va='center', fontsize=9, color='red')

        # Place counts
        ax.text(0.5, 0.5, '8', ha='center', va='center', fontsize=12, weight='bold') # Intersection
        ax.text(center_p[0]-0.2, 0.5, '7', ha='center', va='center', fontsize=12) # P only
        ax.text(center_r[0]+0.2, 0.5, '12', ha='center', va='center', fontsize=12) # R only
        ax.text(0.1, 0.1, '3', ha='center', va='center', fontsize=12) # Neither

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # F2 Measures (Item 31)
    def f2_me_me_ques_floor_area(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Floor Area Calculation")

        # Rectangle
        w, h = 0.8, 0.5 # Represents 5m x 4m
        ax.add_patch(patches.Rectangle((0.1, 0.2), w, h, fill=False, ec='k', lw=1.5))

        # Labels
        ax.text(0.1+w/2, 0.2-0.05, '5 m', ha='center', va='top')
        ax.text(0.1-0.05, 0.2+h/2, '4 m', ha='right', va='center')
        ax.text(0.1+w/2, 0.2+h/2, 'Area = ? cm²', ha='center', va='center', fontsize=9)

        ax.axis('equal'); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        return self._save_plot(fig, filename)

    # F2 Inequalities (Item 38)
    def f2_al_in_notes_ineq_cartesian_plane(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title="Inequality y > x + 1", xlabel="x", ylabel="y", xlim=(-4, 4), ylim=(-4, 4), aspect='equal')

        x = np.linspace(-4, 4, 100)
        y_line = x + 1

        # Boundary line (dashed)
        ax.plot(x, y_line, 'g--', lw=1.5, label='y = x + 1')

        # Shading the unwanted region (y <= x + 1)
        ax.fill_between(x, -4, y_line, color='gray', alpha=0.3, label='y <= x + 1 (Unwanted)')

        # Test point (0,0)
        ax.plot(0, 0, 'ro', markersize=5)
        ax.text(0.1, -0.2, 'Test Point (0,0)\n0 > 0+1 is False\n=> Shade this side', fontsize=8, color='red')

        ax.text(-2, 3, 'Wanted Region\n(y > x + 1)', ha='center', va='center', color='green', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        return self._save_plot(fig, filename)

    def f2_al_in_work_ineq_x_le_neg1_numline(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 1.5), facecolor='white')
        ax.set_title("x ≤ -1", fontsize=10)
        ax.axhline(0, color='k', lw=1.5)
        boundary = -1
        min_lim, max_lim = -4, 2
        ax.set_xlim(min_lim - 0.5, max_lim + 0.5)
        ax.set_ylim(-0.5, 0.5)

        ticks = np.arange(int(min_lim), int(max_lim) + 1)
        for tick in ticks:
            ax.plot([tick, tick], [-0.05, 0.05], color='k', lw=1)
            ax.text(tick, -0.15, str(tick), ha='center', va='top', fontsize=8)

        circle = patches.Circle((boundary, 0), radius=0.1, facecolor='black', edgecolor='black', zorder=5) # Closed circle
        ax.add_patch(circle)
        ax.arrow(boundary, 0, min_lim - boundary - 0.3, 0, head_width=0.1, head_length=0.25, fc='k', ec='k', lw=1.5, length_includes_head=True, zorder=4)

        ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_al_in_work_ineq_x_ge_2_plane(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Inequality x ≥ 2", xlabel="x", ylabel="y", xlim=(-1, 5), ylim=(-3, 3), aspect='auto')

        # Boundary line x=2 (solid)
        ax.axvline(2, color='g', linestyle='-', lw=1.5, label='x = 2')

        # Shading wanted region (x >= 2)
        x_fill = np.linspace(2, 5, 10)
        y_fill_lower = np.full_like(x_fill, -3)
        y_fill_upper = np.full_like(x_fill, 3)
        ax.fill_betweenx(np.linspace(-3, 3, 10), 2, 5, color='lightgreen', alpha=0.4, label='x ≥ 2 (Wanted)')

        ax.legend(fontsize=8)
        return self._save_plot(fig, filename)

    def f2_al_in_work_ineq_y_lt_3_plane(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Inequality y < 3", xlabel="x", ylabel="y", xlim=(-2, 5), ylim=(-1, 6), aspect='auto')

        # Boundary line y=3 (dashed)
        ax.axhline(3, color='purple', linestyle='--', lw=1.5, label='y = 3')

        # Shading wanted region (y < 3)
        x_fill = np.linspace(-2, 5, 10)
        ax.fill_between(x_fill, -1, 3, color='thistle', alpha=0.4, label='y < 3 (Wanted)')

        # Test point (0,0)
        ax.plot(0,0,'ro', markersize=4)
        ax.text(0.1, 0.1, '(0,0): 0<3 True', fontsize=8, color='red')

        ax.legend(fontsize=8)
        return self._save_plot(fig, filename)

    # F2: Measures - Measures
    def f2_me_me_notes_area_conversion(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Area Conversion: 1 m² = 10,000 cm²")

        # Large square (1m x 1m)
        ax.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='black', lw=2))
        ax.text(0.5, 0.90, '1 m', ha='center', va='bottom', fontsize=10)
        ax.text(0.05, 0.5, '1 m', ha='right', va='center', fontsize=10, rotation='vertical')
        ax.text(0.5, 0.5, 'Area = 1 m²', ha='center', va='center', fontsize=12)

        # Indicate cm conversion
        ax.text(0.5, 0.05, '(= 100 cm)', ha='center', va='top', fontsize=9)
        ax.text(0.95, 0.5, '(= 100 cm)', ha='left', va='center', fontsize=9, rotation='vertical')

        # Grid lines (suggestive)
        num_lines = 5
        for i in np.linspace(0.1, 0.9, num_lines + 1):
            ax.plot([i, i], [0.1, 0.9], color='gray', linestyle=':', lw=0.5)
            ax.plot([0.1, 0.9], [i, i], color='gray', linestyle=':', lw=0.5)

        ax.text(0.5, -0.05, '1 m² = 100 cm × 100 cm = 10,000 cm²', ha='center', va='top', fontsize=10)

        ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_me_me_notes_volume_conversion(self):
        filename = get_filename_from_caller()
        fig = plt.figure(figsize=(5, 5), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white') # Ensure 3D background is white

        ax.set_title("Volume Conversion: 1 m³ = 1,000,000 cm³")

        # Draw cube frame
        r = [0, 1]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                 ax.plot3D(*zip(s, e), color="black", linewidth=1.5)

        # Labels
        ax.text(0.5, 1.1, 0, '1 m', ha='center', va='bottom', fontsize=10)
        ax.text(1.1, 0.5, 0, '1 m', ha='left', va='center', fontsize=10)
        ax.text(1.1, 1.1, 0.7, '1 m', ha='left', va='center', fontsize=10) # Height label

        ax.text(0.5, 0.3, 0.3, 'Volume = 1 m³', ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.5, -0.5, '1 m³ = 100cm×100cm×100cm = 1,000,000 cm³', ha='center', va='top', fontsize=9)


        # Remove axes details for clarity
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_axis_off() # Turn off the axis lines and labels

        # Set view angle
        ax.view_init(elev=20, azim=30)

        return self._save_plot(fig, filename)

    # F2: Measures - Mensuration
    def f2_me_ms_notes_volume_cuboid_cube(self):
        filename = get_filename_from_caller()
        fig = plt.figure(figsize=(8, 4), facecolor='white')
        fig.suptitle("Volume Calculations")

        # --- Cuboid ---
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_facecolor('white')
        ax1.set_title("Cuboid")

        # Dimensions
        l, w, h = 1, 0.6, 0.4
        origin = [0,0,0]
        points = np.array([
            origin, [l, 0, 0], [l, w, 0], [0, w, 0],
            [0, 0, h], [l, 0, h], [l, w, h], [0, w, h]
        ])

        # Edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], # top face
            [0, 4], [1, 5], [2, 6], [3, 7] # vertical edges
        ]

        for edge in edges:
            ax1.plot3D(points[edge, 0], points[edge, 1], points[edge, 2], color="blue", linewidth=1)

        # Labels
        ax1.text(l/2, -0.1, 0, 'l', ha='center', va='top', fontsize=10)
        ax1.text(l+0.05, w/2, 0, 'w', ha='left', va='center', fontsize=10)
        ax1.text(0, w+0.05, h/2, 'h', ha='center', va='center', fontsize=10) # Height label vertical
        ax1.text(l/2, w/2, h+0.3, 'V = lwh', ha='center', va='bottom', fontsize=10)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.grid(False)
        ax1.set_axis_off()
        ax1.view_init(elev=20, azim=25)
        ax1.set_box_aspect([l, w, h]) # Set aspect ratio based on dimensions

        # --- Cube ---
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_facecolor('white')
        ax2.set_title("Cube")

        s = 0.7
        origin = [0,0,0]
        points = np.array([
            origin, [s, 0, 0], [s, s, 0], [0, s, 0],
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s]
        ])

        for edge in edges: # Same edge indices apply
             ax2.plot3D(points[edge, 0], points[edge, 1], points[edge, 2], color="green", linewidth=1)

        # Labels
        ax2.text(s/2, -0.1 * s, 0, 's', ha='center', va='top', fontsize=10)
        ax2.text(s + 0.05 * s, s/2, 0, 's', ha='left', va='center', fontsize=10)
        ax2.text(0, s + 0.05 * s, s/2, 's', ha='center', va='center', fontsize=10)
        ax2.text(s/2, s/2, s + 0.2 * s, 'V = s³', ha='center', va='bottom', fontsize=10)

        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.grid(False)
        ax2.set_axis_off()
        ax2.view_init(elev=20, azim=25)
        ax2.set_box_aspect([s, s, s]) # Equal aspect ratio

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)

    # F2: Graphs - Functional Graphs
    def f2_gr_fg_work_cartesian_plane_diff_scales(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 8), facecolor='white') # Adjust figsize based on scale diff
        # NOTE: Actual cm scale depends on figsize and DPI. We simulate the *ratio* of scales.
        # Let 1 unit on y = 1 'unit' height. Then 1 unit on x = 2 'units' width.
        self._setup_ax(ax, title="Cartesian Plane (x: 2cm/unit, y: 1cm/unit)", xlabel="x", ylabel="y", xlim=(-2.5, 2.5), ylim=(-3.5, 5.5), aspect=2.0) # aspect = y_scale / x_scale => 1/2 is wrong, should be x_scale / y_scale = 2/1 = 2
        ax.set_xticks(np.arange(-2, 3))
        ax.set_yticks(np.arange(-3, 6))
        ax.plot(0, 0, 'ko') # Mark origin
        ax.grid(True, linestyle='--', alpha=0.6) # Keep grid for clarity

        return self._save_plot(fig, filename)

    def f2_gr_fg_work_plot_points_y_eq_x_min_2(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Plot Points for y=x-2", xlabel="x", ylabel="y", xlim=(-1.5, 3.5), ylim=(-3.5, 1.5), aspect='equal')
        ax.set_xticks(np.arange(-1, 4))
        ax.set_yticks(np.arange(-3, 2))

        x_vals = [-1, 0, 1, 2, 3]
        y_vals = [-3, -2, -1, 0, 1]

        ax.plot(x_vals, y_vals, 'bo', markersize=6)
        for x, y in zip(x_vals, y_vals):
             ax.text(x+0.1, y+0.1, f'({x},{y})', fontsize=8)

        return self._save_plot(fig, filename)

    def f2_gr_fg_work_draw_graph_y_eq_x_min_2(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Graph of y = x - 2", xlabel="x", ylabel="y", xlim=(-1.5, 3.5), ylim=(-3.5, 1.5), aspect='equal')
        ax.set_xticks(np.arange(-1, 4))
        ax.set_yticks(np.arange(-3, 2))

        x_vals = np.array([-1, 3]) # Use endpoints for line
        y_vals = x_vals - 2

        # Plot points used for construction
        xp = [-1, 0, 1, 2, 3]
        yp = [-3, -2, -1, 0, 1]
        ax.plot(xp, yp, 'bo', markersize=5, alpha=0.5)

        # Plot line
        ax.plot(x_vals, y_vals, 'r-', lw=2, label='y = x - 2')
        ax.legend()

        return self._save_plot(fig, filename)

    def f2_gr_fg_work_draw_graph_y_eq_4_min_x(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 6), facecolor='white')
        self._setup_ax(ax, title="Graph of y = 4 - x", xlabel="x", ylabel="y", xlim=(-1.5, 2.5), ylim=(1.5, 5.5), aspect='equal')
        ax.set_xticks(np.arange(-1, 3))
        ax.set_yticks(np.arange(2, 6))

        x_vals = np.array([-1, 2]) # Endpoints for line
        y_vals = 4 - x_vals

        # Plot points used for construction
        xp = [-1, 0, 1, 2]
        yp = [5, 4, 3, 2]
        ax.plot(xp, yp, 'bo', markersize=5, alpha=0.5)

        # Plot line
        ax.plot(x_vals, y_vals, 'g-', lw=2, label='y = 4 - x')
        ax.legend()

        return self._save_plot(fig, filename)

    # F2: Graphs - Travel Graphs
    def f2_gr_tg_work_journey_graph_1(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        self._setup_ax(ax, title="Journey Graph", xlabel="Time (hours)", ylabel="Distance (km)", xlim=(-0.2, 5.2), ylim=(-5, 70))
        ax.set_xticks(np.arange(0, 6))

        time = [0, 2, 3, 5]
        dist = [0, 60, 60, 0]

        ax.plot(time, dist, 'm-o', markersize=5)
        # Optional: Add labels for points O(0,0), A(2,60), B(3,60), C(5,0)

        return self._save_plot(fig, filename)

    # Re-use for subsequent questions
    f2_gr_tg_work_stationary = f2_gr_tg_work_journey_graph_1
    f2_gr_tg_work_return_speed = f2_gr_tg_work_journey_graph_1
    f2_gr_tg_work_avg_speed_total = f2_gr_tg_work_journey_graph_1
    f2_gr_tg_ques_distance_at_t4 = f2_gr_tg_work_journey_graph_1
    f2_gr_tg_ques_avg_speed_moving = f2_gr_tg_work_journey_graph_1

    def f2_gr_tg_work_draw_walk_rest_return(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        self._setup_ax(ax, title="Walker's Journey", xlabel="Time (hours)", ylabel="Distance from home (km)", xlim=(-0.1, 3.2), ylim=(-0.5, 4.5))
        ax.set_xticks(np.arange(0, 3.5, 0.5))
        ax.set_yticks(np.arange(0, 5))

        time = [0, 1, 1.5, 3]
        dist = [0, 4, 4, 0]

        ax.plot(time, dist, 'c-o', markersize=5)
        ax.text(0.5, 2, 'Walk Out (4km/h)', color='c', rotation=30, fontsize=8)
        ax.text(1.25, 4.1, 'Rest', color='c', fontsize=8)
        ax.text(2.25, 2, 'Walk Home\n(approx 2.7km/h)', color='c', rotation=-30, fontsize=8, ha='center')


        return self._save_plot(fig, filename)

    def f2_gr_tg_ques_car_journey_grid(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        self._setup_ax(ax, title="Car Journey Grid", xlabel="Time (hours)", ylabel="Distance (km)", xlim=(-0.2, 4.2), ylim=(-10, 155))
        ax.set_xticks(np.arange(0, 5))
        ax.set_yticks(np.arange(0, 151, 25))

        return self._save_plot(fig, filename)

    # F2: Variation
    def f2_va_notes_direct_variation_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Direct Variation", xlabel="x", ylabel="y", xlim=(0, 5), ylim=(0, 10), aspect='auto')

        x = np.array([0, 5])
        k = 1.5 # Example slope
        y = k * x
        ax.plot(x, y, 'm-', lw=2, label='y = kx (k>0)')
        ax.text(2.5, 5, 'Constant Ratio y/x = k\nStraight line through origin', color='m', ha='center')
        ax.legend()

        return self._save_plot(fig, filename)

    # F2: Geometry - Points Lines Angles
    def f2_ge_pl_notes_parallel_lines(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
        self._setup_ax_no_origin(ax, title="Parallel Lines L1 || L2")

        ax.plot([0.1, 0.9], [0.6, 0.6], 'k-', lw=1.5, label='L1')
        ax.plot([0.1, 0.9], [0.4, 0.4], 'k-', lw=1.5, label='L2')

        # Arrows for parallel lines
        ax.plot([0.45, 0.55], [0.6, 0.6], 'k>', markersize=5) # Arrow on L1
        ax.plot([0.45, 0.55], [0.4, 0.4], 'k>', markersize=5) # Arrow on L2

        ax.legend(loc='center left', bbox_to_anchor=(0.1, 0.8), fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_pl_notes_transversal_line(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Parallel Lines and Transversal")

        # Parallel lines
        y1, y2 = 0.7, 0.3
        ax.plot([0.1, 0.9], [y1, y1], 'k-', lw=1.5)
        ax.plot([0.1, 0.9], [y2, y2], 'k-', lw=1.5)
        ax.plot([0.45, 0.55], [y1, y1], 'k>', markersize=5) # Arrow on L1
        ax.plot([0.45, 0.55], [y2, y2], 'k>', markersize=5) # Arrow on L2
        ax.text(0.05, y1, 'L1', va='center', ha='right')
        ax.text(0.05, y2, 'L2', va='center', ha='right')

        # Transversal T
        x_start, y_start = 0.2, 0.1
        x_end, y_end = 0.8, 0.9
        ax.plot([x_start, x_end], [y_start, y_end], 'b-', lw=1.5)
        ax.text(x_end+0.02, y_end+0.02, 'T', color='blue')

        # Intersection points
        m = (y_end - y_start) / (x_end - x_start)
        c = y_start - m * x_start
        x_int1 = (y1 - c) / m
        x_int2 = (y2 - c) / m
        G = (x_int1, y1)
        H = (x_int2, y2)

        # Label angles (example locations)
        ax.text(G[0]-0.05, G[1]+0.05, '1', ha='right', va='bottom', color='red', fontsize=8)
        ax.text(G[0]+0.05, G[1]+0.05, '2', ha='left', va='bottom', color='red', fontsize=8)
        ax.text(G[0]-0.05, G[1]-0.05, '3', ha='right', va='top', color='red', fontsize=8)
        ax.text(G[0]+0.05, G[1]-0.05, '4', ha='left', va='top', color='red', fontsize=8)

        ax.text(H[0]-0.05, H[1]+0.05, '5', ha='right', va='bottom', color='green', fontsize=8)
        ax.text(H[0]+0.05, H[1]+0.05, '6', ha='left', va='bottom', color='green', fontsize=8)
        ax.text(H[0]-0.05, H[1]-0.05, '7', ha='right', va='top', color='green', fontsize=8)
        ax.text(H[0]+0.05, H[1]-0.05, '8', ha='left', va='top', color='green', fontsize=8)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_pl_work_corresponding_angles(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Corresponding Angles Example")

        # Parallel lines
        y1, y2 = 0.7, 0.3
        x_min, x_max = 0.1, 0.9
        ax.plot([x_min, x_max], [y1, y1], 'k-', lw=1.5) # AB
        ax.plot([x_min, x_max], [y2, y2], 'k-', lw=1.5) # CD
        ax.plot([0.45, 0.55], [y1, y1], 'k>'); ax.plot([0.45, 0.55], [y2, y2], 'k>') # Arrows
        ax.text(x_min - 0.05, y1, 'A', va='center', ha='right'); ax.text(x_max + 0.05, y1, 'B', va='center', ha='left')
        ax.text(x_min - 0.05, y2, 'C', va='center', ha='right'); ax.text(x_max + 0.05, y2, 'D', va='center', ha='left')

        # Transversal EF
        x_start, y_start = 0.2, 0.1; x_end, y_end = 0.8, 0.9
        ax.plot([x_start, x_end], [y_start, y_end], 'b-', lw=1.5)
        ax.text(x_start, y_start-0.05, 'E', color='blue', ha='center'); ax.text(x_end, y_end+0.05, 'F', color='blue', ha='center')

        # Intersection points G and H
        m = (y_end - y_start) / (x_end - x_start); c = y_start - m * x_start
        x_int1 = (y1 - c) / m; x_int2 = (y2 - c) / m
        G = (x_int1, y1); H = (x_int2, y2)
        ax.plot(G[0], G[1], 'ko'); ax.text(G[0]+0.03, G[1]+0.02, 'G') # Offset label
        ax.plot(H[0], H[1], 'ko'); ax.text(H[0]+0.03, H[1]+0.02, 'H') # Offset label

        # --- Corresponding Angles ---
        angle_val = 60 # Given angle value for EGA
        # Calculate angles of relevant vectors (relative to G or H)
        vec_GA = (x_min - G[0], y1 - G[1]) # Points left from G
        angle_GA_deg = np.rad2deg(np.arctan2(vec_GA[1], vec_GA[0])) # 180

        # Vector from G towards E on the transversal
        vec_GE = (x_start - G[0], y_start - G[1]) # Points down-left from G
        angle_GE_deg = np.rad2deg(np.arctan2(vec_GE[1], vec_GE[0]))

        # Vector HC: Points left from H
        vec_HC = (x_min - H[0], y2 - H[1]) # Points left from H
        angle_HC_deg = np.rad2deg(np.arctan2(vec_HC[1], vec_HC[0])) # 180

        # Vector from H towards E on the transversal
        vec_HE = (x_start - H[0], y_start - H[1]) # Points down-left from H
        angle_HE_deg = np.rad2deg(np.arctan2(vec_HE[1], vec_HE[0]))
        # Note: angle_HE_deg should be same as angle_GE_deg due to line slope

        arc_radius = 0.12
        label_offset = arc_radius + 0.05

        # Angle EGA (Red arc, 60 degrees)
        # Between GE (angle_GE_deg) and GA (180 deg)
        theta1_ega = angle_GE_deg
        theta2_ega = angle_GA_deg
        # Ensure counter-clockwise sweep by making theta2 > theta1
        if theta2_ega < theta1_ega : theta2_ega += 360

        arc_ega = patches.Arc(G, arc_radius*2, arc_radius*2, angle=0, theta1=theta2_ega, theta2=theta1_ega, color='red', lw=1)
        ax.add_patch(arc_ega)
        # Label EGA = 60
        mid_angle_ega_rad = np.deg2rad((theta1_ega + theta2_ega) / 2)
        ax.text(G[0] - label_offset * np.cos(mid_angle_ega_rad), G[1] - label_offset * np.sin(mid_angle_ega_rad),
                f'{angle_val}°', color='red', ha='center', va='center', fontsize=9)

        # Angle CHE (or GHC, labeled x) (Blue arc, x) - Corresponds to EGA
        # Between HE (angle_HE_deg) and HC (180 deg)
        theta1_che = angle_HE_deg
        theta2_che = angle_HC_deg
        # Ensure counter-clockwise sweep by making theta2 > theta1
        if theta2_che < theta1_che : theta2_che += 360

        arc_che = patches.Arc(H, arc_radius*2, arc_radius*2, angle=0, theta1=theta2_che, theta2=theta1_che, color='blue', lw=1)
        ax.add_patch(arc_che)
        # Label CHE = x
        mid_angle_che_rad = np.deg2rad((theta1_che + theta2_che) / 2)
        ax.text(H[0] - label_offset * np.cos(mid_angle_che_rad), H[1] - label_offset * np.sin(mid_angle_che_rad),
                'x', color='blue', ha='center', va='center', fontsize=9)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)


    def f2_ge_pl_work_alternate_interior_angles(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Alternate Interior Angles Example")
        # Setup lines
        y1, y2 = 0.7, 0.3
        x_min, x_max = 0.1, 0.9
        ax.plot([x_min, x_max], [y1, y1], 'k-', lw=1.5); ax.plot([x_min, x_max], [y2, y2], 'k-', lw=1.5)
        ax.plot([0.45, 0.55], [y1, y1], 'k>'); ax.plot([0.45, 0.55], [y2, y2], 'k>') # Arrows
        ax.text(x_min - 0.05, y1, 'A', va='center', ha='right'); ax.text(x_max + 0.05, y1, 'B', va='center', ha='left')
        ax.text(x_min - 0.05, y2, 'C', va='center', ha='right'); ax.text(x_max + 0.05, y2, 'D', va='center', ha='left')
        # Transversal
        x_start, y_start = 0.2, 0.1; x_end, y_end = 0.8, 0.9
        ax.plot([x_start, x_end], [y_start, y_end], 'b-', lw=1.5)
        ax.text(x_start, y_start-0.05, 'E', color='blue', ha='center'); ax.text(x_end, y_end+0.05, 'F', color='blue', ha='center')
        # Calculate intersections and angles
        m = (y_end - y_start) / (x_end - x_start); c = y_start - m * x_start
        x_int1 = (y1 - c) / m; x_int2 = (y2 - c) / m
        G = (x_int1, y1); H = (x_int2, y2)
        ax.plot(G[0], G[1], 'ko'); ax.text(G[0]+0.03, G[1]+0.02, 'G') # Offset G label slightly
        ax.plot(H[0], H[1], 'ko'); ax.text(H[0]+0.03, H[1]+0.02, 'H') # Offset H label slightly

        # --- Alternate Interior Angles --- 
        angle_val = 110 # Given angle value (assuming it refers to BGH as per image)
        # Calculate angles of vectors originating from vertices G and H
        vec_GB = (x_max - G[0], y1 - G[1])
        angle_GB_deg = np.rad2deg(np.arctan2(vec_GB[1], vec_GB[0])) # Should be 0

        vec_GH = (H[0] - G[0], H[1] - G[1])
        angle_GH_deg = np.rad2deg(np.arctan2(vec_GH[1], vec_GH[0])) # Angle of segment GH

        vec_HG = (G[0] - H[0], G[1] - H[1])
        angle_HG_deg = np.rad2deg(np.arctan2(vec_HG[1], vec_HG[0])) # Angle of segment HG

        vec_HC = (x_min - H[0], y2 - H[1])
        angle_HC_deg = np.rad2deg(np.arctan2(vec_HC[1], vec_HC[0])) # Should be 180

        arc_radius = 0.12 # Adjusted radius
        label_offset = arc_radius + 0.05

        # Angle BGH (Red arc, 110 degrees)
        # Between GB (0 deg) and GH (angle_GH_deg)
        theta1_bgh = angle_GH_deg
        theta2_bgh = angle_GB_deg
        # Ensure counter-clockwise sweep for the interior angle
        if theta2_bgh < theta1_bgh : theta2_bgh += 360 # Make sure end angle is larger for sweep

        arc_bgh = patches.Arc(G, arc_radius*2, arc_radius*2, angle=0, theta1=theta1_bgh, theta2=theta2_bgh, color='red', lw=1)
        ax.add_patch(arc_bgh)
        # Label BGH = 110
        mid_angle_bgh_rad = np.deg2rad((theta1_bgh + theta2_bgh) / 2)
        ax.text(G[0] + label_offset * np.cos(mid_angle_bgh_rad), G[1] + label_offset * np.sin(mid_angle_bgh_rad),
                f'{angle_val}°', color='red', ha='center', va='center', fontsize=9)

        # Angle GHC (Blue arc, y)
        # Between HG (angle_HG_deg) and HC (180 deg)
        theta1_ghc = angle_HG_deg
        theta2_ghc = angle_HC_deg
        # Ensure counter-clockwise sweep for the interior angle
        if theta2_ghc < theta1_ghc : theta2_ghc += 360 # Make sure end angle is larger

        arc_ghc = patches.Arc(H, arc_radius*2, arc_radius*2, angle=0, theta1=theta1_ghc, theta2=theta2_ghc, color='blue', lw=1)
        ax.add_patch(arc_ghc)
        # Label GHC = y
        mid_angle_ghc_rad = np.deg2rad((theta1_ghc + theta2_ghc) / 2)
        ax.text(H[0] + label_offset * np.cos(mid_angle_ghc_rad), H[1] + label_offset * np.sin(mid_angle_ghc_rad),
                'y', color='blue', ha='center', va='center', fontsize=9)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)


    def f2_ge_pl_work_co_interior_angles(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Co-interior Angles Example")
        # Setup lines
        y1, y2 = 0.7, 0.3
        x_min, x_max = 0.1, 0.9
        ax.plot([x_min, x_max], [y1, y1], 'k-', lw=1.5); ax.plot([x_min, x_max], [y2, y2], 'k-', lw=1.5)
        ax.plot([0.45, 0.55], [y1, y1], 'k>'); ax.plot([0.45, 0.55], [y2, y2], 'k>') # Arrows
        ax.text(x_min - 0.05, y1, 'A', va='center', ha='right'); ax.text(x_max + 0.05, y1, 'B', va='center', ha='left')
        ax.text(x_min - 0.05, y2, 'C', va='center', ha='right'); ax.text(x_max + 0.05, y2, 'D', va='center', ha='left')
        # Transversal
        x_start, y_start = 0.2, 0.1; x_end, y_end = 0.8, 0.9
        ax.plot([x_start, x_end], [y_start, y_end], 'b-', lw=1.5)
        ax.text(x_start, y_start-0.05, 'E', color='blue', ha='center'); ax.text(x_end, y_end+0.05, 'F', color='blue', ha='center')
        # Calculate intersections
        m = (y_end - y_start) / (x_end - x_start); c = y_start - m * x_start
        x_int1 = (y1 - c) / m; x_int2 = (y2 - c) / m
        G = (x_int1, y1); H = (x_int2, y2)
        ax.plot(G[0], G[1], 'ko'); ax.text(G[0]+0.03, G[1]+0.02, 'G') # Offset G label slightly
        ax.plot(H[0], H[1], 'ko'); ax.text(H[0]+0.03, H[1]+0.02, 'H') # Offset H label slightly

        # --- Co-Interior Angles ---
        angle_val = 75 # Given angle value for BGH
        # Calculate angles of relevant vectors
        # Vector GB: Points right from G
        vec_GB = (x_max - G[0], y1 - G[1])
        angle_GB_deg = np.rad2deg(np.arctan2(vec_GB[1], vec_GB[0])) # Should be 0

        # Vector GH: Points from G down-left to H
        vec_GH = (H[0] - G[0], H[1] - G[1])
        angle_GH_deg = np.rad2deg(np.arctan2(vec_GH[1], vec_GH[0])) # Angle of segment GH (negative angle)

        # Vector HD: Points right from H
        vec_HD = (x_max - H[0], y2 - H[1])
        angle_HD_deg = np.rad2deg(np.arctan2(vec_HD[1], vec_HD[0])) # Should be 0

        # Vector HG: Points from H up-right to G
        vec_HG = (G[0] - H[0], G[1] - H[1])
        angle_HG_deg = np.rad2deg(np.arctan2(vec_HG[1], vec_HG[0])) # Angle of segment HG (positive angle)

        arc_radius = 0.12
        label_offset = arc_radius + 0.05

        # Angle BGH (Red arc, 75 degrees) - Co-interior 1
        # Between GB (0 deg) and GH (angle_GH_deg)
        theta1_bgh = angle_GH_deg
        theta2_bgh = 0 # Use 0 for the horizontal line pointing right
        # Adjust theta2 to 360 if needed for counter-clockwise sweep calculation from GH angle
        if theta2_bgh < theta1_bgh : theta2_bgh += 360

        arc_bgh = patches.Arc(G, arc_radius*2, arc_radius*2, angle=0, theta1=theta1_bgh, theta2=theta2_bgh, color='red', lw=1)
        ax.add_patch(arc_bgh)
        # Label BGH = 75
        mid_angle_bgh_rad = np.deg2rad((theta1_bgh + theta2_bgh) / 2)
        ax.text(G[0] + label_offset * np.cos(mid_angle_bgh_rad), G[1] + label_offset * np.sin(mid_angle_bgh_rad),
                f'{angle_val}°', color='red', ha='center', va='center', fontsize=9)

        # Angle DHG (Blue arc, z) - Co-interior 2
        # Between HD (0 deg) and HG (angle_HG_deg)
        theta1_dhg = angle_HD_deg # Starts at 0 (vector HD)
        theta2_dhg = angle_HG_deg # Ends at angle of vector HG
        # theta1 is already less than theta2, so sweep is correct

        arc_dhg = patches.Arc(H, arc_radius*2, arc_radius*2, angle=0, theta1=theta1_dhg, theta2=theta2_dhg, color='blue', lw=1)
        ax.add_patch(arc_dhg)
        # Label DHG = z
        mid_angle_dhg_rad = np.deg2rad((theta1_dhg + theta2_dhg) / 2)
        ax.text(H[0] + label_offset * np.cos(mid_angle_dhg_rad), H[1] + label_offset * np.sin(mid_angle_dhg_rad),
                'z', color='blue', ha='center', va='center', fontsize=9)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # Re-use function for subsequent example
    f2_ge_pl_work_find_dhf = f2_ge_pl_work_co_interior_angles


    def f2_ge_pl_work_solve_for_x_parallel(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Solve for x (Parallel Lines)")
        # Setup lines
        y1, y2 = 0.7, 0.3
        x_min, x_max = 0.1, 0.9
        ax.plot([x_min, x_max], [y1, y1], 'k-', lw=1.5); ax.plot([x_min, x_max], [y2, y2], 'k-', lw=1.5)
        ax.plot([0.45, 0.55], [y1, y1], 'k>'); ax.plot([0.45, 0.55], [y2, y2], 'k>') # Arrows
        # No need for A, B, C, D labels here

        # Transversal
        x_start, y_start = 0.2, 0.1; x_end, y_end = 0.8, 0.9
        ax.plot([x_start, x_end], [y_start, y_end], 'b-', lw=1.5)
        # No need for E, F labels here

        # Calculate intersections
        m = (y_end - y_start) / (x_end - x_start); c = y_start - m * x_start
        x_int1 = (y1 - c) / m; x_int2 = (y2 - c) / m
        G = (x_int1, y1); H = (x_int2, y2)
        ax.plot(G[0], G[1], 'ko')
        ax.plot(H[0], H[1], 'ko')

        # --- Alternate Exterior Angles ---
        # Calculate angles of relevant vectors (relative to G or H)
        vec_GB = (x_max - G[0], y1 - G[1]) # Points right from G
        angle_GB_deg = np.rad2deg(np.arctan2(vec_GB[1], vec_GB[0])) # 0

        vec_GF = (x_end - G[0], y_end - G[1]) # Points up-right from G (towards F)
        angle_GF_deg = np.rad2deg(np.arctan2(vec_GF[1], vec_GF[0]))

        vec_HC = (x_min - H[0], y2 - H[1]) # Points left from H
        angle_HC_deg = np.rad2deg(np.arctan2(vec_HC[1], vec_HC[0])) # 180

        vec_HE = (x_start - H[0], y_start - H[1]) # Points down-left from H (towards E)
        angle_HE_deg = np.rad2deg(np.arctan2(vec_HE[1], vec_HE[0]))

        arc_radius = 0.12
        label_offset = arc_radius + 0.08 # Slightly more offset for expressions

        # Angle CHE (bottom-left exterior) (Red arc, (2x+10))
        # Between HC (180 deg) and HE (angle_HE_deg)
        theta1_che = angle_HE_deg
        theta2_che = angle_HC_deg
        # Ensure counter-clockwise sweep
        if theta2_che < theta1_che : theta2_che += 360

        arc_che = patches.Arc(H, arc_radius*2, arc_radius*2, angle=0, theta1=180+theta2_che, theta2=180+theta1_che, color='red', lw=1, alpha=0.7)
        ax.add_patch(arc_che)
        # Label CHE
        mid_angle_che_rad = np.deg2rad((theta1_che + theta2_che) / 2)
        ax.text(H[0] + label_offset * np.cos(mid_angle_che_rad), H[1] + label_offset * np.sin(mid_angle_che_rad),
                '(2x+10)°', color='red', ha='center', va='center', fontsize=9)

        # Angle BGF (top-right exterior) (Blue arc, (3x-20))
        # Between GB (0 deg) and GF (angle_GF_deg)
        theta1_bgf = angle_GB_deg # 0
        theta2_bgf = angle_GF_deg
        # Ensure counter-clockwise sweep
        # if theta2_bgf < theta1_bgf : theta2_bgf += 360 # Not needed as GF angle > 0

        arc_bgf = patches.Arc(G, arc_radius*2, arc_radius*2, angle=0, theta1=theta2_che, theta2=theta1_che, color='blue', lw=1, alpha=0.7)
        ax.add_patch(arc_bgf)
        # Label BGF
        mid_angle_bgf_rad = np.deg2rad((theta1_bgf + theta2_bgf) / 2)
        ax.text(G[0] - label_offset * np.cos(mid_angle_bgf_rad), G[1] - label_offset * np.sin(mid_angle_bgf_rad),
                '(3x-20)°', color='blue', ha='center', va='center', fontsize=9)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)


    def f2_ge_pl_ques_parallel_lines_angles_abc(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Find Angles a, b, c")

        # Parallel vertical lines m, n
        x1, x2 = 0.3, 0.7
        ax.plot([x1, x1], [0.1, 0.9], 'k-', lw=1.5); ax.text(x1, 0.95, 'm')
        ax.plot([x2, x2], [0.1, 0.9], 'k-', lw=1.5); ax.text(x2, 0.95, 'n')
        # Arrows for parallel
        ax.plot([x1, x1], [0.45, 0.55], 'kv', markersize=5)
        ax.plot([x2, x2], [0.45, 0.55], 'kv', markersize=5)

        # Transversal
        y_start, y_end = 0.2, 0.8
        m_trans = 1.5 # slope
        c_trans = y_start - m_trans * x1 # Intersects line m at y_start approximately
        x_int_m = (y_start - c_trans) / m_trans
        x_int_n = (y_end - c_trans) / m_trans
        # Adjust transversal points to cross nicely
        tx1 = 0.1; ty1 = m_trans*tx1 + c_trans
        tx2 = 0.9; ty2 = m_trans*tx2 + c_trans
        ax.plot([tx1, tx2], [ty1, ty2], 'b-', lw=1.5)

        # Intersection points
        int_m_y = m_trans*x1 + c_trans
        int_n_y = m_trans*x2 + c_trans
        P = (x1, int_m_y) # Intersection on m
        Q = (x2, int_n_y) # Intersection on n

        # Angles
        ax.text(P[0]+0.05, P[1], '70°', ha='left', va='bottom', color='red', fontsize=9)
        ax.text(Q[0]-0.05, Q[1], 'a', ha='right', va='bottom', color='blue', fontsize=9)
        ax.text(Q[0]-0.03, Q[1]-0.1, 'b', ha='right', va='top', color='green', fontsize=9)
        ax.text(Q[0]+0.05, Q[1]-0.05, 'c', ha='left', va='top', color='purple', fontsize=9)

        # Add arcs (optional)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)


    def f2_ge_pl_ques_parallel_lines_find_x(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='white') # Slightly more height for title
        # Use suptitle for better spacing
        fig.suptitle("Find Angle x (Zig-Zag)", fontsize=10)
        self._setup_ax_no_origin(ax) # Remove title from here

        # Parallel Lines AB, DE (horizontal)
        y_ab, y_de = 0.8, 0.2
        # Extend lines slightly for clarity with labels
        ax.plot([0.0, 0.6], [y_ab, y_ab], 'k-', lw=1.5); ax.text(0.0, y_ab + 0.02, 'A'); ax.text(0.6 + 0.02, y_ab + 0.02, 'B')
        ax.plot([0.4, 1.0], [y_de, y_de], 'k-', lw=1.5); ax.text(0.4 - 0.02, y_de + 0.02, 'D', ha='right'); ax.text(1.0, y_de + 0.02, 'E')
        # Adjust arrow positions (slightly different spacing)
        ax.plot([0.2, 0.3], [y_ab, y_ab], 'k>'); ax.plot([0.35, 0.45], [y_ab, y_ab], 'k>') # Arrows on AB
        ax.plot([0.65, 0.75], [y_de, y_de], 'k>'); ax.plot([0.8, 0.9], [y_de, y_de], 'k>') # Arrows on DE


        # Point C and connecting lines BC, DC
        B = (0.6, y_ab); D = (0.4, y_de)
        C = (0.7, 0.5)
        ax.plot([B[0], C[0]], [B[1], C[1]], 'b-', lw=1.5)
        ax.plot([D[0], C[0]], [D[1], C[1]], 'b-', lw=1.5)
        ax.plot(C[0], C[1], 'bo'); ax.text(C[0]+0.03, C[1]+0.01, 'C') # Slightly offset C label

        # --- Angles ---
        # Angle at B: Interior angle ABC = 50 (corresponding to exterior 130)
        angle_BC_segment_rad = np.arctan2(C[1]-B[1], C[0]-B[0])
        arc_B_start_deg = np.rad2deg(angle_BC_segment_rad)
        arc_B_end_deg = 180 # Angle of line BA pointing left
        interior_angle_B = 130 # Derived from 180-130

        # Place 50 degree label inside the angle ABC
        mid_angle_B_rad = np.deg2rad((arc_B_start_deg + arc_B_end_deg) / 2)
        ax.text(B[0] - 0.12 * np.cos(mid_angle_B_rad), B[1] - 0.12 * np.sin(mid_angle_B_rad),
                f'{interior_angle_B}°', ha='center', va='center', color='red', fontsize=9)
        # Optional: Add arc for interior 50 deg angle
        arc_B = patches.Arc(B, 0.18, 0.18, angle=0, theta1=arc_B_end_deg, theta2=arc_B_start_deg, color='red', lw=1)
        ax.add_patch(arc_B)


        # Angle at D: Interior angle CDE = 140
        angle_DC_segment_rad = np.arctan2(C[1]-D[1], C[0]-D[0])
        arc_D_start_deg = 0 # Angle of line DE pointing right
        arc_D_end_deg = np.rad2deg(angle_DC_segment_rad)
        arc_D = patches.Arc(D, 0.25, 0.25, angle=0, theta1=arc_D_start_deg, theta2=arc_D_end_deg, color='green', lw=1)
        ax.add_patch(arc_D)
        # Place 140 degree label inside the angle CDE
        mid_angle_D_rad = np.deg2rad((arc_D_start_deg + arc_D_end_deg) / 2)
        ax.text(D[0] + 0.15 * np.cos(mid_angle_D_rad), D[1] + 0.15 * np.sin(mid_angle_D_rad),
                '40°', ha='center', va='center', color='green', fontsize=9)

        # Angle BCD (x) - Place clearly inside
        angle_CB_rad = np.arctan2(B[1]-C[1], B[0]-C[0]) # Vector CB from C
        angle_CD_rad = np.arctan2(D[1]-C[1], D[0]-C[0]) # Vector CD from C
        arc_x_start_deg = np.rad2deg(angle_CD_rad)
        arc_x_end_deg = np.rad2deg(angle_CB_rad)

        # Ensure correct order for arc drawing theta1 to theta2 counter-clockwise
        delta_angle = arc_x_end_deg - arc_x_start_deg
        if delta_angle < -180:
            delta_angle += 360
        elif delta_angle > 180:
             delta_angle -= 360

        # Adjust end angle based on calculated delta for arc
        arc_x_end_deg = arc_x_start_deg + delta_angle

        # Calculate midpoint for label placement based on adjusted angles
        mid_angle_C_rad = np.deg2rad(arc_x_start_deg + delta_angle / 2)

        # Position label inside the angle
        ax.text(C[0] - 0.1 * np.cos(mid_angle_C_rad), C[1] + 0.1 * np.sin(mid_angle_C_rad),
                'x', ha='center', va='center', color='purple', fontsize=11)

        # Add arc for x (relative to C)
        arc_x = patches.Arc(C, 0.15, 0.15, angle=0, theta1=arc_x_start_deg, theta2=arc_x_end_deg, color='purple', lw=1)
        ax.add_patch(arc_x)


        # Hint line (optional)
        ax.plot([0.2, 0.85], [C[1], C[1]], 'k--', lw=0.5, alpha=0.7, label='Hint: Line || AB, DE')
        # Adjust legend position
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, 0.95))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        plt.subplots_adjust(top=0.88) # Adjust top margin for suptitle
        return self._save_plot(fig, filename)

    # F2: Geometry - Bearing
    def f2_ge_be_notes_compass_rose(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Compass Points")

        center = (0.5, 0.5)
        radius = 0.4

        # Cardinal points
        points = {
            'N': (center[0], center[1] + radius),
            'S': (center[0], center[1] - radius),
            'E': (center[0] + radius, center[1]),
            'W': (center[0] - radius, center[1]),
        }
        # Intercardinal points
        diag_offset = radius / np.sqrt(2)
        points.update({
            'NE': (center[0] + diag_offset, center[1] + diag_offset),
            'SE': (center[0] + diag_offset, center[1] - diag_offset),
            'SW': (center[0] - diag_offset, center[1] - diag_offset),
            'NW': (center[0] - diag_offset, center[1] + diag_offset),
        })

        # Draw lines from center
        for point in points:
            ax.plot([center[0], points[point][0]], [center[1], points[point][1]], 'k-', lw=1)

        # Draw labels
        for label, pos in points.items():
            offset = 0.05
            if 'N' in label: pos = (pos[0], pos[1])
            if 'S' in label: pos = (pos[0], pos[1] - offset)
            if 'E' in label: pos = (pos[0] + offset, pos[1])
            if 'W' in label: pos = (pos[0] - offset, pos[1])
            if len(label) == 1: # Adjust single letter positions
                if label == 'N': pos = (pos[0], pos[1]+offset*0.5)
                if label == 'S': pos = (pos[0], pos[1]-offset*0.5)
                if label == 'E': pos = (pos[0]+offset*0.5, pos[1])
                if label == 'W': pos = (pos[0]-offset*0.5, pos[1])

            ax.text(pos[0], pos[1], label, ha='center', va='center', fontweight='bold')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)
    def f1_ge_cl_work_construct_equal_segment_2(self):
        filename = get_filename_from_caller("_b") # Suffix 'b' for second image
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Constructing PQ equal to MN")

        # Use context if needed, or redraw MN for clarity
        M = (0.1, 0.7); N = (0.5, 0.7) # Position doesn't matter, only length
        length_mn = N[0] - M[0]
        ax.text(0.5, 0.8, '(Assume MN length from previous step)', fontsize=8, color='grey')

        # Ray from P
        P = (0.1, 0.3)
        ax.plot(P[0], P[1], 'bo', markersize=5)
        ax.text(P[0], P[1]-0.05, 'P', ha='center', va='top')
        ax.arrow(P[0], P[1], 0.8, 0, head_width=0.03, head_length=0.05, fc='black', ec='black', length_includes_head=True, lw=1)

        # Compass arc from P
        Q = (P[0] + length_mn, P[1])
        arc = patches.Arc(P, length_mn*2, length_mn*2, angle=0, theta1=-45, theta2=45, color='red', linestyle='--', lw=1, label='Arc with radius MN')
        ax.add_patch(arc)
        ax.plot(Q[0], Q[1], 'bo', markersize=5) # Intersection point
        ax.text(Q[0], Q[1]-0.05, 'Q', ha='center', va='top')

        # Resulting segment PQ
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'b-', lw=2, label='Constructed PQ')

        ax.text((P[0]+Q[0])/2, P[1]+0.05, 'PQ = MN', ha='center', va='bottom')
        ax.legend(fontsize=8, loc='lower right')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_be_notes_compass_bearing_examples(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(1, 3, figsize=(9, 3.5), facecolor='white')
        fig.suptitle("Compass Bearing Examples")

        def draw_compass_bearing(ax, bearing_str, angle_deg, from_dir):
            # Use the provided bearing string as the title
            self._setup_ax_no_origin(ax, title=bearing_str)
            center = (0.5, 0.5)
            line_length = 0.35 # Length of the blue bearing line
            arc_radius = 0.15  # Radius for the red arc
            label_offset = 0.06 # How far the label is from the arc

            # Axes lines
            axis_lim = 0.4
            ax.plot([center[0], center[0]], [center[1]-axis_lim, center[1]+axis_lim], 'k--', lw=0.5, alpha=0.6) # N-S
            ax.plot([center[0]-axis_lim, center[0]+axis_lim], [center[1], center[1]], 'k--', lw=0.5, alpha=0.6) # E-W
            # Cardinal direction labels
            ax.text(center[0], center[1]+axis_lim+0.05, 'N', ha='center', va='bottom', fontsize=10)
            ax.text(center[0], center[1]-axis_lim-0.05, 'S', ha='center', va='top', fontsize=10)
            ax.text(center[0]+axis_lim+0.05, center[1], 'E', ha='left', va='center', fontsize=10)
            ax.text(center[0]-axis_lim-0.05, center[1], 'W', ha='right', va='center', fontsize=10)

            # Calculate bearing line angle (standard math angle, 0=East, CCW)
            if from_dir == 'N':
                final_angle_deg = 90 - angle_deg if 'E' in bearing_str else 90 + angle_deg
            else: # From S
                final_angle_deg = 270 + angle_deg if 'E' in bearing_str else 270 - angle_deg

            # Draw bearing line
            rad = np.deg2rad(final_angle_deg)
            end_point = (center[0] + line_length * np.cos(rad), center[1] + line_length * np.sin(rad))
            ax.plot([center[0], end_point[0]], [center[1], end_point[1]], 'b-', lw=2)

            # Calculate arc angles (absolute degrees) and draw arc
            if from_dir == 'N':
                theta1 = 90
                theta2 = final_angle_deg
            else: # From S
                theta1 = 270
                theta2 = final_angle_deg

            # Ensure theta1 and theta2 are ordered correctly for Arc
            arc = patches.Arc(center, arc_radius*2, arc_radius*2, angle=0,
                              theta1=min(theta1, theta2), theta2=max(theta1, theta2),
                              color='red', lw=1.2)
            ax.add_patch(arc)

            # Calculate label position based on mid-arc angle
            mid_arc_angle_deg = (theta1 + theta2) / 2
            label_radius = arc_radius + label_offset
            label_angle_rad = np.deg2rad(mid_arc_angle_deg)
            label_x = center[0] + label_radius * np.cos(label_angle_rad)
            label_y = center[1] + label_radius * np.sin(label_angle_rad)

            # Place angle label near the arc
            ax.text(label_x, label_y, f'{angle_deg}' + r'$^\circ$', color='red', ha='center', va='center', fontsize=9)

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('equal'); ax.axis('off')

        draw_compass_bearing(axs[0], "N 30° E", 30, 'N')
        draw_compass_bearing(axs[1], "S 45° W", 45, 'S')
        draw_compass_bearing(axs[2], "N 70° W", 70, 'N')

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        return self._save_plot(fig, filename)

    def f2_ge_be_notes_three_figure_bearing(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(1, 3, figsize=(9, 3.5), facecolor='white')
        fig.suptitle("Three-Figure Bearing Examples")

        def draw_3fig_bearing(ax, bearing_deg, label):
            self._setup_ax_no_origin(ax, title=label)
            center = (0.5, 0.5)
            length = 0.4

            # North Line
            ax.plot([center[0], center[0]], [center[1], center[1]+length], 'k-', lw=1)
            ax.text(center[0], center[1]+length+0.05, 'N', ha='center', va='bottom')

            # Direction line angle (clockwise from North)
            # Convert math angle (anti-clock from East) to bearing angle
            math_angle = 90 - bearing_deg
            rad = np.deg2rad(math_angle)
            end_point = (center[0] + length * np.cos(rad), center[1] + length * np.sin(rad))
            ax.plot([center[0], end_point[0]], [center[1], end_point[1]], 'b-', lw=2) # Direction line

            # Angle arc (clockwise from North)
            arc = patches.Arc(center, length*0.5, length*0.5, angle=90, theta1=-bearing_deg, theta2=0, color='red', lw=1)
            ax.add_patch(arc)
            # Arc label position needs care
            label_angle_rad = np.deg2rad(90 - bearing_deg / 2)
            label_r = length * 0.3
            if bearing_deg ==135:
                ax.text(center[0]+label_r*np.cos(label_angle_rad), center[1]+label_r*np.sin(label_angle_rad),
                        f'     {bearing_deg:03d}°', color='red', ha='center', va='center', fontsize=9)
            else:
                ax.text(center[0]+label_r*np.cos(label_angle_rad), center[1]+label_r*np.sin(label_angle_rad),
                    f'{bearing_deg:03d}°', color='red', ha='center', va='center', fontsize=9)


            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('equal'); ax.axis('off')

        draw_3fig_bearing(axs[0], 60, "Bearing 060°")
        draw_3fig_bearing(axs[1], 135, "Bearing 135°")
        draw_3fig_bearing(axs[2], 310, "Bearing 310°")

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        return self._save_plot(fig, filename)

    def f2_ge_be_work_back_bearing(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5.5), facecolor='white') # Slightly more height for legend
        self._setup_ax_no_origin(ax, title="Bearing and Back Bearing")

        S = (0.3, 0.6) # Ship position adjusted
        L = (0.5, 0.68) # Lighthouse position adjusted
        north_line_len = 0.25 # Length for North lines
        arc_radius = 0.12    # Radius for angle arcs
        label_offset = 0.05  # Offset for angle labels from arc

        # Points and connecting line
        # Use plot for points to include in legend easily
        ax.plot(S[0], S[1], 'bo', markersize=7, label='Ship (S)', zorder=3)
        ax.plot(L[0], L[1], 'r^', markersize=7, label='Lighthouse (L)', zorder=3)
        ax.plot([S[0], L[0]], [S[1], L[1]], 'k--', lw=1, zorder=1) # Line connecting S and L

        # --- Bearing of L from S (070°) ---
        ax.plot([S[0], S[0]], [S[1], S[1]+north_line_len], 'k-', lw=1, zorder=2) # North at S
        ax.text(S[0], S[1]+north_line_len+0.02, 'N', ha='center', va='bottom')
        bearing1 = 70
        # Calculate angles relative to East (0 degrees CCW)
        north_angle = 90
        line_SL_angle = north_angle - bearing1 # 90 - 70 = 20 degrees
        # Arc sweeps from line SL (20) to North (90)
        arc1 = patches.Arc(S, arc_radius, arc_radius, angle=0,
                           theta1=line_SL_angle, theta2=north_angle, color='red', lw=1.2)
        ax.add_patch(arc1)
        # Label position at mid-arc angle
        mid_angle1_rad = np.deg2rad((line_SL_angle + north_angle) / 2)
        label1_x = S[0] + 0.05 
        label1_y = S[1] + 0.05 
        ax.text(label1_x, label1_y, f'{bearing1:03d}' + r'$^\circ$', color='red', ha='center', va='center', fontsize=9)

        # --- Bearing of S from L (250°) ---
        ax.plot([L[0], L[0]], [L[1], L[1]+north_line_len], 'k-', lw=1, zorder=2) # North at L
        ax.text(L[0], L[1]+north_line_len+0.02, 'N', ha='center', va='bottom')
        bearing2 = 250
        # Calculate angle of line LS relative to East
        line_LS_angle = north_angle - bearing2 # 90 - 250 = -160 degrees, same as 200 degrees
        if line_LS_angle < 0: line_LS_angle += 360 # Ensure positive angle (200)
        # Arc sweeps from North (90) CCW around to line LS (200)
        arc2 = patches.Arc(L, arc_radius, arc_radius, angle=0, # Slightly larger radius
                           theta1=line_LS_angle, theta2=north_angle, color='blue', lw=1.2)
        ax.add_patch(arc2)
        # Label position at mid-arc angle (handle angle wrap-around)
        mid_angle2_deg = (north_angle + line_LS_angle) / 2
        # If sweep crosses 360/0 degree line, adjust mid-angle calculation
        if line_LS_angle < north_angle: # e.g., theta1=270, theta2=45
             mid_angle2_deg = (north_angle + line_LS_angle + 360) / 2
        if mid_angle2_deg >= 360: mid_angle2_deg -= 360

        mid_angle2_rad = np.deg2rad(mid_angle2_deg)
        label2_x = L[0] + 0.03 
        label2_y = L[1] 
        ax.text(label2_x, label2_y, f'{bearing2:03d}' + r'$^\circ$', color='blue', ha='center', va='center', fontsize=9)


        ax.legend(fontsize=9, loc='upper right') # Move legend
        ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 1.0) # Adjusted limits
        ax.axis('equal'); ax.axis('off')
        # Add padding below title
        plt.subplots_adjust(top=0.90)
        return self._save_plot(fig, filename)


    def f2_ge_be_work_angle_bac_bearing(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Angle Between Bearings")

        A = (0.5, 0.5)
        length = 0.4
        ax.plot(A[0], A[1], 'ko'); ax.text(A[0]-0.02, A[1], 'A', ha='right', va='top')

        # North Line at A
        ax.plot([A[0], A[0]], [A[1], A[1]+length], 'k-', lw=1)
        ax.text(A[0]+0.02, A[1]+length, 'N', ha='center', va='bottom')

        # Bearing AB (120°)
        bearing_b = 120
        math_angle_b = 90 - bearing_b
        rad_b = np.deg2rad(math_angle_b)
        B = (A[0] + length * np.cos(rad_b), A[1] + length * np.sin(rad_b))
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b-', lw=1.5)
        ax.text(B[0], B[1], 'B', color='blue')
        arc_b = patches.Arc(A, length*0.4, length*0.4, angle=90, theta1=-bearing_b, theta2=0, color='blue', lw=1, linestyle=':')
        ax.add_patch(arc_b)
        ax.text(A[0]+0.15*np.cos(np.deg2rad(30)), A[1]+0.15*np.sin(np.deg2rad(30)), 'NAB=120°', color='blue', fontsize=8)


        # Bearing AC (210°)
        bearing_c = 210
        math_angle_c = 90 - bearing_c
        rad_c = np.deg2rad(math_angle_c)
        C = (A[0] + length * np.cos(rad_c), A[1] + length * np.sin(rad_c))
        ax.plot([A[0], C[0]], [A[1], C[1]], 'r-', lw=1.5)
        ax.text(C[0]+0.02, C[1], 'C', color='red')
        arc_c = patches.Arc(A, length*0.6, length*0.6, angle=90, theta1=-bearing_c, theta2=0, color='red', lw=1, linestyle=':')
        ax.add_patch(arc_c)
        ax.text(A[0], A[1]-0.15, 'NAC=210°', color='red', fontsize=8)


        # Angle BAC
        angle_bac = bearing_c - bearing_b
        arc_bac = patches.Arc(A, length*0.2, length*0.2, angle=90-bearing_c, theta1=0, theta2=angle_bac, color='green', lw=1.5)
        ax.add_patch(arc_bac)
        label_angle_rad = np.deg2rad(90 - (bearing_b + angle_bac/2))
        ax.text(A[0]+0.1*np.cos(label_angle_rad), A[1]+0.1*np.sin(label_angle_rad), 'BAC=90°', color='green', fontsize=9, ha='center')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # F2: Geometry - Polygons Circles
    def f2_ge_pc_notes_triangle_types(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(2, 2, figsize=(7, 7), facecolor='white')
        fig.suptitle("Types of Triangles")
        axs_flat = axs.flatten()

        # Equilateral
        ax = axs_flat[0]; self._setup_ax_no_origin(ax, title="Equilateral")
        side = 0.6; height = side * np.sqrt(3)/2
        pts = [[0.2, 0.2], [0.2+side, 0.2], [0.2+side/2, 0.2+height]]
        ax.add_patch(patches.Polygon(pts, fill=False, ec='k', lw=1.5))
        # Markings
        ax.text((pts[0][0]+pts[1][0])/2, pts[0][1]-0.02, '—', ha='center', va='top', fontsize=12)
        ax.text((pts[0][0]+pts[2][0])/2-0.02, (pts[0][1]+pts[2][1])/2, '/', ha='right', va='center', rotation=-60, fontsize=12)
        ax.text((pts[1][0]+pts[2][0])/2+0.02, (pts[1][1]+pts[2][1])/2, '\\', ha='left', va='center', rotation=60, fontsize=12)
        ax.text(pts[0][0]+0.05, pts[0][1]+0.05, '60°', fontsize=8)
        ax.text(pts[1][0]-0.05, pts[1][1]+0.05, '60°', fontsize=8)
        ax.text(pts[2][0], pts[2][1]-0.05, '60°', fontsize=8)
        ax.axis('equal'); ax.axis('off')

        # Isosceles
        ax = axs_flat[1]; self._setup_ax_no_origin(ax, title="Isosceles")
        pts = [[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]]
        ax.add_patch(patches.Polygon(pts, fill=False, ec='k', lw=1.5))
        # Markings
        ax.text((pts[0][0]+pts[2][0])/2-0.02, (pts[0][1]+pts[2][1])/2, '//', ha='right', va='center', rotation=-70, fontsize=12)
        ax.text((pts[1][0]+pts[2][0])/2+0.02, (pts[1][1]+pts[2][1])/2, '//', ha='left', va='center', rotation=70, fontsize=12)
        ax.add_patch(patches.Arc(pts[0], 0.2, 0.2, theta1=0, theta2=60, color='r')) # Angle marker
        ax.add_patch(patches.Arc(pts[1], 0.2, 0.2, theta1=120, theta2=180, color='r')) # Angle marker
        ax.axis('equal'); ax.axis('off')

        # Scalene
        ax = axs_flat[2]; self._setup_ax_no_origin(ax, title="Scalene")
        pts = [[0.1, 0.2], [0.8, 0.3], [0.4, 0.7]]
        ax.add_patch(patches.Polygon(pts, fill=False, ec='k', lw=1.5))
        # No markings
        ax.axis('equal'); ax.axis('off')

        # Right-angled
        ax = axs_flat[3]; self._setup_ax_no_origin(ax, title="Right-angled")
        pts = [[0.1, 0.2], [0.7, 0.2], [0.1, 0.8]]
        ax.add_patch(patches.Polygon(pts, fill=False, ec='k', lw=1.5))
        # Right angle symbol
        ax.add_patch(patches.Rectangle((pts[0][0], pts[0][1]), 0.05, 0.05, fill=False, edgecolor='r', lw=1))
        ax.axis('equal'); ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)


    def f2_ge_pc_notes_quadrilateral_types(self): # CORRECTED VERSION
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(2, 3, figsize=(9, 6.5), facecolor='white')
        fig.suptitle("Types of Quadrilaterals")
        axs_flat = axs.flatten()

        # Square
        ax = axs_flat[0]; self._setup_ax_no_origin(ax, title="Square")
        s = 0.6; c = 0.2
        ax.add_patch(patches.Rectangle((c, c), s, s, fill=False, ec='k', lw=1.5))
        ax.text(c+s/2, c-0.02, '|'); ax.text(c-0.02, c+s/2, '|'); ax.text(c+s+0.02, c+s/2, '|'); ax.text(c+s/2, c+s+0.02, '|')
        size=0.05
        ax.add_patch(patches.Rectangle((c, c), size, size, fill=False, ec='r'))
        ax.add_patch(patches.Rectangle((c+s-size, c), size, size, fill=False, ec='r'))
        ax.add_patch(patches.Rectangle((c, c+s-size), size, size, fill=False, ec='r'))
        ax.add_patch(patches.Rectangle((c+s-size, c+s-size), size, size, fill=False, ec='r'))
        ax.axis('equal'); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)


        # Rectangle
        ax = axs_flat[1]; self._setup_ax_no_origin(ax, title="Rectangle")
        w, h = 0.7, 0.5; c = 0.15
        ax.add_patch(patches.Rectangle((c, 0.25), w, h, fill=False, ec='k', lw=1.5))
        ax.text(c+w/2, 0.25-0.02, '|'); ax.text(c+w/2, 0.25+h+0.02, '|')
        ax.text(c-0.02, 0.25+h/2, '//'); ax.text(c+w+0.02, 0.25+h/2, '//')
        size=0.05
        ax.add_patch(patches.Rectangle((c, 0.25), size, size, fill=False, ec='r'))
        ax.add_patch(patches.Rectangle((c+w-size, 0.25), size, size, fill=False, ec='r'))
        ax.add_patch(patches.Rectangle((c, 0.25+h-size), size, size, fill=False, ec='r'))
        ax.add_patch(patches.Rectangle((c+w-size, 0.25+h-size), size, size, fill=False, ec='r'))
        ax.axis('equal'); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)

        # Parallelogram
        ax = axs_flat[2]; self._setup_ax_no_origin(ax, title="Parallelogram")
        x = [0.1, 0.7, 0.9, 0.3]; y = [0.3, 0.3, 0.7, 0.7]
        ax.add_patch(patches.Polygon(list(zip(x,y)), fill=False, ec='k', lw=1.5))
        ax.text((x[0]+x[1])/2, y[0]-0.02, '|'); ax.text((x[2]+x[3])/2, y[2]+0.02, '|')
        ax.text((x[1]+x[2])/2+0.02, (y[1]+y[2])/2, '//'); ax.text((x[0]+x[3])/2-0.02, (y[0]+y[3])/2, '//')
        # Use simple arrows for parallel
        mid_x1, mid_y1 = (x[0]+x[1])/2, y[0]; ax.plot(mid_x1, mid_y1, 'k>', markersize=4, fillstyle='none')
        mid_x2, mid_y2 = (x[2]+x[3])/2, y[2]; ax.plot(mid_x2, mid_y2, 'k>', markersize=4, fillstyle='none')
        mid_x3, mid_y3 = (x[0]+x[3])/2, (y[0]+y[3])/2; ax.plot(mid_x3, mid_y3, 'k^', markersize=4, fillstyle='none') # Different marker
        mid_x4, mid_y4 = (x[1]+x[2])/2, (y[1]+y[2])/2; ax.plot(mid_x4, mid_y4, 'k^', markersize=4, fillstyle='none')
        ax.axis('equal'); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)

        # Rhombus
        ax = axs_flat[3]; self._setup_ax_no_origin(ax, title="Rhombus")
        cx, cy, dx, dy = 0.5, 0.5, 0.3, 0.2
        x = [cx-dx, cx, cx+dx, cx]; y = [cy, cy+dy, cy, cy-dy]
        ax.add_patch(patches.Polygon(list(zip(x,y)), fill=False, ec='k', lw=1.5))
        ax.text((x[0]+x[1])/2-0.01, (y[0]+y[1])/2+0.01, '|'); ax.text((x[1]+x[2])/2+0.01, (y[1]+y[2])/2-0.01, '|')
        ax.text((x[2]+x[3])/2+0.01, (y[2]+y[3])/2-0.01, '|'); ax.text((x[3]+x[0])/2-0.01, (y[3]+y[0])/2+0.01, '|')
        ax.axis('equal'); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)

        # Trapezium
        ax = axs_flat[4]; self._setup_ax_no_origin(ax, title="Trapezium")
        x = [0.1, 0.9, 0.7, 0.3]; y = [0.3, 0.3, 0.7, 0.7]
        ax.add_patch(patches.Polygon(list(zip(x,y)), fill=False, ec='k', lw=1.5))
        mid_x1, mid_y1 = (x[0]+x[1])/2, y[0]; ax.plot(mid_x1, mid_y1, 'k>', markersize=4, fillstyle='none')
        mid_x2, mid_y2 = (x[2]+x[3])/2, y[2]; ax.plot(mid_x2, mid_y2, 'k>', markersize=4, fillstyle='none')
        ax.axis('equal'); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)

        # Kite
        ax = axs_flat[5]; self._setup_ax_no_origin(ax, title="Kite")
        x = [0.5, 0.8, 0.5, 0.2]; y = [0.9, 0.5, 0.1, 0.5]
        ax.add_patch(patches.Polygon(list(zip(x,y)), fill=False, ec='k', lw=1.5))
        ax.plot([x[1], x[3]], [y[1], y[3]], 'k--', lw=0.5) # Diagonals
        ax.plot([x[0], x[2]], [y[0], y[2]], 'k--', lw=0.5)
        ax.text((x[0]+x[1])/2+0.01, (y[0]+y[1])/2+0.01, '|'); ax.text((x[0]+x[3])/2-0.01, (y[0]+y[3])/2+0.01, '|')
        ax.text((x[2]+x[1])/2+0.01, (y[2]+y[1])/2-0.01, '//'); ax.text((x[2]+x[3])/2-0.01, (y[2]+y[3])/2-0.01, '//')
        int_x, int_y = 0.5, 0.5
        ax.add_patch(patches.Rectangle((int_x-0.02, int_y-0.02), 0.04, 0.04, fill=False, ec='r', angle=0)) # Simple square for perp
        ax.axis('equal'); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)
    
    # F2: Geometry - Similarity Congruency

    # F2: Geometry - Similarity Congruency
    def f2_ge_sc_notes_similar_triangles(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')
        fig.suptitle("Similar Triangles (ΔABC ~ ΔPQR)")

        # --- Helper to draw angle arc (same as in congruent) --- 
        def draw_angle_arc(ax, center, p1, p2, radius=0.1, color='k', style='-'):
            vec1 = (p1[0] - center[0], p1[1] - center[1])
            vec2 = (p2[0] - center[0], p2[1] - center[1])
            angle1 = np.rad2deg(np.arctan2(vec1[1], vec1[0]))
            angle2 = np.rad2deg(np.arctan2(vec2[1], vec2[0]))
            # Ensure counter-clockwise sweep
            delta_angle = angle2 - angle1
            if delta_angle > 180:
                delta_angle -= 360
            elif delta_angle < -180:
                 delta_angle += 360
            if delta_angle < 0:
                angle1, angle2 = angle2, angle1 # Swap
                delta_angle = -delta_angle
            angle2_adjusted = angle1 + delta_angle

            # Handle single/double/triple arcs via linestyle variation or multiple calls
            if style == '=': # Double arc
                arc = patches.Arc(center, radius*2, radius*2, angle=0, theta1=angle1, theta2=angle2_adjusted, color=color, lw=1)
                ax.add_patch(arc)
                arc2 = patches.Arc(center, (radius+0.02)*2, (radius+0.02)*2, angle=0, theta1=angle1, theta2=angle2_adjusted, color=color, lw=1)
                ax.add_patch(arc2)
            elif style == '\≡': # Triple arc (using unicode symbol is tricky in code, simulate)
                arc = patches.Arc(center, radius*2, radius*2, angle=0, theta1=angle1, theta2=angle2_adjusted, color=color, lw=1)
                ax.add_patch(arc)
                arc2 = patches.Arc(center, (radius+0.02)*2, (radius+0.02)*2, angle=0, theta1=angle1, theta2=angle2_adjusted, color=color, lw=1)
                ax.add_patch(arc2)
                arc3 = patches.Arc(center, (radius+0.04)*2, (radius+0.04)*2, angle=0, theta1=angle1, theta2=angle2_adjusted, color=color, lw=1)
                ax.add_patch(arc3)
            else: # Single arc (default)
                arc = patches.Arc(center, radius*2, radius*2, angle=0, theta1=angle1, theta2=angle2_adjusted, color=color, lw=1)
                ax.add_patch(arc)

        # Triangle ABC
        self._setup_ax_no_origin(ax1, title="ΔABC")
        A, B, C = (0.1, 0.1), (0.7, 0.1), (0.3, 0.6)
        ax1.add_patch(patches.Polygon([A, B, C], fill=False, ec='b', lw=1.5))
        ax1.text(A[0]-0.02, A[1]-0.04, 'A'); ax1.text(B[0]+0.02, B[1]-0.04, 'B'); ax1.text(C[0], C[1]+0.02, 'C')
        # Angle markers (using helper)
        draw_angle_arc(ax1, A, B, C, color='r', style='-') # Angle A (single arc)
        draw_angle_arc(ax1, B, A, C, color='g', style='=') # Angle B (double arc)
        draw_angle_arc(ax1, C, A, B, color='m', style='\≡') # Angle C (triple arc)
        ax1.axis('equal'); ax1.axis('off')

        # Triangle PQR (larger)
        self._setup_ax_no_origin(ax2, title="ΔPQR")
        scale = 1.5
        P, Q, R = (0.1, 0.1), (0.1 + (B[0]-A[0])*scale, 0.1), (0.1 + (C[0]-A[0])*scale, 0.1 + (C[1]-A[1])*scale)
        ax2.add_patch(patches.Polygon([P, Q, R], fill=False, ec='b', lw=1.5))
        ax2.text(P[0]-0.02, P[1]-0.04, 'P'); ax2.text(Q[0]+0.02, Q[1]-0.04, 'Q'); ax2.text(R[0], R[1]+0.02, 'R')
        # Matching angle markers (using helper)
        draw_angle_arc(ax2, P, Q, R, color='r', style='-', radius=0.1) # Angle P = A
        draw_angle_arc(ax2, Q, P, R, color='g', style='=', radius=0.1) # Angle Q = B
        draw_angle_arc(ax2, R, P, Q, color='m', style='\≡', radius=0.1) # Angle R = C
        ax2.axis('equal'); ax2.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)



    def f2_ge_sc_notes_congruent_triangles(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')
        fig.suptitle("Congruent Triangles (ΔDEF ≅ ΔSTU)")

        # --- Helper to draw angle arc --- 
        def draw_angle_arc(ax, center, p1, p2, radius=0.1, color='k'):
            vec1 = (p1[0] - center[0], p1[1] - center[1])
            vec2 = (p2[0] - center[0], p2[1] - center[1])
            angle1 = np.rad2deg(np.arctan2(vec1[1], vec1[0]))
            angle2 = np.rad2deg(np.arctan2(vec2[1], vec2[0]))
            # Ensure counter-clockwise sweep
            if angle2 < angle1: angle2 += 360
            # Handle potential > 180 sweep for interior angle
            if angle2 - angle1 > 180:
                angle1, angle2 = angle2, angle1 # Swap
                if angle2 < angle1: angle2 += 360 # Adjust again after swap

            arc = patches.Arc(center, radius*2, radius*2, angle=0, theta1=angle1, theta2=angle2, color=color, lw=1)
            ax.add_patch(arc)

        # --- Triangle DEF --- 
        self._setup_ax_no_origin(ax1, title="ΔDEF")
        D, E, F = (0.1, 0.1), (0.8, 0.1), (0.3, 0.7)
        ax1.add_patch(patches.Polygon([D, E, F], fill=False, ec='g', lw=1.5))
        ax1.text(D[0]-0.04, D[1]-0.04, 'D'); ax1.text(E[0]+0.02, E[1]-0.02, 'E'); ax1.text(F[0], F[1]+0.02, 'F')
        # Side markers
        ax1.text((D[0]+E[0])/2, E[1]-0.02, '|', color='r') # DE
        ax1.text((E[0]+F[0])/2+0.02, (E[1]+F[1])/2, '//', color='b') # EF
        ax1.text((F[0]+D[0])/2-0.02, (F[1]+D[1])/2, '///', color='m') # FD
        # Angle markers (Corrected)
        draw_angle_arc(ax1, D, E, F, color='r') # Angle D
        draw_angle_arc(ax1, E, D, F, color='b') # Angle E
        draw_angle_arc(ax1, F, D, E, color='m') # Angle F
        ax1.axis('equal'); ax1.axis('off')

        # --- Triangle STU --- 
        self._setup_ax_no_origin(ax2, title="ΔSTU")
        S, T, U = (0.8, 0.8), (0.1, 0.8), (0.6, 0.2) # Rotated roughly
        ax2.add_patch(patches.Polygon([S, T, U], fill=False, ec='g', lw=1.5))
        ax2.text(S[0]+0.02, S[1]+0.02, 'S'); ax2.text(T[0]-0.04, T[1]+0.02, 'T'); ax2.text(U[0], U[1]-0.04, 'U')
        # Corresponding Side markers
        ax2.text((S[0]+T[0])/2, T[1]+0.02, '|', color='r') # ST = DE
        ax2.text((T[0]+U[0])/2-0.02, (T[1]+U[1])/2, '///', color='m') # TU = FD
        ax2.text((U[0]+S[0])/2+0.02, (U[1]+S[1])/2, '//', color='b') # US = EF
        # Corresponding Angle markers (Corrected)
        draw_angle_arc(ax2, S, T, U, color='r') # Angle S = D
        draw_angle_arc(ax2, T, S, U, color='b') # Angle T = E
        draw_angle_arc(ax2, U, S, T, color='m') # Angle U = F
        ax2.axis('equal'); ax2.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)


    def f2_ge_sc_work_rectangles_similar(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')
        fig.suptitle("Are Rectangles Similar?")

        # Rectangle A (4x6)
        self._setup_ax_no_origin(ax1, title="Rectangle A")
        w, h = 0.6, 0.4 # Relative sizes
        ax1.add_patch(patches.Rectangle((0.2, 0.3), w, h, fill=False, ec='b', lw=1.5))
        ax1.text(0.2+w/2, 0.3-0.05, '6 cm', ha='center', va='top')
        ax1.text(0.2-0.05, 0.3+h/2, '4 cm', ha='right', va='center')
        ax1.axis('equal'); ax1.axis('off')

        # Rectangle B (6x9)
        self._setup_ax_no_origin(ax2, title="Rectangle B")
        w2, h2 = 0.9, 0.6 # Proportional: 9 = 1.5*6, 6 = 1.5*4
        ax2.add_patch(patches.Rectangle((0.05, 0.2), w2, h2, fill=False, ec='r', lw=1.5))
        ax2.text(0.05+w2/2, 0.2-0.05, '9 cm', ha='center', va='top')
        ax2.text(0.05-0.05, 0.2+h2/2, '6 cm', ha='right', va='center')
        ax2.axis('equal'); ax2.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)

    def f2_ge_sc_work_triangles_congruent_sss(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')
        fig.suptitle("Congruency Check (SSS)")

        # Triangle ABC
        self._setup_ax_no_origin(ax1, title="ΔABC")
        A, B, C = (0.1, 0.1), (0.8, 0.1), (0.3, 0.7) # Adjust vertex C slightly maybe
        ax1.add_patch(patches.Polygon([A, B, C], fill=False, ec='b', lw=1.5))
        ax1.text(A[0]-0.02, A[1]-0.02, 'A'); ax1.text(B[0]+0.02, B[1]-0.02, 'B'); ax1.text(C[0], C[1]+0.02, 'C')
        # Labels
        ax1.text((A[0]+B[0])/2, B[1]-0.05, '7 cm', ha='center', va='top') # BC in description, but looks like AB here. Let's match description.
        ax1.text((B[0]+C[0])/2+0.03, (B[1]+C[1])/2, '6 cm', ha='left', va='center') # AC
        ax1.text((C[0]+A[0])/2-0.03, (C[1]+A[1])/2, '5 cm', ha='right', va='center') # AB

        ax1.axis('equal'); ax1.axis('off')

        # Triangle PQR
        self._setup_ax_no_origin(ax2, title="ΔPQR")
        P, Q, R = (0.1, 0.1), (0.8, 0.1), (0.3, 0.7) # Same vertices for congruence
        ax2.add_patch(patches.Polygon([P, Q, R], fill=False, ec='r', lw=1.5))
        ax2.text(P[0]-0.02, P[1]-0.02, 'P'); ax2.text(Q[0]+0.02, Q[1]-0.02, 'Q'); ax2.text(R[0], R[1]+0.02, 'R')
        # Labels matching description order to ABC
        ax2.text((R[0]+P[0])/2-0.03, (R[1]+P[1])/2, '5 cm', ha='right', va='center') # PR = AB
        ax2.text((P[0]+Q[0])/2, Q[1]-0.05, '7 cm', ha='center', va='top') # PQ = BC
        ax2.text((Q[0]+R[0])/2+0.03, (Q[1]+R[1])/2, '6 cm', ha='left', va='center') # QR = AC

        ax2.axis('equal'); ax2.axis('off')


        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)

    def f2_ge_sc_work_triangles_congruent_sas(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')
        fig.suptitle("Congruency Check (SAS)")

        # Triangle DEF
        self._setup_ax_no_origin(ax1, title="ΔDEF")
        D, E, F = (0.1, 0.1), (0.9, 0.1), (0.4, 0.6) # Made EF longer
        ax1.add_patch(patches.Polygon([D, E, F], fill=False, ec='b', lw=1.5))
        ax1.text(D[0]-0.04, D[1]-0.02, 'D'); ax1.text(E[0]+0.02, E[1]-0.02, 'E'); ax1.text(F[0], F[1]+0.02, 'F')
        # Labels
        ax1.text((D[0]+E[0])/2, E[1]-0.05, '8 cm', ha='center', va='top') # DE
        ax1.add_patch(patches.Arc(E, 0.15, 0.15, theta1=135, theta2=180, color='r')) # Angle E
        ax1.text(E[0]-0.08, E[1]+0.05, '40°', color='r', ha='right', va='bottom')
        ax1.text((E[0]+F[0])/2+0.03, (E[1]+F[1])/2, '10 cm', ha='left', va='center') # EF

        ax1.axis('equal'); ax1.axis('off')

        # Triangle XYZ
        self._setup_ax_no_origin(ax2, title="ΔXYZ")
        X, Y, Z = (0.1, 0.1), (0.9, 0.1), (0.4, 0.6) # Same vertices
        ax2.add_patch(patches.Polygon([X, Y, Z], fill=False, ec='r', lw=1.5))
        ax2.text(X[0]-0.02, X[1]-0.02, 'X'); ax2.text(Y[0]+0.02, Y[1]-0.02, 'Y'); ax2.text(Z[0], Z[1]+0.02, 'Z')
        # Labels
        ax2.text((X[0]+Y[0])/2, Y[1]-0.05, '8 cm', ha='center', va='top') # XY = DE
        ax2.add_patch(patches.Arc(Y, 0.15, 0.15, theta1=135, theta2=180, color='r')) # Angle Y = E
        ax2.text(Y[0]-0.08, Y[1]+0.05, '40°', color='r', ha='right', va='bottom')
        ax2.text((Y[0]+Z[0])/2+0.03, (Y[1]+Z[1])/2, '10 cm', ha='left', va='center') # YZ = EF

        ax2.axis('equal'); ax2.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)

    def f2_ge_sc_work_similar_triangles_lmn_rst(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor='white')
        fig.suptitle("Similar Triangles Problem (ΔLMN ~ ΔRST)")

        # Triangle LMN
        self._setup_ax_no_origin(ax1, title="ΔLMN")
        L, M, N = (0.1, 0.1), (0.6, 0.1), (0.3, 0.5) # Example shape
        ax1.add_patch(patches.Polygon([L, M, N], fill=False, ec='b', lw=1.5))
        ax1.text(L[0]-0.02, L[1]-0.02, 'L'); ax1.text(M[0]+0.02, M[1]-0.02, 'M'); ax1.text(N[0], N[1]+0.02, 'N')
        # Labels
        ax1.text((L[0]+M[0])/2, M[1]-0.05, 'LM=4', ha='center', va='top')
        ax1.text((M[0]+N[0])/2+0.03, (M[1]+N[1])/2, 'MN=6', ha='left', va='center')
        ax1.text((N[0]+L[0])/2-0.03, (N[1]+L[1])/2, 'LN=7', ha='right', va='center')
        ax1.axis('equal'); ax1.axis('off')

        # Triangle RST (scaled by 1.5)
        self._setup_ax_no_origin(ax2, title="ΔRST")
        scale = 1.5
        R, S, T = (0.1, 0.1), (0.1+(M[0]-L[0])*scale, 0.1), (0.1+(N[0]-L[0])*scale, 0.1+(N[1]-L[1])*scale)
        ax2.add_patch(patches.Polygon([R, S, T], fill=False, ec='r', lw=1.5))
        ax2.text(R[0]-0.02, R[1]-0.02, 'R'); ax2.text(S[0]+0.02, S[1]-0.02, 'S'); ax2.text(T[0], T[1]+0.02, 'T')
        # Labels
        ax2.text((R[0]+S[0])/2, S[1]-0.05, 'RS=6', ha='center', va='top')
        ax2.text((S[0]+T[0])/2+0.03, (S[1]+T[1])/2, 'ST=?', ha='left', va='center')
        ax2.text((T[0]+R[0])/2-0.03, (T[1]+R[1])/2, 'RT=?', ha='right', va='center')
        ax2.axis('equal'); ax2.axis('off')


        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)

    def f2_ge_sc_ques_similar_triangles_pqr_pst(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Similar Triangles (ΔPQR ~ ΔPST)")

        # --- Helper to draw angle arc ---
        def draw_angle_arc(ax, center, p1, p2, radius=0.05, color='k'):
            """Draws an arc representing the interior angle at 'center' formed by segments center-p1 and center-p2."""
            vec1 = (p1[0] - center[0], p1[1] - center[1])
            vec2 = (p2[0] - center[0], p2[1] - center[1])
            angle1 = np.rad2deg(np.arctan2(vec1[1], vec1[0]))
            angle2 = np.rad2deg(np.arctan2(vec2[1], vec2[0]))

            # Ensure the arc sweeps counter-clockwise less than 180 degrees
            delta_angle = angle2 - angle1
            while delta_angle <= -180: delta_angle += 360
            while delta_angle > 180: delta_angle -= 360

            if delta_angle < 0:
                angle1, angle2 = angle2, angle1 # Swap start and end
                delta_angle = -delta_angle

            angle2_adjusted = angle1 + delta_angle

            arc = patches.Arc(center, radius*2, radius*2, angle=0, theta1=angle1, theta2=angle2_adjusted, color=color, lw=1)
            ax.add_patch(arc)

        # Points - Define based on relative lengths for plotting
        # Let P be at (0.1, 0.1)
        # Let PS have length 8 units vertically, ST length 12 units horizontally
        scale_factor = 0.08 # Scale units to plot coordinates
        P = np.array([0.1, 0.1])
        PS_len = 8
        ST_len = 12
        PQ_len = 5
        QS_len = 3

        S = P + np.array([0, PS_len * scale_factor])
        T = S + np.array([ST_len * scale_factor, 0])
        Q = P + np.array([0, PQ_len * scale_factor])

        # Find R using similarity: PQ/PS = QR/ST = PR/PT = 5/8
        ratio = PQ_len / PS_len
        QR_len = ST_len * ratio
        R = Q + np.array([QR_len * scale_factor, 0])

        # Draw triangles
        ax.add_patch(patches.Polygon([P, S, T], fill=False, ec='black', lw=1.5)) # PST
        ax.plot([Q[0], R[0]], [Q[1], R[1]], 'b-', lw=1.5) # QR
        # Also draw PQ segment explicitly if needed (it's part of PS)
        # ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-', lw=1.5) # Already drawn as part of PS

        # Labels for vertices
        ax.text(P[0]-0.02, P[1]-0.02, 'P')
        ax.text(S[0]-0.02, S[1]+0.02, 'S')
        ax.text(T[0]+0.02, T[1]+0.02, 'T')
        ax.text(Q[0]-0.03, Q[1], 'Q', ha='right', va='center') # Adjust Q label position
        ax.text(R[0]+0.02, R[1], 'R', va='center')

        # Labels for Dimensions
        ax.text(P[0]-0.05, (P[1]+Q[1])/2, 'PQ=5', ha='right', va='center', fontsize=9)
        ax.text(Q[0]-0.05, (Q[1]+S[1])/2, 'QS=3', ha='right', va='center', fontsize=9)
        ax.text((S[0]+T[0])/2, S[1]+0.03, 'ST=12', ha='center', va='bottom', fontsize=9)
        ax.text((Q[0]+R[0])/2, Q[1]-0.03, 'QR=x', ha='center', va='top', color='blue', fontsize=9)

        # Angle markers for similarity (using helper)
        # Angle PQR = PST = 90 degrees
        draw_angle_arc(ax, Q, P, R, color='r', radius=0.04) # Angle PQR
        draw_angle_arc(ax, S, P, T, color='r', radius=0.04) # Angle PST

        # Angle PRQ = PTS
        draw_angle_arc(ax, R, P, Q, color='g', radius=0.04) # Angle PRQ
        draw_angle_arc(ax, T, P, S, color='g', radius=0.04) # Angle PTS


        ax.set_xlim(0, P[0] + (ST_len + 1)*scale_factor)
        ax.set_ylim(0, P[1] + (PS_len + 1)*scale_factor)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)


    # F2: Geometry - Construction Loci
    def f2_ge_cl_notes_perpendicular_bisector(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Construction: Perpendicular Bisector")

        # Line Segment AB
        A = (0.1, 0.5); B = (0.9, 0.5)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', lw=1.5)
        ax.plot(A[0], A[1], 'ko'); ax.text(A[0]-0.02, A[1]-0.02, 'A')
        ax.plot(B[0], B[1], 'ko'); ax.text(B[0]+0.02, B[1]-0.02, 'B')

        # Construction Arcs
        radius = 0.5 # Must be > 0.4 (half length)
        arc1 = patches.Arc(A, radius*2, radius*2, theta1=30, theta2=150, color='r', ls='--')
        arc2 = patches.Arc(A, radius*2, radius*2, theta1=210, theta2=330, color='r', ls='--')
        arc3 = patches.Arc(B, radius*2, radius*2, theta1=30, theta2=150, color='b', ls='--')
        arc4 = patches.Arc(B, radius*2, radius*2, theta1=210, theta2=330, color='b', ls='--')
        ax.add_patch(arc1); ax.add_patch(arc2); ax.add_patch(arc3); ax.add_patch(arc4)

        # Intersection points P, Q (approximate based on radius)
        intersect_angle = np.rad2deg(np.arccos(((B[0]-A[0])/2)/radius))
        P = ( (A[0]+B[0])/2, A[1] + radius*np.sin(np.deg2rad(intersect_angle)) )
        Q = ( (A[0]+B[0])/2, A[1] - radius*np.sin(np.deg2rad(intersect_angle)) )
        ax.plot(P[0], P[1], 'go'); ax.text(P[0], P[1]+0.02, 'P')
        ax.plot(Q[0], Q[1], 'go'); ax.text(Q[0], Q[1]-0.02, 'Q', va='top')

        # Perpendicular Bisector line
        M = ((A[0]+B[0])/2, A[1])
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'g-', lw=1.5)
        # Right angle symbol
        ax.add_patch(patches.Rectangle((M[0], M[1]), 0.04, 0.04, fill=False, ec='g'))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_cl_notes_angle_bisector(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Construction: Angle Bisector")

        # Angle XYZ
        Y = np.array([0.2, 0.2]) # Vertex
        angle = 70
        arm_length = 0.7
        X = Y + np.array([arm_length, 0]) # Point on arm YX
        rad = np.deg2rad(angle)
        Z_unit = np.array([np.cos(rad), np.sin(rad)])
        Z = Y + arm_length * Z_unit # Point on arm YZ

        ax.plot([Y[0], X[0]], [Y[1], X[1]], 'k-', lw=1.5)
        ax.plot([Y[0], Z[0]], [Y[1], Z[1]], 'k-', lw=1.5)
        ax.plot(Y[0], Y[1], 'ko'); ax.text(Y[0]-0.02, Y[1]-0.02, 'Y')
        ax.text(X[0]+0.02, X[1], 'X'); ax.text(Z[0], Z[1]+0.02, 'Z')

        # 1. First Arc from Y
        arc_rad1 = 0.25 # Slightly larger radius
        arc1 = patches.Arc(Y, arc_rad1*2, arc_rad1*2, theta1=0, theta2=angle, color='r', ls='--')
        ax.add_patch(arc1)
        # Intersection points P, Q
        P = Y + np.array([arc_rad1, 0])
        Q = Y + arc_rad1 * Z_unit
        ax.plot(P[0], P[1], 'ro', markersize=5); ax.text(P[0]+0.02, P[1]-0.03, 'P')
        ax.plot(Q[0], Q[1], 'ro', markersize=5); ax.text(Q[0]-0.02, Q[1]+0.03, 'Q')

        # 2. Define radius for arcs from P and Q
        # Radius must be > distance PQ / 2. Let's use arc_rad1 * 1.5 for visibility
        arc_rad2 = arc_rad1 * 1.5

        # 3. Calculate Intersection R of arcs from P and Q *first*
        d_sq = np.sum((P - Q)**2)
        d = np.sqrt(d_sq)
        R = None # Initialize R
        R_calculated = False

        # Check if intersection is possible
        if d <= 2 * arc_rad2 and d > 1e-6: # Add small tolerance for d==0
            a = (arc_rad2**2 - arc_rad2**2 + d_sq) / (2 * d) # This simplifies to d_sq / (2*d) = d / 2
            a = d / 2 # Midpoint distance along PQ
            h_sq = arc_rad2**2 - a**2
            if h_sq >= 0: # Check if h is real
                h = np.sqrt(h_sq)
                # Midpoint between P and Q
                M = P + a * (Q - P) / d
                # Two potential intersection points perpendicular to PQ at M
                Rx1 = M[0] + h * (Q[1] - P[1]) / d
                Ry1 = M[1] - h * (Q[0] - P[0]) / d
                Rx2 = M[0] - h * (Q[1] - P[1]) / d
                Ry2 = M[1] + h * (Q[0] - P[0]) / d

                # Pick the point further from Y
                dist_YR1_sq = np.sum((np.array([Rx1, Ry1]) - Y)**2)
                dist_YR2_sq = np.sum((np.array([Rx2, Ry2]) - Y)**2)

                if dist_YR1_sq > dist_YR2_sq:
                     R = np.array([Rx1, Ry1])
                else:
                     R = np.array([Rx2, Ry2])
                R_calculated = True

        if not R_calculated:
            print("Warning: Arcs from P and Q might not intersect; using approximate bisector.")
            # Fallback to approximate R if no intersection
            bisector_angle = angle / 2
            R_rad_approx = np.deg2rad(bisector_angle)
            R = Y + arm_length*0.7 * np.cos(R_rad_approx), Y[1] + arm_length*0.7 * np.sin(R_rad_approx)
            R = np.array(R) # Ensure R is a numpy array

        ax.plot(R[0], R[1], 'mo', markersize=5); ax.text(R[0]+0.02, R[1]+0.02, 'R')

        # 4. Draw Intersecting arcs centered at P and Q that visually pass through R
        if R_calculated:
            # Calculate angle from P to R and Q to R
            vec_PR = R - P
            angle_PR_deg = np.rad2deg(np.arctan2(vec_PR[1], vec_PR[0]))
            vec_QR = R - Q
            angle_QR_deg = np.rad2deg(np.arctan2(vec_QR[1], vec_QR[0]))

            # Define sweep angles around the target angles to ensure R is covered
            sweep_deg = 40 # Draw 40 degrees sweep
            theta1_p = angle_PR_deg - sweep_deg / 2
            theta2_p = angle_PR_deg + sweep_deg / 2
            theta1_q = angle_QR_deg - sweep_deg / 2
            theta2_q = angle_QR_deg + sweep_deg / 2

            arc_p = patches.Arc(P, arc_rad2*2, arc_rad2*2, angle=0, theta1=theta1_p, theta2=theta2_p, color='b', ls='--')
            arc_q = patches.Arc(Q, arc_rad2*2, arc_rad2*2, angle=0, theta1=theta1_q, theta2=theta2_q, color='g', ls='--')
            ax.add_patch(arc_p); ax.add_patch(arc_q)
        else:
            # If R wasn't calculated, draw plausible arcs (might not intersect at fallback R)
            arc_p = patches.Arc(P, arc_rad2*2, arc_rad2*2, theta1=angle*0.3, theta2=angle*0.7 + 30, color='b', ls='--')
            arc_q = patches.Arc(Q, arc_rad2*2, arc_rad2*2, theta1=180 - (angle*0.7 + 30), theta2=180 - angle*0.3, color='g', ls='--')
            ax.add_patch(arc_p); ax.add_patch(arc_q)


        # 5. Bisector Ray YR (extend slightly beyond R)
        vec_YR = R - Y
        dist_YR = np.linalg.norm(vec_YR)
        if dist_YR > 1e-6 : # Avoid division by zero if R coincides with Y
             norm_YR = vec_YR / dist_YR
             R_ext = Y + norm_YR * (dist_YR + 0.1) # Extend 0.1 units beyond R
             ax.plot([Y[0], R_ext[0]], [Y[1], R_ext[1]], 'm-', lw=1.5)
        else: # If R is at Y, just draw a default bisector line
             bisector_angle_rad = np.deg2rad(angle / 2)
             R_ext = Y + np.array([np.cos(bisector_angle_rad), np.sin(bisector_angle_rad)]) * (arm_length * 0.8)
             ax.plot([Y[0], R_ext[0]], [Y[1], R_ext[1]], 'm-', lw=1.5)


        # 6. Mark equal angles
        bisector_angle = angle / 2
        mark_rad = 0.1
        ax.add_patch(patches.Arc(Y, mark_rad*2, mark_rad*2, theta1=0, theta2=bisector_angle, color='purple', lw=1))
        ax.add_patch(patches.Arc(Y, mark_rad*2, mark_rad*2, theta1=bisector_angle, theta2=angle, color='purple', lw=1))


        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_cl_work_construct_perp_bisector_8cm(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white') # Slightly wider for labels
        self._setup_ax_no_origin(ax, title="Construct Perpendicular Bisector (PQ=8cm)")

        # Line Segment PQ = 8 cm (relative units)
        length = 0.8 # Represents 8cm
        P = (0.1, 0.5); Q = (P[0]+length, 0.5)
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-', lw=1.5)
        ax.plot(P[0], P[1], 'ko'); ax.text(P[0], P[1]-0.05, 'P')
        ax.plot(Q[0], Q[1], 'ko'); ax.text(Q[0], Q[1]-0.05, 'Q')
        ax.text((P[0]+Q[0])/2, P[1]-0.05, '8 cm', ha='center', va='top')

        # Construction Arcs (Radius > 4cm, e.g., 5cm -> 0.5 units)
        radius = 0.5
        arc1 = patches.Arc(P, radius*2, radius*2, theta1=45, theta2=135, color='r', ls='--')
        arc2 = patches.Arc(P, radius*2, radius*2, theta1=225, theta2=315, color='r', ls='--')
        arc3 = patches.Arc(Q, radius*2, radius*2, theta1=45, theta2=135, color='b', ls='--')
        arc4 = patches.Arc(Q, radius*2, radius*2, theta1=225, theta2=315, color='b', ls='--')
        ax.add_patch(arc1); ax.add_patch(arc2); ax.add_patch(arc3); ax.add_patch(arc4)

        # Intersection points R, S (approximate based on radius)
        intersect_angle = np.rad2deg(np.arccos(((Q[0]-P[0])/2)/radius))
        R = ( (P[0]+Q[0])/2, P[1] + radius*np.sin(np.deg2rad(intersect_angle)) )
        S = ( (P[0]+Q[0])/2, P[1] - radius*np.sin(np.deg2rad(intersect_angle)) )
        ax.plot(R[0], R[1], 'go'); ax.text(R[0], R[1]+0.02, 'R')
        ax.plot(S[0], S[1], 'go'); ax.text(S[0], S[1]-0.02, 'S', va='top')

        # Perpendicular Bisector line
        M = ((P[0]+Q[0])/2, P[1])
        ax.plot([R[0], S[0]], [R[1], S[1]], 'g-', lw=1.5)
        # Right angle symbol
        ax.add_patch(patches.Rectangle((M[0], M[1]), 0.04, 0.04, fill=False, ec='g'))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_cl_work_construct_60_degrees(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Constructing 60° Angle")

        # Ray OA
        O = (0.2, 0.3)
        arm_length = 0.6
        A = (O[0] + arm_length, O[1])
        ax.plot([O[0], A[0]], [O[1], A[1]], 'k-', lw=1.5)
        ax.plot(O[0], O[1], 'ko'); ax.text(O[0]-0.02, O[1]-0.02, 'O')
        ax.text(A[0]+0.02, A[1], 'A')

        # Construction arcs
        radius = 0.3
        arc1 = patches.Arc(O, radius*2, radius*2, theta1=-10, theta2=70, color='r', ls='--')
        ax.add_patch(arc1)
        # Intersection P
        P = (O[0] + radius, O[1])
        ax.plot(P[0], P[1], 'ro'); ax.text(P[0], P[1]-0.05, 'P')

        # Arc from P
        arc2 = patches.Arc(P, radius*2, radius*2, theta1=100, theta2=140, color='b', ls='--')
        ax.add_patch(arc2)
        # Intersection Q
        Q_rad = np.deg2rad(60)
        Q = (O[0] + radius * np.cos(Q_rad), O[1] + radius * np.sin(Q_rad))
        ax.plot(Q[0], Q[1], 'bo'); ax.text(Q[0]-0.02, Q[1]+0.02, 'Q')

        # Ray OQ
        Q_end = (O[0] + arm_length * np.cos(Q_rad), O[1] + arm_length * np.sin(Q_rad))
        ax.plot([O[0], Q_end[0]], [O[1], Q_end[1]], 'm-', lw=1.5)

        # Mark angle
        arc_angle = patches.Arc(O, 0.15, 0.15, theta1=0, theta2=60, color='m')
        ax.add_patch(arc_angle)
        ax.text(O[0]+0.1*np.cos(np.deg2rad(30)), O[1]+0.1*np.sin(np.deg2rad(30)), '60°', color='m')


        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_cl_work_bisect_80_degree_angle_initial(self): # Renamed and modified
        filename = get_filename_from_caller() # Generates f2_ge_cl_work_bisect_80_degree_angle_initial.png
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Given Angle ABC (~80°)")

        # Angle ABC (~80 deg)
        B = (0.2, 0.2) # Vertex
        angle = 80
        arm_length = 0.7
        A = (B[0] + arm_length, B[1])
        rad = np.deg2rad(angle)
        C = (B[0] + arm_length * np.cos(rad), B[1] + arm_length * np.sin(rad))
        ax.plot([B[0], A[0]], [B[1], A[1]], 'k-', lw=1.5)
        ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', lw=1.5)
        ax.plot(B[0], B[1], 'ko'); ax.text(B[0]-0.02, B[1]-0.02, 'B')
        ax.text(A[0]+0.02, A[1], 'A'); ax.text(C[0], C[1]+0.02, 'C')

        # Arc for angle
        arc_angle = patches.Arc(B, 0.2*2, 0.2*2, theta1=0, theta2=angle, color='grey')
        ax.add_patch(arc_angle)
        mid_angle_rad = np.deg2rad(angle/2)
        ax.text(B[0]+0.15*np.cos(mid_angle_rad), B[1]+0.15*np.sin(mid_angle_rad), '~80°', color='grey')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)


    def f2_ge_cl_work_bisect_80_degree_angle_construction(self):
        filename = get_filename_from_caller() # Generates f2_ge_cl_work_bisect_80_degree_angle_construction.png
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Bisecting Angle ABC")

        # Angle ABC (~80 deg) - Use numpy arrays for easier vector math
        B = np.array([0.2, 0.2]) # Vertex
        angle = 80
        arm_length = 0.7
        A = B + np.array([arm_length, 0])
        rad = np.deg2rad(angle)
        C_unit = np.array([np.cos(rad), np.sin(rad)])
        C = B + arm_length * C_unit

        ax.plot([B[0], A[0]], [B[1], A[1]], 'k-', lw=1.5)
        ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', lw=1.5)
        ax.plot(B[0], B[1], 'ko'); ax.text(B[0]-0.02, B[1]-0.02, 'B')
        ax.text(A[0]+0.02, A[1], 'A'); ax.text(C[0], C[1]+0.02, 'C')

        # --- Construction arcs ---
        # 1. First arc from B
        arc_rad1 = 0.2
        arc1 = patches.Arc(B, arc_rad1*2, arc_rad1*2, theta1=0, theta2=angle, color='r', ls='--')
        ax.add_patch(arc1)
        P = B + np.array([arc_rad1, 0])
        Q = B + arc_rad1 * C_unit
        ax.plot(P[0], P[1], 'ro', markersize=3); ax.text(P[0]+0.02, P[1]-0.02, 'P')
        ax.plot(Q[0], Q[1], 'ro', markersize=3); ax.text(Q[0]-0.02, Q[1]+0.02, 'Q')

        # 2. Define radius for arcs from P and Q
        arc_rad2 = 0.4
        R = None # Initialize R
        R_calculated = False

        # 3. Calculate Intersection R precisely
        d_sq = np.sum((P - Q)**2)
        d = np.sqrt(d_sq)

        if d <= 2 * arc_rad2 and d > 1e-6:
            a = d / 2
            h_sq = arc_rad2**2 - a**2
            if h_sq >= 0:
                h = np.sqrt(h_sq)
                M = P + a * (Q - P) / d # Midpoint
                # Perpendicular vector (scaled by h/d)
                perp_scaled = h * np.array([Q[1] - P[1], P[0] - Q[0]]) / d
                R1 = M + perp_scaled
                R2 = M - perp_scaled

                # Pick the point further from B
                if np.sum((R1 - B)**2) > np.sum((R2 - B)**2):
                     R = R1
                else:
                     R = R2
                R_calculated = True

        if not R_calculated:
            print("Warning: Arcs from P and Q might not intersect in bisect_80_degree; using approximate bisector.")
            bisector_angle_rad = np.deg2rad(angle / 2)
            R = B + np.array([np.cos(bisector_angle_rad), np.sin(bisector_angle_rad)]) * (arm_length * 0.6) # Approx R

        ax.plot(R[0], R[1], 'mo', markersize=4); ax.text(R[0]+0.02, R[1]+0.02, 'R')

        # 4. Draw arcs centered at P and Q passing through calculated R
        if R_calculated:
            vec_PR = R - P
            angle_PR_deg = np.rad2deg(np.arctan2(vec_PR[1], vec_PR[0]))
            vec_QR = R - Q
            angle_QR_deg = np.rad2deg(np.arctan2(vec_QR[1], vec_QR[0]))

            sweep_deg = 45 # Define arc sweep
            theta1_p = angle_PR_deg - sweep_deg / 2
            theta2_p = angle_PR_deg + sweep_deg / 2
            theta1_q = angle_QR_deg - sweep_deg / 2
            theta2_q = angle_QR_deg + sweep_deg / 2

            arc_p = patches.Arc(P, arc_rad2*2, arc_rad2*2, angle=0, theta1=theta1_p, theta2=theta2_p, color='b', ls='--')
            arc_q = patches.Arc(Q, arc_rad2*2, arc_rad2*2, angle=0, theta1=theta1_q, theta2=theta2_q, color='g', ls='--')
            ax.add_patch(arc_p); ax.add_patch(arc_q)
        # else: Draw fallback arcs if calculation failed (optional)

        # 5. Bisector Ray BR
        vec_BR = R - B
        dist_BR = np.linalg.norm(vec_BR)
        if dist_BR > 1e-6:
             R_end = B + vec_BR / dist_BR * (dist_BR + 0.15) # Extend slightly beyond R
             ax.plot([B[0], R_end[0]], [B[1], R_end[1]], 'm-', lw=1.5)
        else: # Fallback if R is at B
             bisector_angle_rad = np.deg2rad(angle / 2)
             R_end = B + np.array([np.cos(bisector_angle_rad), np.sin(bisector_angle_rad)]) * (arm_length * 0.8)
             ax.plot([B[0], R_end[0]], [B[1], R_end[1]], 'm-', lw=1.5)

        # Mark equal angles
        bisector_angle = angle / 2
        ax.add_patch(patches.Arc(B, 0.1, 0.1, theta1=0, theta2=bisector_angle, color='purple'))
        ax.add_patch(patches.Arc(B, 0.1, 0.1, theta1=bisector_angle, theta2=angle, color='purple'))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_cl_work_construct_90_degrees(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Constructing 90° at P on line XY")

        # Line XY with point P
        line_y = 0.3
        P = (0.5, line_y)
        X = (0.1, line_y); Y = (0.9, line_y)
        ax.plot([X[0], Y[0]], [X[1], Y[1]], 'k-', lw=1.5)
        ax.text(X[0]-0.02, X[1], 'X'); ax.text(Y[0]+0.02, Y[1], 'Y')
        ax.plot(P[0], P[1], 'ko'); ax.text(P[0], P[1]-0.05, 'P')

        # Arcs from P to get A, B
        radius1 = 0.2
        A = (P[0]-radius1, P[1]); B = (P[0]+radius1, P[1])
        arc_p = patches.Arc(P, radius1*2, radius1*2, theta1=0, theta2=180, color='r', ls='--')
        ax.add_patch(arc_p)
        ax.plot(A[0], A[1], 'ro'); ax.text(A[0], A[1]-0.05, 'A')
        ax.plot(B[0], B[1], 'ro'); ax.text(B[0], B[1]-0.05, 'B')

        # Arcs from A, B to get Q
        radius2 = 0.3 # Wider radius
        arc_a = patches.Arc(A, radius2*2, radius2*2, theta1=45, theta2=135, color='b', ls='--')
        arc_b = patches.Arc(B, radius2*2, radius2*2, theta1=45, theta2=135, color='g', ls='--')
        ax.add_patch(arc_a); ax.add_patch(arc_b)
        # Intersection Q (directly above P)
        Q = (P[0], P[1] + np.sqrt(radius2**2 - (radius1)**2) ) # Approx height
        ax.plot(Q[0], Q[1], 'mo'); ax.text(Q[0], Q[1]+0.02, 'Q')

        # Line PQ
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'm-', lw=1.5)

        # Right angle symbol
        ax.add_patch(patches.Rectangle((P[0], P[1]), 0.04, 0.04, fill=False, ec='m'))


        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)


    def f2_ge_cl_work_construct_30_degrees(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Constructing 30° Angle (Bisect 60°)")

        # --- Construct 60 deg first (lightly) ---
        O = np.array([0.2, 0.3]) # Vertex O
        arm_length = 0.7
        A = O + np.array([arm_length, 0]) # Point on base ray OA
        ax.plot([O[0], A[0]], [O[1], A[1]], 'k-', lw=1) # Base ray OA
        ax.plot(O[0], O[1], 'ko', markersize=5); ax.text(O[0]-0.03, O[1]-0.03, 'O')
        ax.text(A[0]+0.02, A[1], 'A')

        # Arc from O to find P and Q
        radius_60 = 0.35 # Radius for 60 deg construction
        arc1_60 = patches.Arc(O, radius_60*2, radius_60*2, theta1=-10, theta2=70, color='gray', ls=':', lw=0.8)
        ax.add_patch(arc1_60)
        P = O + np.array([radius_60, 0]) # Intersection on OA
        ax.plot(P[0], P[1], 'ro', markersize=3) # Mark P lightly

        # Arc from P to find Q
        arc2_60 = patches.Arc(P, radius_60*2, radius_60*2, theta1=100, theta2=140, color='gray', ls=':', lw=0.8)
        ax.add_patch(arc2_60)

        # Point Q at 60 degrees
        Q_rad_60 = np.deg2rad(60)
        Q = O + radius_60 * np.array([np.cos(Q_rad_60), np.sin(Q_rad_60)])
        ax.plot(Q[0], Q[1], 'ro', markersize=3) # Mark Q lightly

        # Draw 60 deg ray faintly
        Q_end = O + arm_length * np.array([np.cos(Q_rad_60), np.sin(Q_rad_60)])
        ax.plot([O[0], Q_end[0]], [O[1], Q_end[1]], 'k:', lw=0.5) # 60 deg ray (light)

        # --- Now bisect the 60 deg angle AOQ ---
        # We have points P and Q on the arms at equal distance (radius_60) from O.
        # Draw intersecting arcs from P and Q. Radius must be > PQ/2.
        arc_rad_bisect = radius_60 * 0.8 # Radius for bisecting arcs
        R = None # Initialize R
        R_calculated = False

        # Calculate Intersection R precisely
        d_sq = np.sum((P - Q)**2)
        d = np.sqrt(d_sq)

        # Check if intersection is possible with arc_rad_bisect
        if d <= 2 * arc_rad_bisect and d > 1e-6:
            a = d / 2
            h_sq = arc_rad_bisect**2 - a**2
            if h_sq >= 0:
                h = np.sqrt(h_sq)
                M = P + a * (Q - P) / d # Midpoint
                # Perpendicular vector (scaled by h/d)
                perp_scaled = h * np.array([Q[1] - P[1], P[0] - Q[0]]) / d
                R1 = M + perp_scaled
                R2 = M - perp_scaled
                # Pick the point further from O
                if np.sum((R1 - O)**2) > np.sum((R2 - O)**2):
                     R = R1
                else:
                     R = R2
                R_calculated = True

        if not R_calculated:
            print("Warning: Arcs from P and Q might not intersect in construct_30_degrees; using approximate bisector.")
            R_rad_approx = np.deg2rad(30) # Fallback to 30 degrees
            R = O + np.array([np.cos(R_rad_approx), np.sin(R_rad_approx)]) * (arm_length * 0.6)

        ax.plot(R[0], R[1], 'mo', markersize=4); ax.text(R[0]+0.02, R[1]+0.02, 'R')

        # Draw bisecting arcs centered at P and Q passing through calculated R
        if R_calculated:
            vec_PR = R - P
            angle_PR_deg = np.rad2deg(np.arctan2(vec_PR[1], vec_PR[0]))
            vec_QR = R - Q
            angle_QR_deg = np.rad2deg(np.arctan2(vec_QR[1], vec_QR[0]))

            sweep_deg = 50 # Define arc sweep
            theta1_p = angle_PR_deg - sweep_deg / 2
            theta2_p = angle_PR_deg + sweep_deg / 2
            theta1_q = angle_QR_deg - sweep_deg / 2
            theta2_q = angle_QR_deg + sweep_deg / 2

            arc_p_bisect = patches.Arc(P, arc_rad_bisect*2, arc_rad_bisect*2, angle=0, theta1=theta1_p, theta2=theta2_p, color='b', ls='--')
            arc_q_bisect = patches.Arc(Q, arc_rad_bisect*2, arc_rad_bisect*2, angle=0, theta1=theta1_q, theta2=theta2_q, color='g', ls='--')
            ax.add_patch(arc_p_bisect); ax.add_patch(arc_q_bisect)

        # Bisector Ray OR (The 30 deg line)
        vec_OR = R - O
        dist_OR = np.linalg.norm(vec_OR)
        if dist_OR > 1e-6:
             R_end = O + vec_OR / dist_OR * arm_length # Extend to arm_length
             ax.plot([O[0], R_end[0]], [O[1], R_end[1]], 'm-', lw=1.5)
        else: # Fallback if R is at O
             R_rad_30 = np.deg2rad(30)
             R_end = O + np.array([np.cos(R_rad_30), np.sin(R_rad_30)]) * arm_length
             ax.plot([O[0], R_end[0]], [O[1], R_end[1]], 'm-', lw=1.5)


        # Mark 30 degree angle
        arc_angle = patches.Arc(O, 0.18, 0.18, theta1=0, theta2=30, color='m') # Adjusted radius
        ax.add_patch(arc_angle)
        label_angle_rad = np.deg2rad(15) # Midpoint for label
        label_radius = 0.18 + 0.05
        ax.text(O[0] + label_radius * np.cos(label_angle_rad),
                O[1] + label_radius * np.sin(label_angle_rad),
                '30' + r'$^\circ$', color='m', ha='center', va='center')

        ax.set_xlim(0, 1); ax.set_ylim(0.1, 0.9) # Adjusted ylim
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # F2: Geometry - Symmetry
    def f2_ge_sy_notes_lines_of_symmetry(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(2, 3, figsize=(9, 6), facecolor='white')
        fig.suptitle("Lines of Symmetry")
        axs_flat = axs.flatten()

        titles = ["Isosceles Triangle (1)", "Equilateral Triangle (3)", "Square (4)",
                  "Rectangle (2)", "Rhombus (2)", "Regular Hexagon (6)"]

        # Isosceles
        ax = axs_flat[0]
        pts = [[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]]
        ax.add_patch(patches.Polygon(pts, fill=False, ec='k', lw=1))
        ax.plot([0.5, 0.5], [0.2, 0.8], 'r--', lw=1) # Line of symmetry
        ax.axis('equal'); ax.axis('off'); ax.set_title(titles[0], fontsize=9)

        # Equilateral
        ax = axs_flat[1]
        s = 0.7; h = s * np.sqrt(3)/2
        pts = [[0.15, 0.2], [0.15+s, 0.2], [0.15+s/2, 0.2+h]]
        ax.add_patch(patches.Polygon(pts, fill=False, ec='k', lw=1))
        ax.plot([pts[2][0], (pts[0][0]+pts[1][0])/2], [pts[2][1], pts[0][1]], 'r--', lw=1) # Vert bisector
        ax.plot([pts[0][0], (pts[1][0]+pts[2][0])/2], [pts[0][1], (pts[1][1]+pts[2][1])/2], 'r--', lw=1)
        ax.plot([pts[1][0], (pts[0][0]+pts[2][0])/2], [pts[1][1], (pts[0][1]+pts[2][1])/2], 'r--', lw=1)
        ax.axis('equal'); ax.axis('off'); ax.set_title(titles[1], fontsize=9)

        # Square
        ax = axs_flat[2]
        s = 0.6; c = 0.2
        ax.add_patch(patches.Rectangle((c, c), s, s, fill=False, ec='k', lw=1))
        ax.plot([c+s/2, c+s/2], [c, c+s], 'r--', lw=1) # Vertical
        ax.plot([c, c+s], [c+s/2, c+s/2], 'r--', lw=1) # Horizontal
        ax.plot([c, c+s], [c, c+s], 'b--', lw=1) # Diagonal \
        ax.plot([c, c+s], [c+s, c], 'b--', lw=1) # Diagonal /
        ax.axis('equal'); ax.axis('off'); ax.set_title(titles[2], fontsize=9)

        # Rectangle
        ax = axs_flat[3]
        w, h = 0.7, 0.5; c = 0.15
        ax.add_patch(patches.Rectangle((c, 0.25), w, h, fill=False, ec='k', lw=1))
        ax.plot([c+w/2, c+w/2], [0.25, 0.25+h], 'r--', lw=1) # Vertical
        ax.plot([c, c+w], [0.25+h/2, 0.25+h/2], 'r--', lw=1) # Horizontal
        ax.axis('equal'); ax.axis('off'); ax.set_title(titles[3], fontsize=9)

        # Rhombus
        ax = axs_flat[4]
        cx, cy, dx, dy = 0.5, 0.5, 0.4, 0.2
        x = [cx-dx, cx, cx+dx, cx]; y = [cy, cy+dy, cy, cy-dy]
        ax.add_patch(patches.Polygon(list(zip(x,y)), fill=False, ec='k', lw=1))
        ax.plot([x[1], x[3]], [y[1], y[3]], 'r--', lw=1) # Vertical Diagonal
        ax.plot([x[0], x[2]], [y[0], y[2]], 'r--', lw=1) # Horizontal Diagonal
        ax.axis('equal'); ax.axis('off'); ax.set_title(titles[4], fontsize=9)

        # Regular Hexagon
        ax = axs_flat[5]
        center = (0.5, 0.5); radius = 0.3
        verts = [(center[0] + radius * np.cos(np.pi/3 * i), center[1] + radius * np.sin(np.pi/3 * i)) for i in range(6)]
        ax.add_patch(patches.Polygon(verts, fill=False, ec='k', lw=1))
        # Lines of symmetry (vertex to opposite vertex, midpoint to opposite midpoint)
        for i in range(3):
            ax.plot([verts[i][0], verts[i+3][0]], [verts[i][1], verts[i+3][1]], 'r--', lw=1)
            mid1_x = (verts[i][0]+verts[i+1][0])/2; mid1_y = (verts[i][1]+verts[i+1][1])/2
            mid2_x = (verts[i+3][0]+verts[(i+4)%6][0])/2; mid2_y = (verts[i+3][1]+verts[(i+4)%6][1])/2
            ax.plot([mid1_x, mid2_x], [mid1_y, mid2_y], 'b--', lw=1)
        ax.axis('equal'); ax.axis('off'); ax.set_title(titles[5], fontsize=9)


        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)

    def f2_ge_sy_work_square_symmetry(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Square (4 Lines of Symmetry)")
        s = 0.7; c = 0.15
        ax.add_patch(patches.Rectangle((c, c), s, s, fill=False, ec='k', lw=1.5))
        ax.plot([c+s/2, c+s/2], [c, c+s], 'r--', lw=1) # Vertical
        ax.plot([c, c+s], [c+s/2, c+s/2], 'r--', lw=1) # Horizontal
        ax.plot([c, c+s], [c, c+s], 'b--', lw=1) # Diagonal \
        ax.plot([c, c+s], [c+s, c], 'b--', lw=1) # Diagonal /
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_sy_work_rectangle_symmetry(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Rectangle (2 Lines of Symmetry)")
        w, h = 0.8, 0.5; c = 0.1
        ax.add_patch(patches.Rectangle((c, 0.25), w, h, fill=False, ec='k', lw=1.5))
        ax.plot([c+w/2, c+w/2], [0.25, 0.25+h], 'r--', lw=1) # Vertical
        ax.plot([c, c+w], [0.25+h/2, 0.25+h/2], 'r--', lw=1) # Horizontal
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_sy_work_letter_h_symmetry(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Letter H (2 Lines of Symmetry)")

        # Draw H
        thick = 0.1
        # Left vertical
        ax.add_patch(patches.Rectangle((0.2, 0.1), thick, 0.8, color='k'))
        # Right vertical
        ax.add_patch(patches.Rectangle((0.7, 0.1), thick, 0.8, color='k'))
        # Horizontal bar
        ax.add_patch(patches.Rectangle((0.2+thick, 0.5-thick/2), 0.7-(0.2+thick), thick, color='k'))

        # Lines of symmetry
        ax.plot([0.5, 0.5], [0, 1], 'r--', lw=1.5) # Vertical
        ax.plot([0, 1], [0.5, 0.5], 'r--', lw=1.5) # Horizontal

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_sy_work_isosceles_symmetry(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Isosceles Triangle (1 Line of Symmetry)")
        pts = [[0.1, 0.2], [0.9, 0.2], [0.5, 0.8]]
        ax.add_patch(patches.Polygon(pts, fill=False, ec='k', lw=1.5))
        ax.plot([0.5, 0.5], [0.2, 0.8], 'r--', lw=1.5) # Line of symmetry
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ge_sy_work_pentagon_symmetry(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Regular Pentagon (5 Lines of Symmetry)")
        center = (0.5, 0.55); radius = 0.4
        verts = [(center[0] + radius * np.cos(np.pi/2 + 2*np.pi/5 * i), center[1] + radius * np.sin(np.pi/2 + 2*np.pi/5 * i)) for i in range(5)]
        ax.add_patch(patches.Polygon(verts, fill=False, ec='k', lw=1.5))

        # Lines of symmetry (vertex to midpoint of opposite side)
        for i in range(5):
            mid_x = (verts[(i+2)%5][0] + verts[(i+3)%5][0]) / 2
            mid_y = (verts[(i+2)%5][1] + verts[(i+3)%5][1]) / 2
            ax.plot([verts[i][0], mid_x], [verts[i][1], mid_y], 'r--', lw=1)

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)


    # F2: Statistics - Data Collection
    def f2_st_dc_notes_bar_chart_example(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')

        fruits = ['Apple', 'Banana', 'Orange']
        counts = [5, 8, 3]

        bars = ax.bar(fruits, counts, color=['red', 'yellow', 'orange'], edgecolor='black')

        self._setup_ax(ax, title="Favorite Fruits of Form 2 Students", xlabel="Favorite Fruit", ylabel="Number of Students", ylim=(0, 10), grid=True)
        ax.set_yticks(np.arange(0, 11, 2))

        # Add values on top of bars
        ax.bar_label(bars, padding=3)

        return self._save_plot(fig, filename)

    def f2_st_dc_notes_pie_chart_example(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

        labels = ['Category A', 'Category B', 'Category C']
        sizes = [50, 33.3, 16.7] # Percentages
        angles = [s * 3.6 for s in sizes] # Convert % to degrees
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        explode = (0, 0, 0) # No explosion

        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=False, startangle=90)

        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title("Distribution of Item Types")

        # Improve label appearance if needed
        plt.setp(autotexts, size=8, weight="bold", color="black")
        plt.setp(texts, size=10)


        return self._save_plot(fig, filename)

    def f2_st_dc_work_bar_chart_pets(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')

        num_pets = ['0', '1', '2', '3'] # Treat as categories
        frequency = [3, 4, 2, 1]

        bars = ax.bar(num_pets, frequency, color='skyblue', edgecolor='black')

        self._setup_ax(ax, title="Number of Pets Owned by Students", xlabel="Number of Pets", ylabel="Frequency", ylim=(0, 5), grid=True)
        ax.set_yticks(np.arange(0, 6))
        ax.bar_label(bars, padding=3)


        return self._save_plot(fig, filename)

    def f2_st_dc_work_pie_chart_pets(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')

        labels = ['0 Pets', '1 Pet', '2 Pets', '3 Pets']
        # Frequencies: 3, 4, 2, 1 (Total 10)
        # Angles: 108, 144, 72, 36
        sizes = [3, 4, 2, 1] # Use frequencies for pie chart function

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct=lambda p: f'{p * sum(sizes) / 100 :.0f}', # Show frequency
                                          startangle=90, counterclock=False, pctdistance=0.8)
        # Alternative to show angles:
        # angles = [108, 144, 72, 36]
        # wedges, texts = ax.pie(angles, labels=labels, startangle=90, counterclock=False)
        # for i, p in enumerate(wedges):
        #      ang = (p.theta2 - p.theta1)/2. + p.theta1
        #      y = np.sin(np.deg2rad(ang))
        #      x = np.cos(np.deg2rad(ang))
        #      ax.text(x * 0.7, y * 0.7, f"{angles[i]}°", ha='center', va='center', fontsize=8)


        ax.axis('equal')
        ax.set_title("Number of Pets Owned by Students")
        plt.setp(autotexts, size=8, weight="bold", color="white")


        return self._save_plot(fig, filename)

    # F2: Vectors - Definition
    def f2_ve_de_notes_vector_notation(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), facecolor='white')
        fig.suptitle("Vector Notation Examples")

        # Geometric Vector AB
        self._setup_ax_no_origin(ax1, title="Geometric Vector")
        A = (0.2, 0.3); B = (0.8, 0.7)
        ax1.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1], head_width=0.05, head_length=0.08, fc='blue', ec='blue', length_includes_head=True)
        ax1.plot(A[0], A[1], 'ko'); ax1.text(A[0]-0.02, A[1]-0.02, 'A', ha='right', va='top')
        ax1.plot(B[0], B[1], 'ko'); ax1.text(B[0]+0.02, B[1]+0.02, 'B', ha='left', va='bottom')
        ax1.text((A[0]+B[0])/2, (A[1]+B[1])/2 + 0.05, '$\\vec{AB}$', color='blue', ha='center', va='bottom', fontsize=12)
        ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.axis('equal'); ax1.axis('off')

        # Column Vector Example
        self._setup_ax(ax2, title="Column Vector [3, 2]", xlabel="x", ylabel="y", xlim=(0, 5), ylim=(0, 4), aspect='equal')
        ax2.set_xticks(np.arange(0, 6)); ax2.set_yticks(np.arange(0, 5))
        start = (1, 1); end = (4, 3)
        dx = end[0] - start[0]; dy = end[1] - start[1]
        ax2.arrow(start[0], start[1], dx, dy, head_width=0.15, head_length=0.2, fc='red', ec='red', length_includes_head=True)
        ax2.plot(start[0], start[1], 'ko'); ax2.text(start[0]-0.2, start[1]-0.2, '(1,1)')
        ax2.plot(end[0], end[1], 'ko'); ax2.text(end[0]+0.1, end[1]+0.1, '(4,3)')
        vector_text = '[3, 2]'
        ax2.text(start[0] + dx/2 + 0.3, start[1] + dy/2-0.1, vector_text, color='red', ha='left', va='bottom', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        return self._save_plot(fig, filename)

    # F2: Vectors - Types
    def f2_ve_ty_notes_equal_vectors(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Equal Vectors")

        A=(0.1, 0.7); B=(0.5, 0.8)
        C=(0.3, 0.3); D=(0.7, 0.4) # Same dx=0.4, dy=0.1

        ax.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1], head_width=0.03, head_length=0.05, fc='b', ec='b', length_includes_head=True)
        ax.text((A[0]+B[0])/2, (A[1]+B[1])/2+0.03, '$\\vec{AB}$', color='b')
        ax.text(A[0]-0.02, A[1], 'A'); ax.text(B[0]+0.02, B[1], 'B')

        ax.arrow(C[0], C[1], D[0]-C[0], D[1]-C[1], head_width=0.03, head_length=0.05, fc='r', ec='r', length_includes_head=True)
        ax.text((C[0]+D[0])/2, (C[1]+D[1])/2+0.03, '$\\vec{CD}$', color='r')
        ax.text(C[0]-0.02, C[1], 'C'); ax.text(D[0]+0.02, D[1], 'D')

        ax.text(0.5, 0.1, '$\\vec{AB} = \\vec{CD}$ (Same magnitude and direction)', ha='center')

        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ve_ty_notes_negative_vectors(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Negative Vectors")

        A=(0.2, 0.4); B=(0.7, 0.7)

        # Vector a = AB
        ax.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1], head_width=0.03, head_length=0.05, fc='b', ec='b', length_includes_head=True)
        ax.text((A[0]+B[0])/2-0.08, (A[1]+B[1])/2+0.03, '$\\vec{AB} = \\mathbf{a}$', color='b')
        ax.text(A[0]-0.02, A[1], 'A'); ax.text(B[0]+0.02, B[1], 'B')

        # Vector -a = BA
        ax.arrow(B[0], B[1], A[0]-B[0], A[1]-B[1], head_width=0.03, head_length=0.05, fc='r', ec='r', length_includes_head=True)
        ax.text((B[0]+A[0])/2, (B[1]+A[1])/2-0.05, '$\\vec{BA} = -\\mathbf{a}$', color='r', va='top')

        ax.text(0.5, 0.1, 'Same magnitude, opposite direction', ha='center')

        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f2_ve_ty_work_draw_vector_neg3_1(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Draw Vector [-3, 1] from (2, 2)", xlabel="x", ylabel="y", xlim=(-1.5, 3.5), ylim=(1.5, 3.5), aspect='equal')
        ax.set_xticks(np.arange(-1, 4)); ax.set_yticks(np.arange(2, 4))

        start = (2, 2); vector = (-3, 1)
        end = (start[0] + vector[0], start[1] + vector[1])

        ax.arrow(start[0], start[1], vector[0], vector[1], head_width=0.1, head_length=0.15, fc='m', ec='m', length_includes_head=True)
        ax.plot(start[0], start[1], 'ko'); ax.text(start[0]+0.1, start[1]-0.05, 'Start (2,2)')
        ax.plot(end[0], end[1], 'ko'); ax.text(end[0]-0.05, end[1]+0.1, 'End (-1,3)', ha='right')


        return self._save_plot(fig, filename)

    def f2_ve_ty_work_identify_equal_vectors(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        self._setup_ax(ax, title="Identify Equal Vectors", xlabel="x", ylabel="y", xlim=(0, 7), ylim=(0, 5), aspect='equal')
        ax.set_xticks(np.arange(0, 8)); ax.set_yticks(np.arange(0, 6))

        vectors = {
            'AB': ((1,1), (3,2)), # <2,1>
            'CD': ((4,1), (6,2)), # <2,1>
            'EF': ((1,3), (3,4)), # <2,1>
            'GH': ((4,4), (6,3))  # <2,-1>
        }
        colors = {'AB': 'b', 'CD': 'r', 'EF': 'g', 'GH': 'm'}

        for label, (start, end) in vectors.items():
            dx = end[0] - start[0]; dy = end[1] - start[1]
            ax.arrow(start[0], start[1], dx, dy, head_width=0.15, head_length=0.2, fc=colors[label], ec=colors[label], length_includes_head=True)
            ax.text(start[0]-0.1, start[1]-0.1, label[0], ha='right', va='top', fontsize=8)
            ax.text(end[0]+0.1, end[1]+0.1, label[1], ha='left', va='bottom', fontsize=8)
            ax.text(start[0]+dx/2-0.1, start[1]+dy/2+0.1, f'$\\vec{{{label}}}$', color=colors[label])

        ax.text(0.5, 4.5, 'Equal: $\\vec{AB}=\\vec{CD}=\\vec{EF}$', color='purple')

        return self._save_plot(fig, filename)


    # F2: Vectors - Operations
    def f2_ve_op_notes_vector_addition(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor='white')
        fig.suptitle("Vector Addition")

        # Triangle Law
        self._setup_ax_no_origin(ax1, title="Triangle Law")
        O = (0.1, 0.2)
        a_vec = (0.4, 0.1)
        b_vec = (0.2, 0.4)
        P = (O[0] + a_vec[0], O[1] + a_vec[1])
        Q = (P[0] + b_vec[0], P[1] + b_vec[1])

        ax1.arrow(O[0], O[1], a_vec[0], a_vec[1], head_width=0.03, head_length=0.05, fc='b', ec='b', length_includes_head=True, label='a')
        ax1.text(O[0]+a_vec[0]/2, O[1]+a_vec[1]/2-0.03, '$\\mathbf{a}$', color='b')
        ax1.arrow(P[0], P[1], b_vec[0], b_vec[1], head_width=0.03, head_length=0.05, fc='r', ec='r', length_includes_head=True, label='b')
        ax1.text(P[0]+b_vec[0]/2+0.02, P[1]+b_vec[1]/2, '$\\mathbf{b}$', color='r')
        ax1.arrow(O[0], O[1], Q[0]-O[0], Q[1]-O[1], head_width=0.03, head_length=0.05, fc='g', ec='g', length_includes_head=True, linestyle='-', label='a+b')
        ax1.text(O[0]+(Q[0]-O[0])/2, O[1]+(Q[1]-O[1])/2+0.03, '$\\mathbf{a+b}$', color='g')
        ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.axis('equal'); ax1.axis('off')

        # Parallelogram Law
        self._setup_ax_no_origin(ax2, title="Parallelogram Law")
        O = (0.2, 0.2)
        a_vec = (0.5, 0.1)
        b_vec = (0.2, 0.4)
        P = (O[0] + a_vec[0], O[1] + a_vec[1])
        R = (O[0] + b_vec[0], O[1] + b_vec[1])
        Q = (O[0] + a_vec[0] + b_vec[0], O[1] + a_vec[1] + b_vec[1])

        ax2.arrow(O[0], O[1], a_vec[0], a_vec[1], head_width=0.03, head_length=0.05, fc='b', ec='b', length_includes_head=True)
        ax2.text(O[0]+a_vec[0]/2, O[1]+a_vec[1]/2-0.03, '$\\mathbf{a}$', color='b')
        ax2.arrow(O[0], O[1], b_vec[0], b_vec[1], head_width=0.03, head_length=0.05, fc='r', ec='r', length_includes_head=True)
        ax2.text(O[0]+b_vec[0]/2-0.03, O[1]+b_vec[1]/2, '$\\mathbf{b}$', color='r')

        # Dashed lines for parallelogram
        ax2.plot([P[0], Q[0]], [P[1], Q[1]], 'r--', lw=1)
        ax2.plot([R[0], Q[0]], [R[1], Q[1]], 'b--', lw=1)

        # Resultant Diagonal
        ax2.arrow(O[0], O[1], Q[0]-O[0], Q[1]-O[1], head_width=0.03, head_length=0.05, fc='g', ec='g', length_includes_head=True, linestyle='-')
        ax2.text(O[0]+(Q[0]-O[0])/2+0.02, O[1]+(Q[1]-O[1])/2+0.02, '$\\mathbf{a+b}$', color='g')
        ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.axis('equal'); ax2.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)

    def f2_ve_op_notes_vector_subtraction(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor='white')
        fig.suptitle("Vector Subtraction: a - b")

        # Geometric Method (Head of b to Head of a)
        self._setup_ax_no_origin(ax1, title="Geometric (Tip-to-Tip)")
        O = (0.2, 0.2)
        a_vec = (0.6, 0.3)
        b_vec = (0.2, 0.5)
        A = (O[0] + a_vec[0], O[1] + a_vec[1]) # Head of a
        B = (O[0] + b_vec[0], O[1] + b_vec[1]) # Head of b

        ax1.arrow(O[0], O[1], a_vec[0], a_vec[1], head_width=0.03, head_length=0.05, fc='b', ec='b', length_includes_head=True)
        ax1.text(O[0]+a_vec[0]/2, O[1]+a_vec[1]/2+0.03, '$\\mathbf{a}$', color='b')
        ax1.arrow(O[0], O[1], b_vec[0], b_vec[1], head_width=0.03, head_length=0.05, fc='r', ec='r', length_includes_head=True)
        ax1.text(O[0]+b_vec[0]/2-0.03, O[1]+b_vec[1]/2, '$\\mathbf{b}$', color='r')

        # Resultant a-b (from head of b to head of a)
        ax1.arrow(B[0], B[1], A[0]-B[0], A[1]-B[1], head_width=0.03, head_length=0.05, fc='g', ec='g', length_includes_head=True, linestyle='-')
        ax1.text(B[0]+(A[0]-B[0])/2+0.02, B[1]+(A[1]-B[1])/2, '$\\mathbf{a-b}$', color='g')
        ax1.set_xlim(0,1); ax1.set_ylim(0,1); ax1.axis('equal'); ax1.axis('off')


        # Addition Method (a + (-b))
        self._setup_ax_no_origin(ax2, title="Addition (a + (-b))")
        O = (0.1, 0.3)
        a_vec = (0.6, 0.3)
        b_vec = (0.2, 0.5)
        neg_b_vec = (-b_vec[0], -b_vec[1])
        A = (O[0] + a_vec[0], O[1] + a_vec[1]) # Head of a
        Q = (A[0] + neg_b_vec[0], A[1] + neg_b_vec[1]) # Head of a+(-b)

        ax2.arrow(O[0], O[1], a_vec[0], a_vec[1], head_width=0.03, head_length=0.05, fc='b', ec='b', length_includes_head=True)
        ax2.text(O[0]+a_vec[0]/2, O[1]+a_vec[1]/2+0.03, '$\\mathbf{a}$', color='b')
        # Draw -b starting from head of a
        ax2.arrow(A[0], A[1], neg_b_vec[0], neg_b_vec[1], head_width=0.03, head_length=0.05, fc='purple', ec='purple', length_includes_head=True)
        ax2.text(A[0]+neg_b_vec[0]/2- 0.03, A[1]+neg_b_vec[1]/2+0.03, '$\\mathbf{-b}$', color='purple')
        # Resultant a+(-b)
        ax2.arrow(O[0], O[1], Q[0]-O[0], Q[1]-O[1], head_width=0.03, head_length=0.05, fc='g', ec='g', length_includes_head=True, linestyle='-')
        ax2.text(O[0]+(Q[0]-O[0])/2, O[1]+(Q[1]-O[1])/2-0.04, '$\\mathbf{a+(-b)}$', color='g', va='top')
        ax2.set_xlim(0,1); ax2.set_ylim(0,1); ax2.axis('equal'); ax2.axis('off')


        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)


    # F2: Matrices - Order
    # (No images specified for Form 2 Matrices)


    # F2: Transformation - Translation
    def f2_tr_tr_work_draw_triangles_lmn(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title="Translation of Triangle LMN", xlabel="x", ylabel="y", xlim=(-1, 6), ylim=(-2, 5), aspect='equal')
        ax.set_xticks(np.arange(-1, 7))
        ax.set_yticks(np.arange(-2, 6))

        # Object Triangle LMN
        L = (0, 1); M = (3, 1); N = (0, 4)
        obj_poly = patches.Polygon([L, M, N], closed=True, fill=False, edgecolor='blue', linestyle='-', lw=1.5, label='Object LMN')
        ax.add_patch(obj_poly)
        ax.text(L[0]-0.1, L[1], 'L(0,1)', color='blue', fontsize=9, ha='right')
        ax.text(M[0]+0.1, M[1], 'M(3,1)', color='blue', fontsize=9, ha='left')
        ax.text(N[0]-0.1, N[1], 'N(0,4)', color='blue', fontsize=9, ha='right')

        # Image Triangle L'M'N'
        L_img = (2, 0); M_img = (5, 0); N_img = (2, 3)
        img_poly = patches.Polygon([L_img, M_img, N_img], closed=True, fill=False, edgecolor='red', linestyle='--', lw=1.5, label="Image L'M'N'")
        ax.add_patch(img_poly)
        ax.text(L_img[0], L_img[1]-0.3, "L'(2,0)", color='red', fontsize=9)
        ax.text(M_img[0], M_img[1]-0.3, "M'(5,0)", color='red', fontsize=9)
        ax.text(N_img[0]+0.1, N_img[1], "N'(2,3)", color='red', fontsize=9)

        # Translation Vectors (optional)
        vec = (2, -1)
        ax.arrow(L[0], L[1], vec[0], vec[1], head_width=0.2, head_length=0.2, fc='gray', ec='gray', length_includes_head=True, lw=0.5, linestyle=':')
        ax.arrow(M[0], M[1], vec[0], vec[1], head_width=0.2, head_length=0.2, fc='gray', ec='gray', length_includes_head=True, lw=0.5, linestyle=':')
        ax.arrow(N[0], N[1], vec[0], vec[1], head_width=0.2, head_length=0.2, fc='gray', ec='gray', length_includes_head=True, lw=0.5, linestyle=':')
        # Use LaTeX for vector notation if possible, otherwise simple text
        try:
            plt.rc('text', usetex=True)
            ax.text(1, 2.5, 'Vector $\\begin{pmatrix} 2 \\\\ -1 \\end{pmatrix}$', color='gray', fontsize=9)
        except RuntimeError:
            plt.rc('text', usetex=False)
            ax.text(1, 2.5, 'Vector [2, -1]', color='gray', fontsize=9)

        ax.legend()
        return self._save_plot(fig, filename)

    # F2: Transformation - Reflection
    def f2_tr_re_notes_reflections_std_lines(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.5), facecolor='white')
        fig.suptitle("Reflections in Standard Lines")

        # Reflection in x-axis
        ax = axs[0]
        self._setup_ax(ax, title="Reflection in x-axis (y=0)", xlabel="x", ylabel="y", xlim=(-1, 4), ylim=(-4, 4), aspect='equal')
        ax.axhline(0, color='black', linewidth=1.5) # Emphasize x-axis
        P = (2, 3); P_img = (2, -3)
        ax.plot(P[0], P[1], 'bo', markersize=6); ax.text(P[0]+0.1, P[1], 'P(2,3)')
        ax.plot(P_img[0], P_img[1], 'ro', markersize=6); ax.text(P_img[0]+0.1, P_img[1], "P'(2,-3)")
        ax.plot([P[0], P_img[0]], [P[1], P_img[1]], 'k--', lw=1) # Line connecting P, P'
        ax.text(P[0]+0.1, 0, '(Mirror Line)', fontsize=8, color='gray')

        # Reflection in y-axis
        ax = axs[1]
        self._setup_ax(ax, title="Reflection in y-axis (x=0)", xlabel="x", ylabel="y", xlim=(-3, 3), ylim=(-1, 4), aspect='equal')
        ax.axvline(0, color='black', linewidth=1.5) # Emphasize y-axis
        Q = (-2, 1); Q_img = (2, 1)
        ax.plot(Q[0], Q[1], 'bo', markersize=6); ax.text(Q[0]-0.1, Q[1], 'Q(-2,1)', ha='right')
        ax.plot(Q_img[0], Q_img[1], 'ro', markersize=6); ax.text(Q_img[0]+0.1, Q_img[1], "Q'(2,1)")
        ax.plot([Q[0], Q_img[0]], [Q[1], Q_img[1]], 'k--', lw=1)
        ax.text(0, -0.5, '(Mirror\nLine)', fontsize=8, color='gray', ha='center')


        # Reflection in y=x
        ax = axs[2]
        self._setup_ax(ax, title="Reflection in y = x", xlabel="x", ylabel="y", xlim=(-1, 5), ylim=(-1, 5), aspect='equal')
        x_line = np.array([-1, 5])
        ax.plot(x_line, x_line, 'g--', lw=1.5, label='y=x (Mirror Line)') # Line y=x
        R = (1, 4); R_img = (4, 1)
        ax.plot(R[0], R[1], 'bo', markersize=6); ax.text(R[0]+0.1, R[1], 'R(1,4)')
        ax.plot(R_img[0], R_img[1], 'ro', markersize=6); ax.text(R_img[0]+0.1, R_img[1], "R'(4,1)")
        ax.plot([R[0], R_img[0]], [R[1], R_img[1]], 'k--', lw=1)
        ax.legend(fontsize=8)


        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._save_plot(fig, filename)

    def f2_tr_re_work_reflect_triangle_x_axis(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title="Reflection of $\Delta$ABC in x-axis", xlabel="x", ylabel="y", xlim=(0, 5), ylim=(-4, 4), aspect='equal')
        ax.axhline(0, color='black', linewidth=1.5, label='x-axis (Mirror Line)') # Emphasize x-axis

        # Object Triangle ABC
        A = (1, 1); B = (4, 1); C = (1, 3)
        obj_poly = patches.Polygon([A, B, C], closed=True, fill=False, edgecolor='blue', linestyle='-', lw=1.5, label='Object ABC')
        ax.add_patch(obj_poly)
        ax.text(A[0], A[1]+0.1, 'A(1,1)', color='blue', fontsize=9)
        ax.text(B[0], B[1]+0.1, 'B(4,1)', color='blue', fontsize=9)
        ax.text(C[0], C[1]+0.1, 'C(1,3)', color='blue', fontsize=9)

        # Image Triangle A'B'C'
        A_img = (1, -1); B_img = (4, -1); C_img = (1, -3)
        img_poly = patches.Polygon([A_img, B_img, C_img], closed=True, fill=False, edgecolor='red', linestyle='--', lw=1.5, label="Image A'B'C'")
        ax.add_patch(img_poly)
        ax.text(A_img[0], A_img[1]-0.3, "A'(1,-1)", color='red', fontsize=9)
        ax.text(B_img[0], B_img[1]-0.3, "B'(4,-1)", color='red', fontsize=9)
        ax.text(C_img[0]+0.1, C_img[1], "C'(1,-3)", color='red', fontsize=9)

        # Lines connecting points to images (optional)
        ax.plot([A[0],A_img[0]], [A[1],A_img[1]], 'k:', lw=0.5)
        ax.plot([B[0],B_img[0]], [B[1],B_img[1]], 'k:', lw=0.5)
        ax.plot([C[0],C_img[0]], [C[1],C_img[1]], 'k:', lw=0.5)

        ax.legend()
        return self._save_plot(fig, filename)

    # F3: Real Numbers - Concepts
    # (No images specified)

    # F3: Real Numbers - Ratios, Rates, Proportions
    # (No images specified)

    # F3: Real Numbers - Standard Form
    # (No images specified)

    # F3: Real Numbers - Number Bases
    # (No images specified)

    # F3: Real Numbers - Scales
    # (No images specified)

    # F3: Sets
    def f3_se_notes_venn_3_sets(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Venn Diagram (3 Sets)")

        # Universal Set
        rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5, edgecolor='black', facecolor='none', zorder=1)
        ax.add_patch(rect)
        ax.text(0.95, 0.9, 'U', ha='right', va='top', fontsize=12)

        # Circles A, B, C
        radius = 0.28
        center_a = (0.4, 0.65)
        center_b = (0.6, 0.65)
        center_c = (0.5, 0.4)

        circle_a = patches.Circle(center_a, radius, linewidth=1.5, edgecolor='blue', facecolor='blue', alpha=0.1, zorder=2)
        circle_b = patches.Circle(center_b, radius, linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.1, zorder=2)
        circle_c = patches.Circle(center_c, radius, linewidth=1.5, edgecolor='green', facecolor='green', alpha=0.1, zorder=2)
        ax.add_patch(circle_a); ax.add_patch(circle_b); ax.add_patch(circle_c)

        # Circle Labels
        ax.text(0.2, 0.8, 'A', ha='center', va='center', fontsize=14, color='blue')
        ax.text(0.8, 0.8, 'B', ha='center', va='center', fontsize=14, color='red')
        ax.text(0.5, 0.15, 'C', ha='center', va='center', fontsize=14, color='green')

        # Region Labels (Numbering 1-8) - Approximate centers
        ax.text(0.2, 0.75, '1 (A only)', fontsize=8, ha='center')
        ax.text(0.8, 0.75, '2 (B only)', fontsize=8, ha='center')
        ax.text(0.5, 0.25, '3 (C only)', fontsize=8, ha='center')
        ax.text(0.5, 0.7, '4 (A$\cap$B only)', fontsize=8, ha='center')
        ax.text(0.3, 0.45, '5 (A$\cap$C only)', fontsize=8, ha='center')
        ax.text(0.7, 0.45, '6 (B$\cap$C only)', fontsize=8, ha='center')
        ax.text(0.5, 0.55, '7\n(A$\cap$B$\cap$C)', fontsize=8, ha='center', va='center')
        ax.text(0.3, 0.1, '8 (Outside)', fontsize=8, ha='center')


        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f3_se_work_venn_3_sets_example(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Venn Diagram Example (Numbers)")

        # Universal Set
        rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5, edgecolor='black', facecolor='none', zorder=1)
        ax.add_patch(rect)
        ax.text(0.9, 0.85, 'U={1..10}', ha='right', va='top', fontsize=9)

        # Circles A, B, C
        radius = 0.28
        center_a = (0.4, 0.65)
        center_b = (0.6, 0.65)
        center_c = (0.5, 0.4)

        circle_a = patches.Circle(center_a, radius, linewidth=1.5, edgecolor='blue', facecolor='none', zorder=2)
        circle_b = patches.Circle(center_b, radius, linewidth=1.5, edgecolor='red', facecolor='none', zorder=2)
        circle_c = patches.Circle(center_c, radius, linewidth=1.5, edgecolor='green', facecolor='none', zorder=2)
        ax.add_patch(circle_a); ax.add_patch(circle_b); ax.add_patch(circle_c)

        # Circle Labels
        ax.text(0.15, 0.90, 'A={1,2,3,4,5}', ha='center', va='center', fontsize=9, color='blue')
        ax.text(0.85, 0.9, 'B={2,4,6,8}', ha='center', va='center', fontsize=9, color='red')
        ax.text(0.5, 0.1, 'C={3,5,7,9}', ha='center', va='center', fontsize=9, color='green')

        # Region Counts (from worked example steps)
        ax.text(0.3, 0.75, '1 (1)', fontsize=10, ha='center') # A only
        ax.text(0.7, 0.75, '2 (6,8)', fontsize=10, ha='center') # B only
        ax.text(0.5, 0.25, '2 (7,9)', fontsize=10, ha='center') # C only
        ax.text(0.5, 0.7, '2 (2,4)', fontsize=10, ha='center') # A&B only
        ax.text(0.35, 0.45, '2 (3,5)', fontsize=10, ha='center') # A&C only
        ax.text(0.65, 0.45, '0', fontsize=10, ha='center') # B&C only
        ax.text(0.5, 0.55, '0', fontsize=10, ha='center', va='center') # A&B&C
        ax.text(0.1, 0.1, '1 (10)', fontsize=10, ha='center') # Outside

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f3_se_work_venn_3_sets_students(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Student Subject Choices (n=40)")

        # Universal Set
        rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5, edgecolor='black', facecolor='none', zorder=1)
        ax.add_patch(rect)
        ax.text(0.95, 0.9, 'U (40)', ha='right', va='top', fontsize=9)

        # Circles P, C, B
        radius = 0.28
        center_p = (0.4, 0.65) # Physics
        center_c = (0.6, 0.65) # Chemistry
        center_b = (0.5, 0.4)  # Biology

        circle_p = patches.Circle(center_p, radius, linewidth=1.5, edgecolor='blue', facecolor='none', zorder=2)
        circle_c = patches.Circle(center_c, radius, linewidth=1.5, edgecolor='red', facecolor='none', zorder=2)
        circle_b = patches.Circle(center_b, radius, linewidth=1.5, edgecolor='green', facecolor='none', zorder=2)
        ax.add_patch(circle_p); ax.add_patch(circle_c); ax.add_patch(circle_b)

        # Circle Labels
        ax.text(0.24, 0.8, 'P (20)', ha='center', va='center', fontsize=10, color='blue')
        ax.text(0.76, 0.8, 'C (25)', ha='center', va='center', fontsize=10, color='red')
        ax.text(0.5, 0.17, 'B (18)', ha='center', va='center', fontsize=10, color='green')

        # Region Counts (from worked example steps)
        ax.text(center_p[0]-0.1, center_p[1]+0.1, '6', fontsize=11, ha='center') # P only
        ax.text(center_c[0]+0.1, center_c[1]+0.1, '8', fontsize=11, ha='center') # C only
        ax.text(center_b[0], center_b[1]-0.15, '2', fontsize=11, ha='center') # B only
        ax.text(0.5, center_p[1]+0.05, '5', fontsize=11, ha='center') # P&C only
        ax.text(center_p[0]-0.05, center_p[1]-0.2, '4', fontsize=11, ha='center') # P&B only
        ax.text(center_c[0]+0.05, center_c[1]-0.2, '7', fontsize=11, ha='center') # C&B only
        ax.text(0.5, 0.55, '5', fontsize=11, ha='center', va='center', weight='bold') # P&C&B
        ax.text(0.1, 0.1, '3', fontsize=11, ha='center') # Outside

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    #***#***#
    # F3: Financial Maths
    def f3_fm_notes_simple_vs_compound(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')

        P = 1000 # Principal
        R_simple = 10 # Simple Interest Rate %
        R_compound = 10 # Compound Interest Rate %
        T = np.arange(0, 11) # Time in years

        # Simple Interest Amount
        A_simple = P * (1 + R_simple/100 * T)

        # Compound Interest Amount (compounded annually)
        A_compound = P * (1 + R_compound/100)**T

        ax.plot(T, A_simple, 'b-', label=f'Simple Interest ({R_simple}%)')
        ax.plot(T, A_compound, 'r-', label=f'Compound Interest ({R_compound}%)')

        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Amount (\\$)")
        ax.set_title(f"Simple vs Compound Interest (P=\\${P}, R={R_compound}%)")
        ax.set_xlim(0, 10)
        ax.set_ylim(P*0.9, max(A_compound)*1.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Annotate difference
        diff_t = 8
        diff_val = P * (1 + R_compound/100)**diff_t - P * (1 + R_simple/100 * diff_t)
        # Use formatted string for annotation text
        ax.annotate(f'Difference grows\n(at T={diff_t}, diff $\\approx$ \\${diff_val:.0f})',
                    xy=(diff_t, P * (1 + R_simple/100 * diff_t) + diff_val/2),
                    xytext=(diff_t - 3, P*1.35),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='black'))

        return self._save_plot(fig, filename)

    # F3: Measures - Mensuration
    def f3_me_ms_notes_combined_shape(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Combined Shape (Rectangle + Semi-circle)")

        rect_w = 0.6 # 6cm
        rect_h = 0.4 # 4cm
        semi_diam = rect_h # 4cm
        semi_rad = semi_diam / 2

        # Rectangle Coordinates
        rect_bl = (0.1, 0.3) # Bottom left
        ax.add_patch(patches.Rectangle(rect_bl, rect_w, rect_h, fill=False, ec='k', lw=1.5))

        # Semi-circle attached to right side
        center_semi = (rect_bl[0]+rect_w, rect_bl[1]+rect_h/2)
        semi_circle = patches.Arc(center_semi, semi_diam, semi_diam, angle=-90, theta1=0, theta2=180, fill=False, ec='k', lw=1.5)
        ax.add_patch(semi_circle)

        # Dimensions
        ax.text(rect_bl[0]+rect_w/2, rect_bl[1]-0.05, '6 cm', ha='center', va='top')
        ax.text(rect_bl[0]-0.05, rect_bl[1]+rect_h/2, '4 cm', ha='right', va='center')
        ax.plot([center_semi[0], center_semi[0]], [center_semi[1]-semi_rad, center_semi[1]+semi_rad], 'r:', lw=1) # Diameter line
        ax.text(center_semi[0]+0.03, center_semi[1], 'Diam=4cm', color='r', fontsize=8)


        # Use mathtext for formula description
        ax.text(0.1, 0.1, r'Perimeter = $6+4+6 + \mathrm{Arc}(d=4)$' + '\n' + r'Area = Area(Rect) + Area(Semi)', fontsize=8)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f3_me_ms_notes_cylinder_volume(self):
        filename = get_filename_from_caller()
        fig = plt.figure(figsize=(4, 5), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        ax.set_title("Cylinder Volume")

        # Cylinder parameters
        r = 0.5
        h = 1.0
        center_x, center_y = 0, 0

        # Cylinder surface
        x = np.linspace(-r, r, 100)
        z = np.linspace(0, h, 50)
        Xc, Zc = np.meshgrid(x, z)
        Yc = np.sqrt(np.maximum(0, r**2 - Xc**2)) # Use maximum to avoid sqrt of negative

        # Draw surfaces
        ax.plot_surface(Xc, Yc + center_y, Zc, alpha=0.3, color='lightblue', rstride=1, cstride=5)
        ax.plot_surface(Xc, -Yc + center_y, Zc, alpha=0.3, color='lightblue', rstride=1, cstride=5)

        # Draw top and bottom circles
        theta = np.linspace(0, 2*np.pi, 100)
        xc = center_x + r * np.cos(theta)
        yc = center_y + r * np.sin(theta)
        ax.plot(xc, yc, 0, color='blue', lw=1) # Bottom circle
        ax.plot(xc, yc, h, color='blue', lw=1) # Top circle

        # Annotations
        # Radius
        ax.plot([center_x, center_x+r], [center_y, center_y], [0, 0], 'r-', lw=1.5)
        ax.text(center_x+r/2, center_y-0.1, 0, 'r', color='red', ha='center', va='top', zdir='y')
        # Height
        ax.plot([center_x+r*1.1, center_x+r*1.1], [center_y, center_y], [0, h], 'g-', lw=1.5)
        ax.text(center_x+r*1.1+0.12, center_y, h/2, 'h', color='g', ha='left', va='center', zdir='x')
        # Formula - Use mathtext
        ax.text2D(0.5, 0.05, r'$V = \pi r^2 h$', transform=ax.transAxes, ha='center', va='top', fontsize=11)


        # Adjust plot limits and view
        ax.set_xlim(-r*1.2, r*1.2)
        ax.set_ylim(-r*1.2, r*1.2)
        ax.set_zlim(0, h*1.2)
        ax.set_box_aspect([r*2.4, r*2.4, h*1.2]) # Adjust aspect ratio
        ax.set_axis_off()
        ax.view_init(elev=20, azim=30)


        return self._save_plot(fig, filename)

    def f3_me_ms_work_combined_shape_area(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax_no_origin(ax, title="Combined Shape Area (Square + Triangle)")

        sq_side = 0.6 # 10m
        tri_h = 0.24 # 4m (scaled relative to 10m)
        tri_b = sq_side

        # Square Coordinates
        sq_bl = (0.2, 0.1)
        sq_verts = [(sq_bl[0], sq_bl[1]), (sq_bl[0]+sq_side, sq_bl[1]),
                    (sq_bl[0]+sq_side, sq_bl[1]+sq_side), (sq_bl[0], sq_bl[1]+sq_side)]
        ax.add_patch(patches.Polygon(sq_verts, fill=False, ec='k', lw=1.5))

        # Triangle Coordinates (attached to top side of square)
        tri_v1 = sq_verts[3] # Top left of square
        tri_v2 = sq_verts[2] # Top right of square
        tri_v3 = ( (tri_v1[0]+tri_v2[0])/2, tri_v1[1]+tri_h ) # Top vertex
        ax.add_patch(patches.Polygon([tri_v1, tri_v2, tri_v3], fill=False, ec='k', lw=1.5))

        # Dimensions
        ax.text(sq_bl[0]+sq_side/2, sq_bl[1]-0.05, '10 m', ha='center', va='top')
        ax.text(sq_bl[0]-0.05, sq_bl[1]+sq_side/2, '10 m', ha='right', va='center')
        # Triangle height
        ax.plot([tri_v3[0], tri_v3[0]], [tri_v1[1], tri_v3[1]], 'r--', lw=1)
        ax.text(tri_v3[0]+0.03, (tri_v1[1]+tri_v3[1])/2, 'h=4m', color='r', ha='left', va='center', fontsize=9)


        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    def f3_me_ms_work_combined_shape_perimeter(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        self._setup_ax_no_origin(ax, title="Combined Shape Perimeter (Rectangle - Semi-circle)")

        rect_w = 0.8 # 8cm
        rect_h = 0.5 # 5cm
        semi_diam = rect_w # 8cm
        semi_rad = semi_diam / 2

        # Outer boundary points
        bl = (0.1, 0.2) # Bottom Left
        br = (bl[0]+rect_w, bl[1]) # Bottom Right
        tr = (br[0], br[1]+rect_h) # Top Right (before cutout)
        tl = (bl[0], bl[1]+rect_h) # Top Left (before cutout)
        center_cutout = (bl[0]+rect_w/2, bl[1]+rect_h) # Center of top side

        # Draw outer boundary lines
        ax.plot([bl[0], br[0]], [bl[1], br[1]], 'b-', lw=2) # Bottom
        ax.plot([br[0], tr[0]], [br[1], tr[1]], 'b-', lw=2) # Right
        ax.plot([tl[0], bl[0]], [tl[1], bl[1]], 'b-', lw=2) # Left

        # Draw semi-circular cutout arc
        cutout_arc = patches.Arc(center_cutout, semi_diam, semi_diam, angle=0, theta1=180, theta2=360, fill=False, ec='b', lw=2)
        ax.add_patch(cutout_arc)

        # Dimensions (Optional)
        ax.text((bl[0]+br[0])/2, bl[1]-0.05, '8 cm', ha='center', va='top')
        ax.text(bl[0]-0.05, (bl[1]+tl[1])/2, '5 cm', ha='right', va='center')
        ax.text(br[0]+0.05, (br[1]+tr[1])/2, '5 cm', ha='left', va='center')
        ax.text(center_cutout[0], center_cutout[1]-0.05, 'Cutout Arc (d=8cm)', ha='center', va='bottom', fontsize=8)

        # Indicate perimeter path
        ax.text(0.5, 0.05, 'Perimeter = Blue Lines', ha='center', va='top', color='blue')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('equal'); ax.axis('off')
        return self._save_plot(fig, filename)

    # F3: Graphs - Functional Graphs
    def f3_gr_fg_notes_intercept_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title="Graph using Intercepts (y = 2x - 4)", xlabel="x", ylabel="y", xlim=(-1, 4), ylim=(-5, 2), aspect='auto') # Not equal aspect usually

        x_intercept = 2
        y_intercept = -4

        # Plot intercepts
        ax.plot(x_intercept, 0, 'ro', markersize=6, label=f'x-int ({x_intercept}, 0)')
        ax.plot(0, y_intercept, 'bo', markersize=6, label=f'y-int (0, {y_intercept})')

        # Draw line
        x_line = np.array([-1, 4])
        y_line = 2*x_line - 4
        ax.plot(x_line, y_line, 'g-', lw=1.5, label='y = 2x - 4')

        ax.legend()
        return self._save_plot(fig, filename)

    def f3_gr_fg_notes_quadratic_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title=r"Quadratic Graph ($y = x^2$)", xlabel="x", ylabel="y", xlim=(-3, 3), ylim=(-1, 5), aspect='auto') # Mathtext

        x = np.linspace(-2.2, 2.2, 100) # Smooth curve
        y = x**2

        # Plot points from example
        xp = [-2, -1, 0, 1, 2]
        yp = [4, 1, 0, 1, 4]
        ax.plot(xp, yp, 'ro', markersize=5, alpha=0.7, label='Calculated points')

        # Plot curve
        ax.plot(x, y, 'b-', lw=1.5, label=r'$y = x^2$') # Mathtext

        ax.legend()
        return self._save_plot(fig, filename)

    def f3_gr_fg_work_graph_y_eq_6_min_2x(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title="Graph of y = 6 - 2x", xlabel="x", ylabel="y", xlim=(-1.5, 4.5), ylim=(-1, 8), aspect='auto')

        # Intercepts
        x_intercept = 3
        y_intercept = 6
        ax.plot(x_intercept, 0, 'ro', markersize=6, label=f'x-int ({x_intercept}, 0)')
        ax.plot(0, y_intercept, 'bo', markersize=6, label=f'y-int (0, {y_intercept})')

        # Draw line covering specified range
        x_line = np.array([-1, 4])
        y_line = 6 - 2*x_line
        ax.plot(x_line, y_line, 'g-', lw=1.5, label='y = 6 - 2x')

        ax.legend()
        return self._save_plot(fig, filename)

    def f3_gr_fg_work_graph_y_eq_x2_min_2x_min_3(self):
        filename = get_filename_from_caller("a") # Suffix for first graph
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        self._setup_ax(ax, title=r"Graph of $y = x^2 - 2x - 3$", xlabel="x", ylabel="y", xlim=(-2.5, 4.5), ylim=(-5, 6), aspect='auto') # Mathtext

        # Plot points from table
        xp = [-2, -1, 0, 1, 2, 3, 4]
        yp = [5, 0, -3, -4, -3, 0, 5]
        ax.plot(xp, yp, 'ro', markersize=5, alpha=0.7, label='Calculated points')

        # Smooth curve
        x_curve = np.linspace(-2, 4, 100)
        y_curve = x_curve**2 - 2*x_curve - 3
        ax.plot(x_curve, y_curve, 'b-', lw=1.5, label=r'$y = x^2 - 2x - 3$') # Mathtext

        ax.legend()
        return self._save_plot(fig, filename)

    def f3_gr_fg_work_sketch_y_eq_x2_min_4x_p_3(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        self._setup_ax(ax, title=r"Sketch of $y = x^2 - 4x + 3$", xlabel="x", ylabel="y", xlim=(-0.5, 4.5), ylim=(-1.5, 4), aspect='auto') # Mathtext

        # Intercepts
        y_intercept = 3
        x_intercepts = [1, 3]
        ax.plot(0, y_intercept, 'bo', markersize=6, label=f'y-int (0, {y_intercept})')
        ax.plot(x_intercepts, [0, 0], 'ro', markersize=6, label=f'x-int ({x_intercepts[0]}, 0), ({x_intercepts[1]}, 0)')

        # Turning point (Vertex) x = -b/2a = -(-4)/(2*1) = 2
        vx = 2
        vy = vx**2 - 4*vx + 3
        ax.plot(vx, vy, 'go', markersize=6, label=f'Vertex ({vx}, {vy})')

        # Sketch curve (smooth U-shape through points)
        x_curve = np.linspace(0, 4, 100)
        y_curve = x_curve**2 - 4*x_curve + 3
        ax.plot(x_curve, y_curve, 'k-', lw=1.5) # Use black for sketch line

        ax.legend(fontsize=8)
        return self._save_plot(fig, filename)

    def f3_gr_fg_work_solve_graphically(self):
        filename = get_filename_from_caller("b") # Suffix for second graph related to example 4
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        # Use stored context if needed, or recreate
        # Recreating the graph from Example 4
        self._setup_ax(ax, title=r"Solving $x^2 - 2x - 3 = 2$ Graphically", xlabel="x", ylabel="y", xlim=(-2.5, 4.5), ylim=(-5, 6), aspect='auto') # Mathtext

        xp = [-2, -1, 0, 1, 2, 3, 4]
        yp = [5, 0, -3, -4, -3, 0, 5]
        # ax.plot(xp, yp, 'ro', markersize=5, alpha=0.7) # Points optional for this

        x_curve = np.linspace(-2, 4, 100)
        y_curve = x_curve**2 - 2*x_curve - 3
        ax.plot(x_curve, y_curve, 'b-', lw=1.5, label=r'$y = x^2 - 2x - 3$') # Mathtext

        # Draw horizontal line y=2
        ax.axhline(2, color='purple', linestyle='--', lw=1.5, label='y = 2')

        # Find intersections (approximate from calculation or visual)
        # x^2 - 2x - 3 = 2  => x^2 - 2x - 5 = 0
        # Using quadratic formula: x = [ -b +/- sqrt(b^2-4ac) ] / 2a
        x_sol1 = 1 - np.sqrt(6) # approx -1.45
        x_sol2 = 1 + np.sqrt(6) # approx 3.45
        y_sol = 2

        ax.plot([x_sol1, x_sol2], [y_sol, y_sol], 'go', markersize=6) # Intersection points
        ax.plot([x_sol1, x_sol1], [-5, y_sol], 'g:', lw=1) # Dashed line to x-axis from bottom
        ax.plot([x_sol2, x_sol2], [-5, y_sol], 'g:', lw=1) # Dashed line to x-axis from bottom

        ax.text(x_sol1, -4.5, fr'$x \approx {x_sol1:.1f}$', color='g', ha='center', va='bottom', fontsize=9) # Mathtext
        ax.text(x_sol2, -4.5, fr'$x \approx {x_sol2:.1f}$', color='g', ha='center', va='bottom', fontsize=9) # Mathtext

        ax.legend()
        return self._save_plot(fig, filename)



# === Main Execution Logic (REVISED based on output analysis) ===
def process_json_data(input_file, output_file):
    """Loads JSON, generates images, and updates JSON with image paths."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{input_file}'. Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading JSON: {e}")
        return

    generator = MathDiagramGenerator()
    processed_count = 0
    # Store generated paths keyed by the EXACT description string
    generated_paths_by_desc = {}

    description_to_function = {
        # F1
        """Generate a simple horizontal number line on a plain white background. Mark integers from -5 to +5 with clear tick marks. Label the points -5, 0, and +5. Indicate the positive direction with an arrowhead on the right end and the negative direction with an arrowhead on the left end. Use moderately thick black lines and clear, legible black labels.""": generator.f1_rn_nc_notes_number_line,
        """Generate a simple vertical number line representing sea level on a plain white background. Mark '0' for sea level. Show positive values (e.g., +25, +50) above sea level and negative values (e.g., -25, -50, -75, -100) below sea level with clear tick marks. Use arrows to indicate the submarine's movements: starting at -50, an upward arrow of length 25, and then a downward arrow of length 40 from the intermediate position. Label the start, intermediate, and end points clearly. Use moderately thick black lines and clear, legible black labels.""": generator.f1_rn_nc_ques_submarine_depth,
        """Generate three representations of scales on a plain white background.
1.  **Ratio Scale:** Display the text 'Ratio Scale: 1 : 100,000' clearly.
2.  **Representative Fraction:** Display the text 'Representative Fraction (R.F.): 1/100,000' clearly below the first one.
3.  **Linear Scale:** Draw a bar scale below the text. The bar should be a rectangle divided into segments. Label the start '0'. Mark divisions at regular intervals, labeling them with corresponding real distances (e.g., '1 km', '2 km', '3 km'). Include a small segment to the left of 0, subdivided further (e.g., showing increments of 100m). Use clear black lines and legible labels.""": generator.f1_rn_sc_notes_types_of_scales,
        """Generate a simple Venn diagram on a plain white background.
1. Draw a large rectangle labeled 'U' (or 'ξ') in the top right corner to represent the universal set.
2. Inside the rectangle, draw two overlapping circles of moderate size.
3. Label the left circle 'A' and the right circle 'B' just outside their boundaries.
4. The overlapping region represents A ∩ B.
5. The total area covered by both circles represents A ∪ B.""": generator.f1_se_ty_notes_venn_diagram_two_sets,
        """Generate a Venn diagram on a plain white background.
1. Draw a large rectangle labeled 'U' in the top right corner.
2. Inside the rectangle, draw two overlapping circles labeled 'A' (left) and 'B' (right).
3. Place the number '3' in the overlapping region (intersection).
4. Place the numbers '1' and '2' inside circle A, but outside the overlap.
5. Place the numbers '4' and '5' inside circle B, but outside the overlap.
6. Place the numbers '6' and '7' inside the rectangle U, but outside both circles.
7. Use clear, legible black labels and numbers.""": generator.f1_se_ty_work_venn_diagram_example,
        """Generate a Venn diagram on a plain white background.
1. Draw a large rectangle labeled 'U' in the top right corner.
2. Inside the rectangle, draw two overlapping circles labeled 'A' (left) and 'B' (right).
3. Place elements 'a' and 'b' in circle A only.
4. Place elements 'c' and 'd' in the overlapping region (intersection).
5. Place elements 'e' and 'f' in circle B only.
6. Place element 'g' inside the rectangle U, but outside both circles.
7. Shade the overlapping region (where 'c' and 'd' are located) with light gray color.
8. Use clear, legible black labels and letters.""": generator.f1_se_ty_ques_venn_diagram_shaded,
        """Generate two simple thermometer illustrations side-by-side on a plain white background. 
1. Left thermometer labeled 'Celsius (°C)'. Show markings from -10°C to 40°C in increments of 10. Indicate the freezing point (0°C) and boiling point (100°C - extend scale if needed or note it). Show the mercury/red line indicating a temperature of 25°C.
2. Right thermometer labeled 'Fahrenheit (°F)'. Show markings corresponding roughly to the Celsius scale, including freezing (32°F) and boiling (212°F). Show the mercury/red line indicating the equivalent of 25°C (which is 77°F).
Use clear, legible labels and moderately thick lines.""": generator.f1_me_me_notes_thermometers,
        """Generate two simple illustrations of liquid containers on a plain white background.
1. Left: A measuring jug with markings in milliliters (mL). Show markings for 250 mL, 500 mL, 750 mL, and 1000 mL (1 L). Show liquid filled up to the 750 mL mark.
2. Right: A standard 1 Liter bottle clearly labeled '1 L' or '1000 mL'.
Use clear, legible labels and moderately thick lines.""": generator.f1_me_me_notes_capacity_containers,
        """Generate diagrams for perimeter calculation on a plain white background.
1.  **Square:** Draw a square with moderately thick black lines. Label one side 's'. Indicate the perimeter calculation P = 4s next to it.
2.  **Rectangle:** Draw a rectangle. Label the longer side 'l' (length) and the shorter side 'w' (width). Indicate the perimeter calculation P = 2(l + w) next to it.
3.  **Triangle:** Draw a general triangle. Label the three sides 'a', 'b', and 'c'. Indicate the perimeter calculation P = a + b + c next to it.""": generator.f1_me_ms_notes_perimeter_diagrams,
        """Generate diagrams for area calculation on a plain white background.
1.  **Square:** Draw a square with moderately thick black lines. Label one side 's'. Indicate the area calculation A = s² next to it. Optionally, show faint grid lines inside to represent square units.
2.  **Rectangle:** Draw a rectangle. Label the longer side 'l' (length) and the shorter side 'w' (width). Indicate the area calculation A = l × w next to it. Optionally, show faint grid lines inside.
3.  **Triangle:** Draw a triangle. Label the base 'b'. Draw a perpendicular line (dashed) from the opposite vertex to the base, label this line 'h' (height). Indicate the area calculation A = ½bh next to it.""": generator.f1_me_ms_notes_area_diagrams,
        """Generate a diagram of a square on a plain white background. Label the area inside the square as 'Area = 49 m²'. Label one side with 's = ?'. Use moderately thick black lines and clear, legible labels.""": generator.f1_me_ms_ques_square_area,
        """Generate a diagram of the Cartesian plane on a plain white background.
1. Draw a horizontal x-axis and a vertical y-axis intersecting at the origin (0,0).
2. Include arrowheads on the positive ends of both axes.
3. Mark and label integer values from -5 to +5 on both axes with clear tick marks. Label 'x' near the positive x-axis arrowhead and 'y' near the positive y-axis arrowhead.
4. Label the origin '(0,0)'.
5. Label the quadrants 'I', 'II', 'III', 'IV' in their respective positions (top-right, top-left, bottom-left, bottom-right).
Use moderately thick black lines and clear, legible labels.""": generator.f1_gr_fg_notes_cartesian_plane,
        """Generate a Cartesian plane with x and y axes ranging from -4 to +4. Use a scale where each unit step is clearly marked (like 1cm per unit). Label the axes 'x' and 'y'. Label the integer values on both axes. The origin (0,0) should be clearly marked. Use a plain white background, moderately thick black lines, and clear, legible labels.""": generator.f1_gr_fg_work_cartesian_plane_scale,
        """Generate a Cartesian plane (ranging roughly from -1 to 5 on both axes). Plot a single point at coordinates (3, 2). Mark the point with a clear dot or small cross and label it 'A(3, 2)'. Show dashed lines from the point A perpendicularly to the x-axis (hitting at x=3) and to the y-axis (hitting at y=2). Use a plain white background, moderately thick black lines, and clear, legible labels.""": generator.f1_gr_fg_work_plot_point_a,
        """Generate a Cartesian plane (ranging roughly from -5 to 5 on both axes). Plot four points: B at (-2, 4), C at (-3, -1), and D at (4, -3). Mark each point clearly and label it with its letter and coordinates (e.g., 'B(-2, 4)'). Use a plain white background, moderately thick black lines, and clear, legible labels.""": generator.f1_gr_fg_work_plot_points_bcd,
        """Generate a Cartesian plane (ranging roughly from -4 to 4 on both axes). Plot three points: P at (2, 3), Q at (-3, 1), R at (1, -2). Mark each point clearly and label it only with its letter (P, Q, R). Use a plain white background, moderately thick black lines, and clear, legible labels.""": generator.f1_gr_fg_work_identify_points_pqr,
        """Generate a blank Cartesian plane grid on a plain white background. Axes should range from -5 to +5, with clear tick marks and labels for each integer. Label the axes 'x' and 'y'. Ensure grid lines (faint) are present for easy plotting.""": generator.f1_gr_fg_ques_blank_grid_neg5_5,
        """Generate a Cartesian plane (ranging roughly from 0 to 5 on both axes). Plot points A(1, 2), B(4, 2), C(4, 4). Label these points. Leave space for point D to be identified. Use a plain white background, moderately thick black lines, and clear, legible labels.""": generator.f1_gr_fg_ques_rectangle_vertices,
        """Generate a sample distance-time graph on a plain white background.
1. Label the horizontal axis 'Time' (e.g., in hours) and the vertical axis 'Distance from Start' (e.g., in km).
2. Show a line segment starting at the origin (0,0) and going up to point A (e.g., 2 hours, 80 km), representing constant speed away.
3. Show a horizontal line segment from point A to point B (e.g., from 2 hours to 3 hours, still at 80 km), representing being stationary.
4. Show a line segment sloping down from point B back towards the time axis, reaching point C (e.g., 5 hours, 0 km), representing constant speed return.
5. Label key points like O(0,0), A, B, C with their time and distance values. Use moderately thick black lines and clear, legible labels.""": generator.f1_gr_tg_notes_distance_time,
        """Generate a distance-time graph. Horizontal axis 'Time (hours)' from 0 to 4. Vertical axis 'Distance (km)' from 0 to 120. The graph consists of a straight line from (0,0) to (2, 100), then a horizontal line from (2, 100) to (3, 100), then a straight line from (3, 100) to (4, 120). Label the axes and origin clearly. Use moderately thick black lines.""": generator.f1_gr_tg_work_car_journey_graph,
        """Reference the same graph described in Worked Example 1.""": generator.f1_gr_tg_work_car_journey_graph, # Key for reuse
        """Generate a blank graph grid with 'Time (hours)' on the x-axis (0 to 5) and 'Distance from home (km)' on the y-axis (0 to 40). Mark intervals clearly. Use a plain white background.""": generator.f1_gr_tg_ques_cyclist_journey_grid,
        """Generate simple diagrams illustrating types of lines on a plain white background.
1.  **Parallel Lines:** Draw two straight lines, roughly horizontal, clearly staying the same distance apart, with arrows on both ends. Use // marks on the lines to indicate parallelism.
2.  **Intersecting Lines:** Draw two straight lines crossing each other at a single point.
3.  **Perpendicular Lines:** Draw two straight lines intersecting at a right angle. Mark the right angle with a small square symbol.
4.  **Line Segment:** Draw a straight line with distinct endpoints (dots or small bars), labeled A and B.
5.  **Ray:** Draw a straight line starting at an endpoint (dot) labeled C, extending in one direction with an arrow, passing through another point D.""": generator.f1_ge_pl_notes_types_of_lines,
        """Generate diagrams illustrating types of angles on a plain white background.
1.  **Acute Angle:** Two rays from a vertex forming an angle clearly less than 90°.
2.  **Right Angle:** Two rays forming a 90° angle, marked with a square symbol at the vertex.
3.  **Obtuse Angle:** Two rays forming an angle clearly greater than 90° but less than 180°.
4.  **Straight Angle:** Two rays forming a straight line (180°).
5.  **Reflex Angle:** Show an angle greater than 180° by indicating the arc on the outside of a smaller angle.
6.  **Full Angle:** Show a ray that has rotated 360° back to its starting position, indicated by a circular arrow.""": generator.f1_ge_pl_notes_types_of_angles,
        """Generate a diagram showing angles on a straight line on a plain white background. Draw a straight horizontal line. Mark a point O on the line. Draw two rays starting from O, going upwards, dividing the angle above the line into three adjacent angles. Label the angles 'a', 'b', and 'c'. State the property: a + b + c = 180°.""": generator.f1_ge_pl_notes_angles_straight_line,
        """Generate a diagram showing angles around a point on a plain white background. Mark a point P. Draw four rays starting from P, going in different directions, dividing the space around P into four angles. Label the angles 'w', 'x', 'y', and 'z'. State the property: w + x + y + z = 360°.""": generator.f1_ge_pl_notes_angles_around_point,
        """Generate a diagram. Draw a straight horizontal line AOB with point O in the middle. Draw a ray OC starting from O upwards, such that angle AOC is labeled '130°' and angle BOC is labeled 'x'. Use a plain white background, clear labels, and moderately thick lines.""": generator.f1_ge_pl_work_angles_straight_line_ex,
        """Generate a diagram. Mark a point P. Draw three rays from P. Label the angle between the first and second ray as 100°, between the second and third as 140°, and between the third and first as 'y'. Use a plain white background, clear labels, and moderately thick lines.""": generator.f1_ge_pl_work_angles_around_point_ex,
        """Generate a diagram of an angle ABC, where B is the vertex. Make the angle approximately 65 degrees. Ray BA goes horizontally to the left, Ray BC goes upwards and slightly right. Label points A, B, C. Use moderately thick black lines on a plain white background.""": generator.f1_ge_pl_work_measure_angle_ex,
        """Generate a diagram. Draw a straight horizontal line. Mark a point O on it. Draw one ray OC upwards from O, dividing the angle above the line into two adjacent angles. Label the left angle '(2x)°' and the right angle '(x+30)°'. Use a plain white background, clear labels, and moderately thick lines.""": generator.f1_ge_pl_ques_angles_straight_line_ex,
        """Generate a diagram. Mark a point P. Draw three rays from P. Indicate one angle is 90° (use square symbol). Label the second angle 135°. Label the third angle 'z'. Use a plain white background, clear labels, and moderately thick lines.""": generator.f1_ge_pl_ques_angles_around_point_ex,
        """Generate a clear diagram of a circle on a plain white background with various parts labeled.
1. Draw a circle with center point labeled 'O'.
2. Draw a radius from O to the edge, label it 'Radius (r)'.
3. Draw a diameter passing through O, label it 'Diameter (d)'.
4. Draw a chord connecting two points on the circle (not through the center), label it 'Chord'.
5. Highlight a portion of the circumference and label it 'Arc'.
6. Draw a line outside the circle that touches it at one point, label it 'Tangent'.
7. Draw a line that cuts through the circle at two points, label it 'Secant'.
8. Shade a region bounded by two radii and an arc, label it 'Sector'.
9. Shade a region bounded by a chord and an arc, label it 'Segment'.
Use clear, legible labels with arrows pointing to the correct parts. Use moderately thick lines for the circle and its parts.""": generator.f1_ge_pc_notes_parts_of_circle,
        """Generate a diagram of a circle with center O. Draw a line segment AB that passes through the center O with endpoints A and B on the circle. Draw another line segment CD connecting two points on the circle but not passing through O. Label the points O, A, B, C, D clearly. Use a plain white background and moderately thick lines.""": generator.f1_ge_pc_work_circle_diameter,
        """Generate a diagram illustrating copying a line segment AB onto a ray starting at point P.
1. Show a line segment AB.
2. Show a separate ray starting at P, extending to the right.
3. Show compasses opened to the length of AB.
4. Show the compass point placed at P, drawing an arc that intersects the ray at point Q.
5. Indicate that segment PQ has the same length as segment AB. Use dashed lines for construction arcs.""": generator.f1_ge_cl_notes_copy_line_segment,
        """Generate an illustration showing a ruler placed on a white background. Draw a line segment starting at the 0 cm mark and ending exactly at the 7 cm mark. Label the endpoints A and B. Show the ruler markings clearly.""": generator.f1_ge_cl_work_draw_line_segment,
        """Generate a diagram showing an angle being constructed using a protractor.
1. Show a horizontal ray VX.
2. Show a protractor placed with its center at V and 0° aligned with VX.
3. Show a small dot marked at the 50° position on the protractor's arc.
4. Show the final angle Y'VX clearly marked as 50°. Use dashed lines for alignment marks if helpful.""": generator.f1_ge_cl_work_draw_angle_protractor,
        """Generate a diagram showing a given line segment MN on a plain white background.""": generator.f1_ge_cl_work_construct_equal_segment,
        """Generate a diagram illustrating the construction process described in the steps above. Show the initial ray from P, the compass arc drawn from P intersecting the ray at Q, and indicate that PQ = MN. Use dashed lines for the construction arc.""": generator.f1_ge_cl_work_construct_equal_segment_2,
        """Generate a Cartesian plane with appropriate axes (e.g., x from -1 to 7, y from -3 to 4).\n1. Plot points P(1,1), Q(4,1), R(1,3) and connect them with solid lines to form triangle PQR. Label the vertices.\n2. Plot points P'(3,-2), Q'(6,-2), R'(3,0) and connect them with dashed lines to form triangle P'Q'R'. Label the vertices.\n3. Optionally, draw faint arrows from P to P', Q to Q', and R to R' showing the translation vector <0xE2><0x8B><0x85>2/-3<0xE2><0x8B><0x86>.\nUse clear labels and distinct line styles for object and image.""": generator.f1_tr_tr_work_draw_triangles_pqr,
        """Generate a Cartesian plane with a simple shape (e.g., an L-shape or a small square) labeled 'A' in one position (e.g., vertices at (1,1), (2,1), (2,2), (1,2)). Show the identical shape labeled 'B' in a translated position (e.g., vertices at (4,3), (5,3), (5,4), (4,4)). Use clear labels and grid lines.""": generator.f1_tr_tr_ques_describe_translation,
        # F1 Inequalities (Item 16)
        """Generate four simple number lines on a plain white background, illustrating inequality representation.
1.  **x > 2:** Show a number line marked around 2. Place an open circle at 2 and draw an arrow pointing to the right.
2.  **x < 0:** Show a number line marked around 0. Place an open circle at 0 and draw an arrow pointing to the left.
3.  **x ≥ -1:** Show a number line marked around -1. Place a closed (solid) circle at -1 and draw an arrow pointing to the right.
4.  **x ≤ 3:** Show a number line marked around 3. Place a closed (solid) circle at 3 and draw an arrow pointing to the left.
Use clear markings, circles, arrows, and legible labels.""": generator.f1_al_in_notes_inequality_number_lines,
        """Generate a number line on a plain white background. Mark integers around -2 (e.g., -4, -3, -2, -1, 0, 1). Place an open circle at -2. Draw a thick arrow starting from the open circle and pointing to the right, extending towards the end of the number line. Use clear, legible labels.""": generator.f1_al_in_work_ineq_p_gt_neg2,
        """Generate a number line on a plain white background. Mark integers around 4 (e.g., 1, 2, 3, 4, 5, 6). Place an open circle at 4. Draw a thick arrow starting from the open circle and pointing to the left, extending towards the end of the number line. Use clear, legible labels.""": generator.f1_al_in_work_ineq_k_lt_4,

        # F2
        """Generate a simple diagram illustrating the concept of square root using prime factorization for √144 on a plain white background.
1. Show the prime factorization: 144 = 2 × 2 × 2 × 2 × 3 × 3.
2. Group the factors into pairs: (2 × 2) × (2 × 2) × (3 × 3).
3. Take one factor from each pair: 2 × 2 × 3.
4. Show the result: = 12. State clearly: √144 = 12.""": generator.f2_rn_nc_notes_sqrt_prime_factorization,
        """Generate a simple diagram illustrating the concept of cube root using prime factorization for ³√216 on a plain white background.
1. Show the prime factorization: 216 = 2 × 2 × 2 × 3 × 3 × 3.
2. Group the factors into triplets: (2 × 2 × 2) × (3 × 3 × 3).
3. Take one factor from each triplet: 2 × 3.
4. Show the result: = 6. State clearly: ³√216 = 6.""": generator.f2_rn_nc_notes_cuberoot_prime_factorization,
        """Generate two simple graphs side-by-side on a plain white background.
1.  **Left Graph (Direct Proportion):** Label axes 'x' and 'y'. Show a straight line starting from the origin (0,0) and going upwards to the right. Label it 'y = kx'.
2.  **Right Graph (Inverse Proportion):** Label axes 'x' and 'y'. Show a curve (hyperbola) in the first quadrant, starting high on the left and curving downwards towards the right, approaching but not touching the axes. Label it 'y = k/x'.
Use clear labels and moderately thick lines.""": generator.f2_rn_rr_notes_proportion_graphs,
        """Generate a diagram illustrating the conversion between m² and cm² on a plain white background.
1. Draw a large square labeled '1 m' on each side.
2. Indicate its area is '1 m²'.
3. Inside the large square, show faint grid lines dividing it into smaller squares.
4. Along the top edge, label subdivisions indicating '100 cm'.
5. Along the left edge, label subdivisions indicating '100 cm'.
6. Add text: '1 m² = 100 cm × 100 cm = 10,000 cm²'.
Use clear, legible labels and moderately thick lines for the main square.""": generator.f2_me_me_notes_area_conversion,
        """Generate a diagram illustrating the conversion between m³ and cm³ on a plain white background.
1. Draw a large cube labeled '1 m' on each edge (length, width, height).
2. Indicate its volume is '1 m³'.
3. Suggest perspective lines showing smaller cubes inside.
4. Add text: '1 m³ = 100 cm × 100 cm × 100 cm = 1,000,000 cm³'.
Use clear, legible labels and moderately thick lines for the main cube.""": generator.f2_me_me_notes_volume_conversion,
        """Generate diagrams for volume calculation on a plain white background.
1.  **Cuboid:** Draw a cuboid in perspective. Label the edges 'l' (length), 'w' (width), and 'h' (height). Indicate the volume calculation V = lwh next to it.
2.  **Cube:** Draw a cube in perspective. Label one edge 's'. Indicate the volume calculation V = s³ next to it.""": generator.f2_me_ms_notes_volume_cuboid_cube,
        """Generate a diagram showing a rectangle drawn accurately on a plain white background. Label the longer side '6 cm' and the shorter side '4 cm'. Below the rectangle, include the text 'Scale 1 : 500'. Use moderately thick lines.""": generator.f2_rn_sc_work_scaled_rectangle, # Item 28
        """Generate a Venn diagram for two sets A and B within a universal set U on a plain white background.
1. Draw a rectangle labeled U.
2. Draw two overlapping circles inside, labeled A and B.
3. Clearly label the four distinct regions:
    - The overlapping part: 'A ∩ B'
    - The part of A not overlapping: 'A only' or 'A ∩ B''
    - The part of B not overlapping: 'B only' or 'B ∩ A''
    - The area inside U but outside both circles: '(A ∪ B)'' or 'Neither A nor B'
Use clear, legible labels and moderately thick lines.""": generator.f2_se_notes_venn_2_sets, # Item 29 Note
        """Generate a Venn diagram for two sets A and B within a universal set U on a plain white background.
1. Draw a rectangle labeled U.
2. Draw two overlapping circles inside, labeled A and B.
3. Place the number '2' in the overlapping region (A ∩ B).
4. Place the number '2' in the part of A not overlapping (A only).
5. Place the number '2' in the part of B not overlapping (B only).
6. Place the number '2' inside U but outside both circles (Neither).
Use clear, legible numbers and labels.""": generator.f2_se_work_venn_2_sets_numbers, # Item 29 WE 1
        """Generate a Venn diagram for two sets P and R within a universal set U (Total=30) on a plain white background.
1. Draw rectangle U.
2. Draw overlapping circles P and R.
3. Place '8' in the overlap (P ∩ R).
4. Place '7' in the P-only region.
5. Place '12' in the R-only region.
6. Place '3' inside U but outside both circles.
Use clear, legible numbers and labels.""": generator.f2_se_work_venn_2_sets_music, # Item 29 WE 2
        """Generate a simple diagram of a rectangle representing a floor on a plain white background. Label the sides '4 m' and '5 m'. Inside the rectangle, write 'Area = ? cm²'.""": generator.f2_me_me_ques_floor_area, # Item 31 Ques 1
        """Generate a Cartesian plane grid on a plain white background. The x-axis should range from -2 to +2, with markings clearly 2 cm apart for each unit. The y-axis should range from -3 to +5, with markings clearly 1 cm apart for each unit. Label the axes 'x' and 'y' and mark the integer values on both axes. Label the origin (0,0).""": generator.f2_gr_fg_work_cartesian_plane_diff_scales,
        """Generate a Cartesian plane (e.g., x from -2 to 4, y from -4 to 2). Plot the points (-1, -3), (0, -2), (1, -1), (2, 0), (3, 1). Mark each point with a clear dot or cross. Label the axes and origin.""": generator.f2_gr_fg_work_plot_points_y_eq_x_min_2,
        """Generate the same Cartesian plane and plotted points as in Example 3's image description. Additionally, draw a solid straight line passing through all five plotted points. Extend the line slightly beyond (-1,-3) and (3,1). Add the label 'y = x - 2' near the line. Use clear labels and moderately thick lines.""": generator.f2_gr_fg_work_draw_graph_y_eq_x_min_2,
        """Generate a Cartesian plane (e.g., x from -2 to 3, y from 0 to 6). Plot the points (-1, 5), (0, 4), (1, 3), (2, 2). Draw a solid straight line passing through these points. Label the line 'y = 4 - x'. Label axes and origin.""": generator.f2_gr_fg_work_draw_graph_y_eq_4_min_x,
        """Generate a distance-time graph. Horizontal axis 'Time (hours)' from 0 to 5. Vertical axis 'Distance (km)' from 0 to 80. Graph consists of: line from (0,0) to (2,60), horizontal line from (2,60) to (3,60), line from (3,60) to (5,0). Label axes and origin clearly.""": generator.f2_gr_tg_work_journey_graph_1,
        """Generate a distance-time graph. Horizontal axis 'Time (hours)' from 0 to 3 (marked 0, 1, 1.5, 2, 3). Vertical axis 'Distance from home (km)' from 0 to 5 (marked 0, 1, 2, 3, 4, 5). Plot points (0,0), (1,4), (1.5,4), (3,0). Connect (0,0) to (1,4) with a line. Connect (1,4) to (1.5,4) with a horizontal line. Connect (1.5,4) to (3,0) with a line. Label axes.""": generator.f2_gr_tg_work_draw_walk_rest_return,
        """Generate a blank graph grid with 'Time (hours)' on the x-axis (0 to 4) and 'Distance (km)' on the y-axis (0 to 150). Mark intervals clearly. Use a plain white background.""": generator.f2_gr_tg_ques_car_journey_grid,
        """Generate a simple graph illustrating direct variation on a plain white background.
1. Draw x and y axes intersecting at the origin (0,0).
2. Draw a straight line starting from the origin and going upwards and to the right into the first quadrant.
3. Label the axes 'x' and 'y'.
4. Add the label 'y = kx' near the line.
Use clear labels and moderately thick lines.""": generator.f2_va_notes_direct_variation_graph,
        """Generate a diagram on a plain white background showing two parallel horizontal lines. Label the top line 'L1' and the bottom line 'L2'. Place matching arrow symbols (e.g., single arrowheads '>'), pointing in the same direction, somewhere along the middle of each line to indicate they are parallel. Use moderately thick black lines.""": generator.f2_ge_pl_notes_parallel_lines,
        """Generate a diagram on a plain white background showing two parallel horizontal lines (L1 and L2, marked with arrows). Draw a third line (T, the transversal) intersecting both parallel lines, slanted upwards from left to right. Label the lines L1, L2, and T. Number the eight angles formed at the intersections from 1 to 8 in a standard manner (e.g., top-left=1, top-right=2, bottom-left=3, bottom-right=4 at the intersection with L1; then top-left=5, top-right=6, bottom-left=7, bottom-right=8 at the intersection with L2).""": generator.f2_ge_pl_notes_transversal_line,
        """Generate a diagram. Draw two parallel horizontal lines AB (top) and CD (bottom), marked with parallel arrows. Draw a transversal EF intersecting AB at G and CD at H, slanted upwards left to right. Angle EGA (top-left angle at intersection G) is labeled 60°. Angle GHC (bottom-left angle at intersection H) is labeled 'x'. Use clear labels and moderately thick lines.""": generator.f2_ge_pl_work_corresponding_angles,
        """Generate a diagram. Draw two parallel horizontal lines AB (top) and CD (bottom), marked with parallel arrows. Draw a transversal EF intersecting AB at G and CD at H, slanted upwards left to right. Angle AGH (bottom-left angle at intersection G) is labeled 110°. Angle GHD (top-right angle at intersection H) is labeled 'y'. Use clear labels and moderately thick lines.""": generator.f2_ge_pl_work_alternate_interior_angles,
        """Generate a diagram. Draw two parallel horizontal lines AB (top) and CD (bottom), marked with parallel arrows. Draw a transversal EF intersecting AB at G and CD at H, slanted upwards left to right. Angle BGH (bottom-right angle at intersection G) is labeled 75°. Angle GHC (bottom-left angle at intersection H) is labeled 'z'. Use clear labels and moderately thick lines.""": generator.f2_ge_pl_work_co_interior_angles,
        """Reference the diagram from Worked Example 3. Angle DHF is the bottom-right angle at intersection H.""": generator.f2_ge_pl_work_find_dhf,
        """Generate a diagram. Draw two parallel horizontal lines. Draw a transversal intersecting them. Mark an interior angle on the left of the transversal between the parallel lines as '(2x + 10)°'. Mark the alternate interior angle (on the right of the transversal, between the parallel lines) as '(3x - 20)°'. Use clear labels.""": generator.f2_ge_pl_work_solve_for_x_parallel,
        """Generate a diagram. Draw two parallel vertical lines m and n. Draw a transversal intersecting them, slanted upwards left to right. Mark the top-right angle formed by the transversal and line m as 70°. Mark the top-left angle formed with line n as 'a', the bottom-left angle with line n as 'b', and the bottom-right angle with line n as 'c'. Use clear labels.""": generator.f2_ge_pl_ques_parallel_lines_angles_abc,
        """Generate a diagram. Draw a horizontal line segment AB. Below it, draw another horizontal line segment DE, parallel to AB. Draw a point C between the lines, slightly to the right. Draw line segment BC and line segment DC, forming an angle ∠BCD labeled 'x'. Angle ABC (formed by extending AB leftwards) is given as 130°. Angle CDE (formed by extending DE rightwards) is given as 140°. Hint: Draw a line through C parallel to AB and DE.""": generator.f2_ge_pl_ques_parallel_lines_find_x,
        """Generate a compass rose diagram on a plain white background.
1. Show North (N) at the top, South (S) at the bottom, East (E) to the right, West (W) to the left.
2. Show the intercardinal points NE, SE, SW, NW in their correct positions between the cardinal points.
3. Optionally, add finer divisions like NNE, ENE etc., but label the 8 main points clearly.
Use clear, bold letters for labels.""": generator.f2_ge_be_notes_compass_rose,
        """Generate three simple diagrams illustrating compass bearings on a plain white background.
1.  **N 30° E:** Draw North/South and East/West axes. Draw a line from the center into the NE quadrant. Mark the angle between the North line and the direction line as 30°. Label the direction line 'N 30° E'.
2.  **S 45° W:** Draw axes. Draw a line from the center into the SW quadrant. Mark the angle between the South line and the direction line as 45°. Label the line 'S 45° W'.
3.  **N 70° W:** Draw axes. Draw a line from the center into the NW quadrant. Mark the angle between the North line and the direction line as 70°. Label the line 'N 70° W'.""": generator.f2_ge_be_notes_compass_bearing_examples,
        """Generate three simple diagrams illustrating three-figure bearings on a plain white background.
1.  **Bearing 060°:** Draw a North line pointing up. Draw a line segment from the starting point into the NE quadrant. Draw a clockwise arc from the North line to the direction line, labeling the angle 060°.
2.  **Bearing 135°:** Draw a North line. Draw a line segment into the SE quadrant. Draw a clockwise arc from North, labeling the angle 135°.
3.  **Bearing 310°:** Draw a North line. Draw a line segment into the NW quadrant. Draw a large clockwise arc from North, labeling the angle 310°.""": generator.f2_ge_be_notes_three_figure_bearing,
        """Generate a diagram showing two points S (Ship) and L (Lighthouse). At S, draw a North line pointing up. Draw the line segment SL such that the clockwise angle from North at S to SL is 70°. At L, draw a North line pointing up. Show the clockwise angle from North at L to LS. This angle should be 250°. Indicate the 70° and 250° angles clearly.""": generator.f2_ge_be_work_back_bearing,
        """Generate a diagram. Draw a point A. Draw a North line vertically upwards from A. Draw a line segment AB such that the clockwise angle from North to AB is 120°. Draw a line segment AC such that the clockwise angle from North to AC is 210°. Mark the angle BAC between AB and AC. Label the bearings 120° and 210° relative to the North line.""": generator.f2_ge_be_work_angle_bac_bearing,
        """Generate diagrams illustrating triangle types on a plain white background.
1.  **Equilateral Triangle:** Draw a triangle with markings on all three sides indicating equality. Mark all three angles as 60° or with identical arc symbols.
2.  **Isosceles Triangle:** Draw a triangle with markings on two sides indicating equality. Mark the two base angles (opposite the equal sides) with identical arc symbols.
3.  **Scalene Triangle:** Draw a triangle with sides of clearly different lengths and angles of clearly different sizes. Use no equality markings.
4.  **Right-angled Triangle:** Draw a triangle with one angle marked as a right angle (90°) using a square symbol.""": generator.f2_ge_pc_notes_triangle_types,
        """Generate simple diagrams illustrating common quadrilateral types on a plain white background.
1.  **Square:** Equal sides marked, right angles marked.
2.  **Rectangle:** Opposite sides marked equal, right angles marked.
3.  **Parallelogram:** Opposite sides marked equal and parallel (use arrows).
4.  **Rhombus:** All sides marked equal, opposite sides parallel.
5.  **Trapezium:** One pair of opposite sides marked parallel.
6.  **Kite:** Adjacent pairs of sides marked equal. Indicate perpendicular diagonals.""": generator.f2_ge_pc_notes_quadrilateral_types,
        """Generate diagrams illustrating similar triangles on a plain white background.
1. Draw a triangle ABC.
2. Draw a larger triangle PQR that has the same shape as ABC (corresponding angles are equal, e.g., ∠A=∠P, ∠B=∠Q, ∠C=∠R). Make the sides of PQR clearly proportionally larger than ABC (e.g., twice as long).
3. Use arc markings to show corresponding equal angles.
4. Add the label 'ΔABC ~ ΔPQR'.""": generator.f2_ge_sc_notes_similar_triangles,
        """Generate diagrams illustrating congruent triangles on a plain white background.
1. Draw a triangle DEF.
2. Draw another triangle STU that is identical in size and shape to DEF, perhaps in a different orientation (e.g., rotated or reflected).
3. Use markings on corresponding sides (e.g., single dash, double dash, triple dash) to show equality.
4. Use arc markings on corresponding angles to show equality.
5. Add the label 'ΔDEF ≅ ΔSTU'.""": generator.f2_ge_sc_notes_congruent_triangles,
        """Generate diagrams of two rectangles on a plain white background.
1. Rectangle A: Dimensions 4 cm by 6 cm.
2. Rectangle B: Dimensions 6 cm by 9 cm.
Label the dimensions clearly.""": generator.f2_ge_sc_work_rectangles_similar,
        """Generate diagrams of two triangles ABC and PQR on a plain white background.
- Triangle ABC: AB = 5 cm, BC = 7 cm, AC = 6 cm.
- Triangle PQR: PQ = 5 cm, QR = 7 cm, PR = 6 cm.
Label the sides with their lengths.""": generator.f2_ge_sc_work_triangles_congruent_sss,
        """Generate diagrams of two triangles DEF and XYZ on a plain white background.
- Triangle DEF: DE = 8 cm, ∠E = 40°, EF = 10 cm.
- Triangle XYZ: XY = 8 cm, ∠Y = 40°, YZ = 10 cm.
Label the sides and angles.""": generator.f2_ge_sc_work_triangles_congruent_sas,
        """Generate sketch diagrams of two similar triangles LMN and RST. Label sides LM=4, MN=6, LN=7 on ΔLMN. Label side RS=6 on ΔRST. Indicate similarity ΔLMN ~ ΔRST.""": generator.f2_ge_sc_work_similar_triangles_lmn_rst,
        """Generate a diagram showing a larger triangle PST. Point Q lies on side PS and point R lies on side PT, such that line segment QR is parallel to line segment ST. Label PQ=5, QS=3, ST=12. Label QR='x'. Mark angles PQR and PST as equal (corresponding), and PRQ and PTS as equal (corresponding).""": generator.f2_ge_sc_ques_similar_triangles_pqr_pst,
        """Generate a diagram illustrating the construction of a perpendicular bisector of line segment AB.
1. Show line segment AB.
2. Show an arc drawn from center A, above and below AB (radius > 1/2 AB).
3. Show an arc drawn from center B with the same radius, intersecting the first arcs at points P (above) and Q (below).
4. Draw a straight line passing through P and Q, intersecting AB at its midpoint M.
5. Mark the line PQ as perpendicular to AB using a right angle symbol at M.
Use dashed lines for construction arcs.""": generator.f2_ge_cl_notes_perpendicular_bisector,
        """Generate a diagram illustrating the construction of an angle bisector for angle XYZ.
1. Show angle XYZ (vertex Y).
2. Show an arc drawn from center Y, intersecting YX at P and YZ at Q.
3. Show an arc drawn from center P inside the angle.
4. Show an arc drawn from center Q with the same radius, intersecting the previous arc at R.
5. Draw the ray YR. Mark angle XYR and angle ZYR with identical arc symbols to show they are equal.
Use dashed lines for construction arcs.""": generator.f2_ge_cl_notes_angle_bisector,
        """Generate a diagram showing a horizontal line segment PQ labeled '8 cm'. Show construction arcs drawn from P and Q (radius > 4cm) intersecting above at R and below at S. Draw a vertical line passing through R and S, intersecting PQ at its midpoint M. Mark the right angle at M. Use dashed lines for arcs.""": generator.f2_ge_cl_work_construct_perp_bisector_8cm,
        """Generate a diagram illustrating the construction of a 60° angle.
1. Show a horizontal ray OA.
2. Show an arc drawn from center O, intersecting OA at P.
3. Show a second arc drawn from center P (same radius), intersecting the first arc at Q.
4. Draw the ray OQ. Mark the angle AOQ as 60°.
Use dashed lines for arcs.""": generator.f2_ge_cl_work_construct_60_degrees,
        """Generate a diagram showing an angle ABC (vertex B) that looks roughly 80°.""": generator.f2_ge_cl_work_bisect_80_degree_angle_initial,
        """Generate a diagram showing the angle ABC from the previous description, along with the construction arcs from B intersecting arms at P and Q, and arcs from P and Q intersecting inside at R. Draw the bisecting ray BR. Mark angles ABR and CBR as equal.""": generator.f2_ge_cl_work_bisect_80_degree_angle_construction,
        """Generate a diagram illustrating the construction of a 90° angle at point P on line XY.
1. Show horizontal line XY with point P marked on it.
2. Show small arcs centered at P intersecting XY at A (left) and B (right).
3. Show larger arcs centered at A and B intersecting above P at Q.
4. Draw the vertical line segment PQ. Mark the angle QPX (or QPY) with a right angle symbol.
Use dashed lines for arcs.""": generator.f2_ge_cl_work_construct_90_degrees,
        """Generate a diagram showing a 60° angle AOQ already constructed. Then show the construction marks for bisecting this angle: arc from O cutting OA at P and OQ at S; intersecting arcs from P and S inside the angle at R. Draw the bisecting ray OR. Mark angle AOR as 30°.""": generator.f2_ge_cl_work_construct_30_degrees,
        """Generate diagrams showing lines of symmetry for various shapes on a plain white background.
1.  **Isosceles Triangle:** Draw an isosceles triangle. Draw its single line of symmetry vertically (dashed line).
2.  **Equilateral Triangle:** Draw an equilateral triangle. Draw its three lines of symmetry (dashed lines).
3.  **Square:** Draw a square. Draw its four lines of symmetry (two vertical/horizontal, two diagonal) (dashed lines).
4.  **Rectangle:** Draw a rectangle. Draw its two lines of symmetry (vertical and horizontal) (dashed lines).
5.  **Rhombus:** Draw a rhombus. Draw its two diagonal lines of symmetry (dashed lines).
6.  **Regular Hexagon:** Draw a regular hexagon. Draw its six lines of symmetry (dashed lines).""": generator.f2_ge_sy_notes_lines_of_symmetry,
        """Generate a diagram of a square on a plain white background. Draw its four lines of symmetry as dashed lines: one vertical, one horizontal (both through the center), and the two diagonals. Mark the lines clearly.""": generator.f2_ge_sy_work_square_symmetry,
        """Generate a diagram of a non-square rectangle on a plain white background. Draw its two lines of symmetry as dashed lines: one vertical and one horizontal, both passing through the center.""": generator.f2_ge_sy_work_rectangle_symmetry,
        """Generate the capital letter 'H' in a clear, block font on a plain white background. Draw a horizontal dashed line through the middle crossbar and a vertical dashed line through the middle of the uprights, indicating the lines of symmetry.""": generator.f2_ge_sy_work_letter_h_symmetry,
        """Generate a diagram of an isosceles triangle (e.g., base horizontal at the bottom, two equal sides meeting at the top vertex). Draw a single vertical dashed line from the top vertex to the midpoint of the base.""": generator.f2_ge_sy_work_isosceles_symmetry,
        """Generate a diagram of a regular pentagon on a plain white background. Draw its five lines of symmetry as dashed lines. Each line should go from a vertex to the midpoint of the opposite side.""": generator.f2_ge_sy_work_pentagon_symmetry,
        """Generate a simple example bar chart on a plain white background.
1. Horizontal axis labeled 'Favorite Fruit' with categories 'Apple', 'Banana', 'Orange'.
2. Vertical axis labeled 'Number of Students' with scale from 0 to 10.
3. Draw rectangular bars of equal width above each fruit category:
    - Apple: Height corresponding to 5.
    - Banana: Height corresponding to 8.
    - Orange: Height corresponding to 3.
4. Ensure gaps between the bars.
5. Add a title: 'Favorite Fruits of Form 2 Students'.
Use clear labels and moderately thick lines.""": generator.f2_st_dc_notes_bar_chart_example,
        """Generate a simple example pie chart on a plain white background.
1. Draw a circle.
2. Divide the circle into three sectors with roughly calculated angles representing hypothetical data (e.g., Category A: 180°, Category B: 120°, Category C: 60°).
3. Label each sector clearly: 'Category A (50%)', 'Category B (33.3%)', 'Category C (16.7%)'. Alternatively label with frequencies if context provides.
4. Add a title: 'Distribution of Item Types'.
Use clear labels and distinct sector boundaries.""": generator.f2_st_dc_notes_pie_chart_example,
        """Generate a bar chart based on the frequency table in Example 1.
1. Title: 'Number of Pets Owned by Students'.
2. Horizontal axis labeled 'Number of Pets' with points 0, 1, 2, 3.
3. Vertical axis labeled 'Frequency' with scale 0, 1, 2, 3, 4.
4. Draw bars: Above 0, height 3. Above 1, height 4. Above 2, height 2. Above 3, height 1.
5. Bars should be of equal width with gaps between them.
Use clear labels and moderately thick lines.""": generator.f2_st_dc_work_bar_chart_pets,
        """Generate a pie chart based on the angles calculated in Example 3.
1. Title: 'Number of Pets Owned by Students'.
2. Draw a circle divided into four sectors.
3. Sector 1: Angle 108°, labeled '0 Pets'.
4. Sector 2: Angle 144°, labeled '1 Pet'.
5. Sector 3: Angle 72°, labeled '2 Pets'.
6. Sector 4: Angle 36°, labeled '3 Pets'.
Use clear labels and distinct sector boundaries.""": generator.f2_st_dc_work_pie_chart_pets,
        """Generate diagrams illustrating vector notation on a plain white background.
1.  **Geometric Vector:** Draw an arrow starting at point A and ending at point B. Label it '<0xE2><0x86><0x92>AB' near the arrow. Label points A and B.
2.  **Column Vector Representation:** Draw a simple 2D grid (or just axes). Show a vector starting at (1,1) and ending at (4,3). Draw the arrow. Label the vector '<0xE2><0x8B><0x85>3/2<0xE2><0x8B><0x86>' representing the movement (4-1=3 horizontally, 3-1=2 vertically).""": generator.f2_ve_de_notes_vector_notation,
        """Generate a diagram showing two equal vectors on a plain white background.
1. Draw a vector <0xE2><0x86><0x92>AB as an arrow.
2. Draw another vector <0xE2><0x86><0x92>CD somewhere else on the page, parallel to <0xE2><0x86><0x92>AB, having the same length, and pointing in the same direction. Label the vectors.
3. Add text: '<0xE2><0x86><0x92>AB = <0xE2><0x86><0x92>CD'.""": generator.f2_ve_ty_notes_equal_vectors,
        """Generate a diagram showing a vector and its negative on a plain white background.
1. Draw a vector <0xE2><0x86><0x92>AB as an arrow from A to B.
2. Draw the vector <0xE2><0x86><0x92>BA as an arrow from B to A, parallel to <0xE2><0x86><0x92>AB and of the same length.
3. Label the first vector '<0xE2><0x86><0x92>AB = **a**'.
4. Label the second vector '<0xE2><0x86><0x92>BA = -**a**'.""": generator.f2_ve_ty_notes_negative_vectors,
        """Generate a Cartesian plane (e.g., x from -2 to 3, y from 0 to 4). Plot point (2, 2) and point (-1, 3). Draw a directed arrow starting at (2, 2) and ending at (-1, 3). Label the points if desired. Label axes and origin.""": generator.f2_ve_ty_work_draw_vector_neg3_1,
        """Generate a diagram on a grid background. Draw four vectors represented by arrows:
- <0xE2><0x86><0x92>AB: Starts at (1,1), ends at (3,2). Vector is <2/1>.
- <0xE2><0x86><0x92>CD: Starts at (4,1), ends at (6,2). Vector is <2/1>.
- <0xE2><0x86><0x92>EF: Starts at (1,3), ends at (3,4). Vector is <2/1>.
- <0xE2><0x86><0x92>GH: Starts at (4,4), ends at (6,3). Vector is <2/-1>.
Label the endpoints A, B, C, D, E, F, G, H.""": generator.f2_ve_ty_work_identify_equal_vectors,
        """Generate diagrams illustrating vector addition on a plain white background.
1.  **Triangle Law:** Draw vector **a** (arrow). From the head of **a**, draw vector **b** (arrow). Draw the resultant vector **a** + **b** as an arrow from the tail of **a** to the head of **b**.
2.  **Parallelogram Law:** Draw vectors **a** and **b** starting from the same point O. Complete the parallelogram using dashed lines parallel to **a** and **b**. Draw the diagonal starting from O, label it **a** + **b**.""": generator.f2_ve_op_notes_vector_addition,
        """Generate a diagram illustrating vector subtraction **a** - **b** on a plain white background.
1. Draw vectors **a** and **b** starting from the same point O.
2. Draw the vector connecting the head of **b** to the head of **a**. Label this vector '**a** - **b**'.
3. Alternatively, show vector **a** and vector -**b** (same magnitude as **b**, opposite direction). Then show the addition **a** + (-**b**) using the triangle law.""": generator.f2_ve_op_notes_vector_subtraction,
        """Generate a Cartesian plane (e.g., x from -1 to 6, y from -2 to 5).\n1. Plot L(0,1), M(3,1), N(0,4). Connect with solid lines to form triangle LMN. Label vertices.\n2. Plot L'(2,0), M'(5,0), N'(2,3). Connect with dashed lines to form triangle L'M'N'. Label vertices.\n3. Optionally, draw arrows showing the translation <0xE2><0x8B><0x85>2/-1<0xE2><0x8B><0x86> from L to L', M to M', N to N'.\nUse clear labels and distinct line styles.""": generator.f2_tr_tr_work_draw_triangles_lmn,
        """Generate diagrams illustrating reflections in standard lines on a plain white background.
1.  **Reflection in x-axis:** Draw axes. Plot point P(2,3). Draw the image P'(2,-3). Show dashed perpendicular line from P and P' to the x-axis. Label P, P', and the x-axis as mirror line.
2.  **Reflection in y-axis:** Draw axes. Plot point Q(-2,1). Draw the image Q'(2,1). Show dashed perpendicular line from Q and Q' to the y-axis. Label Q, Q', and the y-axis as mirror line.
3.  **Reflection in y=x:** Draw axes and the dashed line y=x. Plot point R(1,4). Draw the image R'(4,1). Show dashed perpendicular line connecting R and R', bisected by y=x. Label R, R', and y=x.
Use clear labels and lines.""": generator.f2_tr_re_notes_reflections_std_lines,
        """Generate a Cartesian plane (e.g., x from 0 to 5, y from -4 to 4).\n1. Plot A(1,1), B(4,1), C(1,3). Connect with solid lines to form triangle ABC. Label vertices.\n2. Plot A'(1,-1), B'(4,-1), C'(1,-3). Connect with dashed lines to form triangle A'B'C'. Label vertices.\n3. Highlight the x-axis as the mirror line.\nUse clear labels and distinct line styles.""": generator.f2_tr_re_work_reflect_triangle_x_axis,
        # F2 Inequalities (Item 38) - Cartesian Plane ones
        """Generate a Cartesian plane illustrating the inequality y > x + 1.""": generator.f2_al_in_notes_ineq_cartesian_plane,
        """Generate a number line on a plain white background. Mark integers around -1""": generator.f2_al_in_work_ineq_x_le_neg1_numline,
        """Generate a Cartesian plane (e.g., x and y from -3 to 3). Draw the line x = 2 as a solid vertical line. Shade the region""": generator.f2_al_in_work_ineq_x_ge_2_plane,
        """Generate a Cartesian plane (e.g., x and y from -2 to 5). Draw the line y = 3 as a **dashed horizontal line**. Shade the region""": generator.f2_al_in_work_ineq_y_lt_3_plane,

        # F3
        """Generate a Venn diagram for three sets A, B, and C within a universal set U on a plain white background.
1. Draw a large rectangle labeled U.
2. Inside, draw three overlapping circles labeled A, B, and C, positioned such that there is a central region common to all three, regions common to each pair, and regions belonging to only one set.
3. Clearly label or number the 8 distinct regions:
    - A only
    - B only
    - C only
    - A and B only
    - A and C only
    - B and C only
    - A and B and C (central region)
    - Outside A, B, C but inside U.
Use clear, legible labels and moderately thick lines.""": generator.f3_se_notes_venn_3_sets,
        """Generate a Venn diagram for three sets A, B, C within U.
1. Draw rectangle U and three overlapping circles A, B, C.
2. Place '0' (or leave blank) in the central region (A ∩ B ∩ C).
3. Place '2' in the region for A ∩ B only.
4. Place '2' in the region for A ∩ C only.
5. Place '0' (or leave blank) in the region for B ∩ C only.
6. Place '1' in the region for A only.
7. Place '2' in the region for B only.
8. Place '2' in the region for C only.
9. Place '1' (representing element 10) outside the circles but inside U.
Use clear, legible numbers and labels.""": generator.f3_se_work_venn_3_sets_example,
        """Generate a Venn diagram for three sets P, C, B within U (Total=40).
1. Draw rectangle U and three overlapping circles P, C, B.
2. Place '5' in the central region (P∩C∩B).
3. Place '5' in P∩C only.
4. Place '7' in C∩B only.
5. Place '4' in P∩B only.
6. Place '6' in P only.
7. Place '8' in C only.
8. Place '2' in B only.
9. Place '3' outside the circles but inside U.
Use clear, legible numbers and labels.""": generator.f3_se_work_venn_3_sets_students,
        """Generate a simple graph comparing simple vs compound interest over time on a plain white background.
1. Draw axes: Time (years) on x-axis, Amount ($) on y-axis.
2. Show a starting Principal P at Time=0.
3. Draw a straight line sloping upwards representing Simple Interest growth (A = P(1 + RT/100)).
4. Draw a curve starting at P that curves upwards more steeply than the straight line, representing Compound Interest growth (A = P(1 + R/100)^T).
5. Label the lines 'Simple Interest' and 'Compound Interest'. Show that compound interest grows faster over time.""": generator.f3_fm_notes_simple_vs_compound,
        """Generate a diagram of a combined shape on a plain white background.
1. Draw a shape consisting of a rectangle (e.g., 6cm wide, 4cm high) with a semi-circle attached to one of the shorter sides (diameter=4cm).
2. Indicate the dimensions (6cm, 4cm).
3. Show dashed lines suggesting how to calculate perimeter (sum of 3 straight sides + semi-circular arc) and area (Area of rectangle + Area of semi-circle).""": generator.f3_me_ms_notes_combined_shape,
        """Generate a diagram of a cylinder on a plain white background.
1. Draw a cylinder standing upright.
2. Indicate the radius 'r' of the circular base.
3. Indicate the height 'h' of the cylinder (perpendicular distance between bases).
4. Add the formula 'V = πr²h' next to the cylinder.""": generator.f3_me_ms_notes_cylinder_volume,
        """Generate a diagram. Draw a square with side 10m. On the top side of the square, draw a triangle pointing upwards, using the top side of the square as its base. Indicate the square side length 10m and the perpendicular height of the triangle from its base as 4m.""": generator.f3_me_ms_work_combined_shape_area,
        """Generate a diagram. Draw a rectangle 8cm wide and 5cm high. On the top 8cm side, draw a semi-circular cut-out going downwards into the rectangle, with the 8cm side as its diameter. Indicate dimensions 8cm and 5cm. Shade the remaining area.""": generator.f3_me_ms_work_combined_shape_perimeter,
        """Generate a Cartesian plane. Plot the points (0, -4) and (2, 0). Draw a straight solid line passing through these two points. Label the points and label the line 'y = 2x - 4'. Label the axes.""": generator.f3_gr_fg_notes_intercept_graph,
        """Generate a Cartesian plane. Plot points derived from a simple quadratic like y = x² for x = -2, -1, 0, 1, 2 (Points: (-2,4), (-1,1), (0,0), (1,1), (2,4)). Draw a smooth U-shaped curve passing through these points. Label the curve 'y = x²'. Label the axes.""": generator.f3_gr_fg_notes_quadratic_graph,
        """Generate a Cartesian plane (x from -1 to 4, y from -1 to 7). Plot points (0, 6) and (3, 0). Draw a solid straight line passing through these points. Label the points and the line 'y = 6 - 2x'. Label the axes and origin.""": generator.f3_gr_fg_work_graph_y_eq_6_min_2x,
        """Generate a Cartesian plane (x from -3 to 5, y from -5 to 6). Plot the points (-2, 5), (-1, 0), (0, -3), (1, -4), (2, -3), (3, 0), (4, 5). Draw a smooth U-shaped parabola passing through these points. Label the curve 'y = x² - 2x - 3'. Label the axes and origin.""": generator.f3_gr_fg_work_graph_y_eq_x2_min_2x_min_3, # Graph a
        """Generate a sketch of a Cartesian plane. Mark the x-axis intercepts at x=1 and x=3. Mark the y-axis intercept at y=3. Draw a smooth U-shaped parabola passing through these three points, with its minimum point somewhere between x=1 and x=3 and below the x-axis (specifically at x=2, y=-1). Label the intercepts (1,0), (3,0), (0,3). Label axes.""": generator.f3_gr_fg_work_sketch_y_eq_x2_min_4x_p_3,
        """Use the graph generated for Example 4 (y = x² - 2x - 3). Draw a horizontal dashed line at y = 2. Mark the points where this line intersects the parabola. Draw vertical dashed lines from these intersection points down to the x-axis. Indicate the approximate x-values where these vertical lines meet the x-axis (approx -1.4 and 3.4).""": generator.f3_gr_fg_work_solve_graphically, # Graph b

    }
    # --- End of mapping dictionary ---


    # Iterate through the data structure
    for item_index, item in enumerate(data):
        print(f"Processing item {item_index+1}/{len(data)}")
        data_types = ['notes', 'worked_examples', 'questions']

        for data_type in data_types:
            elements_container = item.get(data_type)
            if not elements_container: continue

            elements_to_process = []
            # Notes sections are nested differently
            if data_type == 'notes':
                if 'sections' in elements_container and isinstance(elements_container['sections'], list):
                    elements_to_process = elements_container['sections']
                else: continue
            elif isinstance(elements_container, list): # worked_examples or questions
                elements_to_process = elements_container
            else: continue

            for element_index, element in enumerate(elements_to_process):
                 if isinstance(element, dict):
                    desc_path_map = {
                        'image_description': 'image_path',
                        'image_description_2': 'image_path_2'
                    }

                    for desc_key, path_key in desc_path_map.items():
                        if desc_key in element:
                            desc = element[desc_key]
                            if not desc or not isinstance(desc, str): continue

                            # Use direct lookup first using the exact description string
                            func_to_call = description_to_function.get(desc)

                            if func_to_call:
                                try:
                                    # Check reuse dictionary keyed by description string
                                    if desc in generated_paths_by_desc:
                                        image_path = generated_paths_by_desc[desc]
                                        print(f"Reusing path for {desc_key} (desc: {desc[:30]}...): {image_path}")
                                    else:
                                        image_path = func_to_call() # Generate image
                                        if image_path: # Check if generation was successful
                                            generated_paths_by_desc[desc] = image_path # Store path using description as key
                                            processed_count += 1
                                        else:
                                            print(f"--- Warning: Function {func_to_call.__name__} did not return a valid path for {desc_key}. ---")
                                            continue # Skip adding the key if path is invalid

                                    element[path_key] = image_path # Add/update the path key

                                except Exception as e:
                                    print(f"--- Error processing {data_type} {element_index} ({path_key}) in Item {item_index+1} ---")
                                    print(f"Description:\n{desc}")
                                    print(f"Function: {func_to_call.__name__ if func_to_call else 'N/A'}")
                                    print(f"Error: {e}")
                                    print("-----------------------------------------------------\n")
                            else:
                                # Print warning only if the key is not for reuse (avoid duplicate warnings)
                                if desc != "Reference the same graph described in Worked Example 1.":
                                     print(f"Warning: No function mapped for {data_type} description ({path_key}) (Item {item_index+1}, Element {element_index}):\n{desc}\n")


    # Write the updated data structure back to a new JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\nSuccessfully processed {processed_count} unique images and saved updated data to '{output_file}'.")
    except IOError:
        print(f"Error: Could not write to output file '{output_file}'.")
    except TypeError as e:
        print(f"Error: Could not serialize data to JSON. Error: {e}")

if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use('Agg')
        print("Using Agg backend for Matplotlib.")
    except Exception as e:
        print(f"Could not set Agg backend: {e}")

    # Add necessary imports if they are only used here
    from matplotlib.path import Path
    from mpl_toolkits.mplot3d import Axes3D

    process_json_data(INPUT_JSON_FILE, OUTPUT_JSON_FILE)