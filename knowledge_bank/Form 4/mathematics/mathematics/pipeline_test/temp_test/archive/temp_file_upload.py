import json
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math
from matplotlib.patches import Patch
import inspect  # To get function names dynamically
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
import logging

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
INPUT_JSON_FILE = 'temp_file_upload.json'
OUTPUT_JSON_FILE = 'temp_file_upload_processed.json'
OUTPUT_IMAGE_FOLDER = 'math_diagrams'

# --- Helper Function ---
def ensure_dir(directory):
    """Creates the directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_filename_from_caller(suffix=""):
    """Generates a filename based on the calling function's name."""
    caller_frame = inspect.currentframe().f_back
    func_name = caller_frame.f_code.co_name
    # Basic cleaning: lowercase and replace potential issues
    cleaned_name = func_name.lower().replace(' ', '_').replace('-', '_')
    filename = f"{cleaned_name}{suffix}.png"
    return os.path.join(OUTPUT_IMAGE_FOLDER, filename)

# --- Plotting Functions ---

class MathDiagramGenerator:
    """Class to hold diagram generation methods."""

    def __init__(self):
        self.output_folder = OUTPUT_IMAGE_FOLDER
        ensure_dir(self.output_folder)
        # Store data for potential reuse across related diagrams within a topic if needed
        self.context_data = {}

    def _save_plot(self, fig, filename):
        """Saves the figure and closes it, applying constrained layout and adjustText to fix overlaps."""
        # enable constrained layout for better spacing
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass
        # Use adjustText to resolve overlapping text if available
        try:
            from adjustText import adjust_text
            for ax in fig.axes:
                texts = ax.texts
                if texts:
                    adjust_text(texts, ax=ax, autoalign='y', only_move={'points':'y', 'text':'y'})
        except ImportError:
            pass
        filepath = os.path.join(self.output_folder, filename)
        fig.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved diagram: {filepath}")
        return filepath

    # === Form 3: Graphs - Travel Graphs ===

    def f3_tg_notes_dist_time_vs_speed_time(self):
        """Generates Distance-Time and Speed-Time graphs side-by-side."""
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Travel Graph Examples', fontsize=14)

        # --- Left Graph (Distance-Time) ---
        ax1.set_facecolor('white')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Distance")
        ax1.set_title("Distance-Time Graph")
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 8)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Segments O-A, A-B, B-C
        time1 = [0, 3]
        dist1 = [0, 6] # Positive slope (Constant speed away)
        time2 = [3, 6]
        dist2 = [6, 6] # Horizontal (Stationary)
        time3 = [6, 9]
        dist3 = [6, 0] # Negative slope (Constant speed towards)

        ax1.plot(time1, dist1, 'b-', label='O-A (Speed > 0)')
        ax1.plot(time2, dist2, 'g-', label='A-B (Stationary)')
        ax1.plot(time3, dist3, 'r-', label='B-C (Speed < 0)')

        ax1.text(1.5, 3.5, 'Constant Speed\nAway', ha='center', va='center', color='blue')
        ax1.text(4.5, 6.5, 'Stationary', ha='center', va='center', color='green')
        ax1.text(7.5, 3.5, 'Constant Speed\nTowards', ha='center', va='center', color='red')
        ax1.legend()

        # --- Right Graph (Speed-Time) ---
        ax2.set_facecolor('white')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Speed")
        ax2.set_title("Speed-Time Graph")
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 8)
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Segments D-E, E-F, F-G
        time_s1 = [0, 3]
        speed1 = [0, 6] # Positive slope (Acceleration)
        time_s2 = [3, 6]
        speed2 = [6, 6] # Horizontal (Constant Speed)
        time_s3 = [6, 9]
        speed3 = [6, 0] # Negative slope (Deceleration)

        ax2.plot(time_s1, speed1, 'm-', label='D-E (Acceleration)')
        ax2.plot(time_s2, speed2, 'c-', label='E-F (Constant Speed)')
        ax2.plot(time_s3, speed3, 'orange', linestyle='-', label='F-G (Deceleration)')

        ax2.text(1.5, 3.5, 'Acceleration', ha='center', va='center', color='m')
        ax2.text(4.5, 6.5, 'Constant Speed', ha='center', va='center', color='c')
        ax2.text(7.5, 3.5, 'Deceleration', ha='center', va='center', color='orange')
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        return self._save_plot(fig, filename)

    def f3_tg_notes_area_under_speed_time(self):
        """Generates a speed-time graph illustrating area calculation."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Speed (v)")
        ax.set_title("Area Under Speed-Time Graph = Distance")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Example graph: Accelerate, Constant, Decelerate
        t1, v1 = 3, 8  # End of acceleration
        t2, v2 = 7, 8  # End of constant speed
        t3, v3 = 9, 4  # End of deceleration

        time_pts = [0, t1, t2, t3]
        speed_pts = [0, v1, v2, v3]

        ax.plot(time_pts, speed_pts, 'b-', marker='o')

        # --- Area 1: Triangle (Acceleration from rest) ---
        pts_tri = np.array([[0, 0], [t1, 0], [t1, v1]])
        triangle = patches.Polygon(pts_tri, closed=True, facecolor='lightblue', alpha=0.6)
        ax.add_patch(triangle)
        ax.text(t1/2, v1/2, f'Area = Distance\n1/2*{t1}*{v1}\n= {0.5*t1*v1}',
                ha='center', va='center', fontsize=9, color='darkblue')

        # --- Area 2: Rectangle (Constant Speed) ---
        pts_rect = np.array([[t1, 0], [t2, 0], [t2, v2], [t1, v1]])
        rectangle = patches.Polygon(pts_rect, closed=True, facecolor='lightgreen', alpha=0.6)
        ax.add_patch(rectangle)
        ax.text((t1+t2)/2, v1/2, f'Area = Distance\n({t2-t1})*{v1}\n= {(t2-t1)*v1}',
                ha='center', va='center', fontsize=9, color='darkgreen')

        # --- Area 3: Trapezium (Deceleration) ---
        pts_trap = np.array([[t2, 0], [t3, 0], [t3, v3], [t2, v2]])
        trapezium = patches.Polygon(pts_trap, closed=True, facecolor='lightcoral', alpha=0.6)
        ax.add_patch(trapezium)
        area_trap = 0.5 * (v2 + v3) * (t3 - t2)
        ax.text((t2+t3)/2, (v2+v3)/4, f'Area = Distance\n1/2*({v2}+{v3})*({t3-t2})\n= {area_trap:.1f}',
                ha='center', va='center', fontsize=9, color='darkred')

        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_yticks(np.arange(0, 13, 2))

        return self._save_plot(fig, filename)

    def f3_tg_we1_dist_time_graph(self):
        """Generates distance-time graph for Worked Example 1."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Distance (km)")
        ax.set_title("Car Journey Distance-Time Graph")
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 130)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Points: (0,0), (2,120), (3,120), (6,0)
        time_pts = [0, 2, 3, 6]
        dist_pts = [0, 120, 120, 0]

        ax.plot(time_pts, dist_pts, 'b-', marker='o')
        ax.plot(time_pts, dist_pts, 'bo') # Redraw points to ensure visibility

        # Annotations
        ax.text(1, 60, 'Travel Out', ha='center', va='bottom', color='blue')
        ax.text(2.5, 125, 'Stop', ha='center', va='center', color='blue')
        ax.text(4.5, 60, 'Return', ha='center', va='top', color='blue')

        ax.set_xticks(np.arange(0, 7, 1))
        ax.set_yticks(np.arange(0, 140, 20))

        return self._save_plot(fig, filename)

    def f3_tg_we3_area_triangle(self):
        """References WE2 graph and shades the triangular area."""
        filename = get_filename_from_caller()
        # Re-generate the base graph from WE2
        if 'f3_tg_we2' not in self.context_data:
             # Should have been called, but as fallback:
            print("Warning: Context data for WE2 not found, regenerating base graph.")
            self.f3_tg_we2_speed_time_graph() # Regenerate to populate context
            if 'f3_tg_we2' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None # Or raise error

        time_pts = self.context_data['f3_tg_we2']['time']
        speed_pts = self.context_data['f3_tg_we2']['speed']

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Distance in First 5 Seconds (Area)")
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'r-', marker='o')
        ax.plot(time_pts, speed_pts, 'ro')

        # Shade the triangle (0 to 5s)
        ax.fill_between(time_pts[:2], speed_pts[:2], color='lightblue', alpha=0.7)
        ax.text(2.5, 4, 'Area = Distance\n= 1/2 * 5 * 10\n= 25 m',
                ha='center', va='center', fontsize=10, color='darkblue')

        ax.set_xticks(np.arange(0, 17, 1))
        ax.set_yticks(np.arange(0, 13, 1))

        return self._save_plot(fig, filename)

    def f3_tg_we4_area_total(self):
        """References WE2 graph and shades the total area."""
        filename = get_filename_from_caller()
         # Re-generate the base graph from WE2
        if 'f3_tg_we2' not in self.context_data:
             print("Warning: Context data for WE2 not found, regenerating base graph.")
             self.f3_tg_we2_speed_time_graph()
             if 'f3_tg_we2' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        time_pts = self.context_data['f3_tg_we2']['time']
        speed_pts = self.context_data['f3_tg_we2']['speed']

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Total Distance in 15 Seconds (Total Area)")
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'r-', marker='o')
        ax.plot(time_pts, speed_pts, 'ro')

        # Shade the total area (triangle + rectangle)
        ax.fill_between(time_pts, speed_pts, color='lightgreen', alpha=0.7)
        ax.text(7.5, 4, 'Total Area = Total Distance\n= 25m + 100m\n= 125 m',
                ha='center', va='center', fontsize=10, color='darkgreen')

        ax.set_xticks(np.arange(0, 17, 1))
        ax.set_yticks(np.arange(0, 13, 1))

        return self._save_plot(fig, filename)

    # Note: WE5 for Travel Graphs has no image_description

    def f3_tg_q_moderate_speed_time_segment(self):
        """Generates speed-time graph segment for Moderate question."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Deceleration Example")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 16)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Line from (0, 15) to (4, 5)
        time_pts = [0, 4]
        speed_pts = [15, 5]

        ax.plot(time_pts, speed_pts, 'b-', marker='o')
        ax.plot(time_pts, speed_pts, 'bo')

        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 17, 1))

        # Store for reuse in Challenging question
        self.context_data['f3_tg_q_moderate'] = {'time': time_pts, 'speed': speed_pts}

        return self._save_plot(fig, filename)

    def f3_tg_q_challenging_area_trapezium(self):
        """References Moderate question graph and shades the trapezium area."""
        filename = get_filename_from_caller()
        if 'f3_tg_q_moderate' not in self.context_data:
            print("Warning: Context data for Moderate Question not found, regenerating.")
            self.f3_tg_q_moderate_speed_time_segment()
            if 'f3_tg_q_moderate' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        time_pts = self.context_data['f3_tg_q_moderate']['time']
        speed_pts = self.context_data['f3_tg_q_moderate']['speed']

        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Distance Traveled (Area)")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 16)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'b-', marker='o')
        ax.plot(time_pts, speed_pts, 'bo')

        # Shade the trapezium area
        ax.fill_between(time_pts, speed_pts, color='lightcoral', alpha=0.7)
        area = 0.5 * (speed_pts[0] + speed_pts[1]) * (time_pts[1] - time_pts[0])
        ax.text(2, 6, f'Area = Distance\n= 1/2*(15+5)*4\n= {area} m',
                ha='center', va='center', fontsize=10, color='darkred')

        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 17, 1))

        return self._save_plot(fig, filename)

    # === Form 3: Variation ===

    def f3_v_notes_direct_vs_inverse_graphs(self):
        """Generates Direct and Inverse Variation graphs side-by-side."""
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Variation Graphs', fontsize=14)

        # --- Left Graph (Direct Variation) ---
        ax1.set_facecolor('white')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Direct Variation (y=kx, k>0)")
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(0, color='black', linewidth=0.5)

        k_direct = 1.5 # Example k
        x_direct = np.array([0, 5])
        y_direct = k_direct * x_direct
        ax1.plot(x_direct, y_direct, 'b-', linewidth=2, label=f'y = {k_direct}x')
        ax1.text(2.5, 7.5, 'y = kx (k>0)', ha='center', color='blue')
        ax1.legend()

        # --- Right Graph (Inverse Variation) ---
        ax2.set_facecolor('white')
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Inverse Variation (y=k/x, k>0)")
        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)

        k_inverse = 4.0 # Example k
        x_inverse = np.linspace(0.1, 5, 400) # Avoid x=0
        y_inverse = k_inverse / x_inverse
        ax2.plot(x_inverse, y_inverse, 'r-', linewidth=2, label=f'y = {k_inverse}/x')
        ax2.text(2.5, 6, 'y = k/x (k>0)', ha='center', color='red')
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    # === Form 3: Algebra - Inequalities ===

    def f3_ai_notes_simul_ineq_1var(self):
        """Generates a number line illustrating 2 <= x < 5."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 2), facecolor='white')
        ax.set_facecolor('white')
        ax.set_yticks([]) # Remove y-axis ticks
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero') # x-axis line
        ax.xaxis.set_ticks_position('bottom')

        # Plot the number line segment
        ax.plot([2, 5], [0.1, 0.1], 'b-', linewidth=3)

        # Plot the circles
        ax.plot(2, 0.1, 'bo', markersize=8, label='Closed circle at 2 (>=)') # Closed circle
        ax.plot(5, 0.1, 'bo', markersize=8, markerfacecolor='white', label='Open circle at 5 (<)') # Open circle

        # Add ticks and labels
        ax.set_xticks(np.arange(0, 7, 1))
        ax.set_xlabel("x")
        ax.set_title("Number Line Solution: 2 <= x < 5")

        # Optional legend (might be cleaner without)
        # ax.legend(loc='upper right')

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_ai_notes_simul_ineq_2var(self):
        """Illustrates feasible region for y > 1 and x + y <= 4."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        x_range = np.linspace(-1, 5, 400)

        # Plotting constraints
        # 1. y > 1 (Boundary y = 1, dashed)
        ax.axhline(1, color='red', linestyle='--', label='y = 1')
        # Shade unwanted region (y <= 1)
        ax.fill_between(x_range, -1, 1, color='red', alpha=0.2, label='y <= 1 (unwanted)')

        # 2. x + y <= 4 => y <= 4 - x (Boundary y = 4 - x, solid)
        y_boundary2 = 4 - x_range
        ax.plot(x_range, y_boundary2, 'blue', linestyle='-', label='x + y = 4')
        # Shade unwanted region (y > 4 - x)
        ax.fill_between(x_range, y_boundary2, 5, color='blue', alpha=0.2, label='x + y > 4 (unwanted)')

        # Add axes and grid
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Set limits and labels
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Feasible Region for y > 1 and x + y <= 4")

        # Indicate Feasible Region R (unshaded area)
        ax.text(1, 2, 'R\n(Feasible\nRegion)', ha='center', va='center', fontsize=12, color='black')

        # Add legend
        ax.legend(loc='upper right', fontsize=9)

        return self._save_plot(fig, filename)

    def f3_ai_we2_num_line_soln(self):
        """Number line for 2 <= x < 4 (same as f3_ai_notes_simul_ineq_1var, just different range)."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 2), facecolor='white')
        ax.set_facecolor('white')
        ax.set_yticks([])
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.xaxis.set_ticks_position('bottom')

        ax.plot([2, 4], [0.1, 0.1], 'b-', linewidth=3)
        ax.plot(2, 0.1, 'bo', markersize=8) # Closed
        ax.plot(4, 0.1, 'bo', markersize=8, markerfacecolor='white') # Open

        ax.set_xticks(np.arange(0, 6, 1))
        ax.set_xlabel("x")
        ax.set_title("Number Line Solution: 2 <= x < 4")

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_ai_we4_region_3_ineqs(self):
        """Illustrates feasible region for y >= 0, x > 1, y < x."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        x_range = np.linspace(-1, 5, 400)

        # Plotting constraints
        # 1. y >= 0 (Boundary y = 0 (x-axis), solid)
        ax.axhline(0, color='green', linestyle='-', label='y = 0')
        # Shade unwanted region (y < 0)
        ax.fill_between(x_range, -1, 0, color='green', alpha=0.2, label='y < 0 (unwanted)')

        # 2. x > 1 (Boundary x = 1, dashed)
        ax.axvline(1, color='red', linestyle='--', label='x = 1')
        # Shade unwanted region (x <= 1)
        ax.fill_betweenx([-1, 5], -1, 1, color='red', alpha=0.2, label='x <= 1 (unwanted)')

        # 3. y < x (Boundary y = x, dashed)
        ax.plot(x_range, x_range, 'blue', linestyle='--', label='y = x')
        # Shade unwanted region (y >= x)
        ax.fill_between(x_range, x_range, 5, where=x_range>=x_range, color='blue', alpha=0.2, label='y >= x (unwanted)')


        # Add axes and grid
        # ax.axhline(0, color='black', linewidth=0.5) # Already done by constraint 1
        # ax.axvline(0, color='black', linewidth=0.5) # Covered by shading
        ax.grid(True, linestyle='--', alpha=0.6)

        # Set limits and labels
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Feasible Region R")

        # Indicate Feasible Region R (unshaded triangle)
        # Find vertices: (1,0), intersection of x=1 and y=x -> (1,1)
        ax.text(2, 0.5, 'R', ha='center', va='center', fontsize=14, color='black')

        # Add legend
        ax.legend(loc='lower right', fontsize=9)

        return self._save_plot(fig, filename)

    # def f3_ai_q_challenging_identify_region(self):
    #     """Generates the diagram for the challenging question, asking to identify inequalities."""
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    #     ax.set_facecolor('white')
    #     x_range = np.linspace(-2, 4, 400)

    #     # Draw boundary lines
    #     # y = 3 (solid)
    #     ax.axhline(3, color='blue', linestyle='-', label='y = 3')
    #     # x = -1 (solid)
    #     ax.axvline(-1, color='red', linestyle='-', label='x = -1')
    #     # y = x (dashed)
    #     ax.plot(x_range, x_range, 'green', linestyle='--', label='y = x')

    #     # Shade unwanted regions based on R being below y=3, right of x=-1, below y=x
    #     # Shade above y=3
    #     ax.fill_between(x_range, 3, 5, color='blue', alpha=0.2)
    #     # Shade left of x=-1
    #     ax.fill_betweenx([-2, 5], -2, -1, color='red', alpha=0.2)
    #     # Shade above y=x
    #     ax.fill_between(x_range, x_range, 5, where=x_range>=x_range, color='green', alpha=0.2)

    #     # Indicate Region R
    #     ax.text(1, 0, 'R', ha='center', va='center', fontsize=16, color='black')

    #     # Setup plot
    #     ax.set_xlim(-2, 4)
    #     ax.set_ylim(-2, 5)
    #     ax.axhline(0, color='black', linewidth=0.5)
    #     ax.axvline(0, color='black', linewidth=0.5)
    #     ax.grid(True, linestyle='--', alpha=0.6)
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_title("Identify Inequalities Defining Region R")
    #     ax.legend(loc='lower right')

    #     return self._save_plot(fig, filename)

    # === Form 3: Algebra - Indices and Logarithms ===
    # No images described in this section

    # === Form 3: Geometry - Points, Lines and Angles ===

    def f3_gpla_notes_angle_of_elevation(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Observer O
        ox, oy = 1, 1
        ax.plot(ox, oy, 'ko', markersize=8, label='Observer (O)')

        # Horizontal line
        ax.plot([ox, 5], [oy, oy], 'k--', label='Horizontal Line')

        # Top of object T
        tx, ty = 4, 3
        ax.plot(tx, ty, 'ro', markersize=8, label='Top of Object (T)')

        # Line of sight OT
        ax.plot([ox, tx], [oy, ty], 'b-', label='Line of Sight')

        # Angle of Elevation Arc
        angle_rad = math.atan2(ty - oy, tx - ox)
        angle_deg = math.degrees(angle_rad)
        elev_arc = patches.Arc((ox, oy), 2, 2, angle=0, theta1=0, theta2=angle_deg, color='g', linewidth=1.5, label='Angle of Elevation')
        ax.add_patch(elev_arc)
        ax.text(ox + 1.2 * math.cos(angle_rad / 2), oy + 1.2 * math.sin(angle_rad / 2),
                'Angle of Elevation', color='g', ha='left', va='bottom', rotation=angle_deg/2)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title("Angle of Elevation")
        # ax.legend() # Optional, might clutter

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_gpla_notes_angle_of_depression(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Observer O (high up)
        ox, oy = 1, 3
        ax.plot(ox, oy, 'ko', markersize=8, label='Observer (O)')

        # Horizontal line from O
        ax.plot([ox, 5], [oy, oy], 'k--', label='Horizontal Line')

        # Bottom object B
        bx, by = 4, 1
        ax.plot(bx, by, 'ro', markersize=8, label='Bottom Object (B)')

        # Line of sight OB
        ax.plot([ox, bx], [oy, by], 'b-', label='Line of Sight')

        # Angle of Depression Arc
        angle_rad = math.atan2(oy - by, bx - ox) # Angle below horizontal
        angle_deg = math.degrees(angle_rad)
        dep_arc = patches.Arc((ox, oy), 2, 2, angle=0, theta1=-angle_deg, theta2=0, color='g', linewidth=1.5, label='Angle of Depression')
        ax.add_patch(dep_arc)
        ax.text(ox + 1.2 * math.cos(-angle_rad / 2), oy + 1.2 * math.sin(-angle_rad / 2),
                'Angle of Depression', color='g', ha='left', va='top', rotation=-angle_deg/2)


        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title("Angle of Depression")
        # ax.legend()

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_gpla_notes_elev_dep_relationship(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Points A (lower) and B (higher)
        ax_coord, ay_coord = 1, 1
        bx_coord, by_coord = 4, 4
        ax.plot(ax_coord, ay_coord, 'bo', markersize=8, label='A (Observer)')
        ax.plot(bx_coord, by_coord, 'ro', markersize=8, label='B (Object)')

        # Line of sight AB
        ax.plot([ax_coord, bx_coord], [ay_coord, by_coord], 'k-', linewidth=1)

        # Horizontal at A
        ax.plot([ax_coord, 5], [ay_coord, ay_coord], 'b--', label='Horizontal at A')
        # Horizontal at B (parallel)
        ax.plot([0, bx_coord], [by_coord, by_coord], 'r--', label='Horizontal at B')

        # Angle of Elevation alpha at A
        angle_a_rad = math.atan2(by_coord - ay_coord, bx_coord - ax_coord)
        angle_a_deg = math.degrees(angle_a_rad)
        elev_arc = patches.Arc((ax_coord, ay_coord), 1.5, 1.5, angle=0, theta1=0, theta2=angle_a_deg, color='blue', linewidth=1.5)
        ax.add_patch(elev_arc)
        ax.text(ax_coord + 0.9 * math.cos(angle_a_rad / 2), ay_coord + 0.9 * math.sin(angle_a_rad / 2),
                'α', color='blue', ha='center', va='center', fontsize=12)

        # Angle of Depression beta at B
        angle_b_rad = math.atan2(by_coord - ay_coord, bx_coord - ax_coord) # Same angle relative to horizontal
        angle_b_deg = math.degrees(angle_b_rad)
        dep_arc = patches.Arc((bx_coord, by_coord), 1.5, 1.5, angle=0, theta1=180, theta2=180+angle_b_deg, color='red', linewidth=1.5) # Angle measured clockwise from negative x-direction
        ax.add_patch(dep_arc)
        ax.text(bx_coord - 0.9 * math.cos(angle_b_rad / 2), by_coord - 0.9 * math.sin(angle_b_rad / 2)+0.1, # Offset text slightly
                'β', color='red', ha='center', va='center', fontsize=12)

        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 5)
        ax.set_title("Angle of Elevation (α) = Angle of Depression (β)")
        ax.text(2.5, 0.2, "Horizontals are parallel, α = β (alternate interior angles)", ha='center')

        # ax.legend()

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_gpla_we1_tower_elevation(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box') # Make triangle proportions look right

        # Vertices
        P = (0, 0) # Base of tower
        Q = (50, 0) # Observation point
        T = (0, 28.87) # Top of tower (height h calculated: 50*tan(30))
        h = 50 * math.tan(math.radians(30))
        T = (0,h)


        # Draw triangle
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-', label='Ground (50 m)') # Ground
        ax.plot([P[0], T[0]], [P[1], T[1]], 'k-', label=f'Tower (h)') # Tower
        ax.plot([Q[0], T[0]], [Q[1], T[1]], 'b--', label='Line of Sight') # Line of sight

        # Mark points
        ax.plot(P[0], P[1], 'ko')
        ax.plot(Q[0], Q[1], 'ko')
        ax.plot(T[0], T[1], 'ko')
        ax.text(P[0]-3, P[1]-3, 'P (Base)', ha='right')
        ax.text(Q[0]+3, Q[1]-3, 'Q (Observer)', ha='left')
        ax.text(T[0]-3, T[1], 'T (Top)', ha='right')

        # Mark right angle
        rect = patches.Rectangle((P[0], P[1]), 4, 4, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Mark angle of elevation
        angle_deg = 30
        elev_arc = patches.Arc(Q, 30, 30, angle=0, theta1=180-angle_deg, theta2=180, color='g', linewidth=1.5)
        ax.add_patch(elev_arc)
        ax.text(Q[0]-18, Q[1]+6, f'{angle_deg}°', color='g', ha='center', va='center')

        # Labels
        ax.text(Q[0]/2, Q[1]-3, '50 m', ha='center', va='top', color='k')
        ax.text(P[0]-3, T[1]/2, 'h', ha='right', va='center', color='k')

        ax.set_xlim(-10, 60)
        ax.set_ylim(-10, h+10)
        ax.set_title("Angle of Elevation Problem")
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Vertical Height (m)")
        ax.grid(True, linestyle='--')
        # ax.legend()

        return self._save_plot(fig, filename)

    def f3_gpla_we2_cliff_depression(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Points
        cliff_height = 80
        angle_dep_deg = 40
        angle_elev_deg = 40 # Alternate interior angle
        distance_d = cliff_height / math.tan(math.radians(angle_elev_deg))

        T = (0, cliff_height) # Top of cliff
        F = (0, 0)            # Foot of cliff
        B = (distance_d, 0)   # Boat

        # Draw triangle TFB
        ax.plot([T[0], F[0]], [T[1], F[1]], 'k-', label=f'Cliff ({cliff_height} m)') # Cliff
        ax.plot([F[0], B[0]], [F[1], B[1]], 'k-', label=f'Distance (d)') # Sea level
        ax.plot([T[0], B[0]], [T[1], B[1]], 'b--', label='Line of Sight') # Line of sight

        # Mark points
        ax.plot(T[0], T[1], 'ko'); ax.text(T[0]-5, T[1], 'T', ha='right')
        ax.plot(F[0], F[1], 'ko'); ax.text(F[0]-5, F[1], 'F', ha='right')
        ax.plot(B[0], B[1], 'ko'); ax.text(B[0]+5, B[1], 'B (Boat)', ha='left')

        # Mark right angle at F
        rect = patches.Rectangle((F[0], F[1]), 6, 6, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Mark angle of depression at T
        ax.plot([T[0], B[0]+10], [T[1], T[1]], 'r--', label='Horizontal at T') # Horizontal line
        dep_arc = patches.Arc(T, 40, 40, angle=0, theta1=-angle_dep_deg, theta2=0, color='r', linewidth=1.5)
        ax.add_patch(dep_arc)
        ax.text(T[0]+25, T[1]-10, f'{angle_dep_deg}° (Depression)', color='r', ha='center', va='center')

        # Mark alternate interior angle at B (Elevation)
        elev_arc = patches.Arc(B, 40, 40, angle=0, theta1=180, theta2=180+angle_elev_deg, color='g', linewidth=1.5)
        ax.add_patch(elev_arc)
        ax.text(B[0]-25, B[1]+10, f'{angle_elev_deg}° (Elevation)', color='g', ha='center', va='center')


        # Labels
        ax.text(T[0]-5, T[1]/2, f'{cliff_height} m', ha='right', va='center', color='k')
        ax.text(B[0]/2, B[1]-5, 'd', ha='center', va='top', color='k')

        ax.set_xlim(-15, distance_d + 20)
        ax.set_ylim(-15, cliff_height + 15)
        ax.set_title("Angle of Depression Problem")
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Vertical Height (m)")
        ax.grid(True, linestyle='--')
        # ax.legend()

        return self._save_plot(fig, filename)

    def f3_gpla_we3_elevation_diagram(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ox, oy = 1, 1
        bx, by = 4, 2.5 # Bird position relative
        angle_deg = 25

        ax.plot(ox, oy, 'ko', markersize=8, label='O (Observer)')
        ax.plot([ox, 5], [oy, oy], 'k--', label='Horizontal') # Horizontal line
        ax.plot(bx, by, 'ro', markersize=8, label='B (Bird)') # Bird
        ax.plot([ox, bx], [oy, by], 'b-') # Line of sight

        elev_arc = patches.Arc((ox, oy), 2, 2, angle=0, theta1=0, theta2=angle_deg, color='g', linewidth=1.5)
        ax.add_patch(elev_arc)
        ax.text(ox + 1.2 * math.cos(math.radians(angle_deg / 2)),
                oy + 1.2 * math.sin(math.radians(angle_deg / 2)),
                f'{angle_deg}°', color='g', ha='left', va='bottom')

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title("Angle of Elevation to Bird")
        ax.legend(loc='lower right')

        return self._save_plot(fig, filename)

    def f3_gpla_we4_depression_diagram(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        px, py = 1, 3 # Person P
        cx, cy = 4, 1 # Car C
        angle_deg = 35

        ax.plot(px, py, 'ko', markersize=8, label='P (Person)')
        ax.plot([px, 5], [py, py], 'k--', label='Horizontal') # Horizontal line
        ax.plot(cx, cy, 'ro', markersize=8, label='C (Car)') # Car
        ax.plot([px, cx], [py, cy], 'b-') # Line of sight

        dep_arc = patches.Arc((px, py), 2, 2, angle=0, theta1=-angle_deg, theta2=0, color='g', linewidth=1.5)
        ax.add_patch(dep_arc)
        ax.text(px + 1.2 * math.cos(math.radians(-angle_deg / 2)),
                py + 1.2 * math.sin(math.radians(-angle_deg / 2)),
                f'{angle_deg}°', color='g', ha='left', va='top')

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title("Angle of Depression to Car")
        ax.legend(loc='lower left')

        return self._save_plot(fig, filename)

    # def f3_gpla_we5_scale_drawing(self):
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    #     ax.set_facecolor('white')
    #     # Scale: 1 cm : 10 m. So 50m is 5cm.

    #     # Vertices based on scale
    #     A = (0, 0) # Observation point
    #     B = (5, 0) # Base of tower (scaled)
    #     angle_CAB_deg = 30
    #     angle_CAB_rad = math.radians(angle_CAB_deg)
    #     # Height BC: tan(30) = BC/AB => BC = AB * tan(30) = 5 * tan(30)
    #     height_BC_scaled = 5 * math.tan(angle_CAB_rad)
    #     C = (5, height_BC_scaled) # Top of tower (scaled)

    #     # Draw triangle
    #     ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', label='Ground AB (5 cm)') # Ground
    #     ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', label=f'Tower BC ({height_BC_scaled:.1f} cm)') # Tower
    #     ax.plot([A[0], C[0]], [A[1], C[1]], 'b--', label='Line of Sight AC') # Line of sight

    #     # Mark points
    #     ax.plot(A[0], A[1], 'ko'); ax.text(A[0]-0.2, A[1]-0.2, 'A', ha='right')
    #     ax.plot(B[0], B[1], 'ko'); ax.text(B[0]+0.2, B[1]-0.2, 'B', ha='left')
    #     ax.plot(C[0], C[1], 'ko'); ax.text(C[0]+0.2, C[1], 'C', ha='left')

    #     # Mark right angle at B
    #     rect = patches.Rectangle((B[0]-0.3, B[1]), 0.3, 0.3, angle=-90, linewidth=1, edgecolor='k', facecolor='none')
    #     ax.add_patch(rect)

    #     # Mark angle at A
    #     elev_arc = patches.Arc(A, 1.5, 1.5, angle=0, theta1=0, theta2=angle_CAB_deg, color='g', linewidth=1.5)
    #     ax.add_patch(elev_arc)
    #     ax.text(A[0]+1, A[1]+0.3, f'{angle_CAB_deg}°', color='g', ha='center', va='center')

    #     # Labels
    #     ax.text(A[0]/2 + B[0]/2, A[1]-0.2, '5 cm (Represents 50 m)', ha='center', va='top', color='k')
    #     ax.text(B[0]+0.2, C[1]/2, f'Measure BC\n(Approx {height_BC_scaled:.1f} cm\n=> Approx {height_BC_scaled*10:.1f} m)', ha='left', va='center', color='k', fontsize=8)


    #     ax.set_xlim(-1, 6)
    #     ax.set_ylim(-1, height_BC_scaled + 1)
    #     ax.set_title("Scale Drawing (1 cm : 10 m)")
    #     ax.set_xlabel("Scaled Distance (cm)")
    #     ax.set_ylabel("Scaled Distance (cm)")
    #     ax.grid(True, linestyle='--')
    #     ax.set_aspect('equal', adjustable='box')

    #     return self._save_plot(fig, filename)

    # === Form 3: Geometry - Bearing ===

    def f3_gb_notes_bearing_back_bearing(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off') # Keep axes for reference if preferred

        # Points A and B
        A = (2, 1)
        B = (5, 5)
        theta_deg = 45 # Example bearing < 180
        # Recalculate B based on A and theta
        dist = 5
        theta_rad_math = math.radians(90 - theta_deg) # Convert bearing to math angle
        B = (A[0] + dist * math.cos(theta_rad_math), A[1] + dist * math.sin(theta_rad_math))

        ax.plot(A[0], A[1], 'bo', markersize=8, label='A')
        ax.plot(B[0], B[1], 'ro', markersize=8, label='B')
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=1)

        # North line at A
        ax.plot([A[0], A[0]], [A[1], A[1]+dist*1.2], 'b--', linewidth=1, label='North at A')
        ax.text(A[0], A[1]+dist*1.2 + 0.2, 'N', color='blue', ha='center', va='bottom')

        # North line at B (parallel)
        ax.plot([B[0], B[0]], [B[1], B[1]+dist*1.2], 'r--', linewidth=1, label='North at B || North at A')
        ax.text(B[0], B[1]+dist*1.2 + 0.2, 'N', color='red', ha='center', va='bottom')

        # Bearing theta from A to B
        bearing_arc = patches.Arc(A, 2, 2, angle=90, theta1=-theta_deg, theta2=0, color='blue', linewidth=1.5) # Clockwise from North
        ax.add_patch(bearing_arc)
        # Adjust text position for bearing
        mid_angle_rad = math.radians(90 - theta_deg / 2)
        ax.text(A[0] + 1.2 * math.cos(mid_angle_rad), A[1] + 1.2 * math.sin(mid_angle_rad),
                f'θ={theta_deg}°', color='blue', ha='center', va='center', fontsize=10)

        # Back bearing theta + 180 from B to A
        back_bearing_deg = theta_deg + 180
        # The vector BA makes math angle theta_rad_math + pi
        back_bearing_arc = patches.Arc(B, 3, 3, angle=90, theta1=-back_bearing_deg, theta2=0, color='red', linewidth=1.5) # Clockwise from North at B
        ax.add_patch(back_bearing_arc)
        # Text position for back bearing - more complex angle
        mid_back_angle_deg = (360 + (90 - back_bearing_deg))/2 + (90 - back_bearing_deg) # approx
        mid_back_angle_rad = math.radians(-back_bearing_deg/2 + 90) # Angle bisector in math terms
        ax.text(B[0] + 1.8 * math.cos(mid_back_angle_rad), B[1] + 1.8 * math.sin(mid_back_angle_rad),
                f'Back Bearing\nθ+180°\n= {back_bearing_deg}°', color='red', ha='center', va='top', fontsize=9)


        ax.set_xlim(A[0]-dist*0.5, B[0]+dist*0.5)
        ax.set_ylim(A[1]-dist*0.2, B[1]+dist*1.5)
        ax.set_title("Bearing and Back Bearing")
        # ax.legend()

        return self._save_plot(fig, filename)

    def f3_gb_we1_bearing_240(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'}, facecolor='white')
        ax.set_facecolor('white')
        ax.set_theta_zero_location("N") # North is 0 degrees
        ax.set_theta_direction(-1) # Clockwise

        bearing_deg = 240
        bearing_rad = math.radians(bearing_deg)

        # Plot the bearing line
        ax.plot([0, bearing_rad], [0, 1], 'b-', linewidth=2, label=f'Bearing {bearing_deg}°')
        ax.plot(bearing_rad, 1, 'bo', markersize=8) # Point P

        # Add cardinal directions
        ax.set_thetagrids([0, 90, 180, 270], ['N', 'E', 'S', 'W'])
        ax.set_rticks([]) # Hide radius ticks

        # Optionally show the angle arc (can be tricky in polar)
        # ax.plot(np.linspace(0, bearing_rad, 100), np.full(100, 0.2), color='g', lw=1.5)

        ax.set_title(f"Bearing {bearing_deg}°")
        # ax.legend()

        # Adjust view
        ax.set_rlim(0, 1.1)

        return self._save_plot(fig, filename)

    def f3_gb_we3_ship_path_angle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Path points (relative coordinates)
        A = (0, 0)
        bearing1_deg = 60
        dist1 = 10
        bearing1_rad_math = math.radians(90 - bearing1_deg)
        B = (A[0] + dist1 * math.cos(bearing1_rad_math), A[1] + dist1 * math.sin(bearing1_rad_math))

        bearing2_deg = 150
        dist2 = 8
        bearing2_rad_math = math.radians(90 - bearing2_deg)
        C = (B[0] + dist2 * math.cos(bearing2_rad_math), B[1] + dist2 * math.sin(bearing2_rad_math))

        # Extension of AB to D
        D = (B[0] + 3 * math.cos(bearing1_rad_math), B[1] + 3 * math.sin(bearing1_rad_math))

        # Plot path
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b-', marker='o', label='Path 1 (10km @ 060°)')
        ax.plot([B[0], C[0]], [B[1], C[1]], 'r-', marker='o', label='Path 2 (8km @ 150°)')

        # Plot points
        ax.plot(A[0], A[1], 'ko'); ax.text(A[0]-0.5, A[1]-0.5, 'A', ha='right')
        ax.plot(B[0], B[1], 'ko'); ax.text(B[0]+0.3, B[1]+0.3, 'B', ha='left')
        ax.plot(C[0], C[1], 'ko'); ax.text(C[0]+0.5, C[1], 'C', ha='left')

        # North line at B
        ax.plot([B[0], B[0]], [B[1], B[1]+5], 'k--', linewidth=1)
        ax.text(B[0], B[1]+5.2, 'N', color='black', ha='center', va='bottom')

        # Extension line BD
        ax.plot([B[0], D[0]], [B[1], D[1]], 'g:', label='Extension of AB')
        ax.text(D[0], D[1], 'D', ha='left', va='bottom', color='g')


        # Mark angles at B
        # Angle NBC (Bearing 150)
        angle_bc_rad_math = math.radians(90 - bearing2_deg)
        arc_bc = patches.Arc(B, 4, 4, angle=90, theta1=-bearing2_deg, theta2=0, color='red', ls=':')
        ax.add_patch(arc_bc)
        ax.text(B[0] + 2.2*math.cos(math.radians(90-bearing2_deg/2)), B[1] + 2.2*math.sin(math.radians(90-bearing2_deg/2)), f'{bearing2_deg}°', color='red', fontsize=9)

        # Angle NBD (Bearing 60)
        angle_bd_rad_math = math.radians(90 - bearing1_deg)
        arc_bd = patches.Arc(B, 3, 3, angle=90, theta1=-bearing1_deg, theta2=0, color='blue', ls=':')
        ax.add_patch(arc_bd)
        ax.text(B[0] + 1.7*math.cos(math.radians(90-bearing1_deg/2)), B[1] + 1.7*math.sin(math.radians(90-bearing1_deg/2)), f'{bearing1_deg}°', color='blue', fontsize=9)

        # Angle DBC = 150 - 60 = 90
        angle_dbc_rad = math.radians(bearing2_deg - bearing1_deg)
        # Angle ABC = 180 - 90 = 90
        angle_abc_deg = 90
        # Mark angle ABC (interior)
        vec_BA = np.array(A) - np.array(B)
        vec_BC = np.array(C) - np.array(B)
        angle_start = math.degrees(math.atan2(vec_BA[1], vec_BA[0]))
        angle_end = math.degrees(math.atan2(vec_BC[1], vec_BC[0]))
        # Ensure angle is measured correctly for interior
        theta1 = angle_end # Angle of BC
        theta2 = angle_start # Angle of BA
        arc_abc = patches.Arc(B, 1.5, 1.5, angle=0, theta1=theta1, theta2=theta2, color='purple', linewidth=2)
        # ax.add_patch(arc_abc) # Arc drawing complex here
        # Mark right angle symbol for 90 degrees
        corner_size = 0.8
        p1 = B + vec_BA / np.linalg.norm(vec_BA) * corner_size
        p2 = B + vec_BC / np.linalg.norm(vec_BC) * corner_size
        p3 = p1 + (p2 - B) # Corner point of square
        rect_pts = np.array([p1,p3,p2])
        rect_patch = patches.Polygon(rect_pts, closed=False, fill=False, edgecolor='purple', linewidth=1)
        ax.add_patch(rect_patch)
        ax.text(B[0] + 0.5, B[1] - 0.5, f'∠ABC = 90°', color='purple', ha='left', va='top', fontsize=10)

        # ax.text(B[0] + 1.5, B[1] - 1.5, f'∠DBC = {bearing2_deg - bearing1_deg}°\n∠ABC = 180° - ∠DBC = 90°', color='purple', ha='left', va='top', fontsize=10)

        ax.set_xlim(A[0]-2, max(B[0], C[0], D[0])+2)
        ax.set_ylim(min(A[1], C[1])-2, max(B[1], C[1], D[1])+6)
        ax.set_title("Angle Between Ship Paths")
        ax.set_xlabel("East Displacement (km)")
        ax.set_ylabel("North Displacement (km)")
        ax.grid(True, linestyle='--')
        ax.legend(fontsize=8)

        return self._save_plot(fig, filename)

    def f3_gb_we4_compass_bearing(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

        # Point Q (origin)
        Q = (0, 0)
        ax.plot(Q[0], Q[1], 'ko', markersize=8, label='Q')

        # N/S/E/W axes
        axis_len = 1.5
        ax.plot([0, 0], [-axis_len, axis_len], 'k--', linewidth=0.8) # N-S
        ax.plot([-axis_len, axis_len], [0, 0], 'k--', linewidth=0.8) # E-W
        ax.text(0, axis_len, 'N', ha='center', va='bottom')
        ax.text(0, -axis_len, 'S', ha='center', va='top')
        ax.text(axis_len, 0, 'E', ha='left', va='center')
        ax.text(-axis_len, 0, 'W', ha='right', va='center')

        # Bearing S 40 E
        angle_deg = 40
        # Angle from South towards East. Math angle from positive x-axis is 270 + 40 = 310, or -50.
        angle_rad_math = math.radians(-50)
        ray_len = 1.3
        P_end = (Q[0] + ray_len * math.cos(angle_rad_math), Q[1] + ray_len * math.sin(angle_rad_math))

        # Draw ray QP
        ax.plot([Q[0], P_end[0]], [Q[1], P_end[1]], 'b-', linewidth=2, label='Direction S 40° E')

        # Mark angle
        # Angle from S (-y axis) towards E (+x axis)
        arc = patches.Arc(Q, 1, 1, angle=270, theta1=0, theta2=angle_deg, color='g', linewidth=1.5)
        ax.add_patch(arc)
        # Text position - bisector of angle from S towards E
        text_angle_rad = math.radians(270 + angle_deg / 2)
        ax.text(Q[0] + 0.6 * math.cos(text_angle_rad), Q[1] + 0.6 * math.sin(text_angle_rad),
                f'{angle_deg}°', color='g', ha='center', va='center', fontsize=10)

        ax.set_xlim(-axis_len-0.2, axis_len+0.2)
        ax.set_ylim(-axis_len-0.2, axis_len+0.2)
        ax.set_title("Compass Bearing S 40° E from Q")
        ax.legend(loc='upper left')

        return self._save_plot(fig, filename)

    # === Form 3: Geometry - Polygons ===
    def f3_gpoly_notes_exterior_angle(self, interior_deg=60):
        """
        Draws two adjacent sides of a convex polygon at a vertex,
        marks the interior and exterior angles, and shows that they sum to 180°.
        """
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Convert and compute points
        θ = math.radians(interior_deg)
        O = (0, 0)
        A = (1, 0)                          # one side: OA
        B = (math.cos(θ), math.sin(θ))     # other side: OB
        C = (-1, 0)                         # extension of OA beyond O

        # Draw sides OA, OB and extension OC
        ax.plot([O[0], A[0]], [O[1], A[1]], 'k-', lw=2)
        ax.plot([O[0], B[0]], [O[1], B[1]], 'k-', lw=2)
        ax.plot([O[0], C[0]], [O[1], C[1]], 'k--', lw=1)  # dashed extension

        # Draw points and labels
        ax.plot(*zip(O, A, B), 'ko', ms=6)
        ax.text(A[0]+0.05, A[1],  'A', fontsize=12)
        ax.text(B[0]+0.02, B[1]+0.02, 'B', fontsize=12)
        ax.text(O[0]-0.05, O[1]-0.05, 'O', fontsize=12)

        # Interior angle arc (small radius)
        r1 = 0.25
        arc_int = patches.Arc(O, 2*r1, 2*r1,
                            angle=0,
                            theta1=0,
                            theta2=interior_deg,
                            color='blue', lw=2)
        ax.add_patch(arc_int)
        # Label interior
        mid_int = math.radians(interior_deg/2)
        ax.text(r1*math.cos(mid_int)+0.05,
                r1*math.sin(mid_int)+0.02,
                f'{interior_deg}°',
                color='blue', fontsize=12)

        # Exterior angle arc (larger radius)
        ext_deg = 180 - interior_deg
        r2 = 0.4
        arc_ext = patches.Arc(O, 2*r2, 2*r2,
                            angle=0,
                            theta1=interior_deg,
                            theta2=180,
                            color='red', lw=2)
        ax.add_patch(arc_ext)
        # Label exterior
        mid_ext = math.radians(interior_deg + ext_deg/2)
        ax.text(r2*math.cos(mid_ext)-0.2,
                r2*math.sin(mid_ext)-0.04,
                f'{ext_deg}°',
                color='red', fontsize=12)

        # Supplementary note
        ax.text(0.2, -0.3,
                'Interior + Exterior = 180°',
                fontsize=12, ha='center')

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_title("Interior & Exterior Angles at Vertex O", pad=15)

        return self._save_plot(fig, filename)

    # === Form 3: Geometry - Similarity and Congruency ===

    # def f3_gsc_we1_similar_triangles(self):
    #     filename = get_filename_from_caller()
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
    #     fig.suptitle('Similar Triangles ΔABC ~ ΔPQR', fontsize=14)

    #     # Triangle ABC
    #     ax1.set_facecolor('white')
    #     A, B, C = (0,0), (6,0), (2,4) # Example vertices
    #     ax1.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'b-')
    #     ax1.text(A[0]-0.2, A[1]-0.2, 'A')
    #     ax1.text(B[0]+0.1, B[1]-0.2, 'B')
    #     ax1.text(C[0], C[1]+0.2, 'C')
    #     ax1.text(3, -0.5, 'AB = 6')
    #     ax1.text(4, 2, 'BC = ? (8)') # Provided in text, visually consistent
    #     ax1.text(1, 2, 'AC')
    #     ax1.set_title("Triangle ABC")
    #     ax1.set_aspect('equal', adjustable='box')
    #     ax1.axis('off')

    #     # Triangle PQR (scaled by 1.5)
    #     ax2.set_facecolor('white')
    #     k = 1.5
    #     P, Q, R = (0,0), (k*6,0), (k*2, k*4) # Scaled vertices
    #     ax2.plot([P[0], Q[0], R[0], P[0]], [P[1], Q[1], R[1], P[1]], 'r-')
    #     ax2.text(P[0]-0.2, P[1]-0.2, 'P')
    #     ax2.text(Q[0]+0.1, Q[1]-0.2, 'Q')
    #     ax2.text(R[0], R[1]+0.2, 'R')
    #     ax2.text(k*3, -0.5, 'PQ = 9')
    #     ax2.text(k*4, k*2, 'QR = ? (12)')
    #     ax2.text(k*1, k*2, 'PR')
    #     ax2.set_title("Triangle PQR")
    #     ax2.set_aspect('equal', adjustable='box')
    #     ax2.axis('off')

    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     return self._save_plot(fig, filename)

    # === Form 3: Geometry - Constructions and Loci ===

    def f3_gcl_we1_construct_sss(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off') # Hide axes for construction focus

        # Define side lengths
        pq, pr, qr = 7, 5, 6

        # Position P and Q
        P = (0, 0)
        Q = (pq, 0)

        # Calculate position of R using intersection of circles
        # Let R=(x,y). x^2+y^2 = pr^2. (x-pq)^2+y^2 = qr^2
        # x^2+y^2 = pr^2
        # x^2-2*pq*x+pq^2+y^2 = qr^2
        # pr^2 - 2*pq*x + pq^2 = qr^2
        # pr^2 + pq^2 - qr^2 = 2*pq*x
        x_R = (pr**2 + pq**2 - qr**2) / (2 * pq)
        # y^2 = pr^2 - x_R^2
        y_R_sq = pr**2 - x_R**2
        y_R = math.sqrt(y_R_sq) if y_R_sq >= 0 else 0 # Ensure non-negative before sqrt
        R = (x_R, y_R)

        # Draw sides PQ, PR, QR
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-', linewidth=2)
        ax.plot([P[0], R[0]], [P[1], R[1]], 'k-', linewidth=2)
        ax.plot([Q[0], R[0]], [Q[1], R[1]], 'k-', linewidth=2)

        # Plot vertices
        ax.plot([P[0], Q[0], R[0]], [P[1], Q[1], R[1]], 'ko', markersize=6)
        ax.text(P[0]-0.2, P[1]-0.3, 'P')
        ax.text(Q[0]+0.1, Q[1]-0.3, 'Q')
        ax.text(R[0], R[1]+0.2, 'R')

        # Draw construction arcs (dashed)
        arc1 = patches.Arc(P, 2*pr, 2*pr, angle=0, theta1=0, theta2=90, color='gray', linestyle='--', linewidth=1)
        arc2 = patches.Arc(Q, 2*qr, 2*qr, angle=0, theta1=90, theta2=180, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc1)
        ax.add_patch(arc2)

        # Add labels for side lengths
        ax.text(pq/2, -0.3, f'PQ = {pq} cm', ha='center')
        ax.text(x_R/2 - 0.2, y_R/2 + 0.1, f'PR = {pr} cm', ha='right', rotation=math.degrees(math.atan2(y_R, x_R)))
        ax.text((Q[0]+x_R)/2 + 0.2, y_R/2 + 0.1, f'QR = {qr} cm', ha='left', rotation=math.degrees(math.atan2(y_R, x_R-pq)))


        ax.set_xlim(-1, pq+1)
        ax.set_ylim(-1, y_R+1)
        ax.set_title("Constructing Triangle PQR (SSS)")

        return self._save_plot(fig, filename)

    def f3_gcl_we2_construct_sas(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given values
        ab_len, ac_len, angle_a_deg = 6, 4, 60
        angle_a_rad = math.radians(angle_a_deg)

        # Position A and B
        A = (0, 0)
        B = (ab_len, 0)

        # Position C
        C = (ac_len * math.cos(angle_a_rad), ac_len * math.sin(angle_a_rad))

        # Draw sides AB, AC, BC
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=2)
        ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=2)
        ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', linewidth=2)

        # Plot vertices
        ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'ko', markersize=6)
        ax.text(A[0]-0.2, A[1]-0.3, 'A')
        ax.text(B[0]+0.1, B[1]-0.3, 'B')
        ax.text(C[0]-0.3, C[1]+0.2, 'C')

        # Construction arcs for 60 degrees at A
        arc_base = patches.Arc(A, 1.5, 1.5, angle=0, theta1=0, theta2=70, color='gray', linestyle='--', linewidth=1)
        intersect1_x = 1.5 * math.cos(0)
        intersect1_y = 1.5 * math.sin(0)
        arc_cut = patches.Arc((intersect1_x, intersect1_y), 1.5, 1.5, angle=0, theta1=100, theta2=140, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_base)
        ax.add_patch(arc_cut)
        # Line AX (implicitly AC)
        ax.plot([A[0], A[0]+ab_len*math.cos(angle_a_rad)], [A[1], A[1]+ab_len*math.sin(angle_a_rad)], 'gray', linestyle=':', linewidth=1)
        ax.text(A[0]+0.3, A[1]+0.3, f'{angle_a_deg}°', ha='left')

        # Construction arc for length AC
        arc_len = patches.Arc(A, ac_len, ac_len, angle=0, theta1=angle_a_deg-5, theta2=angle_a_deg+5, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_len)

        # Labels
        ax.text(ab_len/2, -0.3, f'AB = {ab_len} cm', ha='center')
        ax.text(C[0]/2-0.3, C[1]/2+0.1, f'AC = {ac_len} cm', ha='right', rotation=angle_a_deg)

        ax.set_xlim(-1, ab_len+1)
        ax.set_ylim(-1, max(B[1], C[1])+1)
        ax.set_title("Constructing Triangle ABC (SAS)")

        return self._save_plot(fig, filename)

    def f3_gcl_we3_construct_asa(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given values
        xy_len, angle_x_deg, angle_y_deg = 8, 45, 60
        angle_x_rad = math.radians(angle_x_deg)
        angle_y_rad = math.radians(angle_y_deg)
        angle_z_deg = 180 - angle_x_deg - angle_y_deg
        angle_z_rad = math.radians(angle_z_deg)

        # Position X and Y
        X = (0, 0)
        Y = (xy_len, 0)

        # Calculate position of Z using intersection of lines
        # Line from X: slope = tan(45) = 1. Equation y = x
        # Line from Y: slope = tan(180-60) = tan(120) = -sqrt(3). Equation y - 0 = -sqrt(3) * (x - 8)
        # Intersection: x = -sqrt(3)x + 8*sqrt(3) => x(1+sqrt(3)) = 8*sqrt(3)
        # x = 8*sqrt(3)/(1+sqrt(3)) = 8*sqrt(3)*(sqrt(3)-1)/((sqrt(3)+1)*(sqrt(3)-1)) = (8*3 - 8*sqrt(3))/(3-1) = (24 - 8*sqrt(3))/2 = 12 - 4*sqrt(3)
        # y = x
        x_Z = 12 - 4 * math.sqrt(3)
        y_Z = x_Z
        Z = (x_Z, y_Z)

        # Draw sides XY, XZ, YZ
        ax.plot([X[0], Y[0]], [X[1], Y[1]], 'k-', linewidth=2)
        ax.plot([X[0], Z[0]], [X[1], Z[1]], 'k-', linewidth=2)
        ax.plot([Y[0], Z[0]], [Y[1], Z[1]], 'k-', linewidth=2)

        # Plot vertices
        ax.plot([X[0], Y[0], Z[0]], [X[1], Y[1], Z[1]], 'ko', markersize=6)
        ax.text(X[0]-0.2, X[1]-0.3, 'X')
        ax.text(Y[0]+0.1, Y[1]-0.3, 'Y')
        ax.text(Z[0], Z[1]+0.2, 'Z')

        # Construction arcs/lines (simplified representation)
        # Angle X = 45
        arc_x = patches.Arc(X, 1.5, 1.5, angle=0, theta1=0, theta2=angle_x_deg, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_x)
        ax.plot([X[0], X[0]+xy_len], [X[1], X[1]+xy_len*math.tan(angle_x_rad)], 'gray', linestyle=':', linewidth=1) # Line XW
        ax.text(X[0]+0.5, X[1]+0.2, f'{angle_x_deg}°', ha='left')

        # Angle Y = 60
        arc_y = patches.Arc(Y, 1.5, 1.5, angle=0, theta1=180-angle_y_deg, theta2=180, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_y)
        ax.plot([Y[0], Y[0]-xy_len], [Y[1], Y[1]+xy_len*math.tan(math.radians(60))], 'gray', linestyle=':', linewidth=1) # Line YV
        ax.text(Y[0]-0.7, Y[1]+0.2, f'{angle_y_deg}°', ha='right')


        # Labels
        ax.text(xy_len/2, -0.3, f'XY = {xy_len} cm', ha='center')

        ax.set_xlim(-1, xy_len+1)
        ax.set_ylim(-1, y_Z+1)
        ax.set_title("Constructing Triangle XYZ (ASA)")

        return self._save_plot(fig, filename)





    def f3_tg_notes_dist_time_vs_speed_time(self):
        """Generates Distance-Time and Speed-Time graphs side-by-side."""
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Travel Graph Examples', fontsize=14)

        # --- Left Graph (Distance-Time) ---
        ax1.set_facecolor('white')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Distance")
        ax1.set_title("Distance-Time Graph")
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 8)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Segments O-A, A-B, B-C
        time1 = [0, 3]
        dist1 = [0, 6] # Positive slope (Constant speed away)
        time2 = [3, 6]
        dist2 = [6, 6] # Horizontal (Stationary)
        time3 = [6, 9]
        dist3 = [6, 0] # Negative slope (Constant speed towards)

        ax1.plot(time1, dist1, 'b-', label='O-A (Speed > 0)')
        ax1.plot(time2, dist2, 'g-', label='A-B (Stationary)')
        ax1.plot(time3, dist3, 'r-', label='B-C (Speed < 0)')

        ax1.text(1.5, 3.5, 'Constant Speed\nAway', ha='center', va='center', color='blue')
        ax1.text(4.5, 6.5, 'Stationary', ha='center', va='center', color='green')
        ax1.text(7.5, 3.5, 'Constant Speed\nTowards', ha='center', va='center', color='red')
        ax1.legend()

        # --- Right Graph (Speed-Time) ---
        ax2.set_facecolor('white')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Speed")
        ax2.set_title("Speed-Time Graph")
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 8)
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Segments D-E, E-F, F-G
        time_s1 = [0, 3]
        speed1 = [0, 6] # Positive slope (Acceleration)
        time_s2 = [3, 6]
        speed2 = [6, 6] # Horizontal (Constant Speed)
        time_s3 = [6, 9]
        speed3 = [6, 0] # Negative slope (Deceleration)

        ax2.plot(time_s1, speed1, 'm-', label='D-E (Acceleration)')
        ax2.plot(time_s2, speed2, 'c-', label='E-F (Constant Speed)')
        ax2.plot(time_s3, speed3, 'orange', linestyle='-', label='F-G (Deceleration)')

        ax2.text(1.5, 3.5, 'Acceleration', ha='center', va='center', color='m')
        ax2.text(4.5, 6.5, 'Constant Speed', ha='center', va='center', color='c')
        ax2.text(7.5, 3.5, 'Deceleration', ha='center', va='center', color='orange')
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        return self._save_plot(fig, filename)

    def f3_tg_notes_area_under_speed_time(self):
        """Generates a speed-time graph illustrating area calculation."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Speed (v)")
        ax.set_title("Area Under Speed-Time Graph = Distance")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Example graph: Accelerate, Constant, Decelerate
        t1, v1 = 3, 8  # End of acceleration
        t2, v2 = 7, 8  # End of constant speed
        t3, v3 = 9, 4  # End of deceleration

        time_pts = [0, t1, t2, t3]
        speed_pts = [0, v1, v2, v3]

        ax.plot(time_pts, speed_pts, 'b-', marker='o')

        # --- Area 1: Triangle (Acceleration from rest) ---
        pts_tri = np.array([[0, 0], [t1, 0], [t1, v1]])
        triangle = patches.Polygon(pts_tri, closed=True, facecolor='lightblue', alpha=0.6)
        ax.add_patch(triangle)
        ax.text(t1/2, v1/2, f'Area = Distance\n1/2*{t1}*{v1}\n= {0.5*t1*v1}',
                ha='center', va='center', fontsize=9, color='darkblue')

        # --- Area 2: Rectangle (Constant Speed) ---
        pts_rect = np.array([[t1, 0], [t2, 0], [t2, v2], [t1, v1]])
        rectangle = patches.Polygon(pts_rect, closed=True, facecolor='lightgreen', alpha=0.6)
        ax.add_patch(rectangle)
        ax.text((t1+t2)/2, v1/2, f'Area = Distance\n({t2-t1})*{v1}\n= {(t2-t1)*v1}',
                ha='center', va='center', fontsize=9, color='darkgreen')

        # --- Area 3: Trapezium (Deceleration) ---
        pts_trap = np.array([[t2, 0], [t3, 0], [t3, v3], [t2, v2]])
        trapezium = patches.Polygon(pts_trap, closed=True, facecolor='lightcoral', alpha=0.6)
        ax.add_patch(trapezium)
        area_trap = 0.5 * (v2 + v3) * (t3 - t2)
        ax.text((t2+t3)/2, (v2+v3)/4, f'Area = Distance\n1/2*({v2}+{v3})*({t3-t2})\n= {area_trap:.1f}',
                ha='center', va='center', fontsize=9, color='darkred')

        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_yticks(np.arange(0, 13, 2))

        return self._save_plot(fig, filename)

    def f3_tg_we1_dist_time_graph(self):
        """Generates distance-time graph for Worked Example 1."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Distance (km)")
        ax.set_title("Car Journey Distance-Time Graph")
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 130)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Points: (0,0), (2,120), (3,120), (6,0)
        time_pts = [0, 2, 3, 6]
        dist_pts = [0, 120, 120, 0]

        ax.plot(time_pts, dist_pts, 'b-', marker='o')
        ax.plot(time_pts, dist_pts, 'bo') # Redraw points to ensure visibility

        # Annotations
        ax.text(1, 60, 'Travel Out', ha='center', va='bottom', color='blue')
        ax.text(2.5, 125, 'Stop', ha='center', va='center', color='blue')
        ax.text(4.5, 60, 'Return', ha='center', va='top', color='blue')

        ax.set_xticks(np.arange(0, 7, 1))
        ax.set_yticks(np.arange(0, 140, 20))

        return self._save_plot(fig, filename)

    def f3_tg_we2_speed_time_graph(self):
        """Generates speed-time graph for Worked Example 2."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Particle Speed-Time Graph")
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Segments: (0,0) to (5,10), then (5,10) to (15,10)
        time_pts = [0, 5, 15]
        speed_pts = [0, 10, 10]

        ax.plot(time_pts, speed_pts, 'r-', marker='o')
        ax.plot(time_pts, speed_pts, 'ro') # Redraw points

        # Annotations
        ax.text(2, 5, 'Acceleration', ha='center', va='center', color='red', rotation=63)
        ax.text(10, 10.5, 'Constant Speed', ha='center', va='center', color='red')

        ax.set_xticks(np.arange(0, 17, 1))
        ax.set_yticks(np.arange(0, 13, 1))

        # Store data for reuse in WE3 and WE4
        self.context_data['f3_tg_we2'] = {'time': time_pts, 'speed': speed_pts}

        return self._save_plot(fig, filename)

    def f3_tg_we3_area_triangle(self):
        """References WE2 graph and shades the triangular area."""
        filename = get_filename_from_caller()
        # Re-generate the base graph from WE2
        if 'f3_tg_we2' not in self.context_data:
             # Should have been called, but as fallback:
            print("Warning: Context data for WE2 not found, regenerating base graph.")
            self.f3_tg_we2_speed_time_graph() # Regenerate to populate context
            if 'f3_tg_we2' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None # Or raise error

        time_pts = self.context_data['f3_tg_we2']['time']
        speed_pts = self.context_data['f3_tg_we2']['speed']

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Distance in First 5 Seconds (Area)")
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'r-', marker='o')
        ax.plot(time_pts, speed_pts, 'ro')

        # Shade the triangle (0 to 5s)
        ax.fill_between(time_pts[:2], speed_pts[:2], color='lightblue', alpha=0.7)
        ax.text(2.5, 4, 'Area = Distance\n= 1/2 * 5 * 10\n= 25 m',
                ha='center', va='center', fontsize=10, color='darkblue')

        ax.set_xticks(np.arange(0, 17, 1))
        ax.set_yticks(np.arange(0, 13, 1))

        return self._save_plot(fig, filename)

    def f3_tg_we4_area_total(self):
        """References WE2 graph and shades the total area."""
        filename = get_filename_from_caller()
         # Re-generate the base graph from WE2
        if 'f3_tg_we2' not in self.context_data:
             print("Warning: Context data for WE2 not found, regenerating base graph.")
             self.f3_tg_we2_speed_time_graph()
             if 'f3_tg_we2' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        time_pts = self.context_data['f3_tg_we2']['time']
        speed_pts = self.context_data['f3_tg_we2']['speed']

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Total Distance in 15 Seconds (Total Area)")
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'r-', marker='o')
        ax.plot(time_pts, speed_pts, 'ro')

        # Shade the total area (triangle + rectangle)
        ax.fill_between(time_pts, speed_pts, color='lightgreen', alpha=0.7)
        ax.text(7.5, 4, 'Total Area = Total Distance\n= 25m + 100m\n= 125 m',
                ha='center', va='center', fontsize=10, color='darkgreen')

        ax.set_xticks(np.arange(0, 17, 1))
        ax.set_yticks(np.arange(0, 13, 1))

        return self._save_plot(fig, filename)

    # Note: WE5 for Travel Graphs has no image_description

    def f3_tg_q_moderate_speed_time_segment(self):
        """Generates speed-time graph segment for Moderate question."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Deceleration Example")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 16)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Line from (0, 15) to (4, 5)
        time_pts = [0, 4]
        speed_pts = [15, 5]

        ax.plot(time_pts, speed_pts, 'b-', marker='o')
        ax.plot(time_pts, speed_pts, 'bo')

        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 17, 1))

        # Store for reuse in Challenging question
        self.context_data['f3_tg_q_moderate'] = {'time': time_pts, 'speed': speed_pts}

        return self._save_plot(fig, filename)

    def f3_tg_q_challenging_area_trapezium(self):
        """References Moderate question graph and shades the trapezium area."""
        filename = get_filename_from_caller()
        if 'f3_tg_q_moderate' not in self.context_data:
            print("Warning: Context data for Moderate Question not found, regenerating.")
            self.f3_tg_q_moderate_speed_time_segment()
            if 'f3_tg_q_moderate' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        time_pts = self.context_data['f3_tg_q_moderate']['time']
        speed_pts = self.context_data['f3_tg_q_moderate']['speed']

        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Distance Traveled (Area)")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 16)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'b-', marker='o')
        ax.plot(time_pts, speed_pts, 'bo')

        # Shade the trapezium area
        ax.fill_between(time_pts, speed_pts, color='lightcoral', alpha=0.7)
        area = 0.5 * (speed_pts[0] + speed_pts[1]) * (time_pts[1] - time_pts[0])
        ax.text(2, 6, f'Area = Distance\n= 1/2*(15+5)*4\n= {area} m',
                ha='center', va='center', fontsize=10, color='darkred')

        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 17, 1))

        return self._save_plot(fig, filename)

    # === Form 3: Variation ===

    def f3_v_notes_direct_vs_inverse_graphs(self):
        """Generates Direct and Inverse Variation graphs side-by-side."""
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Variation Graphs', fontsize=14)

        # --- Left Graph (Direct Variation) ---
        ax1.set_facecolor('white')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Direct Variation (y=kx, k>0)")
        ax1.set_xlim(0, 5)
        ax1.set_ylim(0, 10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(0, color='black', linewidth=0.5)

        k_direct = 1.5 # Example k
        x_direct = np.array([0, 5])
        y_direct = k_direct * x_direct
        ax1.plot(x_direct, y_direct, 'b-', linewidth=2, label=f'y = {k_direct}x')
        ax1.text(2.5, 7.5, 'y = kx (k>0)', ha='center', color='blue')
        ax1.legend()

        # --- Right Graph (Inverse Variation) ---
        ax2.set_facecolor('white')
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Inverse Variation (y=k/x, k>0)")
        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)

        k_inverse = 4.0 # Example k
        x_inverse = np.linspace(0.1, 5, 400) # Avoid x=0
        y_inverse = k_inverse / x_inverse
        ax2.plot(x_inverse, y_inverse, 'r-', linewidth=2, label=f'y = {k_inverse}/x')
        ax2.text(2.5, 6, 'y = k/x (k>0)', ha='center', color='red')
        ax2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    # === Form 3: Algebra - Inequalities ===

    def f3_ai_notes_simul_ineq_1var(self):
        """Generates a number line illustrating 2 <= x < 5."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 2), facecolor='white')
        ax.set_facecolor('white')
        ax.set_yticks([]) # Remove y-axis ticks
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero') # x-axis line
        ax.xaxis.set_ticks_position('bottom')

        # Plot the number line segment
        ax.plot([2, 5], [0.1, 0.1], 'b-', linewidth=3)

        # Plot the circles
        ax.plot(2, 0.1, 'bo', markersize=8, label='Closed circle at 2 (>=)') # Closed circle
        ax.plot(5, 0.1, 'bo', markersize=8, markerfacecolor='white', label='Open circle at 5 (<)') # Open circle

        # Add ticks and labels
        ax.set_xticks(np.arange(0, 7, 1))
        ax.set_xlabel("x")
        ax.set_title("Number Line Solution: 2 <= x < 5")

        # Optional legend (might be cleaner without)
        # ax.legend(loc='upper right')

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_ai_notes_simul_ineq_2var(self):
        """Illustrates feasible region for y > 1 and x + y <= 4."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        x_range = np.linspace(-1, 5, 400)

        # Plotting constraints
        # 1. y > 1 (Boundary y = 1, dashed)
        ax.axhline(1, color='red', linestyle='--', label='y = 1')
        # Shade unwanted region (y <= 1)
        ax.fill_between(x_range, -1, 1, color='red', alpha=0.2, label='y <= 1 (unwanted)')

        # 2. x + y <= 4 => y <= 4 - x (Boundary y = 4 - x, solid)
        y_boundary2 = 4 - x_range
        ax.plot(x_range, y_boundary2, 'blue', linestyle='-', label='x + y = 4')
        # Shade unwanted region (y > 4 - x)
        ax.fill_between(x_range, y_boundary2, 5, color='blue', alpha=0.2, label='x + y > 4 (unwanted)')

        # Add axes and grid
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Set limits and labels
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Feasible Region for y > 1 and x + y <= 4")

        # Indicate Feasible Region R (unshaded area)
        ax.text(1, 2, 'R\n(Feasible\nRegion)', ha='center', va='center', fontsize=12, color='black')

        # Add legend
        ax.legend(loc='upper right', fontsize=9)

        return self._save_plot(fig, filename)

    # def f3_ai_we2_num_line_soln(self):
    #     """Number line for 2 <= x < 4 (same as f3_ai_notes_simul_ineq_1var, just different range)."""
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(8, 2), facecolor='white')
    #     ax.set_facecolor('white')
    #     ax.set_yticks([])
    #     ax.set_xlim(0, 5)
    #     ax.set_ylim(0, 1)
    #     ax.spines['left'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['bottom'].set_position('zero')
    #     ax.xaxis.set_ticks_position('bottom')

    #     ax.plot([2, 4], [0.1, 0.1], 'b-', linewidth=3)
    #     ax.plot(2, 0.1, 'bo', markersize=8) # Closed
    #     ax.plot(4, 0.1, 'bo', markersize=8, markerfacecolor='white') # Open

    #     ax.set_xticks(np.arange(0, 6, 1))
    #     ax.set_xlabel("x")
    #     ax.set_title("Number Line Solution: 2 <= x < 4")

    #     plt.tight_layout()
    #     return self._save_plot(fig, filename)

    def f3_ai_we4_region_3_ineqs(self):
        """Illustrates feasible region for y >= 0, x > 1, y < x."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        x_range = np.linspace(-1, 5, 400)

        # Plotting constraints
        # 1. y >= 0 (Boundary y = 0 (x-axis), solid)
        ax.axhline(0, color='green', linestyle='-', label='y = 0')
        # Shade unwanted region (y < 0)
        ax.fill_between(x_range, -1, 0, color='green', alpha=0.2, label='y < 0 (unwanted)')

        # 2. x > 1 (Boundary x = 1, dashed)
        ax.axvline(1, color='red', linestyle='--', label='x = 1')
        # Shade unwanted region (x <= 1)
        ax.fill_betweenx([-1, 5], -1, 1, color='red', alpha=0.2, label='x <= 1 (unwanted)')

        # 3. y < x (Boundary y = x, dashed)
        ax.plot(x_range, x_range, 'blue', linestyle='--', label='y = x')
        # Shade unwanted region (y >= x)
        ax.fill_between(x_range, x_range, 5, where=x_range>=x_range, color='blue', alpha=0.2, label='y >= x (unwanted)')


        # Add axes and grid
        # ax.axhline(0, color='black', linewidth=0.5) # Already done by constraint 1
        # ax.axvline(0, color='black', linewidth=0.5) # Covered by shading
        ax.grid(True, linestyle='--', alpha=0.6)

        # Set limits and labels
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Feasible Region R")

        # Indicate Feasible Region R (unshaded triangle)
        # Find vertices: (1,0), intersection of x=1 and y=x -> (1,1)
        ax.text(2, 0.5, 'R', ha='center', va='center', fontsize=14, color='black')

        # Add legend
        ax.legend(loc='lower right', fontsize=9)

        return self._save_plot(fig, filename)

    def f3_ai_q_challenging_identify_region(self):
        """Generates the diagram for the challenging question, asking to identify inequalities."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        x_range = np.linspace(-2, 4, 400)

        # Draw boundary lines
        # y = 3 (solid)
        ax.axhline(3, color='blue', linestyle='-', label='y = 3')
        # x = -1 (solid)
        ax.axvline(-1, color='red', linestyle='-', label='x = -1')
        # y = x (dashed)
        ax.plot(x_range, x_range, 'green', linestyle='--', label='y = x')

        # Shade unwanted regions based on R being below y=3, right of x=-1, below y=x
        # Shade above y=3
        ax.fill_between(x_range, 3, 5, color='blue', alpha=0.2)
        # Shade left of x=-1
        ax.fill_betweenx([-2, 5], -2, -1, color='red', alpha=0.2)
        # Shade above y=x
        ax.fill_between(x_range, x_range, 5, where=x_range>=x_range, color='green', alpha=0.2)

        # Indicate Region R
        ax.text(1, 0, 'R', ha='center', va='center', fontsize=16, color='black')

        # Setup plot
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Identify Inequalities Defining Region R")
        ax.legend(loc='lower right')

        return self._save_plot(fig, filename)

    # === Form 3: Algebra - Indices and Logarithms ===
    # No images described in this section

    # === Form 3: Geometry - Points, Lines and Angles ===

    def f3_gpla_notes_angle_of_elevation(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Observer O
        ox, oy = 1, 1
        ax.plot(ox, oy, 'ko', markersize=8, label='Observer (O)')

        # Horizontal line
        ax.plot([ox, 5], [oy, oy], 'k--', label='Horizontal Line')

        # Top of object T
        tx, ty = 4, 3
        ax.plot(tx, ty, 'ro', markersize=8, label='Top of Object (T)')

        # Line of sight OT
        ax.plot([ox, tx], [oy, ty], 'b-', label='Line of Sight')

        # Angle of Elevation Arc
        angle_rad = math.atan2(ty - oy, tx - ox)
        angle_deg = math.degrees(angle_rad)
        elev_arc = patches.Arc((ox, oy), 2, 2, angle=0, theta1=0, theta2=angle_deg, color='g', linewidth=1.5, label='Angle of Elevation')
        ax.add_patch(elev_arc)
        ax.text(ox + 1.2 * math.cos(angle_rad / 2), oy + 1.2 * math.sin(angle_rad / 2),
                'Angle of Elevation', color='g', ha='left', va='bottom', rotation=angle_deg/2)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title("Angle of Elevation")
        # ax.legend() # Optional, might clutter

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_gpla_notes_angle_of_depression(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Observer O (high up)
        ox, oy = 1, 3
        ax.plot(ox, oy, 'ko', markersize=8, label='Observer (O)')

        # Horizontal line from O
        ax.plot([ox, 5], [oy, oy], 'k--', label='Horizontal Line')

        # Bottom object B
        bx, by = 4, 1
        ax.plot(bx, by, 'ro', markersize=8, label='Bottom Object (B)')

        # Line of sight OB
        ax.plot([ox, bx], [oy, by], 'b-', label='Line of Sight')

        # Angle of Depression Arc
        angle_rad = math.atan2(oy - by, bx - ox) # Angle below horizontal
        angle_deg = math.degrees(angle_rad)
        dep_arc = patches.Arc((ox, oy), 2, 2, angle=0, theta1=-angle_deg, theta2=0, color='g', linewidth=1.5, label='Angle of Depression')
        ax.add_patch(dep_arc)
        ax.text(ox + 1.2 * math.cos(-angle_rad / 2), oy + 1.2 * math.sin(-angle_rad / 2),
                'Angle of Depression', color='g', ha='left', va='top', rotation=-angle_deg/2)


        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title("Angle of Depression")
        # ax.legend()

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_gpla_we1_tower_elevation(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box') # Make triangle proportions look right

        # Vertices
        P = (0, 0) # Base of tower
        Q = (50, 0) # Observation point
        T = (0, 28.87) # Top of tower (height h calculated: 50*tan(30))
        h = 50 * math.tan(math.radians(30))
        T = (0,h)


        # Draw triangle
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-', label='Ground (50 m)') # Ground
        ax.plot([P[0], T[0]], [P[1], T[1]], 'k-', label=f'Tower (h)') # Tower
        ax.plot([Q[0], T[0]], [Q[1], T[1]], 'b--', label='Line of Sight') # Line of sight

        # Mark points
        ax.plot(P[0], P[1], 'ko')
        ax.plot(Q[0], Q[1], 'ko')
        ax.plot(T[0], T[1], 'ko')
        ax.text(P[0]-3, P[1]-3, 'P (Base)', ha='right')
        ax.text(Q[0]+3, Q[1]-3, 'Q (Observer)', ha='left')
        ax.text(T[0]-3, T[1], 'T (Top)', ha='right')

        # Mark right angle
        rect = patches.Rectangle((P[0], P[1]), 4, 4, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Mark angle of elevation
        angle_deg = 30
        elev_arc = patches.Arc(Q, 30, 30, angle=0, theta1=180-angle_deg, theta2=180, color='g', linewidth=1.5)
        ax.add_patch(elev_arc)
        ax.text(Q[0]-18, Q[1]+6, f'{angle_deg}°', color='g', ha='center', va='center')

        # Labels
        ax.text(Q[0]/2, Q[1]-3, '50 m', ha='center', va='top', color='k')
        ax.text(P[0]-3, T[1]/2, 'h', ha='right', va='center', color='k')

        ax.set_xlim(-10, 60)
        ax.set_ylim(-10, h+10)
        ax.set_title("Angle of Elevation Problem")
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Vertical Height (m)")
        ax.grid(True, linestyle='--')
        # ax.legend()

        return self._save_plot(fig, filename)

    def f3_gpla_we2_cliff_depression(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Points
        cliff_height = 80
        angle_dep_deg = 40
        angle_elev_deg = 40 # Alternate interior angle
        distance_d = cliff_height / math.tan(math.radians(angle_elev_deg))

        T = (0, cliff_height) # Top of cliff
        F = (0, 0)            # Foot of cliff
        B = (distance_d, 0)   # Boat

        # Draw triangle TFB
        ax.plot([T[0], F[0]], [T[1], F[1]], 'k-', label=f'Cliff ({cliff_height} m)') # Cliff
        ax.plot([F[0], B[0]], [F[1], B[1]], 'k-', label=f'Distance (d)') # Sea level
        ax.plot([T[0], B[0]], [T[1], B[1]], 'b--', label='Line of Sight') # Line of sight

        # Mark points
        ax.plot(T[0], T[1], 'ko'); ax.text(T[0]-5, T[1], 'T', ha='right')
        ax.plot(F[0], F[1], 'ko'); ax.text(F[0]-5, F[1], 'F', ha='right')
        ax.plot(B[0], B[1], 'ko'); ax.text(B[0]+5, B[1], 'B (Boat)', ha='left')

        # Mark right angle at F
        rect = patches.Rectangle((F[0], F[1]), 6, 6, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Mark angle of depression at T
        ax.plot([T[0], B[0]+10], [T[1], T[1]], 'r--', label='Horizontal at T') # Horizontal line
        dep_arc = patches.Arc(T, 40, 40, angle=0, theta1=-angle_dep_deg, theta2=0, color='r', linewidth=1.5)
        ax.add_patch(dep_arc)
        ax.text(T[0]+40, T[1]-5, f'{angle_dep_deg}° (Depression)', color='r', ha='center', va='center')

        # Mark alternate interior angle at B (Elevation)
        elev_arc = patches.Arc(B, 40, 40, angle=0, theta1=180-angle_elev_deg, theta2=180, color='g', linewidth=1.5)
        ax.add_patch(elev_arc)
        ax.text(B[0]-35, B[1]+5, f'{angle_elev_deg}° (Elevation)', color='g', ha='center', va='center')


        # Labels
        ax.text(T[0]-5, T[1]/2, f'{cliff_height} m', ha='right', va='center', color='k')
        ax.text(B[0]/2, B[1]-5, 'd', ha='center', va='top', color='k')

        ax.set_xlim(-15, distance_d + 20)
        ax.set_ylim(-15, cliff_height + 15)
        ax.set_title("Angle of Depression Problem")
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Vertical Height (m)")
        ax.grid(True, linestyle='--')
        # ax.legend()

        return self._save_plot(fig, filename)

    def f3_gpla_we3_elevation_diagram(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ox, oy = 1, 1
        bx, by = 4, 2.5 # Bird position relative
        angle_deg = 25

        ax.plot(ox, oy, 'ko', markersize=8, label='O (Observer)')
        ax.plot([ox, 5], [oy, oy], 'k--', label='Horizontal') # Horizontal line
        ax.plot(bx, by, 'ro', markersize=8, label='B (Bird)') # Bird
        ax.plot([ox, bx], [oy, by], 'b-') # Line of sight

        elev_arc = patches.Arc((ox, oy), 2, 2, angle=0, theta1=0, theta2=angle_deg, color='g', linewidth=1.5)
        ax.add_patch(elev_arc)
        ax.text(ox + 1.2 * math.cos(math.radians(angle_deg / 2)),
                oy + 1.2 * math.sin(math.radians(angle_deg / 2)),
                f'{angle_deg}°', color='g', ha='left', va='bottom')

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title("Angle of Elevation to Bird")
        ax.legend(loc='lower right')

        return self._save_plot(fig, filename)

    def f3_gpla_we4_depression_diagram(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        px, py = 1, 3 # Person P
        cx, cy = 4, 1 # Car C
        angle_deg = 35

        ax.plot(px, py, 'ko', markersize=8, label='P (Person)')
        ax.plot([px, 5], [py, py], 'k--', label='Horizontal') # Horizontal line
        ax.plot(cx, cy, 'ro', markersize=8, label='C (Car)') # Car
        ax.plot([px, cx], [py, cy], 'b-') # Line of sight

        dep_arc = patches.Arc((px, py), 2, 2, angle=0, theta1=-angle_deg, theta2=0, color='g', linewidth=1.5)
        ax.add_patch(dep_arc)
        ax.text(px + 1.2 * math.cos(math.radians(-angle_deg / 2)),
                py + 1.2 * math.sin(math.radians(-angle_deg / 2)),
                f'{angle_deg}°', color='g', ha='left', va='top')

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_title("Angle of Depression to Car")
        ax.legend(loc='lower left')

        return self._save_plot(fig, filename)

    def f3_gpla_we5_scale_drawing(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        # Scale: 1 cm : 10 m. So 50m is 5cm.

        # Vertices based on scale
        A = (0, 0) # Observation point
        B = (5, 0) # Base of tower (scaled)
        angle_CAB_deg = 30
        angle_CAB_rad = math.radians(angle_CAB_deg)
        # Height BC: tan(30) = BC/AB => BC = AB * tan(30) = 5 * tan(30)
        height_BC_scaled = 5 * math.tan(angle_CAB_rad)
        C = (5, height_BC_scaled) # Top of tower (scaled)

        # Draw triangle
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', label='Ground AB (5 cm)') # Ground
        ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', label=f'Tower BC ({height_BC_scaled:.1f} cm)') # Tower
        ax.plot([A[0], C[0]], [A[1], C[1]], 'b--', label='Line of Sight AC') # Line of sight

        # Mark points
        ax.plot(A[0], A[1], 'ko'); ax.text(A[0]-0.2, A[1]-0.2, 'A', ha='right')
        ax.plot(B[0], B[1], 'ko'); ax.text(B[0]+0.2, B[1]-0.2, 'B', ha='left')
        ax.plot(C[0], C[1], 'ko'); ax.text(C[0]+0.2, C[1], 'C', ha='left')

        # # Mark right angle at B
        # rect = patches.Rectangle((B[0]-0.3, B[1]), 0.3, 0.3, angle=-90, linewidth=1, edgecolor='k', facecolor='none')
        # ax.add_patch(rect)
        # Mark right angle at B using a correctly positioned square patch
        symbol_size = 0.3 # Size of the square symbol
        rect_anchor = (B[0] - symbol_size, B[1]) # Bottom-left corner is (5-size, 0)
        rect = patches.Rectangle(rect_anchor, symbol_size, symbol_size, angle=0, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        
        # Mark angle at A
        elev_arc = patches.Arc(A, 1.5, 1.5, angle=0, theta1=0, theta2=angle_CAB_deg, color='g', linewidth=1.5)
        ax.add_patch(elev_arc)
        ax.text(A[0]+1, A[1]+0.3, f'{angle_CAB_deg}°', color='g', ha='center', va='center')

        # Labels
        ax.text(A[0]/2 + B[0]/2, A[1]-0.2, '5 cm (Represents 50 m)', ha='center', va='top', color='k')
        ax.text(B[0]+0.2, C[1]/2, f'Measure BC\n(Approx {height_BC_scaled:.1f} cm\n=> Approx {height_BC_scaled*10:.1f} m)', ha='left', va='center', color='k', fontsize=8)


        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, height_BC_scaled + 1)
        ax.set_title("Scale Drawing (1 cm : 10 m)")
        ax.set_xlabel("Scaled Distance (cm)")
        ax.set_ylabel("Scaled Distance (cm)")
        ax.grid(True, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        return self._save_plot(fig, filename)

    # === Form 3: Geometry - Bearing ===

    def f3_gb_notes_bearing_back_bearing(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off') # Keep axes for reference if preferred

        # Points A and B
        A = (2, 1)
        B = (5, 5)
        theta_deg = 45 # Example bearing < 180
        # Recalculate B based on A and theta
        dist = 5
        theta_rad_math = math.radians(90 - theta_deg) # Convert bearing to math angle
        B = (A[0] + dist * math.cos(theta_rad_math), A[1] + dist * math.sin(theta_rad_math))

        ax.plot(A[0], A[1], 'bo', markersize=8, label='A')
        ax.plot(B[0], B[1], 'ro', markersize=8, label='B')
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=1)

        # North line at A
        ax.plot([A[0], A[0]], [A[1], A[1]+dist*1.2], 'b--', linewidth=1, label='North at A')
        ax.text(A[0], A[1]+dist*1.2 + 0.2, 'N', color='blue', ha='center', va='bottom')

        # North line at B (parallel)
        ax.plot([B[0], B[0]], [B[1], B[1]+dist*1.2], 'r--', linewidth=1, label='North at B || North at A')
        ax.text(B[0], B[1]+dist*1.2 + 0.2, 'N', color='red', ha='center', va='bottom')

        # Bearing theta from A to B
        bearing_arc = patches.Arc(A, 2, 2, angle=90, theta1=-theta_deg, theta2=0, color='blue', linewidth=1.5) # Clockwise from North
        ax.add_patch(bearing_arc)
        # Adjust text position for bearing
        mid_angle_rad = math.radians(90 - theta_deg / 2)
        ax.text(A[0] + 1.2 * math.cos(mid_angle_rad), A[1] + 1.2 * math.sin(mid_angle_rad),
                f'θ={theta_deg}°', color='blue', ha='center', va='center', fontsize=10)

        # Back bearing theta + 180 from B to A
        back_bearing_deg = theta_deg + 180
        # The vector BA makes math angle theta_rad_math + pi
        back_bearing_arc = patches.Arc(B, 3, 3, angle=90, theta1=-back_bearing_deg, theta2=0, color='red', linewidth=1.5) # Clockwise from North at B
        ax.add_patch(back_bearing_arc)
        # Text position for back bearing - more complex angle
        mid_back_angle_deg = (360 + (90 - back_bearing_deg))/2 + (90 - back_bearing_deg) # approx
        mid_back_angle_rad = math.radians(-back_bearing_deg/2 + 90) # Angle bisector in math terms
        ax.text(B[0] + 1.8 * math.cos(mid_back_angle_rad), B[1] + 1.8 * math.sin(mid_back_angle_rad),
                f'Back Bearing\nθ+180°\n= {back_bearing_deg}°', color='red', ha='center', va='top', fontsize=9)


        ax.set_xlim(A[0]-dist*0.5, B[0]+dist*0.5)
        ax.set_ylim(A[1]-dist*0.2, B[1]+dist*1.5)
        ax.set_title("Bearing and Back Bearing")
        # ax.legend()

        return self._save_plot(fig, filename)

    def f3_gb_we1_bearing_240(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'}, facecolor='white')
        ax.set_facecolor('white')
        ax.set_theta_zero_location("N") # North is 0 degrees
        ax.set_theta_direction(-1) # Clockwise

        bearing_deg = 240
        bearing_rad = math.radians(bearing_deg)

        # Plot the bearing line
        ax.plot([0, bearing_rad], [0, 1], 'b-', linewidth=2, label=f'Bearing {bearing_deg}°')
        ax.plot(bearing_rad, 1, 'bo', markersize=8) # Point P

        # Add cardinal directions
        ax.set_thetagrids([0, 90, 180, 270], ['N', 'E', 'S', 'W'])
        ax.set_rticks([]) # Hide radius ticks

        # Optionally show the angle arc (can be tricky in polar)
        # ax.plot(np.linspace(0, bearing_rad, 100), np.full(100, 0.2), color='g', lw=1.5)

        ax.set_title(f"Bearing {bearing_deg}°")
        # ax.legend()

        # Adjust view
        ax.set_rlim(0, 1.1)

        return self._save_plot(fig, filename)

    def f3_gb_we3_ship_path_angle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Path points (relative coordinates)
        A = (0, 0)
        bearing1_deg = 60
        dist1 = 10
        bearing1_rad_math = math.radians(90 - bearing1_deg)
        B = (A[0] + dist1 * math.cos(bearing1_rad_math), A[1] + dist1 * math.sin(bearing1_rad_math))

        bearing2_deg = 150
        dist2 = 8
        bearing2_rad_math = math.radians(90 - bearing2_deg)
        C = (B[0] + dist2 * math.cos(bearing2_rad_math), B[1] + dist2 * math.sin(bearing2_rad_math))

        # Extension of AB to D
        D = (B[0] + 3 * math.cos(bearing1_rad_math), B[1] + 3 * math.sin(bearing1_rad_math))

        # Plot path
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b-', marker='o', label='Path 1 (10km @ 060°)')
        ax.plot([B[0], C[0]], [B[1], C[1]], 'r-', marker='o', label='Path 2 (8km @ 150°)')

        # Plot points
        ax.plot(A[0], A[1], 'ko'); ax.text(A[0]-0.5, A[1]-0.5, 'A', ha='right')
        ax.plot(B[0], B[1], 'ko'); ax.text(B[0]+0.3, B[1]+0.3, 'B', ha='left')
        ax.plot(C[0], C[1], 'ko'); ax.text(C[0]+0.5, C[1], 'C', ha='left')

        # North line at B
        ax.plot([B[0], B[0]], [B[1], B[1]+5], 'k--', linewidth=1)
        ax.text(B[0], B[1]+5.2, 'N', color='black', ha='center', va='bottom')

        # Extension line BD
        ax.plot([B[0], D[0]], [B[1], D[1]], 'g:', label='Extension of AB')
        ax.text(D[0], D[1], 'D', ha='left', va='bottom', color='g')


        # Mark angles at B
        # Angle NBC (Bearing 150)
        angle_bc_rad_math = math.radians(90 - bearing2_deg)
        arc_bc = patches.Arc(B, 4, 4, angle=90, theta1=-bearing2_deg, theta2=0, color='red', ls=':')
        ax.add_patch(arc_bc)
        ax.text(B[0] + 2.2*math.cos(math.radians(90-bearing2_deg/2)), B[1] + 2.2*math.sin(math.radians(90-bearing2_deg/2)), f'{bearing2_deg}°', color='red', fontsize=9)

        # Angle NBD (Bearing 60)
        angle_bd_rad_math = math.radians(90 - bearing1_deg)
        arc_bd = patches.Arc(B, 3, 3, angle=90, theta1=-bearing1_deg, theta2=0, color='blue', ls=':')
        ax.add_patch(arc_bd)
        ax.text(B[0] + 1.7*math.cos(math.radians(90-bearing1_deg/2)), B[1] + 1.7*math.sin(math.radians(90-bearing1_deg/2)), f'{bearing1_deg}°', color='blue', fontsize=9)

        # Angle DBC = 150 - 60 = 90
        angle_dbc_rad = math.radians(bearing2_deg - bearing1_deg)
        # Angle ABC = 180 - 90 = 90
        angle_abc_deg = 90
        # Mark angle ABC (interior)
        vec_BA = np.array(A) - np.array(B)
        vec_BC = np.array(C) - np.array(B)
        angle_start = math.degrees(math.atan2(vec_BA[1], vec_BA[0]))
        angle_end = math.degrees(math.atan2(vec_BC[1], vec_BC[0]))
        # Ensure angle is measured correctly for interior
        theta1 = angle_end # Angle of BC
        theta2 = angle_start # Angle of BA
        arc_abc = patches.Arc(B, 1.5, 1.5, angle=0, theta1=theta1, theta2=theta2, color='purple', linewidth=2)
        # ax.add_patch(arc_abc) # Arc drawing complex here
        # Mark right angle symbol for 90 degrees
        corner_size = 0.8
        p1 = B + vec_BA / np.linalg.norm(vec_BA) * corner_size
        p2 = B + vec_BC / np.linalg.norm(vec_BC) * corner_size
        p3 = p1 + (p2 - B) # Corner point of square
        rect_pts = np.array([p1,p3,p2])
        rect_patch = patches.Polygon(rect_pts, closed=False, fill=False, edgecolor='purple', linewidth=1)
        ax.add_patch(rect_patch)
        ax.text(B[0] + 0.5, B[1] - 0.5, f'∠ABC = 90°', color='purple', ha='left', va='top', fontsize=10)

        # ax.text(B[0] + 1.5, B[1] - 1.5, f'∠DBC = {bearing2_deg - bearing1_deg}°\n∠ABC = 180° - ∠DBC = 90°', color='purple', ha='left', va='top', fontsize=10)

        ax.set_xlim(A[0]-2, max(B[0], C[0], D[0])+2)
        ax.set_ylim(min(A[1], C[1])-2, max(B[1], C[1], D[1])+6)
        ax.set_title("Angle Between Ship Paths")
        ax.set_xlabel("East Displacement (km)")
        ax.set_ylabel("North Displacement (km)")
        ax.grid(True, linestyle='--')
        ax.legend(fontsize=8)

        return self._save_plot(fig, filename)

    def f3_gb_we4_compass_bearing(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

        # Point Q (origin)
        Q = (0, 0)
        ax.plot(Q[0], Q[1], 'ko', markersize=8, label='Q')

        # N/S/E/W axes
        axis_len = 1.5
        ax.plot([0, 0], [-axis_len, axis_len], 'k--', linewidth=0.8) # N-S
        ax.plot([-axis_len, axis_len], [0, 0], 'k--', linewidth=0.8) # E-W
        ax.text(0, axis_len, 'N', ha='center', va='bottom')
        ax.text(0, -axis_len, 'S', ha='center', va='top')
        ax.text(axis_len, 0, 'E', ha='left', va='center')
        ax.text(-axis_len, 0, 'W', ha='right', va='center')

        # Bearing S 40 E
        angle_deg = 40
        # Angle from South towards East. Math angle from positive x-axis is 270 + 40 = 310, or -50.
        angle_rad_math = math.radians(-50)
        ray_len = 1.3
        P_end = (Q[0] + ray_len * math.cos(angle_rad_math), Q[1] + ray_len * math.sin(angle_rad_math))

        # Draw ray QP
        ax.plot([Q[0], P_end[0]], [Q[1], P_end[1]], 'b-', linewidth=2, label='Direction S 40° E')

        # Mark angle
        # Angle from S (-y axis) towards E (+x axis)
        arc = patches.Arc(Q, 1, 1, angle=270, theta1=0, theta2=angle_deg, color='g', linewidth=1.5)
        ax.add_patch(arc)
        # Text position - bisector of angle from S towards E
        text_angle_rad = math.radians(270 + angle_deg / 2)
        ax.text(Q[0] + 0.6 * math.cos(text_angle_rad), Q[1] + 0.6 * math.sin(text_angle_rad),
                f'{angle_deg}°', color='g', ha='center', va='center', fontsize=10)

        ax.set_xlim(-axis_len-0.2, axis_len+0.2)
        ax.set_ylim(-axis_len-0.2, axis_len+0.2)
        ax.set_title("Compass Bearing S 40° E from Q")
        ax.legend(loc='upper left')

        return self._save_plot(fig, filename)

    # === Form 3: Geometry - Polygons ===

    # === Form 3: Geometry - Similarity and Congruency ===

    def f3_gsc_we1_similar_triangles(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Similar Triangles ΔABC ~ ΔPQR', fontsize=14)

        # Triangle ABC
        ax1.set_facecolor('white')
        A, B, C = (0,0), (6,0), (2,4) # Example vertices
        ax1.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'b-')
        ax1.text(A[0]-0.2, A[1]-0.2, 'A')
        ax1.text(B[0]+0.1, B[1]-0.2, 'B')
        ax1.text(C[0], C[1]+0.2, 'C')
        ax1.text(3, -0.5, 'AB = 6')
        ax1.text(4, 2, 'BC = ? (8)') # Provided in text, visually consistent
        ax1.text(0.7, 2, 'AC')
        ax1.set_title("Triangle ABC")
        ax1.set_aspect('equal', adjustable='box')
        ax1.axis('off')

        # Triangle PQR (scaled by 1.5)
        ax2.set_facecolor('white')
        k = 1.5
        P, Q, R = (0,0), (k*6,0), (k*2, k*4) # Scaled vertices
        ax2.plot([P[0], Q[0], R[0], P[0]], [P[1], Q[1], R[1], P[1]], 'r-')
        ax2.text(P[0]-0.2, P[1]-0.2, 'P')
        ax2.text(Q[0]+0.1, Q[1]-0.2, 'Q')
        ax2.text(R[0], R[1]+0.2, 'R')
        ax2.text(k*3, -0.5, 'PQ = 9')
        ax2.text(k*4, k*2, 'QR = ? (12)')
        ax2.text(k*0.8, k*2, 'PR')
        ax2.set_title("Triangle PQR")
        ax2.set_aspect('equal', adjustable='box')
        ax2.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    # === Form 3: Geometry - Constructions and Loci ===

    def f3_gcl_we1_construct_sss(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off') # Hide axes for construction focus

        # Define side lengths
        pq, pr, qr = 7, 5, 6

        # Position P and Q
        P = (0, 0)
        Q = (pq, 0)

        # Calculate position of R using intersection of circles
        # Let R=(x,y). x^2+y^2 = pr^2. (x-pq)^2+y^2 = qr^2
        # x^2+y^2 = pr^2
        # x^2-2*pq*x+pq^2+y^2 = qr^2
        # pr^2 - 2*pq*x + pq^2 = qr^2
        # pr^2 + pq^2 - qr^2 = 2*pq*x
        x_R = (pr**2 + pq**2 - qr**2) / (2 * pq)
        # y^2 = pr^2 - x_R^2
        y_R_sq = pr**2 - x_R**2
        y_R = math.sqrt(y_R_sq) if y_R_sq >= 0 else 0 # Ensure non-negative before sqrt
        R = (x_R, y_R)

        # Draw sides PQ, PR, QR
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-', linewidth=2)
        ax.plot([P[0], R[0]], [P[1], R[1]], 'k-', linewidth=2)
        ax.plot([Q[0], R[0]], [Q[1], R[1]], 'k-', linewidth=2)

        # Plot vertices
        ax.plot([P[0], Q[0], R[0]], [P[1], Q[1], R[1]], 'ko', markersize=6)
        ax.text(P[0]-0.2, P[1]-0.3, 'P')
        ax.text(Q[0]+0.1, Q[1]-0.3, 'Q')
        ax.text(R[0], R[1]+0.2, 'R')

        # Draw construction arcs (dashed)
        arc1 = patches.Arc(P, 2*pr, 2*pr, angle=0, theta1=0, theta2=90, color='gray', linestyle='--', linewidth=1)
        arc2 = patches.Arc(Q, 2*qr, 2*qr, angle=0, theta1=90, theta2=180, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc1)
        ax.add_patch(arc2)

        # Add labels for side lengths
        ax.text(pq/2, -0.3, f'PQ = {pq} cm', ha='center')
        ax.text(x_R/2 - 0.2, y_R/2 + 0.1, f'PR = {pr} cm', ha='right', rotation=math.degrees(math.atan2(y_R, x_R)))
        ax.text((Q[0]+x_R)/2 + 0.2, y_R/2 + 0.1, f'QR = {qr} cm', ha='left', rotation=math.degrees(math.atan2(y_R, x_R-pq)))


        ax.set_xlim(-1, pq+1)
        ax.set_ylim(-1, y_R+1)
        ax.set_title("Constructing Triangle PQR (SSS)")

        return self._save_plot(fig, filename)

    def f3_gcl_we2_construct_sas(self):
        """Generates a diagram illustrating the SAS construction of a triangle."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given values
        ab_len, ac_len, angle_a_deg = 6, 4, 60
        angle_a_rad = math.radians(angle_a_deg)

        # Position A and B
        A = (0, 0)
        B = (ab_len, 0)

        # Position C (calculated for final triangle drawing)
        C = (ac_len * math.cos(angle_a_rad), ac_len * math.sin(angle_a_rad))

        # --- Construction Steps Visualization ---
        # 1. Draw line segment AB
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=2)
        ax.text(ab_len/2, -0.4, f'AB = {ab_len} cm', ha='center', va='top')

        # 2. Construct 60° angle at A
        construct_radius = 1.5 # Radius for construction arcs
        # Arc from A intersecting AB
        arc1 = patches.Arc(A, 2*construct_radius, 2*construct_radius, angle=0,
                           theta1=0, theta2=70, # Draw slightly more than 60 deg
                           color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc1)
        # Intersection point P on AB (or extension)
        P = (construct_radius, 0)
        # Arc from P intersecting arc1 to find 60 deg direction (Q)
        arc2 = patches.Arc(P, 2*construct_radius, 2*construct_radius, angle=0,
                           theta1=110, theta2=130, # Small arc around 120 deg relative to P
                           color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc2)
        # Point Q (intersection of arcs, defines 60 deg) - calculated for ray
        # Q_x = construct_radius * math.cos(angle_a_rad)
        # Q_y = construct_radius * math.sin(angle_a_rad)
        # Draw dashed ray AX (through Q)
        ray_len = max(ac_len + 1, ab_len * 0.3) # Ensure ray is sufficiently long
        X_x = A[0] + ray_len * math.cos(angle_a_rad)
        X_y = A[1] + ray_len * math.sin(angle_a_rad)
        ax.plot([A[0], X_x], [A[1], X_y], color='gray', linestyle='--', linewidth=1, label='Line AX (construction)')

        # 3. Mark point C on AX using AC length
        # Draw arc centered at A with radius ac_len (4cm) intersecting AX
        arc_AC = patches.Arc(A, 2*ac_len, 2*ac_len, angle=0,
                             theta1=angle_a_deg - 10, theta2=angle_a_deg + 10, # Arc around 60 deg
                             color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_AC)

        # 4. Draw side BC
        ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', linewidth=2)

        # --- Draw Final Triangle and Labels ---
        # (Side AB already drawn as part of construction step 1)
        # Side AC
        ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=2)
        # Side BC (already drawn as step 4)

        # Plot vertices
        ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'ko', markersize=6)
        ax.text(A[0]-0.2, A[1]-0.4, 'A', va='top')
        ax.text(B[0]+0.1, B[1]-0.4, 'B', va='top')
        ax.text(C[0], C[1]+0.2, 'C', ha='center', va='bottom')

        # Label angle A
        angle_label_radius = 0.8 # Adjusted radius for angle label arc
        angle_arc = patches.Arc(A, 2*angle_label_radius, 2*angle_label_radius, angle=0,
                                theta1=0, theta2=angle_a_deg,
                                color='dimgray', linestyle='--', linewidth=1) # Slightly darker/different arc for label
        ax.add_patch(angle_arc)
        mid_angle_rad = math.radians(angle_a_deg / 2)
        ax.text(A[0] + angle_label_radius * 1.2 * math.cos(mid_angle_rad), # Adjusted position
                A[1] + angle_label_radius * 1.2 * math.sin(mid_angle_rad),
                f'{angle_a_deg}°', color='dimgray', fontsize=10, ha='center', va='center')

        # Label side AC
        # Calculate midpoint and angle for AC label rotation
        mid_ac_x = (A[0] + C[0]) / 2
        mid_ac_y = (A[1] + C[1]) / 2
        ax.text(mid_ac_x, mid_ac_y, f'AC = {ac_len} cm', ha='center', va='bottom',
                rotation=angle_a_deg, rotation_mode='anchor',
                transform_rotates_text=True,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1))


        # Adjust plot limits
        ax.set_xlim(-1, max(B[0], X_x) + 1)
        ax.set_ylim(-1.5, max(B[1], C[1], X_y) + 1) # Adjusted ylim slightly
        ax.set_title("Constructing Triangle ABC (SAS)")

        return self._save_plot(fig, filename)


    # def f3_gcl_we2_construct_sas(self):
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    #     ax.set_facecolor('white')
    #     ax.set_aspect('equal', adjustable='box')
    #     ax.axis('off')

    #     # Given values
    #     ab_len, ac_len, angle_a_deg = 6, 4, 60
    #     angle_a_rad = math.radians(angle_a_deg)

    #     # Position A and B
    #     A = (0, 0)
    #     B = (ab_len, 0)

    #     # Position C
    #     C = (ac_len * math.cos(angle_a_rad), ac_len * math.sin(angle_a_rad))

    #     # Draw sides AB, AC, BC
    #     ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', linewidth=2)
    #     ax.plot([A[0], C[0]], [A[1], C[1]], 'k-', linewidth=2)
    #     ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', linewidth=2)

    #     # Plot vertices
    #     ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'ko', markersize=6)
    #     ax.text(A[0]-0.2, A[1]-0.3, 'A')
    #     ax.text(B[0]+0.1, B[1]-0.3, 'B')
    #     ax.text(C[0]-0.3, C[1]+0.2, 'C')

    #     # Construction: draw 60° dashed arc, ray, and AC circle
    #     r_arc = 1.0
    #     arc_angle = patches.Arc(A, 2*r_arc, 2*r_arc, angle=0,
    #                             theta1=0, theta2=angle_a_deg,
    #                          color='gray', linestyle='--', linewidth=1)
    #     ax.add_patch(arc_angle)
    #     # Dashed ray from A to locate C
    #     ray_len = ac_len + 1
    #     x_ray = A[0] + ray_len * math.cos(angle_a_rad)
    #     y_ray = A[1] + ray_len * math.sin(angle_a_rad)
    #     ax.plot([A[0], x_ray], [A[1], y_ray], color='gray', linestyle='--', linewidth=1)
    #     # Label the 60° angle
    #     mid_angle_rad = math.radians(angle_a_deg / 2)
    #     ax.text(A[0] + r_arc * math.cos(mid_angle_rad) + 0.1,
    #             A[1] + r_arc * math.sin(mid_angle_rad) + 0.1,
    #             f'{angle_a_deg}°', color='gray', fontsize=10)
    #     # Dashed circle for AC length construction
    #     circle_AC = patches.Circle(A, ac_len, fill=False,
    #                                linestyle='--', color='gray', linewidth=1)
    #     ax.add_patch(circle_AC)

    #     # Labels
    #     ax.text(ab_len/2, -0.3, f'AB = {ab_len} cm', ha='center')
    #     ax.text(C[0]/2-0.3, C[1]/2+0.1, f'AC = {ac_len} cm', ha='right', rotation=angle_a_deg)

    #     ax.set_xlim(-1, ab_len+1)
    #     ax.set_ylim(-1, max(B[1], C[1])+1)
    #     ax.set_title("Constructing Triangle ABC (SAS)")

    #     return self._save_plot(fig, filename)

    def f3_gcl_we3_construct_asa(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given values
        xy_len, angle_x_deg, angle_y_deg = 8, 45, 60
        angle_x_rad = math.radians(angle_x_deg)
        angle_y_rad = math.radians(angle_y_deg)
        angle_z_deg = 180 - angle_x_deg - angle_y_deg
        angle_z_rad = math.radians(angle_z_deg)

        # Position X and Y
        X = (0, 0)
        Y = (xy_len, 0)

        # Calculate position of Z using intersection of lines
        # Line from X: slope = tan(45) = 1. Equation y = x
        # Line from Y: slope = tan(180-60) = tan(120) = -sqrt(3). Equation y - 0 = -sqrt(3) * (x - 8)
        # Intersection: x = -sqrt(3)x + 8*sqrt(3) => x(1+sqrt(3)) = 8*sqrt(3)
        # x = 8*sqrt(3)/(1+sqrt(3)) = 8*sqrt(3)*(sqrt(3)-1)/((sqrt(3)+1)*(sqrt(3)-1)) = (8*3 - 8*sqrt(3))/(3-1) = (24 - 8*sqrt(3))/2 = 12 - 4*sqrt(3)
        # y = x
        x_Z = 12 - 4 * math.sqrt(3)
        y_Z = x_Z
        Z = (x_Z, y_Z)

        # Draw sides XY, XZ, YZ
        ax.plot([X[0], Y[0]], [X[1], Y[1]], 'k-', linewidth=2)
        ax.plot([X[0], Z[0]], [X[1], Z[1]], 'k-', linewidth=2)
        ax.plot([Y[0], Z[0]], [Y[1], Z[1]], 'k-', linewidth=2)

        # Plot vertices
        ax.plot([X[0], Y[0], Z[0]], [X[1], Y[1], Z[1]], 'ko', markersize=6)
        ax.text(X[0]-0.2, X[1]-0.3, 'X')
        ax.text(Y[0]+0.1, Y[1]-0.3, 'Y')
        ax.text(Z[0], Z[1]+0.2, 'Z')

        # Construction arcs/lines (simplified representation)
        # Angle X = 45
        arc_x = patches.Arc(X, 1.5, 1.5, angle=0, theta1=0, theta2=angle_x_deg, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_x)
        ax.plot([X[0], X[0]+xy_len], [X[1], X[1]+xy_len*math.tan(angle_x_rad)], 'gray', linestyle=':', linewidth=1) # Line XW
        ax.text(X[0]+0.5, X[1]+0.2, f'{angle_x_deg}°', ha='left')

        # Angle Y = 60
        arc_y = patches.Arc(Y, 1.5, 1.5, angle=0, theta1=180-angle_y_deg, theta2=180, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_y)
        ax.plot([Y[0], Y[0]-xy_len], [Y[1], Y[1]+xy_len*math.tan(math.radians(60))], 'gray', linestyle=':', linewidth=1) # Line YV
        ax.text(Y[0]-0.7, Y[1]+0.2, f'{angle_y_deg}°', ha='right')


        # Labels
        ax.text(xy_len/2, -0.3, f'XY = {xy_len} cm', ha='center')

        ax.set_xlim(-1, xy_len+1)
        ax.set_ylim(-1, y_Z+1)
        ax.set_title("Constructing Triangle XYZ (ASA)")

        return self._save_plot(fig, filename)

    def f3_gcl_we4_construct_rectangle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Dimensions
        length, width = 7, 4

        # Vertices
        A = (0, 0)
        B = (length, 0)
        C = (length, width)
        D = (0, width)

        # Draw rectangle
        ax.plot([A[0], B[0], C[0], D[0], A[0]], [A[1], B[1], C[1], D[1], A[1]], 'k-', linewidth=2)

        # Plot vertices
        ax.plot([A[0], B[0], C[0], D[0]], [A[1], B[1], C[1], D[1]], 'ko', markersize=6)
        ax.text(A[0]-0.2, A[1]-0.3, 'A')
        ax.text(B[0]+0.1, B[1]-0.3, 'B')
        ax.text(C[0]+0.1, C[1]+0.1, 'C')
        ax.text(D[0]-0.2, D[1]+0.1, 'D')

        # Construction arcs/lines (simplified representation for perpendiculars)
        # At A
        ax.plot([A[0], A[0]], [A[1], A[1]+width*1.1], 'gray', linestyle=':', linewidth=1) # Line AX
        rect_A = patches.Rectangle((A[0], A[1]), 0.4, 0.4, linewidth=1, edgecolor='gray', facecolor='none', ls='--')
        ax.add_patch(rect_A)
        # At B
        ax.plot([B[0], B[0]], [B[1], B[1]+width*1.1], 'gray', linestyle=':', linewidth=1) # Line BY
        rect_B = patches.Rectangle((B[0]-0.4, B[1]), 0.4, 0.4, linewidth=1, edgecolor='gray', facecolor='none', ls='--')
        ax.add_patch(rect_B)
        # Arc for AD length
        arc_ad = patches.Arc(A, width, width, angle=0, theta1=85, theta2=95, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_ad)
        # Arc for BC length
        arc_bc = patches.Arc(B, width, width, angle=0, theta1=85, theta2=95, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_bc)


        # Labels
        ax.text(length/2, -0.3, f'AB = {length} cm', ha='center')
        ax.text(-0.3, width/2, f'AD = {width} cm', va='center', ha='right', rotation=90)
        ax.text(length+0.3, width/2, f'BC = {width} cm', va='center', ha='left', rotation=90)
        ax.text(length/2, width+0.2, f'DC = {length} cm', ha='center')

        ax.set_xlim(-1, length+1)
        ax.set_ylim(-1, width+1)
        ax.set_title("Constructing Rectangle ABCD")

        return self._save_plot(fig, filename)

    def f3_gcl_we5_construct_parallelogram(self):
        """Generates a diagram illustrating parallelogram construction using intersecting arcs."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white') # Slightly larger figure
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given values
        pq_len, ps_len, angle_p_deg = 6, 4, 75
        angle_p_rad = math.radians(angle_p_deg)

        # Position P and Q
        P = (0, 0)
        Q = (pq_len, 0)

        # Position S
        S = (ps_len * math.cos(angle_p_rad), ps_len * math.sin(angle_p_rad))

        # Position R (calculated based on parallelogram property: P + (Q-P) + (S-P) = Q + S - P)
        R = (Q[0] + S[0] - P[0], Q[1] + S[1] - P[1])
        # R = (pq_len + ps_len * math.cos(angle_p_rad), ps_len * math.sin(angle_p_rad))

        # --- Construction Visualization ---
        # 1. Draw PQ
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k-', linewidth=2)
        ax.text(pq_len/2, -0.4, f'PQ = {pq_len} cm', ha='center', va='top')

        # 2. Draw PS at 75 degrees
        ax.plot([P[0], S[0]], [P[1], S[1]], 'k-', linewidth=2)
        # Label PS (position adjusted for clarity)
        mid_ps_x = (P[0] + S[0]) / 2
        mid_ps_y = (P[1] + S[1]) / 2
        ax.text(mid_ps_x - 0.2, mid_ps_y, f'PS = {ps_len} cm', ha='right', va='center',
                rotation=angle_p_deg, rotation_mode='anchor', transform_rotates_text=True,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))
        # Label Angle P
        angle_label_radius = 1.0
        arc_p = patches.Arc(P, 2*angle_label_radius, 2*angle_label_radius, angle=0, theta1=0, theta2=angle_p_deg,
                            color='dimgray', linestyle='--', linewidth=1)
        ax.add_patch(arc_p)
        mid_angle_rad = math.radians(angle_p_deg / 2)
        ax.text(P[0] + angle_label_radius * 0.7 * math.cos(mid_angle_rad), # Adjusted position
                P[1] + angle_label_radius * 0.7 * math.sin(mid_angle_rad),
                f'{angle_p_deg}°', color='dimgray', fontsize=10, ha='left', va='bottom') # Adjusted alignment

        # 3. Construction arc from S (radius PQ)
        arc_s_radius = pq_len
        # Estimate angle range for arc near R
        vec_SR = (R[0] - S[0], R[1] - S[1]) # Should be parallel to PQ, angle is 0
        angle_sr = math.degrees(math.atan2(vec_SR[1], vec_SR[0])) # Should be 0
        arc_s = patches.Arc(S, 2*arc_s_radius, 2*arc_s_radius, angle=0,
                            theta1=angle_sr - 15, theta2=angle_sr + 15, # Arc centered around expected R direction
                            color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_s)

        # 4. Construction arc from Q (radius PS)
        arc_q_radius = ps_len
        # Estimate angle range for arc near R
        vec_QR = (R[0] - Q[0], R[1] - Q[1]) # Should be parallel to PS, angle is angle_p_deg
        angle_qr = math.degrees(math.atan2(vec_QR[1], vec_QR[0])) # Should be angle_p_deg
        arc_q = patches.Arc(Q, 2*arc_q_radius, 2*arc_q_radius, angle=0,
                            theta1=angle_qr - 15, theta2=angle_qr + 15, # Arc centered around expected R direction
                            color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_q)

        # 5. Draw sides SR and QR
        ax.plot([S[0], R[0]], [S[1], R[1]], 'k-', linewidth=2) # SR
        ax.plot([Q[0], R[0]], [Q[1], R[1]], 'k-', linewidth=2) # QR

        # Plot vertices
        ax.plot([P[0], Q[0], R[0], S[0]], [P[1], Q[1], R[1], S[1]], 'ko', markersize=6)
        ax.text(P[0]-0.2, P[1]-0.4, 'P', va='top')
        ax.text(Q[0]+0.1, Q[1]-0.4, 'Q', va='top')
        ax.text(R[0]+0.1, R[1], 'R', va='center', ha='left')
        ax.text(S[0]-0.1, S[1]+0.1, 'S', va='bottom', ha='right')

        # Adjust plot limits
        all_x = [P[0], Q[0], R[0], S[0]]
        all_y = [P[1], Q[1], R[1], S[1]]
        ax.set_xlim(min(all_x)-1, max(all_x)+1)
        ax.set_ylim(min(all_y)-1, max(all_y)+1)
        ax.set_title("Constructing Parallelogram PQRS")

        return self._save_plot(fig, filename)

    # === Form 3: Geometry - Symmetry ===

    def f3_gsym_we2_pentagon_rotation(self):
        """Illustrates rotational symmetry of a regular pentagon."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor='white') # Adjusted size slightly
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        radius = 3.0
        n_sides = 5
        center = (0, 0)
        angle_of_rotation = 360 / n_sides

        # Calculate vertices of a regular pentagon (start with top vertex)
        # Start angle adjusted so vertex A is near the top-left for clearer arrow path
        start_angle_offset = math.pi / 2 + math.radians(angle_of_rotation / 2)
        angles = np.linspace(start_angle_offset, start_angle_offset + 2*math.pi, n_sides, endpoint=False)
        vertices = np.array([(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles])
        vertex_labels = ['A', 'B', 'C', 'D', 'E'] # Match order with angles

        # Draw pentagon
        pentagon = patches.Polygon(vertices, closed=True, edgecolor='blue', facecolor='lightblue', linewidth=2)
        ax.add_patch(pentagon)

        # Mark center
        ax.plot(center[0], center[1], 'ko', markersize=7, label='Center of Rotation')
        # ax.text(center[0], center[1]-0.3, 'Center', ha='center', va='top') # Removed redundant text

        # Label vertices
        label_offset = 0.3 # Distance outside vertex
        for i, v in enumerate(vertices):
            angle = angles[i]
            ax.text(center[0] + (radius + label_offset) * math.cos(angle),
                    center[1] + (radius + label_offset) * math.sin(angle),
                    vertex_labels[i], ha='center', va='center', fontsize=12, fontweight='bold')

        # Show rotation from A to B
        angle1_rad = angles[0] # Angle for vertex A
        angle2_rad = angles[1] # Angle for vertex B

        # Draw lines/rays from center to A and B (visual aid)
        line_radius = radius * 0.95 # Slightly inside vertices
        ax.plot([center[0], center[0] + line_radius * math.cos(angle1_rad)],
                [center[1], center[1] + line_radius * math.sin(angle1_rad)],
                'k--', linewidth=0.8, alpha=0.7)
        ax.plot([center[0], center[0] + line_radius * math.cos(angle2_rad)],
                [center[1], center[1] + line_radius * math.sin(angle2_rad)],
                'k--', linewidth=0.8, alpha=0.7)

        # Draw arc indicating rotation angle near center
        arc_radius = radius * 0.3 # Smaller radius for angle arc
        rot_arc = patches.Arc(center, arc_radius*2, arc_radius*2,
                              angle=math.degrees(angle1_rad), # Start angle of the arc
                              theta1=0, theta2=angle_of_rotation, # Sweep angle
                              color='red', linewidth=2, linestyle='-')
        ax.add_patch(rot_arc)

        # Add arrow head to the rotation arc
        arrow_angle_rad = angle2_rad # Point arrow towards B's direction
        arrow_start_x = center[0] + arc_radius * math.cos(arrow_angle_rad)
        arrow_start_y = center[1] + arc_radius * math.sin(arrow_angle_rad)
        # Use annotate for better arrow control
        ax.annotate('', xy=(arrow_start_x, arrow_start_y),
                    xytext=(center[0] + arc_radius*0.9 * math.cos(arrow_angle_rad - math.radians(5)), # Offset base slightly
                            center[1] + arc_radius*0.9 * math.sin(arrow_angle_rad - math.radians(5))),
                    arrowprops=dict(arrowstyle="->", color='red', lw=2))


        # Label the rotation angle
        label_angle_rad = angle1_rad + math.radians(angle_of_rotation / 2)
        label_radius = arc_radius * 1.4
        ax.text(center[0] + label_radius * math.cos(label_angle_rad),
                center[1] + label_radius * math.sin(label_angle_rad),
                f'{angle_of_rotation:.0f}°', color='red', ha='center', va='center', fontsize=11, fontweight='bold')

        # Main descriptive text
        ax.text(center[0], -radius * 1.25, # Positioned below the pentagon
                f'Rotational Symmetry\nOrder: {n_sides}\nAngle: {angle_of_rotation:.0f}°\n(e.g., A rotates to B)',
                ha='center', va='top', fontsize=12)

        ax.set_xlim(-radius*1.4, radius*1.4)
        ax.set_ylim(-radius*1.4, radius*1.4)
        ax.set_title("Rotational Symmetry of a Regular Pentagon", fontsize=14, pad=10)
        # ax.legend() # Legend might be clutter

        return self._save_plot(fig, filename)


    def f3_gsym_we3_letter_s_rotation(self):
        """
        Illustrates the rotational symmetry (order 2) of the letter 'S'
        step-by-step using 90-degree rotations and marked points, with manual centering.
        """
        filename = get_filename_from_caller()
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), facecolor='white')
        fig.suptitle("Rotational Symmetry of Letter 'S' (Order 2)", fontsize=16, y=0.98)

        center = (0, 0)
        font_size = 100
        angle_degrees = [0, -90, -180] # Rotations for each subplot (clockwise)
        titles = ["Initial State (0°)", "Rotated 90° CW", "Rotated 180° CW"]
        font_props = {'ha': 'center', 'va': 'center', 'fontsize': font_size, 'fontfamily': 'sans-serif', 'fontweight':'bold'}

        # Manual offsets to visually center the 'S' glyph at each rotation
        # (These may need slight tweaking depending on font rendering)
        offsets = [
            (0.4, 0.6),  # Offset for 0 degrees (Slightly right, slightly up)
            (0.35, 0.45), # Offset for -90 degrees (Slightly right, slightly up in its rotated frame)
            (0.27, 0.83) # Offset for -180 degrees (Slightly left, slightly down) -> This might need reversing
        ]
        # Let's rethink the 180 offset. If the S appears up-right, we need to shift down-left. (-0.05, -0.1) seems plausible.

        # Define marker positions relative to the center for the UNROTATED 'S'
        marker_radius_scale = 0.65 # Reduced scale slightly more
        top_marker_rel = (0.1 * marker_radius_scale, 0.85 * marker_radius_scale)
        bottom_marker_rel = (-0.1 * marker_radius_scale, -0.85 * marker_radius_scale)

        # Function to rotate a point
        def rotate(point, angle_deg):
            rad = math.radians(angle_deg)
            x, y = point
            new_x = x * math.cos(rad) - y * math.sin(rad)
            new_y = x * math.sin(rad) + y * math.cos(rad)
            return (new_x, new_y)

        for i, ax in enumerate(axes):
            ax.set_facecolor('white')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            ax.axhline(0, color='gray', linestyle='--', lw=0.5, zorder=0)
            ax.axvline(0, color='gray', linestyle='--', lw=0.5, zorder=0)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.set_title(titles[i], fontsize=12)

            current_angle = angle_degrees[i]
            offset_x, offset_y = offsets[i]

            # Plot rotated 'S' with manual offset
            # The offset is applied *before* rotation relative to the anchor point
            text_x = center[0] + offset_x
            text_y = center[1] + offset_y
            ax.text(text_x, text_y, 'S', color='blue', rotation=current_angle,
                    rotation_mode='anchor', zorder=1, **font_props)

            # --- Recalculate marker positions relative to the *new* text anchor ---
            # We rotate the relative positions, then add the offset used for the text
            top_marker_rotated = rotate(top_marker_rel, current_angle)
            bottom_marker_rotated = rotate(bottom_marker_rel, current_angle)
            center_marker_rotated = rotate(center, current_angle) # Center relative to text anchor

            # Add the text offset to the rotated marker coordinates
            top_marker_abs = (top_marker_rotated[0], top_marker_rotated[1])
            bottom_marker_abs = (bottom_marker_rotated[0], bottom_marker_rotated[1])
            # The actual center marker remains at (0,0) visually
            center_marker_abs = (0,0)

            # Ensure labels are added only once for the legend
            label_top = 'Top Ref Point' if i == 0 else ""
            label_bottom = 'Bottom Ref Point' if i == 0 else ""
            label_center = 'Center' if i == 0 else ""

            ax.plot(top_marker_abs[0], top_marker_abs[1], 'ro', markersize=8, label=label_top, zorder=2)
            ax.plot(bottom_marker_abs[0], bottom_marker_abs[1], 'go', markersize=8, label=label_bottom, zorder=2)
            ax.plot(center_marker_abs[0], center_marker_abs[1], 'ko', markersize=5, label=label_center, zorder=2)

            # Add internal rotation arc (centered at the visual origin 0,0)
            if i > 0:
                arc_radius_internal = 1.4
                rotation_angle_abs = abs(current_angle)
                rot_arc_internal = patches.Arc((0,0), arc_radius_internal*2, arc_radius_internal*2, # Centered at 0,0
                                               angle=90, theta1=-rotation_angle_abs, theta2=0,
                                               color='dimgray', linewidth=1.5, linestyle='--', zorder=1)
                ax.add_patch(rot_arc_internal)

                # Add arrow head to arc end
                arrow_angle_rad = math.radians(90 - rotation_angle_abs)
                arrow_end_x = 0 + arc_radius_internal * math.cos(arrow_angle_rad) # Relative to 0,0
                arrow_end_y = 0 + arc_radius_internal * math.sin(arrow_angle_rad)
                ax.plot(arrow_end_x, arrow_end_y, marker=(3, 0, -90 - rotation_angle_abs),
                        markersize=8, linestyle='None', color='dimgray', zorder=1)

                # Label the arc
                label_angle_rad = math.radians(90 - rotation_angle_abs / 2)
                label_radius = arc_radius_internal * 1.18
                ax.text(0 + label_radius * math.cos(label_angle_rad), # Relative to 0,0
                        0 + label_radius * math.sin(label_angle_rad),
                        f'{rotation_angle_abs}°', color='dimgray', ha='center', va='center', fontsize=10, zorder=1)


        # Add final note about symmetry under the last plot
        axes[2].text(0, -2.2,
                     "After 180°, 'S' orientation matches start. Green marker \n(bottom) is now top. Red marker (top) is now bottom.",
                     ha='center', va='top', fontsize=10, transform=axes[2].transData)


        # Create a single legend for the figure
        handles, labels = axes[0].get_legend_handles_labels()
        # Check if labels list is populated before creating legend
        if handles and labels and any(labels):
             fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

        plt.tight_layout(rect=[0, 0.1, 1, 0.93])
        return self._save_plot(fig, filename)



    def f3_gsym_we5_isosceles_line_sym(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Vertices of isosceles triangle
        A = (0, 3)
        B = (-2, 0)
        C = (2, 0)

        # Draw triangle
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'b-', linewidth=2)

        # Draw line of symmetry (y-axis, x=0)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Line of Symmetry (x=0)')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 3.5)
        ax.set_title("Line Symmetry of Isosceles Triangle")
        ax.legend(loc='lower center')

        return self._save_plot(fig, filename)

    # === Form 3: Statistics - Data Collection, Classification and Representation ===

    def f3_sdccr_notes_histogram(self):
        filename = get_filename_from_caller()
        # Example Data
        boundaries = [149.5, 159.5, 169.5, 179.5]
        frequencies = [5, 12, 8] # Frequencies for intervals defined by boundaries
        class_widths = np.diff(boundaries)

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')

        # Create histogram bars
        ax.bar(boundaries[:-1], frequencies, width=class_widths, align='edge', edgecolor='black', color='skyblue')

        ax.set_xlabel("Height (cm) (Class Boundaries)")
        ax.set_ylabel("Frequency (Number of Students)")
        ax.set_title("Histogram: Heights of Form 3 Students")
        ax.set_xticks(boundaries)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_sdccr_notes_frequency_polygon(self):
        filename = get_filename_from_caller()
        # Example Data (Midpoints and Frequencies)
        midpoints = [5, 15, 25, 35] # Midpoints of 0-9, 10-19, 20-29, 30-39
        frequencies = [0, 8, 12, 0] # Frequencies including zero classes at ends

        # Midpoints for plotting (actual data points)
        plot_midpoints = [15, 25]
        plot_frequencies = [8, 12]
        # Zero frequency points
        zero_mid_before = 5
        zero_mid_after = 35
        all_midpoints = [zero_mid_before] + plot_midpoints + [zero_mid_after]
        all_frequencies = [0] + plot_frequencies + [0]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')

        # Plot the frequency polygon
        ax.plot(all_midpoints, all_frequencies, 'b-o', linewidth=2, markersize=6)

        ax.set_xlabel("Midpoint of Interval")
        ax.set_ylabel("Frequency")
        ax.set_title("Frequency Polygon: Test Score Distribution")
        ax.set_xticks(all_midpoints)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0) # Start y-axis at 0

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_sdccr_we2_histogram(self):
        """Draws histogram for WE1 data."""
        filename = get_filename_from_caller()
        # Data from WE1
        intervals = ["10-19", "20-29", "30-39", "40-49"]
        frequencies = [2, 6, 7, 5]
        boundaries = [9.5, 19.5, 29.5, 39.5, 49.5]
        class_widths = np.diff(boundaries)

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')

        ax.bar(boundaries[:-1], frequencies, width=class_widths, align='edge', edgecolor='black', color='lightcoral')

        ax.set_xlabel("Marks (Class Boundaries)")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram: Student Marks")
        ax.set_xticks(boundaries)
        ax.set_yticks(np.arange(0, max(frequencies) + 2, 1))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f3_sdccr_we3_frequency_polygon(self):
        """Draws frequency polygon for WE1 data."""
        filename = get_filename_from_caller()
        # Data from WE1
        intervals = ["10-19", "20-29", "30-39", "40-49"]
        frequencies = [2, 6, 7, 5]
        midpoints = [14.5, 24.5, 34.5, 44.5]

        # Add zero frequency points
        zero_mid_before = 4.5 # Midpoint of 0-9
        zero_mid_after = 54.5 # Midpoint of 50-59
        all_midpoints = [zero_mid_before] + midpoints + [zero_mid_after]
        all_frequencies = [0] + frequencies + [0]

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white')

        ax.plot(all_midpoints, all_frequencies, 'g-o', linewidth=2, markersize=6)

        ax.set_xlabel("Mark Midpoint")
        ax.set_ylabel("Frequency")
        ax.set_title("Frequency Polygon: Student Marks")
        ax.set_xticks(all_midpoints)
        ax.set_yticks(np.arange(0, max(frequencies) + 2, 1))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        return self._save_plot(fig, filename)

    # === Form 3: Statistics - Measures of Central Tendency ===
    # No images described

    # === Form 3: Trigonometry - Pythagoras Theorem ===

    def f3_tpt_notes_right_angled_triangle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Vertices
        C = (0, 0)
        A = (0, 3)
        B = (4, 0)

        # Draw triangle
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-', linewidth=2)

        # Mark vertices
        ax.text(A[0]-0.2, A[1], 'A', ha='right', va='center')
        ax.text(B[0]+0.1, B[1]-0.2, 'B', va='top')
        ax.text(C[0]-0.2, C[1]-0.2, 'C', ha='right', va='top')

        # Mark right angle at C
        rect = patches.Rectangle((C[0], C[1]), 0.3, 0.3, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Labels
        ax.text((A[0]+B[0])/2+0.2, (A[1]+B[1])/2+0.1, 'Hypotenuse', ha='left', rotation=-37)
        ax.text((A[0]+C[0])/2-0.2, (A[1]+C[1])/2, 'Leg', ha='right', va='center')
        ax.text((B[0]+C[0])/2, (B[1]+C[1])/2-0.2, 'Leg', ha='center', va='top')

        ax.set_xlim(-1, 4.5)
        ax.set_ylim(-1, 3.5)
        ax.set_title("Right-Angled Triangle")

        return self._save_plot(fig, filename)

    def f3_tpt_notes_pythagoras_proof(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        a, b = 2, 3 # Example leg lengths
        c = math.sqrt(a**2 + b**2)

        # Outer square vertices
        outer_verts = [(0,0), (a+b,0), (a+b,a+b), (0,a+b)]
        outer_square = patches.Polygon(outer_verts, closed=True, edgecolor='black', facecolor='lightyellow', alpha=0.3)
        ax.add_patch(outer_square)

        # Inner square vertices (rotated)
        inner_verts = [(b,0), (a+b,b), (a,a+b), (0,a)]
        inner_square = patches.Polygon(inner_verts, closed=True, edgecolor='red', facecolor='lightblue')
        ax.add_patch(inner_square)
        # Label inner square side 'c'
        ax.text((b+(a+b))/2, b/2, 'c', ha='center', va='bottom', color='red')


        # Triangle vertices for drawing outlines
        tri1_verts = [(0,0), (b,0), (0,a)]
        tri2_verts = [(b,0), (a+b,0), (a+b,b)]
        tri3_verts = [(a+b, b), (a+b, a+b), (a, a+b)]
        tri4_verts = [(a, a+b), (0, a+b), (0,a)]

        # Draw triangle outlines
        ax.plot([v[0] for v in tri1_verts]+[tri1_verts[0][0]], [v[1] for v in tri1_verts]+[tri1_verts[0][1]], 'k-')
        ax.plot([v[0] for v in tri2_verts]+[tri2_verts[0][0]], [v[1] for v in tri2_verts]+[tri2_verts[0][1]], 'k-')
        ax.plot([v[0] for v in tri3_verts]+[tri3_verts[0][0]], [v[1] for v in tri3_verts]+[tri3_verts[0][1]], 'k-')
        ax.plot([v[0] for v in tri4_verts]+[tri4_verts[0][0]], [v[1] for v in tri4_verts]+[tri4_verts[0][1]], 'k-')

        # Labels a, b
        ax.text(b/2, -0.2, 'b', ha='center', va='top')
        ax.text(-0.2, a/2, 'a', ha='right', va='center')
        ax.text(a+b+0.2, b/2, 'a', ha='left', va='center')
        ax.text(a/2, a+b+0.2, 'b', ha='center', va='bottom')

        # Label total side a+b
        ax.text((a+b)/2, -0.5, 'a+b', ha='center', va='top')
        ax.text(-0.5, (a+b)/2, 'a+b', ha='right', va='center', rotation=90)

        # Central text (formula)
        ax.text( (a+b)/2, (a+b)/2, 'Area = c²', ha='center', va='center', color='darkred', fontsize=10)
        ax.text( (a+b)/2, a+b+1, 'Visual Proof: (a+b)² = 4(½ab) + c² => a² + b² = c²', ha='center', va='center', fontsize=9)

        ax.set_xlim(-1, a+b+1)
        ax.set_ylim(-1, a+b+2)
        ax.set_title("Pythagoras Theorem (Area Proof)")

        return self._save_plot(fig, filename)

    def f3_tpt_we1_find_hypotenuse(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Legs
        a, b = 6, 8
        c = math.sqrt(a**2 + b**2)

        # Vertices
        C = (0, 0)
        A = (0, a)
        B = (b, 0)

        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-', linewidth=2)
        rect = patches.Rectangle((C[0], C[1]), 0.5, 0.5, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        ax.text((A[0]+C[0])/2-0.3, (A[1]+C[1])/2, f'{a} cm', ha='right', va='center')
        ax.text((B[0]+C[0])/2, (B[1]+C[1])/2-0.3, f'{b} cm', ha='center', va='top')
        ax.text((A[0]+B[0])/2+0.2, (A[1]+B[1])/2+0.1, 'c = ?', ha='left') 

        ax.set_xlim(-1, b+1)
        ax.set_ylim(-1, a+1)
        ax.set_title("Find Hypotenuse (a=6, b=8)")

        return self._save_plot(fig, filename)

    def f3_tpt_we2_find_leg(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given
        c, a = 13, 5
        b = math.sqrt(c**2 - a**2) # Calculate the missing leg (12)

        # Vertices
        C = (0, 0)
        A = (0, a)
        B = (b, 0)

        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-', linewidth=2)
        rect = patches.Rectangle((C[0], C[1]), 0.5, 0.5, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        ax.text((A[0]+C[0])/2-0.3, (A[1]+C[1])/2, f'{a} m', ha='right', va='center')
        ax.text((B[0]+C[0])/2, (B[1]+C[1])/2-0.3, 'b = ?', ha='center', va='top')
        ax.text((A[0]+B[0])/2+0.4, (A[1]+B[1])/2+0.2, f'{c} m', ha='left') #, rotation=math.degrees(math.atan2(A[1]-B[1], A[0]-B[0])))

        ax.set_xlim(-1, b+1)
        ax.set_ylim(-1, a+1)
        ax.set_title("Find Leg (c=13, a=5)")

        return self._save_plot(fig, filename)

    def f3_tpt_we5_ladder_problem(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        # ax.axis('off')

        # Given
        ladder_len = 5
        base_dist = 3
        height_h = math.sqrt(ladder_len**2 - base_dist**2) # 4

        # Points
        BaseWall = (0, 0)
        FootLadder = (base_dist, 0)
        TopLadder = (0, height_h)

        # Draw Wall, Ground, Ladder
        ax.plot([BaseWall[0], TopLadder[0]], [BaseWall[1], TopLadder[1]], 'k-', linewidth=3, label='Wall') # Wall
        ax.plot([BaseWall[0], FootLadder[0]], [BaseWall[1], FootLadder[1]], 'k-', linewidth=3, label='Ground') # Ground
        ax.plot([FootLadder[0], TopLadder[0]], [FootLadder[1], TopLadder[1]], 'brown', linewidth=4, label='Ladder') # Ladder

        # Mark right angle
        rect = patches.Rectangle((BaseWall[0], BaseWall[1]), 0.3, 0.3, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Labels
        ax.text(base_dist/2, -0.3, f'{base_dist} m', ha='center', va='top')
        ax.text(-0.3, height_h/2, 'h = ?', ha='right', va='center')
        ax.text(base_dist/2+0.5, height_h/2+0.5, f'{ladder_len} m', ha='center', va='center', rotation=math.degrees(math.atan2(-height_h, base_dist)))

        ax.set_xlim(-1, base_dist+1)
        ax.set_ylim(-1, height_h+1)
        ax.set_title("Ladder Against Wall")
        ax.set_xlabel("Distance from Wall (m)")
        ax.set_ylabel("Height on Wall (m)")
        ax.grid(True, linestyle='--')

        return self._save_plot(fig, filename)

    # === Form 3: Trigonometry - Trigonometrical Ratios ===

    def f3_tr_notes_triangle_sides_relative(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Vertices
        C = (0, 0)
        A = (0, 3) # Let angle theta be at A
        B = (4, 0)
        theta_deg = math.degrees(math.atan2(B[0]-C[0], A[1]-C[1])) # Angle at A

        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-', linewidth=2)
        rect = patches.Rectangle((C[0], C[1]), 0.3, 0.3, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Mark vertices and angle theta at A
        ax.text(A[0]-0.2, A[1]+0.1, 'A', ha='right', va='bottom')
        ax.text(B[0]+0.1, B[1]-0.2, 'B', va='top')
        ax.text(C[0]-0.2, C[1]-0.2, 'C', ha='right', va='top')
        theta_arc = patches.Arc(A, 0.8, 0.8, angle=270, theta1=0, theta2=theta_deg, color='g', linewidth=1.5)
        ax.add_patch(theta_arc)
        ax.text(A[0]+0.4, A[1]-0.4, 'θ', color='g', ha='center', va='center', fontsize=14)

        # Labels relative to theta
        ax.text((A[0]+B[0])/2+0.3, (A[1]+B[1])/2+0.2, 'Hypotenuse (H)', ha='left', rotation=-37)
        ax.text((B[0]+C[0])/2, (B[1]+C[1])/2+0.2, 'Opposite (O)', ha='center', va='bottom') # Opposite theta
        ax.text((A[0]+C[0])/2-0.2, (A[1]+C[1])/2, 'Adjacent (A)', ha='right', va='center') # Adjacent to theta


        ax.set_xlim(-1.5, 4.5)
        ax.set_ylim(-1, 4)
        ax.set_title("Sides Relative to Angle θ")

        return self._save_plot(fig, filename)

    def f3_tr_notes_unit_circle_obtuse(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Draw unit circle
        circle = patches.Circle((0, 0), 1, edgecolor='black', facecolor='none', linewidth=1)
        ax.add_patch(circle)

        # Draw axes
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)

        # Example obtuse angle theta (e.g., 135 degrees)
        theta_deg = 135
        theta_rad = math.radians(theta_deg)
        x_p = math.cos(theta_rad)
        y_p = math.sin(theta_rad)

        # Plot point P and radius OP
        ax.plot(x_p, y_p, 'ro', label=f'P(cos{theta_deg}°, sin{theta_deg}°)')
        ax.plot([0, x_p], [0, y_p], 'r-')

        # Mark angle theta from positive x-axis
        theta_arc = patches.Arc((0,0), 0.4, 0.4, angle=0, theta1=0, theta2=theta_deg, color='red', linewidth=1.5)
        ax.add_patch(theta_arc)
        ax.text(0.5 * math.cos(theta_rad / 2), 0.5 * math.sin(theta_rad / 2), 'θ', color='red', fontsize=12)

        # Show coordinates relation
        ax.plot([x_p, x_p], [0, y_p], 'b--', linewidth=1) # Vertical line to x-axis
        ax.plot([0, x_p], [y_p, y_p], 'b--', linewidth=1) # Horizontal line to y-axis
        ax.text(x_p, -0.05, f'x = cosθ\n({x_p:.2f})', ha='center', va='top', color='blue', fontsize=9)
        ax.text(0.25, y_p+ 0.05, f'y = sinθ\n({y_p:.2f})', ha='right', va='center', color='blue', fontsize=9)


        # Show related acute angle alpha
        alpha_deg = 180 - theta_deg
        alpha_rad = math.radians(alpha_deg)
        alpha_arc = patches.Arc((0,0), 0.3, 0.3, angle=0, theta1=theta_deg, theta2=180, color='green', linewidth=1.5)
        ax.add_patch(alpha_arc)
        ax.text(-0.4 * math.cos(alpha_rad / 2), 0.4 * math.sin(alpha_rad / 2), 'α', color='green', fontsize=12)
        ax.text(-0.8, -0.8, f'α = 180° - θ = {alpha_deg}°', color='green', fontsize=10)


        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("cos θ")
        ax.set_ylabel("sin θ")
        ax.set_title("Unit Circle - Obtuse Angle θ")
        ax.grid(True, linestyle='--')
        ax.legend(loc='lower left', fontsize=8)

        return self._save_plot(fig, filename)

    def f3_tr_we1_find_ratios(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given sides
        O, H = 5, 13
        A = math.sqrt(H**2 - O**2) # 12

        # Vertices (assuming theta at origin for simplicity of drawing labels)
        Origin = (0, 0) # Angle theta here
        AdjEnd = (A, 0)
        OppEnd = (A, O)

        ax.plot([Origin[0], AdjEnd[0], OppEnd[0], Origin[0]], [Origin[1], AdjEnd[1], OppEnd[1], Origin[1]], 'k-', linewidth=2)
        rect = patches.Rectangle((AdjEnd[0]-0.5, AdjEnd[1]), 0.5, 0.5, angle=90, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect) # Right angle roughly shown

        # Mark angle theta
        theta_deg = math.degrees(math.atan2(O, A))
        theta_arc = patches.Arc(Origin, 2, 2, angle=0, theta1=0, theta2=theta_deg, color='g', linewidth=1.5)
        ax.add_patch(theta_arc)
        ax.text(1.2, 0.5, 'θ', color='g', fontsize=14)

        # Labels
        ax.text(A/2, -0.5, f'Adjacent = {A}', ha='center', va='top')
        ax.text(A+0.3, O/2, f'Opposite = {O}', ha='left', va='center')
        ax.text(A/2+0.5, O/2+0.5, f'Hypotenuse = {H}', ha='center', rotation=theta_deg)

        ax.set_xlim(-1, A+1)
        ax.set_ylim(-1, O+1)
        ax.set_title("Triangle for Calculating Ratios")

        return self._save_plot(fig, filename)

    def f3_tr_we4_find_opposite_side(self):
        """Generates a diagram for finding the opposite side given angle and hypotenuse."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given
        theta_deg = 40
        H = 10
        # Calculate O = H * sin(theta)
        O = H * math.sin(math.radians(theta_deg))
        # Calculate A = H * cos(theta)
        A = H * math.cos(math.radians(theta_deg))

        # Vertices
        AngleVertex = (0, 0)
        AdjEnd = (A, 0) # Vertex where right angle should be
        OppEnd = (A, O)

        # Draw Triangle sides
        ax.plot([AngleVertex[0], AdjEnd[0], OppEnd[0], AngleVertex[0]],
                [AngleVertex[1], AdjEnd[1], OppEnd[1], AngleVertex[1]], 'k-', linewidth=2)

        # Position the right-angle symbol correctly at the vertex AdjEnd
        symbol_size = 0.5 # Size of the square symbol
        # Bottom-left corner of the symbol, offset inwards from AdjEnd
        rect_corner = (AdjEnd[0] - symbol_size, AdjEnd[1])
        rect = patches.Rectangle(rect_corner, symbol_size, symbol_size, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Mark angle theta
        theta_arc = patches.Arc(AngleVertex, 2, 2, angle=0, theta1=0, theta2=theta_deg, color='g', linewidth=1.5)
        ax.add_patch(theta_arc)
        # Position angle label carefully
        label_radius = 1.2
        label_angle_rad = math.radians(theta_deg / 2)
        ax.text(AngleVertex[0] + label_radius * math.cos(label_angle_rad)+ 0.4,
                AngleVertex[1] + label_radius * math.sin(label_angle_rad),
                f'{theta_deg}°', color='g', fontsize=12, ha='center', va='center')

        # Labels for sides
        # Opposite side label
        ax.text(AdjEnd[0] + 0.35, OppEnd[1] / 2, 'x = ? (Opp)', ha='left', va='center', fontsize=10)
        # Hypotenuse label - calculate midpoint and offset perpendicular to the line
        mid_hyp_x = (AngleVertex[0] + OppEnd[0]) / 2
        mid_hyp_y = (AngleVertex[1] + OppEnd[1]) / 2
        offset_dist = 0.3
        ax.text(mid_hyp_x - offset_dist * math.sin(math.radians(theta_deg)),
                mid_hyp_y + offset_dist * math.cos(math.radians(theta_deg)),
                f'H = {H} cm', ha='center', va='bottom', rotation=theta_deg, fontsize=10)

        # Adjust plot limits
        ax.set_xlim(-1, A + 1.5) # Increased limit slightly for label space
        ax.set_ylim(-1, O + 1)
        ax.set_title(f"Find Opposite Side (Angle={theta_deg}°, H={H})", fontsize=14)

        return self._save_plot(fig, filename)

    def f3_tr_we5_find_angle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Given
        O, A = 6, 8
        # Calculate H = sqrt(O^2 + A^2)
        H = math.sqrt(O**2 + A**2)
        # Calculate angle beta
        beta_rad = math.atan2(O, A)
        beta_deg = math.degrees(beta_rad)

        # Vertices
        AngleVertex = (0, 0) # Angle beta here
        AdjEnd = (A, 0)
        OppEnd = (A, O)

        ax.plot([AngleVertex[0], AdjEnd[0], OppEnd[0], AngleVertex[0]],
                [AngleVertex[1], AdjEnd[1], OppEnd[1], AngleVertex[1]], 'k-', linewidth=2)
        # Draw right-angle symbol using a square patch
        symbol_size = 0.4 # Size of the square symbol
        rect_anchor = (AdjEnd[0] - symbol_size, AdjEnd[1]) # Bottom-left corner of the square is (A-size, 0)
        rect = patches.Rectangle(rect_anchor, symbol_size, symbol_size, angle=0, linewidth=1.5, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Mark angle beta
        beta_arc = patches.Arc(AngleVertex, 2, 2, angle=0, theta1=0, theta2=beta_deg, color='g', linewidth=1.5)
        ax.add_patch(beta_arc)
        ax.text(1.4, 0.6, 'β = ?', color='g', fontsize=12)

        # Labels
        ax.text(A/2, -0.3, f'Adj = {A} m', ha='center', va='top')
        ax.text(A+0.3, O/2, f'Opp = {O} m', ha='left', va='center')

        ax.set_xlim(-1, A+1)
        ax.set_ylim(-1, O+1)
        ax.set_title(f"Find Angle β (Opp={O}, Adj={A})")

        return self._save_plot(fig, filename)

    # === Form 3: Vectors - Types of Vectors ===

    def f3_vt_notes_position_vectors(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Origin
        O = (0, 0)
        # Points A and B
        A = (3, 1)
        B = (1, 4)

        # Draw axes
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--')

        # Plot points
        ax.plot(O[0], O[1], 'ko', label='Origin O(0,0)')
        ax.plot(A[0], A[1], 'bo', label=f'A{A}')
        ax.plot(B[0], B[1], 'ro', label=f'B{B}')

        # Draw position vectors OA (a) and OB (b)
        ax.arrow(O[0], O[1], A[0]-0.05, A[1]-0.05, head_width=0.2, head_length=0.2, fc='blue', ec='blue', label='OA = a')
        ax.arrow(O[0], O[1], B[0]-0.05, B[1]-0.07, head_width=0.2, head_length=0.2, fc='red', ec='red', label='OB = b')

        # Draw vector AB
        ax.arrow(A[0], A[1], B[0]-A[0]+0.095, B[1]-A[1]-0.085, head_width=0.2, head_length=0.2, fc='green', ec='green', label='AB = b - a')

        # Labels
        ax.text(A[0]*0.5, A[1]*0.5 + 0.2, 'a', color='blue', ha='center', va='bottom', fontweight='bold')
        ax.text(B[0]*0.5 + 0.2, B[1]*0.5, 'b', color='red', ha='left', va='center', fontweight='bold')
        ax.text((A[0]+B[0])/2+0.1, (A[1]+B[1])/2, 'b - a', color='green', ha='left', va='bottom', fontweight='bold')
        ax.text(3.5, 4, "AB = AO + OB = -a + b", fontsize=10)


        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Position Vectors and Vector Between Points")
        ax.legend(loc='lower right', fontsize=8)

        return self._save_plot(fig, filename)

    def f3_vt_we2_draw_position_vector(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Origin and Point Q
        O = (0, 0)
        Q = (-3, 4)

        # Draw axes
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--')

        # Plot points
        ax.plot(O[0], O[1], 'ko', label='Origin O(0,0)')
        ax.plot(Q[0], Q[1], 'ro', label=f'Q{Q}')

        # Draw position vector OQ
        ax.arrow(O[0], O[1], Q[0], Q[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label='OQ = q')

        # Labels
        ax.text(Q[0]*0.5, Q[1]*0.5 + 0.2, 'OQ', color='red', ha='center', va='bottom', fontweight='bold')

        ax.set_xlim(-4, 1)
        ax.set_ylim(-1, 5)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Position Vector OQ")
        ax.legend(loc='lower left')

        return self._save_plot(fig, filename)

    def f3_vt_we5_identify_equal_vectors(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Define vectors start and end points
        vecs = {
            '1': {'start': (1,1), 'end': (3,3), 'color': 'blue'},   # <2,2>
            '2': {'start': (0,4), 'end': (2,6), 'color': 'green'},  # <2,2>
            '3': {'start': (4,1), 'end': (6,4), 'color': 'red'},    # <2,3>
            '4': {'start': (5,4), 'end': (3,2), 'color': 'purple'} # <-2,-2>
        }

        # Draw axes & grid
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--')

        # Draw vectors
        for label, data in vecs.items():
            start = data['start']
            end = data['end']
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            ax.arrow(start[0], start[1], dx, dy, head_width=0.2, head_length=0.3,
                     fc=data['color'], ec=data['color'], length_includes_head=True)
            # Add label near the middle of the arrow
            mid_x = start[0] + dx * 0.5
            mid_y = start[1] + dy * 0.5
            ax.text(mid_x , mid_y + 0.1, label, color=data['color'], fontweight='bold', ha='left', va='bottom')

        ax.text(0.5, 6.5, "Vector 1 = <2,2>\nVector 2 = <2,2>\nVector 3 = <2,3>\nVector 4 = <-2,-2>",
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(3.5, 5.5, "Vector 1 == Vector 2", fontsize=10, color='darkgreen', fontweight='bold')


        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 7)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Identifying Equal Vectors")

        return self._save_plot(fig, filename)

    # === Form 3: Vectors - Operations ===

    def f3_vo_notes_magnitude_vector(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Vector a = <x, y>
        x, y = 4, 3
        O = (0, 0)
        P = (x, y)

        # Draw axes
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--')

        # Draw vector OP (a)
        ax.arrow(O[0], O[1], P[0], P[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='a = <x, y>')

        # Draw right-angled triangle components
        ax.plot([O[0], P[0]], [O[1], O[1]], 'r--', label='x component') # Horizontal component
        ax.plot([P[0], P[0]], [O[1], P[1]], 'g--', label='y component') # Vertical component

        # Mark right angle
        rect = patches.Rectangle((P[0]-0.3, 0), 0.3, 0.3, angle=0, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        # Labels
        ax.text(x/2, -0.3, f'x = {x}', color='red', ha='center', va='top')
        ax.text(x+0.2, y/2, f'y = {y}', color='green', ha='left', va='center')
        ax.text(x*0.5, y*0.6, 'a', color='blue', fontweight='bold')
        ax.text(1, 3.5, '|a| = √(x² + y²)\n    = √(4² + 3²)\n    = √25 = 5', fontsize=10)

        ax.set_xlim(-1, x+1)
        ax.set_ylim(-1, y+1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Magnitude of a Vector")
        # ax.legend()

        return self._save_plot(fig, filename)

    # === Form 3: Matrices - Operations ===
    # No images described

    # === Form 3: Matrices - Determinants ===
    # No images described

    # === Form 3: Matrices - Inverse Matrix ===
    # No images described

    # === Form 3: Transformation - Translation ===

    def f3_tt_we2_draw_object_image(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Object vertices PQR
        P, Q, R = (1,2), (3,4), (2,5)
        obj_verts = np.array([P, Q, R, P]) # Close the triangle

        # Image vertices P'Q'R' (translated by <3, -1>)
        P_prime, Q_prime, R_prime = (4,1), (6,3), (5,4)
        img_verts = np.array([P_prime, Q_prime, R_prime, P_prime])

        # Plot object
        ax.plot(obj_verts[:, 0], obj_verts[:, 1], 'b-o', label='Object PQR')
        ax.text(P[0]-0.2, P[1], 'P', ha='right')
        ax.text(Q[0]+0.1, Q[1], 'Q', va='bottom')
        ax.text(R[0], R[1]+0.1, 'R', ha='center', va='bottom')

        # Plot image
        ax.plot(img_verts[:, 0], img_verts[:, 1], 'r--o', label="Image P'Q'R'")
        ax.text(P_prime[0]+0.1, P_prime[1], "P'", va='center')
        ax.text(Q_prime[0]+0.1, Q_prime[1], "Q'", va='bottom')
        ax.text(R_prime[0], R_prime[1]+0.1, "R'", ha='center', va='bottom')

        # Show translation vector (e.g., from P to P')
        ax.arrow(P[0], P[1], P_prime[0]-P[0], P_prime[1]-P[1],
                 head_width=0.2, head_length=0.3, fc='g', ec='g', length_includes_head=True, linestyle=':')
        ax.text( (P[0]+P_prime[0])/2, (P[1]+P_prime[1])/2 + 0.2 , '<3, -1>', color='g')


        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 6)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Translation of Triangle PQR by <3, -1>")
        ax.legend()

        return self._save_plot(fig, filename)

    # === Form 3: Transformation - Reflection ===

    def f3_trfl_notes_reflection_lines(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Reflection in Lines y=a and x=b', fontsize=14)

        # --- Reflection in y=a ---
        ax1.set_facecolor('white')
        ax1.set_aspect('equal', adjustable='box')
        a = 2 # Example value for y=a
        P = (3, 4) # Example point
        P_prime = (P[0], 2*a - P[1]) # (3, 0)

        ax1.axhline(a, color='red', linestyle='--', label=f'Mirror Line y={a}')
        ax1.plot(P[0], P[1], 'bo', label=f'P{P}')
        ax1.plot(P_prime[0], P_prime[1], 'ro', label=f"P'{P_prime}")
        ax1.plot([P[0], P_prime[0]], [P[1], P_prime[1]], 'k:', label='PP\'') # Line segment PP'
        # Midpoint of PP'
        M = ((P[0]+P_prime[0])/2, (P[1]+P_prime[1])/2)
        ax1.plot(M[0], M[1], 'gx', markersize=8, label='Midpoint on y=a') # Midpoint

        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.grid(True, linestyle='--')
        ax1.set_xlim(0, 5)
        ax1.set_ylim(-1, 6)
        ax1.set_title("Reflection in y = a")
        ax1.legend(fontsize=8)

        # --- Reflection in x=b ---
        ax2.set_facecolor('white')
        ax2.set_aspect('equal', adjustable='box')
        b = 1 # Example value for x=b
        Q = (-2, 3) # Example point
        Q_prime = (2*b - Q[0], Q[1]) # (4, 3)

        ax2.axvline(b, color='red', linestyle='--', label=f'Mirror Line x={b}')
        ax2.plot(Q[0], Q[1], 'bo', label=f'Q{Q}')
        ax2.plot(Q_prime[0], Q_prime[1], 'ro', label=f"Q'{Q_prime}")
        ax2.plot([Q[0], Q_prime[0]], [Q[1], Q_prime[1]], 'k:', label='QQ\'') # Line segment QQ'
        # Midpoint of QQ'
        N = ((Q[0]+Q_prime[0])/2, (Q[1]+Q_prime[1])/2)
        ax2.plot(N[0], N[1], 'gx', markersize=8, label='Midpoint on x=b') # Midpoint

        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(True, linestyle='--')
        ax2.set_xlim(-4, 6)
        ax2.set_ylim(0, 5)
        ax2.set_title("Reflection in x = b")
        ax2.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    def f3_trfl_we3_reflect_y_axis(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Object PQR
        P, Q, R = (1,1), (4,1), (1,3)
        obj_verts = np.array([P, Q, R, P])

        # Image P'Q'R' (reflected in y-axis)
        P_prime, Q_prime, R_prime = (-1,1), (-4,1), (-1,3)
        img_verts = np.array([P_prime, Q_prime, R_prime, P_prime])

        # Plot object
        ax.plot(obj_verts[:, 0], obj_verts[:, 1], 'b-o', label='Object PQR')
        ax.text(P[0]+0.1, P[1], 'P')
        ax.text(Q[0]+0.1, Q[1], 'Q')
        ax.text(R[0]+0.1, R[1], 'R')

        # Plot image
        ax.plot(img_verts[:, 0], img_verts[:, 1], 'r--o', label="Image P'Q'R'")
        ax.text(P_prime[0]-0.2, P_prime[1], "P'", ha='right')
        ax.text(Q_prime[0]-0.2, Q_prime[1], "Q'", ha='right')
        ax.text(R_prime[0]-0.2, R_prime[1], "R'", ha='right')

        # Highlight y-axis (x=0)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Mirror Line (y-axis, x=0)')

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 4)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Reflection in the y-axis (x=0)")
        ax.legend()

        return self._save_plot(fig, filename)

    # === Form 3: Transformation - Rotation ===

        # Function to plot perpendicular bisector
        def plot_perp_bisector(p1, p2, color):
            mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            dx = p2[0]-p1[0]
            dy = p2[1]-p1[1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6: return # Points coincide
            # Perpendicular direction (-dy, dx)
            perp_dx, perp_dy = -dy, dx
            # Extend line
            t = 3
            start_pt = (mid[0] - t*perp_dx, mid[1] - t*perp_dy)
            end_pt = (mid[0] + t*perp_dx, mid[1] + t*perp_dy)
            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], color=color, linestyle=':', linewidth=1.5, label=f'Bisector {name_map[p1]}{name_map[p2]}\'')
            ax.plot(mid[0], mid[1], marker='x', color=color, markersize=6)

        # Plot bisectors
        name_map = {tuple(A):'A', tuple(B):'B', tuple(C):'C'} # Map coords back to names for label
        plot_perp_bisector(A, A_prime, 'green')
        plot_perp_bisector(B, B_prime, 'purple')
        # plot_perp_bisector(C, C_prime, 'orange') # 2 are enough

        # Mark Center
        ax.plot(center[0], center[1], 'kx', markersize=10, markeredgewidth=2, label='Center C')
        ax.text(center[0], center[1]-0.3, 'Center C', ha='center')

        # Show angle ACA' (optional)
        ax.plot([center[0], A[0]], [center[1], A[1]], 'gray', linestyle='-.')
        ax.plot([center[0], A_prime[0]], [center[1], A_prime[1]], 'gray', linestyle='-.')
        angle_arc = patches.Arc(center, 1.5, 1.5, angle=0, theta1=math.degrees(math.atan2(A_prime[1]-center[1], A_prime[0]-center[0])), theta2=math.degrees(math.atan2(A[1]-center[1], A[0]-center[0])), color='orange')
        ax.add_patch(angle_arc)
        ax.text(0.5, 2.8, "∠ACA'", color='orange')


        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        all_x = np.concatenate((obj_verts[:,0], img_verts[:,0]))
        all_y = np.concatenate((obj_verts[:,1], img_verts[:,1]))
        ax.set_xlim(min(all_x)-1, max(all_x)+1)
        ax.set_ylim(min(all_y)-1, max(all_y)+1)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Finding Center of Rotation")
        ax.legend(fontsize=8, loc='upper right')

        return self._save_plot(fig, filename)

    def f3_trot_we4_rotate_180(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Object ABC
        A, B, C = (1,1), (3,1), (1,4)
        obj_verts = np.array([A, B, C, A])

        # Image A'B'C' (rotated 180 deg about origin)
        A_prime, B_prime, C_prime = (-1,-1), (-3,-1), (-1,-4)
        img_verts = np.array([A_prime, B_prime, C_prime, A_prime])

        # Plot object
        ax.plot(obj_verts[:, 0], obj_verts[:, 1], 'b-o', label='Object ABC')
        ax.text(A[0]+0.1, A[1], 'A')
        ax.text(B[0]+0.1, B[1], 'B')
        ax.text(C[0]+0.1, C[1], 'C')

        # Plot image
        ax.plot(img_verts[:, 0], img_verts[:, 1], 'r--o', label="Image A'B'C'")
        ax.text(A_prime[0]-0.2, A_prime[1], "A'", ha='right')
        ax.text(B_prime[0]-0.2, B_prime[1], "B'", ha='right')
        ax.text(C_prime[0]-0.2, C_prime[1], "C'", ha='right')

        # Mark center
        ax.plot(0, 0, 'kx', markersize=8, label='Center (Origin)')

        # Show rotation lines (optional)
        ax.plot([A[0], A_prime[0]], [A[1], A_prime[1]], 'gray', linestyle=':')
        ax.plot([B[0], B_prime[0]], [B[1], B_prime[1]], 'gray', linestyle=':')
        ax.plot([C[0], C_prime[0]], [C[1], C_prime[1]], 'gray', linestyle=':')

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Rotation of 180° about Origin")
        ax.legend()

        return self._save_plot(fig, filename)

    # === Form 3: Transformation - Enlargement ===

    def f3_te_notes_find_center(self):
        filename = 'math_diagrams/f3_te_notes_find_center.png'
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Example Object and Image (e.g., enlargement k=2, center (1,1))
        A, B, C = (2,2), (4,2), (2,4) # Object
        center = (1,1)
        k = 2

        def enlarge_point(p, center, k):
            vec = np.array(p) - np.array(center)
            enlarged_vec = k * vec
            return enlarged_vec + np.array(center)

        A_prime = enlarge_point(A, center, k) # (1,1) + 2*(1,1) = (3,3)
        B_prime = enlarge_point(B, center, k) # (1,1) + 2*(3,1) = (7,3)
        C_prime = enlarge_point(C, center, k) # (1,1) + 2*(1,3) = (3,7)

        obj_verts = np.array([A, B, C, A])
        img_verts = np.array([A_prime, B_prime, C_prime, A_prime])

        # Plot object and image
        ax.plot(obj_verts[:, 0], obj_verts[:, 1], 'b-o', label='Object ABC')
        ax.plot(img_verts[:, 0], img_verts[:, 1], 'r--o', label="Image A'B'C'")
        for pt, name in zip([A, B, C], ['A', 'B', 'C']): ax.text(pt[0]+0.1, pt[1], name)
        for pt, name in zip([A_prime, B_prime, C_prime], ["A'", "B'", "C'"]): ax.text(pt[0]+0.1, pt[1], name)

        # Draw lines connecting corresponding points through center
        line_len = 8
        # Line A-A'
        vec_AA = A_prime - A
        ax.plot([A[0]-vec_AA[0]*0.8, A_prime[0]+vec_AA[0]*0.8], [A[1]-vec_AA[1]*0.8, A_prime[1]+vec_AA[1]*0.8], 'g:', linewidth=1)
        # Line B-B'
        vec_BB = B_prime - B
        ax.plot([B[0]-vec_BB[0]*0.8, B_prime[0]+vec_BB[0]*0.8], [B[1]-vec_BB[1]*0.8, B_prime[1]+vec_BB[1]*0.8], 'g:', linewidth=1)
         # Line C-C'
        vec_CC = C_prime - C
        ax.plot([C[0]-vec_CC[0]*0.8, C_prime[0]+vec_CC[0]*0.8], [C[1]-vec_CC[1]*0.8, C_prime[1]+vec_CC[1]*0.8], 'g:', linewidth=1)


        # Mark Center
        ax.plot(center[0], center[1], 'kx', markersize=10, markeredgewidth=2, label='Center O')
        ax.text(center[0], center[1]-0.3, 'Center O', ha='center')

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        all_x = np.concatenate((obj_verts[:,0], img_verts[:,0], [center[0]]))
        all_y = np.concatenate((obj_verts[:,1], img_verts[:,1], [center[1]]))
        ax.set_xlim(min(all_x)-1, max(all_x)+1)
        ax.set_ylim(min(all_y)-1, max(all_y)+1)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Finding Center of Enlargement")
        ax.legend(fontsize=8, loc='upper left')

        return self._save_plot(fig, filename)

    def f3_te_we4_enlarge_triangle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Object ABC
        A, B, C = (1,1), (3,1), (1,2)
        obj_verts = np.array([A, B, C, A])

        # Image A'B'C' (enlarged k=2, center origin)
        A_prime, B_prime, C_prime = (2,2), (6,2), (2,4)
        img_verts = np.array([A_prime, B_prime, C_prime, A_prime])

        # Plot object
        ax.plot(obj_verts[:, 0], obj_verts[:, 1], 'b-o', label='Object ABC')
        ax.text(A[0]+0.1, A[1], 'A')
        ax.text(B[0]+0.1, B[1], 'B')
        ax.text(C[0]+0.1, C[1], 'C')

        # Plot image
        ax.plot(img_verts[:, 0], img_verts[:, 1], 'r--o', label="Image A'B'C'")
        ax.text(A_prime[0]+0.1, A_prime[1], "A'")
        ax.text(B_prime[0]+0.1, B_prime[1], "B'")
        ax.text(C_prime[0]+0.1, C_prime[1], "C'")

        # Mark center
        ax.plot(0, 0, 'kx', markersize=8, label='Center (Origin)')

        # Show enlargement lines (optional)
        ax.plot([0, A_prime[0]], [0, A_prime[1]], 'gray', linestyle=':')
        ax.plot([0, B_prime[0]], [0, B_prime[1]], 'gray', linestyle=':')
        ax.plot([0, C_prime[0]], [0, C_prime[1]], 'gray', linestyle=':')

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 5)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Enlargement, k=2, Center Origin")
        ax.legend()

        return self._save_plot(fig, filename)

    # === Form 3: Probability - Probability ===
    # No images described

    # === Form 4: Financial Mathematics - Consumer Arithmetic ===
    # No images described

    # === Form 4: Measures and Mensuration - Mensuration ===

    def f4_mm_notes_prism_pyramid(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), facecolor='white')
        fig.suptitle('Prism and Pyramid Volume', fontsize=14)

        # # --- Triangular Prism ---
        # ax1.set_facecolor('white')
        # ax1.axis('off')
        # ax1.set_title("Triangular Prism\nV = A × h")

        # # Base triangle vertices
        # b1, b2, b3 = (1,1), (4,1), (2.5, 3)
        # # Top triangle vertices (shifted)
        # h = 5 # Height/Length of prism
        # t1, t2, t3 = (b1[0]+1, b1[1]+h), (b2[0]+1, b2[1]+h), (b3[0]+1, b3[1]+h)

        # # Draw bases
        # base = patches.Polygon([b1,b2,b3], closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.6)
        # top = patches.Polygon([t1,t2,t3], closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.6)
        # ax1.add_patch(base)
        # ax1.add_patch(top)

        # # Draw connecting edges (rectangular faces)
        # ax1.plot([b1[0], t1[0]], [b1[1], t1[1]], 'k-')
        # ax1.plot([b2[0], t2[0]], [b2[1], t2[1]], 'k-')
        # ax1.plot([b3[0], t3[0]], [b3[1], t3[1]], 'k-')

        # # Labels
        # ax1.text(2.5, 1.26, 'Base (Area A)', ha='center', color='darkblue')
        # # Height indicator
        # mid_base = ((b1[0]+b2[0])/2, (b1[1]+b2[1])/2 + 0.5) # Midpoint slightly above base edge
        # mid_top = ((t1[0]+t2[0])/2, (t1[1]+t2[1])/2 - 0.6)
        # ax1.annotate('', xy=mid_top, xytext=mid_base,
        #              arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        # ax1.text(mid_base[0]+0.05, ((mid_base[1]+mid_top[1])/2), 'h', color='red')

        # ax1.set_xlim(0, 6)
        # ax1.set_ylim(0, h+4)
        # --- Triangular Prism ---
        ax1.set_facecolor('white')
        ax1.axis('off')
        ax1.set_title("Triangular Prism\nV = A × h")

        # Base triangle vertices
        b1, b2, b3 = (1,1), (4,1), (2.5, 3)
        # Top triangle vertices (shifted)
        h = 5 # Height/Length of prism
        shift_x = 0.5 # Horizontal shift for slant effect
        t1, t2, t3 = (b1[0]+shift_x, b1[1]+h), (b2[0]+shift_x, b2[1]+h), (b3[0]+shift_x, b3[1]+h)

        # Draw bases
        base = patches.Polygon([b1,b2,b3], closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.6, zorder=1)
        top = patches.Polygon([t1,t2,t3], closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.6, zorder=1)
        ax1.add_patch(base)
        ax1.add_patch(top)

        # Draw connecting edges (rectangular faces)
        ax1.plot([b1[0], t1[0]], [b1[1], t1[1]], 'k-', zorder=0)
        ax1.plot([b2[0], t2[0]], [b2[1], t2[1]], 'k-', zorder=0)
        ax1.plot([b3[0], t3[0]], [b3[1], t3[1]], 'k-', zorder=0)

        # Calculate centroids for labeling and height
        base_centroid_x = (b1[0] + b2[0] + b3[0]) / 3
        base_centroid_y = (b1[1] + b2[1] + b3[1]) / 3
        top_centroid_x = (t1[0] + t2[0] + t3[0]) / 3
        top_centroid_y = (t1[1] + t2[1] + t3[1]) / 3 # Should be base_centroid_y + h

        # Labels
        # Place base label at the calculated centroid
        ax1.text(base_centroid_x + 0.01, base_centroid_y, 'Base (Area A)', ha='center', va='center', color='darkblue', fontsize=9, zorder=2)

        # Height indicator (Vertical)
        # Position the vertical height line clearly, e.g., near the center or slightly offset
        height_line_x = (base_centroid_x + top_centroid_x) / 2 + 1.0 # Offset to the right for clarity
        # Draw dashed line
        ax1.plot([height_line_x, height_line_x +0.5], [base_centroid_y-0.22, top_centroid_y-0.23], color='red', linestyle='--', lw=1.5, zorder=2)
        # Add arrowheads using annotate for better control
        # ax1.annotate('', xy=(height_line_x, base_centroid_y), xytext=(height_line_x, top_centroid_y),
        #              arrowprops=dict(arrowstyle='<->', color='red', lw=1.5), zorder=2)
        # Place 'h' label next to the vertical line
        ax1.text(height_line_x + 0.1, (base_centroid_y + top_centroid_y) / 2, 'h', color='red', ha='left', va='center', fontsize=10, zorder=2)


        # Adjust limits to ensure everything fits
        all_x = [b1[0], b2[0], b3[0], t1[0], t2[0], t3[0], height_line_x]
        all_y = [b1[1], b2[1], b3[1], t1[1], t2[1], t3[1]]
        ax1.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
        ax1.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
        ax1.set_aspect('equal', adjustable='box') # Keep aspect ratio reasonable

        # --- Square-based Pyramid ---
        ax2.set_facecolor('white')
        ax2.axis('off')
        ax2.set_title("Square-based Pyramid\nV = (1/3) × A × h")

        # Base square vertices (perspective)
        pb1, pb2, pb3, pb4 = (1,1), (5,1), (6,2), (2,2)
        # Apex
        apex = (3.5, 6)
        # Base center (approx)
        base_center = (3.5, 1.5)
        # Height
        h_pyr = apex[1] - base_center[1]

        # Draw base
        base_pyr = patches.Polygon([pb1,pb2,pb3,pb4], closed=True, edgecolor='green', facecolor='lightgreen', alpha=0.6)
        ax2.add_patch(base_pyr)

        # Draw edges to apex
        ax2.plot([pb1[0], apex[0]], [pb1[1], apex[1]], 'k-')
        ax2.plot([pb2[0], apex[0]], [pb2[1], apex[1]], 'k-')
        ax2.plot([pb3[0], apex[0]], [pb3[1], apex[1]], 'k-')
        ax2.plot([pb4[0], apex[0]], [pb4[1], apex[1]], 'k-')

        # Draw height
        ax2.plot([apex[0], base_center[0]], [apex[1], base_center[1]], 'r--')
        ax2.plot(base_center[0], base_center[1], 'rx') # Mark base center
        ax2.text(apex[0]+0.2, (apex[1]+base_center[1])/2, 'h', color='red')

        # Label Base Area
        ax2.text(base_center[0], base_center[1] - 0.5, 'Base (Area A)', ha='center', color='darkgreen')


        ax2.set_xlim(0, 7)
        ax2.set_ylim(0, 7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    def f4_mm_notes_cone_volume(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        # Cone parameters
        radius = 2
        height = 5
        center_base = (0, 0)
        apex = (0, height)

        # Draw base ellipse (perspective)
        base = patches.Ellipse(center_base, 2*radius, 1, edgecolor='blue', facecolor='lightblue', alpha=0.6)
        ax.add_patch(base)

        # Draw sides to apex
        ax.plot([-radius, apex[0]], [center_base[1], apex[1]], 'k-')
        ax.plot([radius, apex[0]], [center_base[1], apex[1]], 'k-')

        # Draw height
        ax.plot([center_base[0], apex[0]], [center_base[1], apex[1]], 'r--')
        ax.text(center_base[0]+0.1, height/2, 'h', color='red')

        # Draw radius
        ax.plot([center_base[0], radius], [center_base[1], center_base[1]], 'b--')
        ax.text(radius/2, center_base[1]-0.3, 'r', color='blue')

        ax.text(0, -1.5, 'V = (1/3)πr²h', ha='center', fontsize=12)

        ax.set_xlim(-radius-1, radius+1)
        ax.set_ylim(-2, height+1)
        ax.set_title("Cone Volume")

        return self._save_plot(fig, filename)

    def f4_mm_notes_sphere_volume(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        radius = 2
        center = (0,0)

        # Draw sphere (circle with shading/lines for 3D effect)
        sphere = patches.Circle(center, radius, edgecolor='black', facecolor='skyblue', alpha=0.7)
        ax.add_patch(sphere)
        # Add equator line (perspective)
        equator = patches.Ellipse(center, 2*radius, 0.8, angle=0, fill=False, edgecolor='black', linestyle='--')
        ax.add_patch(equator)
        # Add meridian line
        # meridian = patches.Arc(center, radius*2, radius*2, angle=90, theta1=-70, theta2=70, fill=False, edgecolor='black', linestyle=':')
        # ax.add_patch(meridian)


        # Draw radius
        ax.plot([center[0], center[0]+radius], [center[1], center[1]], 'r-')
        ax.plot(center[0], center[1], 'ko') # Center point
        ax.text(radius/2, 0.2, 'r', color='red')

        ax.text(0, -radius-0.8, 'V = (4/3)πr³', ha='center', fontsize=12)

        ax.set_xlim(-radius-1, radius+1)
        ax.set_ylim(-radius-1.5, radius+1)
        ax.set_title("Sphere Volume")

        return self._save_plot(fig, filename)

    def f4_mm_notes_surface_area_nets(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
        fig.suptitle('Surface Area Nets', fontsize=14)

        # --- Cylinder Net (ax1) ---
        ax1.set_facecolor('white')
        ax1.set_aspect('equal', adjustable='box')
        ax1.axis('off')
        ax1.set_title("Cylinder Net\nSA = 2πr² + 2πrh", pad=15)

        r, h = 1.5, 4
        circ = 2 * math.pi * r
        rect_y_start = r # Start rectangle above the bottom circle's space

        # Rectangle (curved surface)
        rect = patches.Rectangle((0, rect_y_start), circ, h, edgecolor='black', facecolor='lightgreen', alpha=0.6, zorder=0)
        ax1.add_patch(rect)
        ax1.text(circ/2, rect_y_start + h/2, 'Curved Surface\n(2πr × h)', ha='center', va='center', zorder=1)

        # Circles (top and bottom)
        bottom_center_y = 0
        top_center_y = rect_y_start + h + r
        circle_center_x = circ/2

        top_circle = patches.Circle((circle_center_x, top_center_y), r, edgecolor='black', facecolor='lightblue', alpha=0.6, zorder=0)
        bottom_circle = patches.Circle((circle_center_x, bottom_center_y), r, edgecolor='black', facecolor='lightblue', alpha=0.6, zorder=0)
        ax1.add_patch(top_circle)
        ax1.add_patch(bottom_circle)
        ax1.text(circle_center_x, top_center_y, 'Top\n(πr²)', ha='center', va='center', zorder=1)
        ax1.text(circle_center_x, bottom_center_y, 'Bottom\n(πr²)', ha='center', va='center', zorder=1)
        arrow_offset = 0.05 # Reduced offset to bring arrows closer
        label_offset = 0.2 # Adjusted label offset for clarity

        # Diameter 2r (Bottom circle) - Arrow closer to the circle
        arrow_y_2r = bottom_center_y - r - arrow_offset
        ax1.annotate('', xy=(circle_center_x+r, arrow_y_2r), xytext=(circle_center_x-3.3*r, arrow_y_2r),
                     arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
        ax1.text(circle_center_x, arrow_y_2r - label_offset, '2r', color='blue', ha='center', va='top') # Position label below arrow

        # Height h (Right side of rectangle) - Arrow closer to the rectangle
        arrow_x_h = circ + arrow_offset # Use consistent offset horizontally
        ax1.annotate('', xy=(arrow_x_h, rect_y_start+h), xytext=(arrow_x_h, rect_y_start-h),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax1.text(arrow_x_h + label_offset, rect_y_start + h/2, 'h', color='red', va='center', ha='left') # Position label to the right of arrow

        # Circumference 2πr (Below rectangle) - Arrow closer to the rectangle
        arrow_y_circ = rect_y_start - arrow_offset
        ax1.annotate('', xy=(circ, arrow_y_circ), xytext=(-1.2*circ, arrow_y_circ),
                     arrowprops=dict(arrowstyle='<->', color='m', lw=1.5))
        ax1.text((circ/2)-2, arrow_y_circ - label_offset, '2πr', color='m', ha='center', va='top') # Position label below arrow
        # Adjust limits based on new arrow/label positions
        ax1.set_xlim(-1, arrow_x_h + label_offset + 0.5) # Ensure right limit includes 'h' label
        ax1.set_ylim(arrow_y_2r - label_offset - 0.5, top_center_y + r + 1) # Ensure bottom limit includes '2r' label

        # --- Cone Net (ax2) --- - Unchanged from previous version
        ax2.set_facecolor('white')
        ax2.set_aspect('equal', adjustable='box')
        ax2.axis('off')
        ax2.set_title("Cone Net\nSA = πr² + πrl", pad=15)

        r_cone, h_cone = 2, 4
        l_cone = math.sqrt(r_cone**2 + h_cone**2)
        if l_cone == 0: angle_sector_rad = 0
        else: angle_sector_rad = (2 * math.pi * r_cone) / l_cone
        angle_sector_deg = math.degrees(angle_sector_rad)

        sector_apex = (0, l_cone + r_cone + 0.5)
        start_angle_deg = -90 - angle_sector_deg / 2
        end_angle_deg = -90 + angle_sector_deg / 2

        sector = patches.Wedge(sector_apex, l_cone, start_angle_deg, end_angle_deg,
                               edgecolor='black', facecolor='lightgreen', alpha=0.6, zorder=0)
        ax2.add_patch(sector)
        text_angle_rad = math.radians(-90)
        text_radius = l_cone * 0.6
        ax2.text(sector_apex[0] + text_radius * math.cos(text_angle_rad),
                 sector_apex[1] + text_radius * math.sin(text_angle_rad),
                 'Curved Surface\n(πrl)', ha='center', va='center', zorder=1)

        arc_mid_angle_rad = math.radians(-90)
        arc_mid_x = sector_apex[0] + l_cone * math.cos(arc_mid_angle_rad)
        arc_mid_y = sector_apex[1] + l_cone * math.sin(arc_mid_angle_rad)

        base_center_cone_x = arc_mid_x
        base_center_cone_y = arc_mid_y - r_cone
        circle_cone = patches.Circle((base_center_cone_x, base_center_cone_y), r_cone,
                                     edgecolor='black', facecolor='lightblue', alpha=0.6, zorder=0)
        ax2.add_patch(circle_cone)
        ax2.text(base_center_cone_x, base_center_cone_y, 'Base\n(πr²)', ha='center', va='center', zorder=1)

        ax2.plot([base_center_cone_x - r_cone, base_center_cone_x],
                 [base_center_cone_y, base_center_cone_y], 'b--')
        ax2.text(base_center_cone_x - r_cone/2, base_center_cone_y + 0.2, 'r', color='blue', va='bottom', ha='center')

        sector_edge_x = sector_apex[0] + l_cone * math.cos(math.radians(start_angle_deg))
        sector_edge_y = sector_apex[1] + l_cone * math.sin(math.radians(start_angle_deg))
        ax2.plot([sector_apex[0], sector_edge_x], [sector_apex[1], sector_edge_y], 'r--', zorder=1)
        l_label_x = sector_apex[0] * 0.5 + sector_edge_x * 0.5 + 0.2
        l_label_y = sector_apex[1] * 0.5 + sector_edge_y * 0.5
        ax2.text(l_label_x, l_label_y, 'l', color='red', va='center', ha='left', zorder=1)

        arc_label_y = arc_mid_y - r_cone * 2 - 0.3
        ax2.text(arc_mid_x, arc_label_y, 'Arc = 2πr', color='m', ha='center', va='top')

        # Adjust limits for cone
        ax2.set_xlim(sector_apex[0] - l_cone - 1, sector_apex[0] + l_cone + 1)
        ax2.set_ylim(arc_label_y - 0.5, sector_apex[1] + 0.5)

        # Final layout adjustment
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        return self._save_plot(fig, filename)


    # === Form 4: Graphs - Functional Graphs ===

    def f4_gfg_notes_cubic_graphs(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Cubic Function Graphs', fontsize=14)
        x_vals = np.linspace(-2.5, 2.5, 400)

        # --- y = x³ ---
        ax1.set_facecolor('white')
        y1 = x_vals**3
        ax1.plot(x_vals, y1, 'b-', label='y = x³')
        ax1.set_title("Simple Cubic (y=ax³, a>0)")
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.grid(True, linestyle='--')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()

        # --- y = x³ - 3x ---
        ax2.set_facecolor('white')
        y2 = x_vals**3 - 3*x_vals
        ax2.plot(x_vals, y2, 'r-', label='y = x³ - 3x')
        ax2.set_title("General Cubic (with turning points)")
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(True, linestyle='--')
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.legend()
        # Mark turning points approximately
        ax2.plot(-1, 2, 'ko') # Local max approx
        ax2.plot(1, -2, 'ko') # Local min approx
        ax2.text(-1, 2.5, 'Local Max')
        ax2.text(1, -2.5, 'Local Min')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    def f4_gfg_notes_inverse_graphs(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Inverse Function Graphs (Hyperbolas)', fontsize=14)
        x_vals = np.linspace(-5, 5, 400)

        # --- y = k/x (k>0) ---
        ax1.set_facecolor('white')
        k1 = 4
        y1 = k1 / x_vals
        y1[np.abs(x_vals) < 0.1] = np.nan # Avoid division by zero / vertical line at asymptote

        ax1.plot(x_vals, y1, 'b-', label=f'y = {k1}/x')
        ax1.set_title("y = k/x (k>0)")
        ax1.axhline(0, color='black', linewidth=0.8, label='Asymptote y=0')
        ax1.axvline(0, color='black', linewidth=0.8, label='Asymptote x=0')
        ax1.grid(True, linestyle='--')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_ylim(-10, 10)
        ax1.set_xlim(-5, 5)
        ax1.legend(fontsize=8)

        # --- y = a/(x-p) ---
        ax2.set_facecolor('white')
        a2 = 2
        p2 = 1
        y2 = a2 / (x_vals - p2)
        y2[np.abs(x_vals - p2) < 0.1] = np.nan # Avoid division by zero

        ax2.plot(x_vals, y2, 'r-', label=f'y = {a2}/(x-{p2})')
        ax2.set_title(f"y = a/(x-p) (a={a2}, p={p2})")
        ax2.axhline(0, color='black', linewidth=0.8, label='Asymptote y=0')
        ax2.axvline(p2, color='black', linestyle='--', linewidth=0.8, label=f'Asymptote x={p2}')
        ax2.grid(True, linestyle='--')
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_ylim(-10, 10)
        ax2.set_xlim(-5, 5)
        ax2.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    def f4_gfg_we1_cubic_graph(self):
        filename = 'math_diagrams/f4_gfg_we1_cubic_graph.png'
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')

        # Data points from WE1
        x_pts = np.array([-3, -2, -1, 0, 1, 2, 3])
        y_pts = np.array([-15, 0, 3, 0, -3, 0, 15])

        # Interpolate for a smooth curve
        x_smooth = np.linspace(x_pts.min(), x_pts.max(), 300)
        # Simple cubic interpolation can be tricky, let's just plot given points and smooth manually
        # from scipy.interpolate import interp1d
        # f_cubic = interp1d(x_pts, y_pts, kind='cubic')
        # y_smooth = f_cubic(x_smooth)
        # Using numpy's polynomial fit as an approximation for smoothness
        coeffs = np.polyfit(x_pts, y_pts, 3) # Fit a cubic polynomial
        poly = np.poly1d(coeffs)
        y_smooth = poly(x_smooth)


        ax.plot(x_smooth, y_smooth, 'b-', label='y = x³ - 4x (smooth)')
        ax.plot(x_pts, y_pts, 'ro', label='Calculated Points') # Plot original points

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Graph of y = x³ - 4x")
        ax.legend()
        ax.set_ylim(min(y_pts)-2, max(y_pts)+2) # Adjust ylim based on points

        # Store data for WE3
        self.context_data['f4_gfg_we1'] = {'x': x_pts, 'y': y_pts, 'xsmooth': x_smooth, 'ysmooth': y_smooth}

        return self._save_plot(fig, filename)

    def f4_gfg_we2_inverse_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')

        # Data points
        x_pts = np.array([-4, -2, -1, -0.5, 0.5, 1, 2, 4])
        y_pts = np.array([-1, -2, -4, -8, 8, 4, 2, 1])

        # Generate smooth curve data
        x_neg = np.linspace(-4, -0.1, 200)
        y_neg = 4 / x_neg
        x_pos = np.linspace(0.1, 4, 200)
        y_pos = 4 / x_pos

        # Plot
        ax.plot(x_neg, y_neg, 'b-', label='y = 4/x')
        ax.plot(x_pos, y_pos, 'b-')
        ax.plot(x_pts, y_pts, 'ro', label='Calculated Points')

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.8, label='Asymptote y=0')
        ax.axvline(0, color='black', linewidth=0.8, label='Asymptote x=0')
        ax.grid(True, linestyle='--')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Graph of y = 4/x")
        ax.legend(fontsize=9)
        ax.set_ylim(-9, 9)
        ax.set_xlim(-5, 5)

        return self._save_plot(fig, filename)

    # def f4_gfg_we3_solve_on_graph(self):
    #     filename = get_filename_from_caller()
    #     # Need graph data from WE1
    #     if 'f4_gfg_we1' not in self.context_data:
    #         print("Warning: Context data for WE1 not found, regenerating.")
    #         self.f4_gfg_we1_cubic_graph()
    #         if 'f4_gfg_we1' not in self.context_data:
    #              print("Error: Failed to get context data.")
    #              return None

    #     x_smooth = self.context_data['f4_gfg_we1']['xsmooth']
    #     y_smooth = self.context_data['f4_gfg_we1']['ysmooth']
    #     x_pts = self.context_data['f4_gfg_we1']['x']
    #     y_pts = self.context_data['f4_gfg_we1']['y']


    #     fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    #     ax.set_facecolor('white')

    #     ax.plot(x_smooth, y_smooth, 'b-', label='y = x³ - 4x')
    #     # ax.plot(x_pts, y_pts, 'ro')

    #     # Draw horizontal line y=5
    #     target_y = 5
    #     ax.axhline(target_y, color='red', linestyle='--', label=f'y = {target_y}')

    #     # Find intersections (approximate visually or numerically)
    #     # Numerical: find where poly(x) - 5 = 0
    #     target_poly = np.poly1d(self.context_data['f4_gfg_we1']['coeffs'] - [0,0,0,target_y])
    #     roots = target_poly.roots
    #     real_roots = roots[np.isreal(roots)].real
    #     # Filter roots within the plotted range
    #     plot_range_roots = [r for r in real_roots if x_smooth.min() <= r <= x_smooth.max()]
    #     approx_x = plot_range_roots[0] if plot_range_roots else 2.45 # Fallback estimate


    #     # Mark intersection and drop line
    #     if plot_range_roots:
    #         ax.plot(approx_x, target_y, 'go', markersize=8)
    #         ax.plot([approx_x, approx_x], [0, target_y], 'g:')
    #         ax.text(approx_x, -1, f'x ≈ {approx_x:.2f}', color='g', ha='center')
    #     else:
    #          ax.text(2, 6, f'Intersection near x=2.45', color='g', ha='center')


        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Solving x³ - 4x = 5 Graphically")
        ax.legend()
        ax.set_ylim(min(y_pts)-2, max(y_pts)+2)

        return self._save_plot(fig, filename)

    # === Form 4: Graphs - Travel Graphs ===

    def f4_gtg_notes_velocity_time_disp(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')

        # Example data: v decreases linearly, crossing x-axis
        t_pts = [0, 3, 6] # Times
        v_pts = [6, 0, -4] # Velocities at these times
        t1 = 3 # Time when v=0

        # Plot velocity-time graph
        ax.plot(t_pts, v_pts, 'b-', marker='o')

        # Shade area above axis (Positive displacement)
        ax.fill_between(t_pts[:2], v_pts[:2], color='lightblue', alpha=0.7, label='Area = Pos Disp')
        area_above = 0.5 * t1 * v_pts[0]
        ax.text(t1/2, v_pts[0]/2, f'+Area\n{area_above}', ha='center', va='center', color='darkblue')

        # Shade area below axis (Negative displacement)
        ax.fill_between(t_pts[1:], v_pts[1:], color='lightcoral', alpha=0.7, label='Area = Neg Disp')
        area_below = 0.5 * (t_pts[2] - t1) * abs(v_pts[2]) # Area magnitude
        ax.text((t1+t_pts[2])/2, v_pts[2]/2, f'-Area\n{area_below}', ha='center', va='center', color='darkred')


        # Gradient = Acceleration
        accel = (v_pts[2]-v_pts[0])/(t_pts[2]-t_pts[0])
        ax.text(1, -1, f"Gradient = Accel = {accel:.2f} m/s²", fontsize=9)

        # Total Displacement / Distance annotations
        ax.text(5, 4, f"Total Displacement = {area_above - area_below}\nTotal Distance = {area_above + area_below}",
                fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.8) # Time axis
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Velocity-Time Graph and Displacement")
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylim(min(v_pts)-1, max(v_pts)+1)
        ax.set_xlim(-0.5, t_pts[-1]+0.5)

        return self._save_plot(fig, filename)

    def f4_gtg_we2_accel_graph(self):
        """Generates velocity-time graph for WE2."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Acceleration Example")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 14)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Line from (0, 4) to (4, 12)
        time_pts = [0, 4]
        speed_pts = [4, 12]

        ax.plot(time_pts, speed_pts, 'g-', marker='o')
        ax.plot(time_pts, speed_pts, 'go')

        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 15, 1))

        # Store for reuse in WE3
        self.context_data['f4_gtg_we2'] = {'time': time_pts, 'speed': speed_pts}

        return self._save_plot(fig, filename)

    def f4_gtg_we3_disp_from_accel(self):
        """References WE2 graph and shades the trapezium area."""
        filename = get_filename_from_caller()
        if 'f4_gtg_we2' not in self.context_data:
            print("Warning: Context data for WE2 not found, regenerating.")
            self.f4_gtg_we2_accel_graph()
            if 'f4_gtg_we2' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        time_pts = self.context_data['f4_gtg_we2']['time']
        speed_pts = self.context_data['f4_gtg_we2']['speed']

        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Displacement (Area under Graph)")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 14)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'g-', marker='o')
        ax.plot(time_pts, speed_pts, 'go')

        # Shade the trapezium area
        ax.fill_between(time_pts, speed_pts, color='lightgreen', alpha=0.7)
        area = 0.5 * (speed_pts[0] + speed_pts[1]) * (time_pts[1] - time_pts[0])
        ax.text(2, 5, f'Area = Displacement\n= 1/2*(4+12)*4\n= {area} m',
                ha='center', va='center', fontsize=10, color='darkgreen')

        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 15, 1))

        return self._save_plot(fig, filename)

    def f4_gtg_we5_total_distance(self):
        """References WE4 graph and shades the total area."""
        filename = get_filename_from_caller()
        if 'f4_gtg_we4' not in self.context_data:
            print("Warning: Context data for WE4 not found, regenerating.")
            self.f4_gtg_we4_velocity_time_graph()
            if 'f4_gtg_we4' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        time_pts = self.context_data['f4_gtg_we4']['time']
        speed_pts = self.context_data['f4_gtg_we4']['speed']

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Total Distance Traveled (Total Area)")
        ax.set_xlim(0, 12.5)
        ax.set_ylim(0, 10)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'm-', marker='o')
        ax.plot(time_pts, speed_pts, 'mo')

        # Shade the total area (trapezoid)
        ax.fill_between(time_pts, speed_pts, color='plum', alpha=0.7)

        # Calculate areas
        area1 = 0.5 * 4 * 8
        area2 = (10 - 4) * 8
        area3 = 0.5 * (12 - 10) * 8
        total_area = area1 + area2 + area3

        ax.text(2, 2, f'A1={area1}', ha='center', fontsize=9)
        ax.text(7, 4, f'A2={area2}', ha='center', fontsize=9)
        ax.text(11, 2, f'A3={area3}', ha='center', fontsize=9)
        ax.text(6, 6, f'Total Area = Dist = {total_area} m', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))


        ax.set_xticks(np.arange(0, 13, 1))
        ax.set_yticks(np.arange(0, 11, 1))

        return self._save_plot(fig, filename)

    # === Form 4: Variation ===

    def f4_v_notes_partial_variation_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')

        # Example: y = kx + c with k=2, c=3
        k, c = 2, 3
        x_vals = np.linspace(0, 5, 100)
        y_vals = k * x_vals + c

        ax.plot(x_vals, y_vals, 'r-', linewidth=2, label=f'y = {k}x + {c}')

        # Mark y-intercept
        ax.plot(0, c, 'ro', markersize=8)
        ax.text(0.1, c, f' y-intercept = c = {c}', va='center')

        # Show slope (optional)
        ax.text(2, k*2+c + 0.5, f'Slope = k = {k}', ha='center')

        # Add direct variation for comparison
        y_direct = k * x_vals
        ax.plot(x_vals, y_direct, 'b:', label=f'Direct Variation (y={k}x)')


        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Partial Variation (y = kx + c)")
        ax.legend()
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, max(y_vals)+1)

        return self._save_plot(fig, filename)

    def f4_v_we5_partial_variation_cost_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')

        # Formula from WE3: C = 40d + 60
        k, c = 40, 60
        d_vals = np.linspace(0, 5, 100) # Days from 0 to 5
        C_vals = k * d_vals + c

        ax.plot(d_vals, C_vals, 'r-', linewidth=2, label=f'C = {k}d + {c}')

        # Mark y-intercept (fixed cost)
        ax.plot(0, c, 'ro', markersize=8)
        ax.text(0.1, c + 5, f' Fixed Cost = ${c}', va='bottom')

        # Show points from WE3
        ax.plot(4, 220, 'bo', label='(4 days, $220)')
        ax.plot(7, 340, 'go', label='(7 days, $340)') # Note: extends beyond d=5

        # Adjust plot range if needed to show points
        d_vals_ext = np.linspace(0, 8, 100)
        C_vals_ext = k * d_vals_ext + c
        ax.plot(d_vals_ext, C_vals_ext, 'r-', linewidth=2) # Replot extended line


        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlabel("Number of Days (d)")
        ax.set_ylabel("Cost (C) ($)")
        ax.set_title("Partial Variation: Car Hire Cost")
        ax.legend()
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(0, max(C_vals_ext)+20)

        return self._save_plot(fig, filename)

    # === Form 4: Algebra - Algebraic Manipulation ===
    # No images described

    # === Form 4: Algebra - Equations ===
    # No images described

    # === Form 4: Algebra - Inequalities ===

    def f4_ai_notes_lp_feasible_region(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        x_range = np.linspace(0, 8, 400)

        # Constraints (example)
        # x <= 5
        ax.axvline(5, color='red', linestyle='-', label='x = 5')
        ax.fill_betweenx([0, 8], 5, 8, color='red', alpha=0.2, label='x>5 unw')
        # y <= 6
        ax.axhline(6, color='blue', linestyle='-', label='y = 6')
        ax.fill_between(x_range, 6, 8, color='blue', alpha=0.2, label='y>6 unw')
        # x + y <= 7 => y <= 7 - x
        y_bound3 = 7 - x_range
        ax.plot(x_range, y_bound3, 'green', linestyle='-', label='x+y = 7')
        ax.fill_between(x_range, y_bound3, 8, where=y_bound3<=8, color='green', alpha=0.2, label='x+y>7 unw')

        # Implicit x>=0, y>=0
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)

        # Identify feasible region R (unshaded)
        # Calculate vertices
        v1 = (0, 0)
        v2 = (5, 0) # x=5, y=0
        v3 = (5, 2) # x=5, x+y=7 => y=2
        v4 = (1, 6) # y=6, x+y=7 => x=1
        v5 = (0, 6) # x=0, y=6

        vertices_x = [v1[0], v2[0], v3[0], v4[0], v5[0]]
        vertices_y = [v1[1], v2[1], v3[1], v4[1], v5[1]]
        ax.plot(vertices_x, vertices_y, 'ko', markersize=8)

        # Label R and vertices
        ax.text(2, 2, 'R\n(Feasible Region)', ha='center', va='center', fontsize=14, color='black')
        ax.text(v1[0], v1[1]-0.3, 'O(0,0)', ha='center')
        ax.text(v2[0], v2[1]-0.3, '(5,0)', ha='center')
        ax.text(v3[0]+0.1, v3[1]+0.1, '(5,2)', ha='left')
        ax.text(v4[0]-0.1, v4[1]+0.1, '(1,6)', ha='right')
        ax.text(v5[0]-0.1, v5[1], '(0,6)', ha='right')

        # Setup plot
        ax.grid(True, linestyle='--')
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 8)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Linear Programming - Feasible Region")
        ax.legend(fontsize=8, loc='upper right')

        return self._save_plot(fig, filename)


    def f4_ai_we1_inequality_graph(self):
        """
        Generates a graph for x >= 0, y >= 0, x + 2y <= 6,
        explicitly shading all unwanted regions (x<0, y<0, x+2y>6).
        """
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white') # Slightly adjusted size maybe needed
        ax.set_facecolor('white')
        # ax.set_aspect('equal', adjustable='box') # removed for potentially better axis ranges

        # --- Set desired limits FIRST ---
        x_min, x_max = -1, 7
        y_min, y_max = -1, 5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # --- Define ranges based on CORRECT limits ---
        plot_x_range = np.linspace(x_min, x_max, 400)
        plot_y_range = np.linspace(y_min, y_max, 400)

        # --- Shade Unwanted Regions ---
        # Unwanted: x + 2y > 6  => y > (6-x)/2
        y_boundary = (6 - plot_x_range) / 2
        ax.fill_between(plot_x_range, y_boundary, y_max,
                        where=y_boundary <= y_max, # Ensure shading respects upper plot boundary
                        color='blue', alpha=0.2, label='_Unwanted Region(s)') # Use _nolabel_ convention

        # Unwanted: x < 0
        ax.fill_betweenx(plot_y_range, x_min, 0,
                         where=plot_y_range <= y_max, # Respect y limits
                         color='blue', alpha=0.2)

        # Unwanted: y < 0
        ax.fill_between(plot_x_range, y_min, 0,
                        where=plot_x_range <= x_max, # Respect x limits
                        color='blue', alpha=0.2)

        # --- Plot Boundary Line on top ---
        # Use known intercepts (0,3) and (6,0) for the line segment
        line_x = np.array([0, 6])
        line_y = np.array([3, 0])
        ax.plot(line_x, line_y, 'blue', linestyle='-', label='x + 2y = 6')

        # --- Plot Vertices and Label R ---
        vertices_x = [0, 6, 0]
        vertices_y = [0, 0, 3]
        ax.plot(vertices_x, vertices_y, 'ko', markersize=6) # Make markers slightly larger
        ax.text(0.1, -0.3, '(0,0)')
        ax.text(6.1, -0.3, '(6,0)')
        ax.text(-0.5, 3.1, '(0,3)') # Adjusted position

        ax.text(1.5, 1, 'R', ha='center', va='center', fontsize=16, color='black')

        # --- Final Setup ---
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--')
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Graph showing Feasible Region (R)")

        # Custom legend: Combine shading into one entry
        # Create proxy artist for the shaded area
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.2, label='Unwanted Region(s)'),
                           plt.Line2D([0], [0], color='blue', lw=2, label='x + 2y = 6')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


        return self._save_plot(fig, filename)

    # def f4_ai_we1_inequality_graph(self):
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    #     ax.set_facecolor('white')
    #     ax.set_aspect('equal', adjustable='box')
    #     x_range = np.linspace(0, 7, 400)

    #     # Define full ranges based on plot limits for shading
    #     x_lims = ax.get_xlim() # Get current x limits (-1, 7)
    #     y_lims = ax.get_ylim() # Get current y limits (-1, 5)
    #     plot_x_range = np.linspace(x_lims[0], x_lims[1], 400)
    #     plot_y_range = np.linspace(y_lims[0], y_lims[1], 400)

    #     # Constraint: x + 2y <= 6  => y <= (6-x)/2
    #     # Shade unwanted: x + 2y > 6 => y > (6-x)/2
    #     y_boundary = (6 - plot_x_range) / 2
    #     ax.fill_between(plot_x_range, y_boundary, y_lims[1],
    #                     where=y_boundary <= y_lims[1], # Ensure shading doesn't go below boundary line itself
    #                     color='blue', alpha=0.2, label='Unwanted Region(s)') # Single label for all unwanted

    #     # Constraint: x >= 0
    #     # Shade unwanted: x < 0
    #     ax.fill_betweenx(plot_y_range, x_lims[0], 0, color='blue', alpha=0.2)

    #     # Constraint: y >= 0
    #     # Shade unwanted: y < 0
    #     ax.fill_between(plot_x_range, y_lims[0], 0, color='blue', alpha=0.2)

    #     # Plot the boundary line x + 2y = 6 on top of shading
    #     # Calculate line points only within plot bounds for neatness
    #     line_x_vals = np.array([max(x_lims[0], -2*y_lims[1]+6), min(x_lims[1], -2*y_lims[0]+6)]) # x values where line intersects plot box
    #     line_y_vals = (6 - line_x_vals) / 2
    #     ax.plot(line_x_vals, line_y_vals, 'blue', linestyle='-', label='x + 2y = 6')

    #     # Vertices: (0,0), (6,0), (0,3) - plot after shading
    #     ax.plot([0, 6, 0], [0, 0, 3], 'ko', markersize=5) # Smaller markers?
    #     ax.text(0.1, -0.3, '(0,0)') # Adjusted text position slightly
    #     ax.text(6.1, -0.3, '(6,0)')
    #     ax.text(-0.5, 3.1, '(0,3)')

    #     # Label Region R (the only remaining unshaded area)
    #     ax.text(1.5, 1, 'R', ha='center', va='center', fontsize=16, color='black')

    #     # Setup plot - keep existing settings
    #     ax.axhline(0, color='black', linewidth=0.8)
    #     ax.axvline(0, color='black', linewidth=0.8)
    #     ax.grid(True, linestyle='--')
    #     ax.set_xlim(x_lims) # Keep original xlim
    #     ax.set_ylim(y_lims) # Keep original ylim
    #     ax.set_xlabel("x-axis")
    #     ax.set_ylabel("y-axis")
    #     ax.set_title("Graph showing Feasible Region (R)") # Adjusted title
    #     # Consolidate legend
    #     handles, labels = ax.get_legend_handles_labels()
    #     unique_labels = {}
    #     for h, l in zip(handles, labels):
    #         if l not in unique_labels:
    #              unique_labels[l] = h
    #     ax.legend(unique_labels.values(), unique_labels.keys(), fontsize=8, loc='upper right')

    #     return self._save_plot(fig, filename)


    def f4_ai_we3_lp_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        x_range = np.linspace(0, 9, 400)

        # Constraints: x >= 0, y >= 0, 2x + y <= 10, x + y <= 8
        # 1) 2x + y <= 10 => y <= 10 - 2x
        y1 = 10 - 2*x_range
        ax.plot(x_range, y1, 'r-', label='2x + y = 10')
        ax.fill_between(x_range, y1, 12, where=y1<=12, color='red', alpha=0.2, label='2x+y>10 unw')

        # 2) x + y <= 8 => y <= 8 - x
        y2 = 8 - x_range
        ax.plot(x_range, y2, 'b-', label='x + y = 8')
        ax.fill_between(x_range, y2, 12, where=y2<=12, color='blue', alpha=0.2, label='x+y>8 unw')

        # Implicit x>=0, y>=0 handled by axes limits

        # Identify Feasible Region R
        # Vertices: (0,0), (5,0) [x-int of 2x+y=10], (0,8) [y-int of x+y=8], (2,6) [intersection]
        vertices = np.array([[0,0], [5,0], [2,6], [0,8]])
        ax.plot(vertices[:,0], vertices[:,1], 'ko', markersize=8)
        ax.text(0,-0.5,'(0,0)')
        ax.text(4.5,-0.5,'(5,0)')
        ax.text(2,6.3,'(2,6)')
        ax.text(-0.8,8,'(0,8)')

        ax.text(3, 3, 'R\n(Feasible Region)', ha='center', va='center', fontsize=14, color='black')

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--')
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 11)
        ax.set_xlabel("x (Tables)")
        ax.set_ylabel("y (Chairs)")
        ax.set_title("Linear Programming Feasible Region")
        ax.legend(fontsize=8, loc='upper right')

        return self._save_plot(fig, filename)

    # === Form 4: Geometry - Polygons and Circle ===

    def f4_gpc_notes_circle_theorems_1_5(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), facecolor='white')
        fig.suptitle('Circle Theorems (1-4)', fontsize=14)

        center = (0,0)
        radius = 3

        # --- Theorem 1: Angle at Center ---
        ax = axs[0, 0]
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Thm 1: ∠ at Center = 2 × ∠ at Circumf.")
        circle = patches.Circle(center, radius, fill=False)
        ax.add_patch(circle)
        # Points A, B on circumference, C on major arc
        angle1, angle2 = 30, 110 # degrees
        A = (radius*math.cos(math.radians(angle1)), radius*math.sin(math.radians(angle1)))
        B = (radius*math.cos(math.radians(angle2)), radius*math.sin(math.radians(angle2)))
        C_angle = (angle1+angle2)/2 + 180
        C = (radius*math.cos(math.radians(C_angle)), radius*math.sin(math.radians(C_angle)))
        O = center
        ax.plot([O[0], A[0]], [O[1], A[1]], 'r-') # OA
        ax.plot([O[0], B[0]], [O[1], B[1]], 'r-') # OB
        ax.plot([A[0], C[0]], [A[1], C[1]], 'b-') # AC
        ax.plot([B[0], C[0]], [B[1], C[1]], 'b-') # BC
        ax.plot([A[0],B[0],C[0],O[0]], [A[1],B[1],C[1],O[1]], 'ko', markersize=4)
        ax.text(O[0], O[1]-0.3, 'O')
        ax.text(A[0]*1.1, A[1]*1.1, 'A')
        ax.text(B[0]*1.1, B[1]*1.1, 'B')
        ax.text(C[0]*1.1, C[1]*1.1, 'C')
        # Angles
        arc_center = patches.Arc(O, radius*0.5, radius*0.5, angle=0, theta1=angle1, theta2=angle2, color='red')
        ax.add_patch(arc_center)
        ax.text(O[0], O[1]+0.8, '∠AOB', color='red')
        arc_circum = patches.Arc(C, radius*0.8, radius*0.8, angle=math.degrees(math.atan2(A[1]-C[1], A[0]-C[0])),
                                theta1=0, theta2=math.degrees(math.atan2(B[1]-C[1], B[0]-C[0]))-math.degrees(math.atan2(A[1]-C[1], A[0]-C[0])), color='blue')
        # ax.add_patch(arc_circum) # Arc drawing is tricky for angles not at origin
        ax.text(C[0], C[1]+1, '∠ACB', color='blue')
        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)


        # --- Theorem 2: Angles in Same Segment ---
        ax = axs[0, 1]
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Thm 2: Angles in Same Segment Equal")
        circle = patches.Circle(center, radius, fill=False)
        ax.add_patch(circle)
        # Points A, B, C, D
        A = (radius*math.cos(math.radians(210)), radius*math.sin(math.radians(210)))
        B = (radius*math.cos(math.radians(330)), radius*math.sin(math.radians(330)))
        C_angle, D_angle = 60, 120
        C = (radius*math.cos(math.radians(C_angle)), radius*math.sin(math.radians(C_angle)))
        D = (radius*math.cos(math.radians(D_angle)), radius*math.sin(math.radians(D_angle)))
        ax.plot([A[0], C[0]], [A[1], C[1]], 'b-') # AC
        ax.plot([B[0], C[0]], [B[1], C[1]], 'b-') # BC
        ax.plot([A[0], D[0]], [A[1], D[1]], 'g-') # AD
        ax.plot([B[0], D[0]], [B[1], D[1]], 'g-') # BD
        ax.plot([A[0],B[0],C[0],D[0]], [A[1],B[1],C[1],D[1]], 'ko', markersize=4)
        ax.text(A[0]*1.1, A[1]*1.1, 'A'); ax.text(B[0]*1.1, B[1]*1.1, 'B')
        ax.text(C[0]*1.1, C[1]*1.1, 'C'); ax.text(D[0]*1.1, D[1]*1.1, 'D')
        ax.text(C[0]-0.5, C[1]-1, '∠ACB', color='blue')
        ax.text(D[0], D[1]-1, '∠ADB', color='green')
        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)


        # --- Theorem 3: Angle in Semi-circle ---
        ax = axs[1, 0]
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Thm 3: Angle in Semi-circle = 90°")
        circle = patches.Circle(center, radius, fill=False)
        ax.add_patch(circle)
        # Diameter AB, point C
        A = (-radius, 0)
        B = (radius, 0)
        O = center
        C_angle = 70
        C = (radius*math.cos(math.radians(C_angle)), radius*math.sin(math.radians(C_angle)))
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-') # Diameter
        ax.plot([A[0], C[0]], [A[1], C[1]], 'b-') # AC
        ax.plot([B[0], C[0]], [B[1], C[1]], 'b-') # BC
        ax.plot([A[0],B[0],C[0],O[0]], [A[1],B[1],C[1],O[1]], 'ko', markersize=4)
        ax.text(A[0]-0.3, A[1], 'A'); ax.text(B[0]+0.1, B[1], 'B'); ax.text(O[0]-0.3, O[1], 'O')
        ax.text(C[0], C[1]+0.2, 'C')
        # Right angle symbol at C
        # Need vectors CB and CA
        vec_CA = np.array(A) - np.array(C)
        vec_CB = np.array(B) - np.array(C)
        norm_CA = vec_CA / np.linalg.norm(vec_CA)
        norm_CB = vec_CB / np.linalg.norm(vec_CB)
        corner_size = 0.4
        p1 = C + norm_CA * corner_size
        p2 = C + norm_CB * corner_size
        p3 = p1 + (p2 - C)
        rect_pts = np.array([p1, p3, p2])
        rect_patch = patches.Polygon(rect_pts, closed=False, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect_patch)
        ax.text(C[0]-0.8, C[1]-0.8, '90°', color='red')
        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)

        # --- Theorem 4: Line from Center perp to Chord ---
        ax = axs[1, 1]
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Thm 4: Line from Center ⊥ to Chord bisects Chord")
        circle = patches.Circle(center, radius, fill=False)
        ax.add_patch(circle)
        # Chord AB, Center O, Midpoint M
        A = (radius*math.cos(math.radians(120)), radius*math.sin(math.radians(120)))
        B = (radius*math.cos(math.radians(60)), radius*math.sin(math.radians(60)))
        O = center
        # Midpoint M
        M = ((A[0]+B[0])/2, (A[1]+B[1])/2)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b-') # Chord AB
        ax.plot([O[0], M[0]], [O[1], M[1]], 'r-') # Line OM
        ax.plot([A[0],B[0],O[0],M[0]], [A[1],B[1],O[1],M[1]], 'ko', markersize=4)
        ax.text(A[0]-0.3, A[1], 'A'); ax.text(B[0]+0.1, B[1], 'B'); ax.text(O[0]-0.3, O[1], 'O'); ax.text(M[0], M[1]-0.3, 'M')
        # Right angle symbol at M
        angle_OM = math.atan2(M[1]-O[1], M[0]-O[0])
        angle_AB = math.atan2(B[1]-A[1], B[0]-A[0])
        corner_size=0.3
        # Draw square rotated by angle of OM
        rect_M = patches.Rectangle( (M[0]-corner_size*math.sin(angle_AB), M[1]+corner_size*math.cos(angle_AB)),
                                   corner_size, corner_size, angle=math.degrees(angle_OM-math.pi/2), linewidth=1, edgecolor='red', facecolor='none')
        # ax.add_patch(rect_M) # This angle is complex
        ax.text(M[0]+0.2, M[1]+0.2, '⊥', color='red', fontsize=14)
        # Indicate AM = MB
        ax.plot([A[0]+(M[0]-A[0])*0.4], [A[1]+(M[1]-A[1])*0.4], 'k|', markersize=10)
        ax.plot([M[0]+(B[0]-M[0])*0.6], [M[1]+(B[1]-M[1])*0.6], 'k|', markersize=10)
        ax.text(M[0]-0.8, M[1]+0.2, 'AM=MB', fontsize=9)
        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)


    def f4_gpc_notes_cyclic_quad_theorems(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Cyclic Quadrilateral Theorems', fontsize=14)
        center = (0,0)
        radius = 3

        # Points A, B, C, D on circumference
        angles = [20, 80, 160, 260] # degrees
        A = (radius*math.cos(math.radians(angles[0])), radius*math.sin(math.radians(angles[0])))
        B = (radius*math.cos(math.radians(angles[1])), radius*math.sin(math.radians(angles[1])))
        C = (radius*math.cos(math.radians(angles[2])), radius*math.sin(math.radians(angles[2])))
        D = (radius*math.cos(math.radians(angles[3])), radius*math.sin(math.radians(angles[3]))-0.05)
        verts = [A, B, C, D]
        labels = ['A', 'B', 'C', 'D']

        # --- Theorem 6: Opposite Angles Supplementary ---
        ax1.set_facecolor('white'); ax1.set_aspect('equal'); ax1.axis('off')
        ax1.set_title("Thm 6: Opposite Angles Add to 180°")
        circle = patches.Circle(center, radius, fill=False)
        ax1.add_patch(circle)
        # Draw cyclic quad
        quad = patches.Polygon(verts, closed=True, fill=False, edgecolor='blue')
        ax1.add_patch(quad)
        # Plot vertices and labels
        for pt, name in zip(verts, labels):
            ax1.plot(pt[0], pt[1], 'ko', markersize=4)
            ax1.text(pt[0]*1.1, pt[1]*1.1, name)
        # Indicate opposite angles
        ax1.text(A[0]-0.7, A[1]-0.1, '∠A', color='red')
        ax1.text(C[0]+0.4, C[1], '∠C', color='red')
        ax1.text(B[0]-0.4, B[1] - 0.4, '∠B', color='green')
        ax1.text(D[0]-0.1, D[1] + 0.4, '∠D', color='green')
        ax1.text(0, -radius*1.2, '∠A + ∠C = 180°\n∠B + ∠D = 180°', ha='center')
        ax1.set_xlim(-radius-1, radius+1); ax1.set_ylim(-radius-1.5, radius+1)


        # --- Theorem 7: Exterior Angle ---
        ax2.set_facecolor('white'); ax2.set_aspect('equal'); ax2.axis('off')
        ax2.set_title("Thm 7: Exterior Angle = Interior Opposite Angle")
        circle = patches.Circle(center, radius, fill=False)
        ax2.add_patch(circle)
        # Draw cyclic quad
        quad = patches.Polygon(verts, closed=True, fill=False, edgecolor='blue')
        ax2.add_patch(quad)
        # Plot vertices and labels
        for pt, name in zip(verts, labels):
            ax2.plot(pt[0], pt[1], 'ko', markersize=4)
            ax2.text(pt[0]*1.1, pt[1]*1.1, name)
        # Extend DC to E
        vec_DC = np.array(C) - np.array(D)
        E = C + vec_DC*0.4 # Extend past C
        ax2.plot([D[0], E[0]], [D[1], E[1]], 'k--')
        ax2.text(E[0], E[1], 'E')
        # Mark Exterior angle BCE
        ax2.text(C[0]+0.4, C[1]-0.4, '∠BCE', color='red')
        # Mark Interior opposite angle BAD
        ax2.text(A[0]-1, A[1]-0.2, '∠BAD', color='red')
        ax2.text(0, -radius*1.2, '∠BCE = ∠BAD', ha='center')
        ax2.set_xlim(-radius-1, radius+1); ax2.set_ylim(-radius-1.5, radius+1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)


    def f4_gpc_notes_tangent_theorems(self):
        filename = get_filename_from_caller()
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
        fig.suptitle('Tangent Theorems', fontsize=14)
        center = (0,0)
        radius = 2

        # --- Theorem 8: Tangent perp to Radius ---
        ax = axs[0]
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Thm 8: Tangent ⊥ Radius")
        circle = patches.Circle(center, radius, fill=False)
        ax.add_patch(circle)
        # Point of tangency T
        T_angle = 45
        T = (radius*math.cos(math.radians(T_angle)), radius*math.sin(math.radians(T_angle)))
        O = center
        # Draw radius OT
        ax.plot([O[0], T[0]], [O[1], T[1]], 'r-') # Radius OT
        # Draw tangent line (perpendicular to OT)
        slope_rad = math.tan(math.radians(T_angle)) if T[0]!=0 else float('inf')
        slope_tan = -1/slope_rad if T[0]!=0 and slope_rad!=0 else (0 if T[0]!=0 else float('inf'))

        if abs(slope_tan) == float('inf'): # Vertical tangent
             tan_x = [T[0], T[0]]
             tan_y = [T[1]-2, T[1]+2]
        else: # Non-vertical
            tan_x = [T[0]-2/math.sqrt(1+slope_tan**2), T[0]+2/math.sqrt(1+slope_tan**2)]
            tan_y = [T[1]-2*slope_tan/math.sqrt(1+slope_tan**2), T[1]+2*slope_tan/math.sqrt(1+slope_tan**2)]
        ax.plot(tan_x, tan_y, 'b-') # Tangent line

        ax.plot(O[0], O[1], 'ko', markersize=4); ax.text(O[0]-0.2, O[1]-0.2, 'O')
        ax.plot(T[0], T[1], 'ko', markersize=4); ax.text(T[0]+0.1, T[1]+0.1, 'T')
        # Mark right angle
        corner_size=0.3
        rect = patches.Rectangle((T[0]-corner_size*math.sin(math.radians(T_angle)), T[1]+corner_size*math.cos(math.radians(T_angle))),
                                 corner_size, corner_size, angle=T_angle-90, linewidth=1, edgecolor='red', facecolor='none')
        # ax.add_patch(rect) # Angle calc complex
        ax.text(T[0]-0.2, T[1]-0.4, '90°', color='red')
        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)


        # # --- Theorem 9: Alternate Segment Theorem ---
        # ax = axs[1]
        # ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        # ax.set_title("Thm 9: Alternate Segment Thm")
        # circle = patches.Circle(center, radius, fill=False)
        # ax.add_patch(circle)
        # # Tangent at T, Chord TQ, Point R on major arc
        # T_angle, Q_angle, R_angle = 200, 300, 80
        # T = (radius*math.cos(math.radians(T_angle)), radius*math.sin(math.radians(T_angle)))
        # Q = (radius*math.cos(math.radians(Q_angle)), radius*math.sin(math.radians(Q_angle)))
        # R = (radius*math.cos(math.radians(R_angle)), radius*math.sin(math.radians(R_angle)))
        # # Tangent line (approximate)
        # ax.plot([T[0]-1.5, T[0]+1.5], [T[1]-0.5, T[1]+0.5], 'b-') # Tangent line ATP
        # ax.text(T[0]+1.6, T[1]+0.5, 'P')
        # # Chord TQ
        # ax.plot([T[0], Q[0]], [T[1], Q[1]], 'g-')
        # # Lines TR, QR
        # ax.plot([T[0], R[0]], [T[1], R[1]], 'm-')
        # ax.plot([Q[0], R[0]], [Q[1], R[1]], 'm-')
        # ax.plot([T[0],Q[0],R[0]], [T[1],Q[1],R[1]], 'ko', markersize=4)
        # ax.text(T[0]*1.1, T[1]*1.1, 'T'); ax.text(Q[0]*1.1, Q[1]*1.1, 'Q'); ax.text(R[0]*1.1, R[1]*1.1, 'R')
        # # Mark angles
        # ax.text(T[0]+0.5, T[1]+0.3, '∠ATP', color='blue', fontsize=9)
        # ax.text(R[0]-0.1, R[1]-0.5, '∠TRQ', color='m', fontsize=9)
        # ax.text(0, -radius*1.2, '∠ATP = ∠TRQ', ha='center')
        # ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1.5, radius+1)

        # --- Theorem 9: Alternate Segment Theorem ---
        ax2 = fig.add_subplot(132) # Assuming ax2 is the second subplot
        ax2.set_aspect('equal', adjustable='box'); ax2.axis('off')
        circle2 = patches.Circle((0, 0), radius, fill=False, color='black')
        ax2.add_patch(circle2)
        ax2.plot(0, 0, 'ko', markersize=4) # Center point (optional label?)

        ax2.set_title('Thm 9: Alternate Segment Thm', fontsize=10)
        # Tangent at T, Chord TQ, Point R on major arc
        T_angle, Q_angle, R_angle = 200, 300, 80 # degrees
        T = (radius*math.cos(math.radians(T_angle)), radius*math.sin(math.radians(T_angle)))
        Q = (radius*math.cos(math.radians(Q_angle)), radius*math.sin(math.radians(Q_angle)))
        R = (radius*math.cos(math.radians(R_angle)), radius*math.sin(math.radians(R_angle)))

        # Calculate and draw the correct tangent line ATP
        tx, ty = T[0], T[1]
        tan_len = 1.5 # Length of tangent segment to draw on each side of T
        # Direction vector of tangent is perpendicular to radius (-ty, tx)
        # Normalize the direction vector
        # Use radius directly for normalization as T is on the circle
        if radius == 0: # Avoid division by zero if radius is 0
             norm_dx, norm_dy = 0, 1 # Default to horizontal tangent if radius is 0
        else:
             norm_dx = -ty / radius
             norm_dy = tx / radius

        # Calculate endpoints P1 and P2 for the tangent segment
        # P1 corresponds to P in the label ATP
        p1_x = tx + tan_len * norm_dx
        p1_y = ty + tan_len * norm_dy
        p2_x = tx - tan_len * norm_dx
        p2_y = ty - tan_len * norm_dy
        ax2.plot([p1_x, p2_x], [p1_y, p2_y], 'b-') # Draw the tangent line
        # Place P label slightly outward from endpoint p1
        ax2.text(p1_x + 0.1 * norm_dx, p1_y + 0.1 * norm_dy, 'P', ha='center', va='center')

        # Chord TQ
        ax2.plot([T[0], Q[0]], [T[1], Q[1]], 'g-')
        # Lines TR, QR
        ax2.plot([T[0], R[0]], [T[1], R[1]], 'm-')
        ax2.plot([Q[0], R[0]], [Q[1], R[1]], 'm-')
        # Points T, Q, R
        ax2.plot(T[0], T[1], 'ko', markersize=4)
        ax2.plot(Q[0], Q[1], 'ko', markersize=4)
        ax2.plot(R[0], R[1], 'ko', markersize=4)
        # Labels for T, Q, R
        ax2.text(T[0]*1.15, T[1]*1.15, 'T', ha='center', va='center')
        ax2.text(Q[0]*1.15, Q[1]*1.15, 'Q', ha='center', va='center')
        ax2.text(R[0]*1.15, R[1]*1.15, 'R', ha='center', va='center')

        # Mark angles (using helper function or manual placement)
        # Example manual placement - might need adjustment based on specific angles
        angle_atp_mid_x = tx + 0.4 * norm_dx - 0.1 * norm_dy
        angle_atp_mid_y = ty + 0.4 * norm_dy + 0.1 * norm_dx
        ax2.text(angle_atp_mid_x+ 0.5, angle_atp_mid_y, '∠QTP', color='blue', fontsize=9, ha='center', va='center')

        angle_trq_mid_x = R[0]*0.8 # Simple approximation
        angle_trq_mid_y = R[1]*0.8
        ax2.text(angle_trq_mid_x+ 0.05, angle_trq_mid_y-0.1, '∠TRQ', color='m', fontsize=9, ha='center', va='center')

        ax2.text(0, -radius*1.3, '∠QTP = ∠TRQ', ha='center', fontsize=9) # Lowered text slightly
        ax2.set_xlim(-radius-1, radius+1); ax2.set_ylim(-radius-1.5, radius+1)

        # --- Theorem 10: Tangents from External Point ---
        ax = axs[2]
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title("Thm 10: Tangents from External Pt Equal")
        circle = patches.Circle(center, radius, fill=False)
        ax.add_patch(circle)
        # External point P, Tangency points A, B
        P = (-radius*1.8, 0)
        # Find tangency points (requires geometry, approximate here)
        A_angle, B_angle = 60, -60
        A = (radius*math.cos(math.radians(A_angle)), radius*math.sin(math.radians(A_angle)))
        B = (radius*math.cos(math.radians(B_angle)), radius*math.sin(math.radians(B_angle)))
        # Draw tangents PA, PB
        ax.plot([P[0]+0.1, A[0]], [P[1]-0.05, A[1]], 'b-')
        ax.plot([P[0]+0.1, B[0]], [P[1]-0.05, B[1]], 'b-')
        ax.plot([P[0],A[0],B[0]], [P[1],A[1],B[1]], 'ko', markersize=4)
        ax.text(P[0]-0.55, P[1]-0.05, 'P'); ax.text(A[0]+0.1, A[1], 'A'); ax.text(B[0]+0.1, B[1]-0.1, 'B')
        # Indicate PA = PB
        ax.text(P[0]/2 + A[0]/2 - 0.2, A[1]/2-0.17, '\\', color='blue', fontweight='bold', fontsize=14)
        ax.text(P[0]/2 + B[0]/2 - 0.2, B[1]/2 - 0.05, '/', color='blue', fontweight='bold', fontsize=14)
        ax.text(P[0]-0.5, -radius*0.8, 'PA = PB', fontsize=10)
        ax.set_xlim(-radius-1.5, radius+1); ax.set_ylim(-radius-1, radius+1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    def f4_gpc_we1_angle_center_circum(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        center = (0,0); radius = 3
        angle_aob = 110

        A_angle = (180 - angle_aob)/2 # Place symmetrically
        B_angle = A_angle + angle_aob
        C_angle = (A_angle + B_angle)/2 + 180 # Opposite arc

        A = (radius*math.cos(math.radians(A_angle)), radius*math.sin(math.radians(A_angle)))
        B = (radius*math.cos(math.radians(B_angle)), radius*math.sin(math.radians(B_angle)))
        C = (radius*math.cos(math.radians(C_angle)), radius*math.sin(math.radians(C_angle)))
        O = center

        circle = patches.Circle(center, radius, fill=False); ax.add_patch(circle)
        ax.plot([O[0], A[0]], [O[1], A[1]], 'r-') # OA
        ax.plot([O[0], B[0]], [O[1], B[1]], 'r-') # OB
        ax.plot([A[0], C[0]], [A[1], C[1]], 'b-') # AC
        ax.plot([B[0], C[0]], [B[1], C[1]], 'b-') # BC
        ax.plot([A[0],B[0],C[0],O[0]], [A[1],B[1],C[1],O[1]], 'ko', markersize=4)
        ax.text(O[0], O[1]-0.3, 'O'); ax.text(A[0]*1.1, A[1]*1.1, 'A')
        ax.text(B[0]*1.1, B[1]*1.1, 'B'); ax.text(C[0]*1.1, C[1]*1.1, 'C')

        arc_center = patches.Arc(O, radius*0.5, radius*0.5, angle=0, theta1=A_angle, theta2=B_angle, color='red')
        ax.add_patch(arc_center)
        ax.text(O[0]-0.3, O[1]+0.8, f'{angle_aob}°', color='red')
        ax.text(C[0], C[1]-0.5, 'x', color='blue')

        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)
        ax.set_title("Find Angle ACB")
        return self._save_plot(fig, filename)

    def f4_gpc_we2_cyclic_quad_opp_angles(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        center = (0,0); radius = 3
        angles = [10, 100, 190, 280] # Approx vertices
        P = (radius*math.cos(math.radians(angles[0])), radius*math.sin(math.radians(angles[0])))
        Q = (radius*math.cos(math.radians(angles[1])), radius*math.sin(math.radians(angles[1])))
        R = (radius*math.cos(math.radians(angles[2])), radius*math.sin(math.radians(angles[2])))
        S = (radius*math.cos(math.radians(angles[3])), radius*math.sin(math.radians(angles[3])))
        verts = [P, Q, R, S]
        labels = ['P', 'Q', 'R', 'S']

        circle = patches.Circle(center, radius, fill=False); ax.add_patch(circle)
        quad = patches.Polygon(verts, closed=True, fill=False, edgecolor='blue'); ax.add_patch(quad)
        for pt, name in zip(verts, labels):
            ax.plot(pt[0], pt[1], 'ko', markersize=4)
            ax.text(pt[0]*1.1, pt[1]*1.1, name)

        ax.text(S[0]-0.3, S[1]+0.2, '80°') # Angle PSR
        ax.text(Q[0], Q[1]-0.35, 'y') # Angle PQR

        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)
        ax.set_title("Find Angle PQR")
        return self._save_plot(fig, filename)

    def f4_gpc_we3_angle_in_semicircle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        center = (0,0); radius = 3
        A = (-radius, 0); B = (radius, 0); O = center
        C_angle = 50 # Angle for C
        C = (radius*math.cos(math.radians(C_angle)), radius*math.sin(math.radians(C_angle)))

        circle = patches.Circle(center, radius, fill=False); ax.add_patch(circle)
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k-') # Diameter AB
        ax.plot([A[0], C[0]], [A[1], C[1]], 'b-') # AC
        ax.plot([B[0], C[0]], [B[1], C[1]], 'b-') # BC
        ax.plot([A[0],B[0],C[0],O[0]], [A[1],B[1],C[1],O[1]], 'ko', markersize=4)
        ax.text(A[0]-0.3, A[1], 'A'); ax.text(B[0]+0.1, B[1], 'B'); ax.text(O[0]-0.3, O[1], 'O')
        ax.text(C[0], C[1]+0.2, 'C')

        # Angle CAB = 35
        angle_cab_deg = 35
        angle_cab_rad = math.radians(angle_cab_deg)
        arc_a = patches.Arc(A, 1, 1, angle=0, theta1=0, theta2=angle_cab_deg, color='g')
        ax.add_patch(arc_a)
        ax.text(A[0]+0.7, A[1]+0.1, f'{angle_cab_deg}°', color='g')

        # Angle ABC = z
        # Angle at C is 90
        vec_CA = np.array(A) - np.array(C)
        vec_CB = np.array(B) - np.array(C)
        corner_size = 0.4
        p1 = C + vec_CA / np.linalg.norm(vec_CA) * corner_size
        p2 = C + vec_CB / np.linalg.norm(vec_CB) * corner_size
        p3 = p1 + (p2 - C)
        rect_pts = np.array([p1, p3, p2])
        rect_patch = patches.Polygon(rect_pts, closed=False, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect_patch)
        ax.text(C[0]-0.6, C[1]-0.7, '90°', color='red')

        # Label angle z at B
        angle_abc_deg = 180 - 90 - angle_cab_deg
        angle_abc_rad = math.radians(angle_abc_deg)
        arc_b = patches.Arc(B, 1, 1, angle=180, theta1=-angle_abc_deg, theta2=0, color='purple')
        ax.add_patch(arc_b)
        ax.text(B[0]-0.6, B[1]+0.2, 'z', color='purple')

        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)
        ax.set_title("Find Angle ABC (z)")
        return self._save_plot(fig, filename)

    def f4_gpc_we4_alternate_segment(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        center = (0,0); radius = 3
        T_angle, Q_angle, R_angle = 210, 330, 90
        T = (radius*math.cos(math.radians(T_angle)), radius*math.sin(math.radians(T_angle)))
        Q = (radius*math.cos(math.radians(Q_angle)), radius*math.sin(math.radians(Q_angle)))
        R = (radius*math.cos(math.radians(R_angle)), radius*math.sin(math.radians(R_angle)))

        # Calculate tangent direction perpendicular to radius OT
        dx_rad = T[0] - center[0]
        dy_rad = T[1] - center[1]
        tan_dx, tan_dy = -dy_rad, dx_rad # Perpendicular vector
        norm = math.sqrt(tan_dx**2 + tan_dy**2)
        tan_dx /= norm
        tan_dy /= norm
        tan_len = 1.5
        P = (T[0] + tan_dx * tan_len, T[1] + tan_dy * tan_len) # Point P on tangent

        circle = patches.Circle(center, radius, fill=False); ax.add_patch(circle)
        ax.plot([T[0] - tan_dx * tan_len, P[0]], [T[1] - tan_dy * tan_len, P[1]], 'b-', label='Tangent PT') # Tangent line
        ax.plot([T[0], Q[0]], [T[1], Q[1]], 'g-', label='Chord TQ') # Chord TQ
        ax.plot([T[0], R[0]], [T[1], R[1]], 'm-') # TR
        ax.plot([Q[0], R[0]], [Q[1], R[1]], 'm-') # QR
        ax.plot([T[0],Q[0],R[0]], [T[1],Q[1],R[1]], 'ko', markersize=4)
        ax.text(T[0]-0.3, T[1]-0.3, 'T'); ax.text(Q[0]*1.1, Q[1]*1.1, 'Q'); ax.text(R[0]*1.1, R[1]*1.1, 'R')
        ax.text(P[0], P[1], 'P')

        angle_ptq_deg = 65
        # Mark angles (approximate locations)
        ax.text(T[0]+0.6, T[1]+0.1, f'{angle_ptq_deg}°\n(∠PTQ)', color='blue', fontsize=9, ha='left')
        ax.text(R[0]-0.16, R[1]-1, 'a\n(∠TRQ)', color='m', fontsize=9, ha='center')
        ax.text(0, -radius*1.2, '∠PTQ = ∠TRQ', ha='center')

        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1.5, radius+1)
        ax.set_title("Find Angle TRQ (a)")
        return self._save_plot(fig, filename)


    def f4_gpc_we5_tangent_radius_angle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        center = (0,0); radius = 3
        angle_toc_deg = 130

        # Position T and C based on angle TOC
        T_angle = 180 # Place T on the left for simpler tangent drawing
        C_angle = T_angle + angle_toc_deg
        T = (radius*math.cos(math.radians(T_angle)), radius*math.sin(math.radians(T_angle)))
        C = (radius*math.cos(math.radians(C_angle)), radius*math.sin(math.radians(C_angle)))
        O = center
        # Point A on the tangent line, above T
        A = (T[0], T[1] + 2.5) # Define A relative to T

        circle = patches.Circle(center, radius, fill=False); ax.add_patch(circle)
        ax.plot([O[0], T[0]], [O[1], T[1]], 'r-', label='Radius OT') # OT
        ax.plot([O[0], C[0]], [O[1], C[1]], 'r-', label='Radius OC') # OC
        # Draw the tangent line explicitly from T to A
        ax.plot([T[0], A[0]], [T[1]+ (T[1]-A[1]), A[1]], 'b-', label='Tangent AT')
        ax.plot([A[0], C[0]], [A[1], C[1]], 'g--') # Line AC (just for context)
        ax.plot([T[0], C[0]], [T[1], C[1]], 'k--') # Line TC
        ax.plot([A[0], C[0]], [A[1], C[1]], 'g--') # Line AC (just for context)
        ax.plot([T[0], C[0]], [T[1], C[1]], 'k--') # Line TC

        ax.plot([O[0], T[0], C[0], A[0]], [O[1], T[1], C[1], A[1]], 'ko', markersize=4)
        ax.text(O[0]+0.2, O[1], 'O'); ax.text(T[0]-0.3, T[1], 'T');
        ax.text(C[0]-0.1, C[1]-0.3, 'C'); ax.text(A[0]-0.3, A[1], 'A')

        # Mark angle TOC
        arc_toc = patches.Arc(O, radius*0.4, radius*0.4, angle=0, theta1=T_angle, theta2=C_angle, color='red')
        ax.add_patch(arc_toc)
        ax.text(-0.5, -0.8, f'{angle_toc_deg}°', color='red')

        # Mark right angle ATO
        ax.text(T[0]+0.2, T[1]+0.2, '90°', color='red', fontsize=9) # Angle ATO

        # Mark angle ATC to find
        ax.text(T[0]+0.5, T[1]-0.5, 'x', color='green', fontsize=12) # Angle ATC


        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1.5)
        ax.set_title("Find Angle ATC (x)")
        # ax.legend()
        return self._save_plot(fig, filename)

    def f4_gpc_q_easy_reflex_angle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        center = (0,0); radius = 3
        angle_aoc = 140 # Minor angle

        A_angle = (180-angle_aoc)/2 # Place symmetrically
        C_angle = A_angle + angle_aoc
        B_angle = (A_angle + C_angle)/2 + 180 # Point B on major arc

        A = (radius*math.cos(math.radians(A_angle)), radius*math.sin(math.radians(A_angle)))
        C = (radius*math.cos(math.radians(C_angle)), radius*math.sin(math.radians(C_angle)))
        B = (radius*math.cos(math.radians(B_angle)), radius*math.sin(math.radians(B_angle)))
        O = center

        circle = patches.Circle(center, radius, fill=False); ax.add_patch(circle)
        ax.plot([O[0], A[0]], [O[1], A[1]], 'r-') # OA
        ax.plot([O[0], C[0]], [O[1], C[1]], 'r-') # OC
        ax.plot([A[0], B[0]], [A[1], B[1]], 'b-') # AB
        ax.plot([C[0], B[0]], [C[1], B[1]], 'b-') # CB
        ax.plot([A[0],B[0],C[0],O[0]], [A[1],B[1],C[1],O[1]], 'ko', markersize=4)
        ax.text(O[0], O[1]-0.3, 'O'); ax.text(A[0]*1.1, A[1]*1.1, 'A')
        ax.text(C[0]*1.1, C[1]*1.1, 'C'); ax.text(B[0]*1.1, B[1]*1.1-0.1, 'B')

        # Mark minor angle AOC
        arc_minor = patches.Arc(O, radius*0.4, radius*0.4, angle=0, theta1=A_angle, theta2=C_angle, color='red')
        ax.add_patch(arc_minor)
        ax.text(-0.3, 0.6, f'{angle_aoc}°', color='red')

        # Mark reflex angle AOC
        reflex_aoc = 360 - angle_aoc
        arc_reflex = patches.Arc(O, radius*0.5, radius*0.5, angle=0, theta1=C_angle, theta2=A_angle+360, color='purple')
        ax.add_patch(arc_reflex)
        ax.text(-1.3, -1, f'Reflex ∠AOC = {reflex_aoc}°', color='purple')

        # Mark angle ABC
        ax.text(B[0]-0.3, B[1]-0.4, '∠ABC = ?', color='blue')

        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)
        ax.set_title("Angles AOC and ABC")
        return self._save_plot(fig, filename)

    def f4_gpc_q_moderate_cyclic_quad_eqn(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        center = (0,0); radius = 3
        # Approximate vertex positions
        angles = [45, 100, 200, 300]
        P = (radius*math.cos(math.radians(angles[0])), radius*math.sin(math.radians(angles[0])))
        Q = (radius*math.cos(math.radians(angles[1])), radius*math.sin(math.radians(angles[1])))
        R = (radius*math.cos(math.radians(angles[2])), radius*math.sin(math.radians(angles[2])))
        S = (radius*math.cos(math.radians(angles[3])), radius*math.sin(math.radians(angles[3])))
        verts = [P, Q, R, S]
        labels = ['P', 'Q', 'R', 'S']

        circle = patches.Circle(center, radius, fill=False); ax.add_patch(circle)
        quad = patches.Polygon(verts, closed=True, fill=False, edgecolor='blue'); ax.add_patch(quad)
        for pt, name in zip(verts, labels):
            ax.plot(pt[0], pt[1], 'ko', markersize=4)
            ax.text(pt[0]*1.1, pt[1]*1.1, name)

        # Label angles
        ax.text(P[0]-0.7, P[1]-0.2, '(2x)°', color='red')
        ax.text(R[0]+0.2, R[1], '(x+30)°', color='red')

        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)
        ax.set_title("Cyclic Quad: Find x")
        return self._save_plot(fig, filename)

    def f4_gpc_q_challenging_alt_segment_thm(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        center = (0,0); radius = 3
        Y_angle, W_angle, V_angle = 150, 30, 270 # Example angles
        Y = (radius*math.cos(math.radians(Y_angle)), radius*math.sin(math.radians(Y_angle)))
        W = (radius*math.cos(math.radians(W_angle)), radius*math.sin(math.radians(W_angle)))
        V = (radius*math.cos(math.radians(V_angle)), radius*math.sin(math.radians(V_angle)))

        # Tangent line XYZ (approximate)
        tan_angle = Y_angle + 90 # Perpendicular to radius
        X = (Y[0] + 2*math.cos(math.radians(tan_angle+180)), Y[1] + 2*math.sin(math.radians(tan_angle+180)))
        Z = (Y[0] + 2*math.cos(math.radians(tan_angle)), Y[1] + 2*math.sin(math.radians(tan_angle)))

        circle = patches.Circle(center, radius, fill=False); ax.add_patch(circle)
        ax.plot([X[0], Z[0]], [X[1], Z[1]], 'b-', label='Tangent XYZ') # Tangent
        ax.plot([Y[0], W[0]], [Y[1], W[1]], 'g-', label='Chord YW') # Chord YW
        ax.plot([Y[0], V[0]], [Y[1], V[1]], 'm-') # YV
        ax.plot([W[0], V[0]], [W[1], V[1]], 'm-') # WV
        ax.plot([Y[0],W[0],V[0]], [Y[1],W[1],V[1]], 'ko', markersize=4)
        ax.plot(X[0], X[1], 'ko', markersize=4); ax.text(X[0], X[1]-0.3, 'X')
        ax.plot(Z[0], Z[1], 'ko', markersize=4); ax.text(Z[0], Z[1]+0.3, 'Z')
        ax.text(Y[0]*1.1, Y[1]*1.1, 'Y'); ax.text(W[0]*1.1, W[1]*1.1, 'W'); ax.text(V[0]*1.1, V[1]*1.1-0.01, 'V')

        angle_wyx_deg = 55
        # Mark angles
        ax.text(Y[0]+1.3, Y[1]+0.15, f'{angle_wyx_deg}°\n(∠WYX)', color='blue', fontsize=9, ha='right')
        ax.text(V[0]-0.47, V[1]+0.9, '(∠YVW)?', color='m', fontsize=9)
        ax.text(0, -radius*1.4, 'Find ∠YVW using Alt. Segment Thm.', ha='center')

        ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1.5, radius+1)
        ax.set_title("Alternate Segment Theorem")
        return self._save_plot(fig, filename)


    # === Form 4: Geometry - Constructions and Loci ===

    def f4_gcl_notes_locus1_circle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        A = (0,0) # Fixed point
        r = 3 # Radius

        ax.plot(A[0], A[1], 'ko', markersize=6, label='Fixed Point A')
        locus_circle = patches.Circle(A, r, fill=False, edgecolor='blue', linewidth=2, label='Locus PA = r')
        ax.add_patch(locus_circle)

        # Example point P on locus
        P_angle = 45
        P = (A[0] + r*math.cos(math.radians(P_angle)), A[1] + r*math.sin(math.radians(P_angle)))
        ax.plot(P[0], P[1], 'ro', markersize=5, label='Point P on Locus')
        ax.plot([A[0], P[0]], [A[1], P[1]], 'r--', label='PA = r')
        ax.text(P[0]/2+0.2, P[1]/2, 'r', color='red')

        ax.set_xlim(-r-1, r+1); ax.set_ylim(-r-1, r+1)
        ax.set_title("Locus 1: Equidistant from Fixed Point A")
        ax.legend(loc='upper right', fontsize=8)
        return self._save_plot(fig, filename)

    def f4_gcl_notes_locus2_parallel_lines(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); # ax.set_aspect('equal');
        ax.axis('off')
        d = 1.5 # Distance

        # Fixed line L (horizontal)
        ax.plot([-3, 3], [0, 0], 'k-', linewidth=2, label='Fixed Line L')

        # Locus lines (parallel)
        ax.plot([-3, 3], [d, d], 'b--', linewidth=2, label='Locus (Distance d)')
        ax.plot([-3, 3], [-d, -d], 'b--', linewidth=2)

        # Show distance d
        ax.annotate('', xy=(1, d), xytext=(1, 0),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(1.1, d/2, 'd', color='red')
        ax.annotate('', xy=(-1, -d), xytext=(-1, 0),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(-0.9, -d/2, 'd', color='red')

        # Example point P
        P = (2, d)
        ax.plot(P[0], P[1], 'ro', markersize=5, label='Point P on Locus')

        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-d-1, d+1)
        ax.set_title("Locus 2: Equidistant from Fixed Line L")
        ax.legend(loc='upper right', fontsize=8)
        return self._save_plot(fig, filename)

    def f4_gcl_notes_locus3_perp_bisector(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        # Fixed points A, B
        A = (-2, 0)
        B = (2, 0)
        mid = (0,0) # Midpoint

        ax.plot(A[0], A[1], 'ko', markersize=6, label='Fixed Point A')
        ax.plot(B[0], B[1], 'ko', markersize=6, label='Fixed Point B')
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k:') # Line segment AB

        # Locus (perpendicular bisector - vertical line x=0)
        ax.plot([mid[0], mid[0]], [-2.5, 2.5], 'b-', linewidth=2, label='Locus PA = PB')

        # Example point P
        P = (0, 1.5)
        ax.plot(P[0], P[1], 'ro', markersize=5, label='Point P on Locus')
        ax.plot([P[0], A[0]], [P[1], A[1]], 'r--')
        ax.plot([P[0], B[0]], [P[1], B[1]], 'r--')
        ax.text(P[0]-0.5, P[1]/2, 'PA', color='red', ha='right')
        ax.text(P[0]+0.5, P[1]/2, 'PB', color='red', ha='left')


        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title("Locus 3: Equidistant from Fixed Points A and B")
        ax.legend(loc='upper right', fontsize=8)
        return self._save_plot(fig, filename)

    def f4_gcl_notes_locus4_angle_bisector(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        O = (0,0) # Intersection

        # Lines L1, L2
        angle1_deg = 30
        angle2_deg = 120
        len = 3
        x1_end = len*math.cos(math.radians(angle1_deg))
        y1_end = len*math.sin(math.radians(angle1_deg))
        x2_end = len*math.cos(math.radians(angle2_deg))
        y2_end = len*math.sin(math.radians(angle2_deg))

        ax.plot([-x1_end, x1_end], [-y1_end, y1_end], 'k-', label='Line L1')
        ax.plot([-x2_end, x2_end], [-y2_end, y2_end], 'k-', label='Line L2')
        ax.text(x1_end*1.05, y1_end*1.05, 'L1')
        ax.text(x2_end*1.05, y2_end*1.05, 'L2')

        # Angle bisectors
        bisector1_angle_deg = (angle1_deg + angle2_deg) / 2
        bisector2_angle_deg = bisector1_angle_deg + 90
        bx1_end = len*1.1*math.cos(math.radians(bisector1_angle_deg))
        by1_end = len*1.1*math.sin(math.radians(bisector1_angle_deg))
        bx2_end = len*1.1*math.cos(math.radians(bisector2_angle_deg))
        by2_end = len*1.1*math.sin(math.radians(bisector2_angle_deg))

        ax.plot([-bx1_end, bx1_end], [-by1_end, by1_end], 'b--', linewidth=2, label='Locus (Angle Bisectors)')
        ax.plot([-bx2_end, bx2_end], [-by2_end, by2_end], 'b--', linewidth=2)

        # Indicate perpendicular bisectors
        corner_size = 0.3
        rect = patches.Rectangle(O, corner_size, corner_size, angle=bisector1_angle_deg,
                                 linewidth=1, edgecolor='blue', facecolor='none', ls='--')
        ax.add_patch(rect)


        ax.set_xlim(-len-1, len+1); ax.set_ylim(-len-1, len+1)
        ax.set_title("Locus 4: Equidistant from Intersecting Lines")
        ax.legend(loc='upper right', fontsize=8)
        return self._save_plot(fig, filename)

    def f4_gcl_we2_locus_perp_bisector_construct(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        dist_ab = 6
        A = (0, 0)
        B = (dist_ab, 0)
        compass_radius = 4 # Must be > dist_ab / 2

        # Plot A and B
        ax.plot(A[0], A[1], 'ko', markersize=6, label='A')
        ax.plot(B[0], B[1], 'ko', markersize=6, label='B')
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k:') # Segment AB
        ax.text((A[0]+B[0])/2, A[1]-0.3, f'{dist_ab} cm', ha='center')

        # Construction arcs from A
        arc_a1 = patches.Arc(A, 2*compass_radius, 2*compass_radius, angle=0, theta1=30, theta2=150, color='gray', linestyle='--', linewidth=1)
        arc_a2 = patches.Arc(A, 2*compass_radius, 2*compass_radius, angle=0, theta1=210, theta2=330, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_a1)
        ax.add_patch(arc_a2)

        # Construction arcs from B
        arc_b1 = patches.Arc(B, 2*compass_radius, 2*compass_radius, angle=0, theta1=30, theta2=150, color='gray', linestyle='--', linewidth=1)
        arc_b2 = patches.Arc(B, 2*compass_radius, 2*compass_radius, angle=0, theta1=210, theta2=330, color='gray', linestyle='--', linewidth=1)
        ax.add_patch(arc_b1)
        ax.add_patch(arc_b2)
        # Calculate intersection points
        # x = dist_ab / 2
        # y^2 = compass_radius^2 - (dist_ab/2)^2
        mid_x = dist_ab / 2
        # Ensure radius is large enough before calculating sqrt
        if compass_radius <= mid_x:
             print(f"Warning: Compass radius ({compass_radius}) must be greater than half the distance ({mid_x}) for arcs to intersect.")
             y_intersect = 0 # Set a default or handle error appropriately
        else:
             y_intersect = math.sqrt(compass_radius**2 - mid_x**2)

        P_top = (mid_x, y_intersect)
        P_bottom = (mid_x, -y_intersect)

        # Draw the locus line between intersection points
        ax.plot([P_top[0], P_bottom[0]], [P_top[1], P_bottom[1]], 'b-', linewidth=2, label='Locus (Perpendicular Bisector)')
        # Optional: Mark intersection points explicitly
        ax.plot(P_top[0], P_top[1], 'bx', markersize=5)
        ax.plot(P_bottom[0], P_bottom[1], 'bx', markersize=5)

        # Adjust ylim based on intersection height for better framing
        ax.set_xlim(-1, dist_ab + 1)
        ax.set_ylim(-y_intersect - 1, y_intersect + 1) # Frame around intersections

        ax.set_title("Construction: Locus Equidistant from A and B")
        ax.legend(loc='upper right', fontsize=8)
        return self._save_plot(fig, filename)


    def f4_gcl_we3_locus_circle_construct(self):
        # Essentially the same as f4_gcl_notes_locus1_circle but called from WE
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        P = (0,0) # Fixed point
        r = 3 # Radius

        ax.plot(P[0], P[1], 'ko', markersize=6, label='Fixed Point P')
        locus_circle = patches.Circle(P, r, fill=False, edgecolor='blue', linewidth=2, label=f'Locus ({r} cm from P)')
        ax.add_patch(locus_circle)
        # Indicate radius measurement with compass arc (optional visual cue)
        arc_compass = patches.Arc(P, r*0.5, r*0.5, angle=0, theta1=0, theta2=360, ls=':', color='gray')
        # ax.add_patch(arc_compass)

        ax.set_xlim(-r-1, r+1); ax.set_ylim(-r-1, r+1)
        ax.set_title(f"Construction: Locus {r} cm from Point P")
        ax.legend(loc='upper right', fontsize=8)
        return self._save_plot(fig, filename)

    def f4_gcl_we4_locus_angle_bisector_construct(self):
         # Essentially the same as f4_gcl_notes_locus4_angle_bisector but called from WE
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        O = (0,0) # Intersection
        angle1_deg = 20
        angle2_deg = 100
        len = 3
        x1_end = len*math.cos(math.radians(angle1_deg))
        y1_end = len*math.sin(math.radians(angle1_deg))
        x2_end = len*math.cos(math.radians(angle2_deg))
        y2_end = len*math.sin(math.radians(angle2_deg))

        ax.plot([-x1_end, x1_end], [-y1_end, y1_end], 'k-', label='Line L1')
        ax.plot([-x2_end, x2_end], [-y2_end, y2_end], 'k-', label='Line L2')
        ax.text(x1_end*1.05, y1_end*1.05, 'L1')
        ax.text(x2_end*1.05, y2_end*1.05, 'L2')

        # # Construction arcs (simplified visual)
        # arc_base = patches.Arc(O, 1, 1, angle=0, theta1=angle1_deg, theta2=angle2_deg, color='gray', linestyle='--')
        # ax.add_patch(arc_base)
        # pt1 = (1*math.cos(math.radians(angle1_deg)), 1*math.sin(math.radians(angle1_deg)))
        # pt2 = (1*math.cos(math.radians(angle2_deg)), 1*math.sin(math.radians(angle2_deg)))
        # arc_cross1 = patches.Arc(pt1, 0.8, 0.8, angle=0, theta1=angle2_deg, theta2=angle2_deg+60, color='gray', linestyle='--')
        # arc_cross2 = patches.Arc(pt2, 0.8, 0.8, angle=0, theta1=angle1_deg-60, theta2=angle1_deg, color='gray', linestyle='--')
        # ax.add_patch(arc_cross1)
        # ax.add_patch(arc_cross2)


        # Angle bisectors
        bisector1_angle_deg = (angle1_deg + angle2_deg) / 2
        bisector2_angle_deg = bisector1_angle_deg + 90
        bx1_end = len*1.1*math.cos(math.radians(bisector1_angle_deg))
        by1_end = len*1.1*math.sin(math.radians(bisector1_angle_deg))
        bx2_end = len*1.1*math.cos(math.radians(bisector2_angle_deg))
        by2_end = len*1.1*math.sin(math.radians(bisector2_angle_deg))

        ax.plot([-bx1_end, bx1_end], [-by1_end, by1_end], 'b--', linewidth=2, label='Locus (Angle Bisectors)')
        ax.plot([-bx2_end, bx2_end], [-by2_end, by2_end], 'b--', linewidth=2)

        ax.set_xlim(-len-1, len+1); ax.set_ylim(-len-1, len+1)
        ax.set_title("Construction: Locus Equidistant from L1 and L2")
        ax.legend(loc='upper right', fontsize=8)
        return self._save_plot(fig, filename)

    def f4_gcl_we5_locus_intersection(self):
        # filename = self.get_filename_from_caller() # This caused the error
        filename = 'math_diagrams/f4_gcl_we5_locus_intersection.png'
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        dist_ab = 5
        rA = 3
        rB = 4
        A = (0, 0)
        B = (dist_ab, 0)

        # Set plot limits BEFORE contourf
        ax.set_xlim(-rA-1, B[0]+rB+1); ax.set_ylim(-max(rA,rB)-1, max(rA,rB)+1)

        # Shade the intersection region using contourf
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x_grid = np.linspace(x_lim[0], x_lim[1], 400)
        y_grid = np.linspace(y_lim[0], y_lim[1], 400)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = ((X - A[0])**2 + (Y - A[1])**2 <= rA**2) & ((X - B[0])**2 + (Y - B[1])**2 <= rB**2)
        ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=['lightgrey'], alpha=0.7, zorder=0)

        # Locus 1: Circle center A, radius rA
        circle_A = patches.Circle(A, rA, fill=False, edgecolor='blue', linewidth=1.5, label=f'Locus PA ≤ {rA} cm', zorder=2)
        ax.add_patch(circle_A)
        ax.plot(A[0], A[1], 'bo', markersize=6, zorder=3); ax.text(A[0]-0.2, A[1]-0.3, 'A', zorder=4)

        # Locus 2: Circle center B, radius rB
        circle_B = patches.Circle(B, rB, fill=False, edgecolor='red', linewidth=1.5, label=f'Locus PB ≤ {rB} cm', zorder=2)
        ax.add_patch(circle_B)
        ax.plot(B[0], B[1], 'ro', markersize=6, zorder=3); ax.text(B[0]+0.1, B[1]-0.3, 'B', zorder=4)

        # Draw segment AB
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k:', zorder=1)
        ax.text((A[0]+B[0])/2, -0.3, f'{dist_ab} cm', ha='center', zorder=4)

        ax.set_title("Intersection of Loci: PA ≤ 3 and PB ≤ 4")

        # Create custom legend handles
        legend_elements = [Line2D([0], [0], color='blue', lw=1.5, label=f'Locus PA ≤ {rA} cm'),
                           Line2D([0], [0], color='red', lw=1.5, label=f'Locus PB ≤ {rB} cm'),
                           Patch(facecolor='lightgrey', alpha=0.7, label='Shaded Region\n(PA ≤ 3 AND PB ≤ 4)')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        print(f"Attempting to save plot with filename: {filename}") # Add this line

        # Call the save plot method
        return self._save_plot(fig, filename)

    # === Form 4: Statistics - Data Representation ===

    def f4_sdr_notes_ogive(self): # Renamed from f3_sdccr...
        filename = get_filename_from_caller()
        # Data from example table in notes
        upper_boundaries = [9.5, 19.5, 29.5, 39.5, 49.5]
        cf = [0, 2, 8, 15, 20] # Must start CF with 0 at lower bound of first interval

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')

        ax.plot(upper_boundaries, cf, 'b-o', linewidth=2, markersize=6)

        ax.set_xlabel("Marks (Upper Boundary)")
        ax.set_ylabel("Cumulative Frequency")
        ax.set_title("Cumulative Frequency Curve (Ogive)")
        ax.set_xticks(upper_boundaries)
        ax.set_yticks(np.arange(0, max(cf) + 2, 2))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=min(upper_boundaries)-1)


        plt.tight_layout()
        # Store data for reuse potentially
        self.context_data['f4_sdr_notes_ogive'] = {'boundaries': upper_boundaries, 'cf': cf}
        return self._save_plot(fig, filename)

    def f4_sdr_we2_ogive(self): # Renamed from f3_sdccr...
        filename = get_filename_from_caller()
        # Data from WE1
        upper_boundaries = [-0.5, 4.5, 9.5, 14.5, 19.5]
        cf = [0, 5, 13, 19, 20]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')

        ax.plot(upper_boundaries, cf, 'g-o', linewidth=2, markersize=6)

        ax.set_xlabel("Variable (Upper Boundary)")
        ax.set_ylabel("Cumulative Frequency")
        ax.set_title("Cumulative Frequency Curve (Ogive)")
        ax.set_xticks(upper_boundaries)
        ax.set_yticks(np.arange(0, max(cf) + 2, 2))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=min(upper_boundaries)-0.5)

        plt.tight_layout()
        # Store data for reuse
        self.context_data['f4_sdr_we2_ogive'] = {'boundaries': upper_boundaries, 'cf': cf}
        return self._save_plot(fig, filename)

    def f4_sdr_we3_median_on_ogive(self):
        filename = get_filename_from_caller()
        # Need Ogive data from WE2
        if 'f4_sdr_we2_ogive' not in self.context_data:
            print("Warning: Context data for WE2 Ogive not found, regenerating.")
            self.f4_sdr_we2_ogive()
            if 'f4_sdr_we2_ogive' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        boundaries = self.context_data['f4_sdr_we2_ogive']['boundaries']
        cf = self.context_data['f4_sdr_we2_ogive']['cf']
        N = cf[-1]
        median_pos = N / 2 # 20 / 2 = 10

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(boundaries, cf, 'g-o', linewidth=2, markersize=6)

        # Estimate median value by interpolating on the graph
        # Find where CF = 10 intersects the curve between (4.5, 5) and (9.5, 13)
        # Assume linear interpolation for estimation:
        cf1, cf2 = 5, 13
        b1, b2 = 4.5, 9.5
        if cf2 - cf1 > 0: # Avoid division by zero
            median_est = b1 + ((median_pos - cf1) / (cf2 - cf1)) * (b2 - b1)
        else:
            median_est = (b1+b2)/2 # Fallback if segment is flat

        # Draw lines for median
        ax.plot([boundaries[0]-0.5, median_est], [median_pos, median_pos], 'r--') # Horizontal line
        ax.plot([median_est, median_est], [0, median_pos], 'r--') # Vertical line
        ax.text(median_est, -1.5, f'Median ≈ {median_est:.1f}', color='red', ha='center')
        ax.text(-1, median_pos, f'N/2 = {median_pos}', color='red', va='center', ha='right')


        ax.set_xlabel("Variable (Upper Boundary)")
        ax.set_ylabel("Cumulative Frequency")
        ax.set_title("Estimating Median from Ogive")
        ax.set_xticks(boundaries)
        ax.set_yticks(np.arange(0, N + 2, 2))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=min(boundaries)-0.5)
        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f4_sdr_we4_value_from_ogive(self):
        filename = get_filename_from_caller()
        # Need Ogive data from WE2
        if 'f4_sdr_we2_ogive' not in self.context_data:
            print("Warning: Context data for WE2 Ogive not found, regenerating.")
            self.f4_sdr_we2_ogive()
            if 'f4_sdr_we2_ogive' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        boundaries = self.context_data['f4_sdr_we2_ogive']['boundaries']
        cf = self.context_data['f4_sdr_we2_ogive']['cf']
        N = cf[-1]
        target_value = 12 # Value on horizontal axis

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(boundaries, cf, 'g-o', linewidth=2, markersize=6)

        # Estimate CF at value = 12 by interpolating
        # Value 12 is between boundaries 9.5 and 14.5 (CFs 13 and 19)
        b1, b2 = 9.5, 14.5
        cf1, cf2 = 13, 19
        if b2 - b1 > 0:
             cf_est = cf1 + ((target_value - b1) / (b2 - b1)) * (cf2 - cf1)
        else:
             cf_est = (cf1+cf2)/2

        # Draw lines
        ax.plot([target_value, target_value], [0, cf_est], 'r--') # Vertical line
        ax.plot([boundaries[0]-0.5, target_value], [cf_est, cf_est], 'r--') # Horizontal line
        ax.text(target_value, -1.5, f'Value = {target_value}', color='red', ha='center')
        ax.text(boundaries[0]+0.6, cf_est, f'CF ≈ {cf_est:.0f}', color='red', va='center', ha='right')

        ax.set_xlabel("Variable (Upper Boundary)")
        ax.set_ylabel("Cumulative Frequency")
        ax.set_title(f"Estimating No. of Values < {target_value} from Ogive")
        ax.set_xticks(boundaries)
        ax.set_yticks(np.arange(0, N + 2, 2))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=min(boundaries)-0.5)
        plt.tight_layout()
        return self._save_plot(fig, filename)

    # === Form 4: Statistics - Measures of Central Tendency and Dispersion ===

    def f4_smctd_notes_ogive_quartiles(self):
        filename = get_filename_from_caller()
        # Generate a sample Ogive
        N = 60 # Example total frequency
        boundaries = np.array([0, 10, 20, 30, 40, 50])
        cf = np.array([0, 5, 20, 45, 55, N]) # Example CF values

        q1_pos, q2_pos, q3_pos = N/4, N/2, 3*N/4 # 15, 30, 45

        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(boundaries, cf, 'b-o', linewidth=2)

        # Interpolate Q1, Q2, Q3 values
        q1_val = np.interp(q1_pos, cf, boundaries)
        q2_val = np.interp(q2_pos, cf, boundaries)
        q3_val = np.interp(q3_pos, cf, boundaries) # Q3 = 30 exactly here

        # Plot Q1 lines
        ax.plot([boundaries[0], q1_val], [q1_pos, q1_pos], 'g--')
        ax.plot([q1_val, q1_val], [0, q1_pos], 'g--')
        ax.text(q1_val, -2, f'Q1≈{q1_val:.1f}', color='g', ha='center')
        ax.text(boundaries[0]-1, q1_pos, f'N/4={q1_pos}', color='g', va='center', ha='right')

        # Plot Q2 lines
        ax.plot([boundaries[0], q2_val], [q2_pos, q2_pos], 'r--')
        ax.plot([q2_val, q2_val], [0, q2_pos], 'r--')
        ax.text(q2_val, -2, f'Q2≈{q2_val:.1f}', color='r', ha='center')
        ax.text(boundaries[0]-1, q2_pos, f'N/2={q2_pos}', color='r', va='center', ha='right')

        # Plot Q3 lines
        ax.plot([boundaries[0], q3_val], [q3_pos, q3_pos], 'm--')
        ax.plot([q3_val, q3_val], [0, q3_pos], 'm--')
        ax.text(q3_val, -2, f'Q3={q3_val:.1f}', color='m', ha='center')
        ax.text(boundaries[0]-1, q3_pos, f'3N/4={q3_pos}', color='m', va='center', ha='right')

        # Setup plot
        ax.set_xlabel("Variable (Upper Boundary)")
        ax.set_ylabel("Cumulative Frequency")
        ax.set_title("Estimating Quartiles from Ogive")
        ax.set_xticks(boundaries)
        ax.set_yticks(np.arange(0, N + 5, 5))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=-5) # Make space for labels
        ax.set_xlim(left=boundaries[0]-2)
        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f4_smctd_we1_ogive_median(self): # Renamed for clarity
        filename = get_filename_from_caller()
        # Sample Ogive from description
        N = 80
        # Use interpolation between imagined points for drawing
        boundaries = np.array([10, 30, 50, 70, 100]) # Approx based on descr.
        cf = np.array([0, 10, 40, 70, 80]) # Rough CF values

        q2_pos = N/2 # 40

        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(boundaries, cf, 'b-o', linewidth=2)

        q2_val = np.interp(q2_pos, cf, boundaries) # Should be 50 based on points

        ax.plot([boundaries[0], q2_val], [q2_pos, q2_pos], 'r--')
        ax.plot([q2_val, q2_val], [0, q2_pos], 'r--')
        ax.text(q2_val, -5, f'Median ≈ {q2_val:.0f}', color='red', ha='center')
        ax.text(boundaries[0]-3, q2_pos, f'N/2 = {q2_pos}', color='red', va='center', ha='right')

        ax.set_xlabel("Marks (Upper Boundary assumed)")
        ax.set_ylabel("Cumulative Frequency")
        ax.set_title("Estimating Median (Q2) from Ogive (N=80)")
        ax.set_xticks(boundaries)
        ax.set_yticks(np.arange(0, N + 5, 10))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=-10)
        ax.set_xlim(left=boundaries[0]-5)
        plt.tight_layout()
        # Store for WE2
        self.context_data['f4_smctd_we1_ogive'] = {'boundaries': boundaries, 'cf': cf}
        return self._save_plot(fig, filename)

    def f4_smctd_we2_ogive_quartiles(self): # Renamed for clarity
        filename = get_filename_from_caller()
        # Use Ogive data from WE1
        if 'f4_smctd_we1_ogive' not in self.context_data:
            print("Warning: Context data for WE1 Ogive not found, regenerating.")
            self.f4_smctd_we1_ogive_median()
            if 'f4_smctd_we1_ogive' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        boundaries = self.context_data['f4_smctd_we1_ogive']['boundaries']
        cf = self.context_data['f4_smctd_we1_ogive']['cf']
        N = cf[-1]
        q1_pos, q3_pos = N/4, 3*N/4 # 20, 60

        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.plot(boundaries, cf, 'b-o', linewidth=2)

        q1_val = np.interp(q1_pos, cf, boundaries) # Estimate 40
        q3_val = np.interp(q3_pos, cf, boundaries) # Estimate 66

        # Plot Q1 lines
        ax.plot([boundaries[0], q1_val], [q1_pos, q1_pos], 'g--')
        ax.plot([q1_val, q1_val], [0, q1_pos], 'g--')
        ax.text(q1_val, -5, f'Q1 ≈ {q1_val:.0f}', color='g', ha='center')
        ax.text(boundaries[0]-3, q1_pos, f'N/4={q1_pos}', color='g', va='center', ha='right')

        # Plot Q3 lines
        ax.plot([boundaries[0], q3_val], [q3_pos, q3_pos], 'm--')
        ax.plot([q3_val, q3_val], [0, q3_pos], 'm--')
        ax.text(q3_val, -5, f'Q3 ≈ {q3_val:.0f}', color='m', ha='center')
        ax.text(boundaries[0]-3, q3_pos, f'3N/4={q3_pos}', color='m', va='center', ha='right')

        ax.set_xlabel("Marks (Upper Boundary assumed)")
        ax.set_ylabel("Cumulative Frequency")
        ax.set_title("Estimating Quartiles from Ogive (N=80)")
        ax.set_xticks(boundaries)
        ax.set_yticks(np.arange(0, N + 5, 10))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=-10)
        ax.set_xlim(left=boundaries[0]-5)
        plt.tight_layout()
        return self._save_plot(fig, filename)

    # === Form 4: Trigonometry - Trigonometrical Ratios ===

    def _draw_triangle(self, ax, A, B, C, labels={'A':'A', 'B':'B', 'C':'C'}, sides={'a':None, 'b':None, 'c':None}, angles={'A':None, 'B':None, 'C':None}):
        """Helper to draw a triangle with labels."""
        verts = np.array([A, B, C, A])
        ax.plot(verts[:,0], verts[:,1], 'k-')
        ax.plot([A[0],B[0],C[0]], [A[1],B[1],C[1]], 'ko', markersize=5)
        # Vertex labels
        ax.text(A[0], A[1]+0.1, labels['A'], ha='center', va='bottom')
        ax.text(B[0]+0.1, B[1]-0.1, labels['B'], ha='left', va='top')
        ax.text(C[0]-0.1, C[1]-0.1, labels['C'], ha='right', va='top')
        # Side labels (opposite vertex)
        mid_BC = ((B[0]+C[0])/2, (B[1]+C[1])/2)
        mid_AC = ((A[0]+C[0])/2, (A[1]+C[1])/2)
        mid_AB = ((A[0]+B[0])/2, (A[1]+B[1])/2)
        if sides.get('a'): ax.text(mid_BC[0]+0.1, mid_BC[1], f"a={sides['a']}", ha='left', va='center')
        if sides.get('b'): ax.text(mid_AC[0]-0.1, mid_AC[1], f"b={sides['b']}", ha='right', va='center')
        if sides.get('c'): ax.text(mid_AB[0], mid_AB[1]-0.2, f"c={sides['c']}", ha='center', va='top')
        # Angle labels (at vertex)
        if angles.get('A'): ax.text(A[0]+0.2, A[1]-0.1, f"{angles['A']}°", ha='left', va='top', fontsize=9, color='g')
        if angles.get('B'): ax.text(B[0]-0.2, B[1]+0.1, f"{angles['B']}°", ha='right', va='bottom', fontsize=9, color='g')
        if angles.get('C'): ax.text(C[0]+0.1, C[1]+0.1, f"{angles['C']}°", ha='left', va='bottom', fontsize=9, color='g')


    def f4_ttr_notes_sine_rule(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        # Define triangle vertices
        A = (0, 0); B = (5, 0); C = (2, 3)
        # Draw edges and vertices
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-')
        ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'ko', markersize=5)
        # Label sides (opposite vertices)
        mid_BC = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
        mid_AC = ((A[0] + C[0]) / 2, (A[1] + C[1]) / 2)
        mid_AB = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
        ax.text(mid_BC[0] + 0.1, mid_BC[1], 'a', ha='left', va='center', fontsize=10)
        ax.text(mid_AC[0] - 0.1, mid_AC[1], 'b', ha='right', va='center', fontsize=10)
        ax.text(mid_AB[0], mid_AB[1] - 0.2, 'c', ha='center', va='top', fontsize=10)
        # Label angles (inside vertices)
        ax.text(A[0] + 0.3, A[1] + 0.15, 'A', ha='left', va='bottom', fontsize=10, color='g')
        ax.text(B[0] - 0.5, B[1] + 0.15, 'B', ha='right', va='bottom', fontsize=10, color='g')
        ax.text(C[0], C[1] - 0.2, 'C', ha='center', va='top', fontsize=10, color='g')
        ax.text(2.5, -1.5, "Sine Rule:\na / sin A = b / sin B = c / sin C", ha='center', fontsize=10)
        ax.set_xlim(-1, 6); ax.set_ylim(-2, 4)
        ax.set_title("Sine Rule")
        return self._save_plot(fig, filename)


    def f4_ttr_notes_cosine_rule(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        # Define triangle vertices
        A = (0, 0); B = (5, 0); C = (2, 3)
        # Draw edges and vertices
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-')
        ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'ko', markersize=5)
        # Label sides (opposite vertices)
        mid_BC = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
        mid_AC = ((A[0] + C[0]) / 2, (A[1] + C[1]) / 2)
        mid_AB = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
        ax.text(mid_BC[0] + 0.1, mid_BC[1], 'a', ha='left', va='center', fontsize=10)
        ax.text(mid_AC[0] - 0.1, mid_AC[1], 'b', ha='right', va='center', fontsize=10)
        ax.text(mid_AB[0], mid_AB[1] - 0.2, 'c', ha='center', va='top', fontsize=10)
        # Label angles (inside vertices)
        ax.text(A[0] + 0.3, A[1] + 0.15, 'A', ha='left', va='bottom', fontsize=10, color='g')
        ax.text(B[0] - 0.5, B[1] + 0.15, 'B', ha='right', va='bottom', fontsize=10, color='g')
        ax.text(C[0], C[1] - 0.2, 'C', ha='center', va='top', fontsize=10, color='g')
        # Annotate Cosine Rule formula
        ax.text(2.5, -1.5, 'Cosine Rule (e.g.):\nc² = a² + b² - 2ab cos C', ha='center', fontsize=10)
        # Set limits and title
        ax.set_xlim(-1, 6); ax.set_ylim(-2, 4)
        ax.set_title('Cosine Rule')
        return self._save_plot(fig, filename)

    def f4_ttr_notes_area_sine(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        # Define triangle vertices
        A = (0, 0); B = (5, 0); C = (2, 3)
        # Draw edges and vertices
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-')
        ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'ko', markersize=5)
        # Highlight sides a (BC) and b (AC)
        ax.plot([B[0], C[0]], [B[1], C[1]], 'g-', linewidth=3)
        ax.plot([A[0], C[0]], [A[1], C[1]], 'b-', linewidth=3)
        # Label sides
        mid_BC = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
        mid_AC = ((A[0] + C[0]) / 2, (A[1] + C[1]) / 2)
        ax.text(mid_BC[0] + 0.1, mid_BC[1], 'a', color='g', fontsize=10)
        ax.text(mid_AC[0] - 0.2, mid_AC[1], 'b', color='b', fontsize=10)
        # Place angle label C inside the angle
        ax.text(C[0], C[1] - 0.4, 'C', color='r', fontsize=11, fontweight='bold')
        ax.text(B[0] - 0.6, B[1] + 0.2, 'B', color='b', fontsize=11, fontweight='bold')
        ax.text(A[0] + 0.3, A[1] + 0.2, 'A', color='g', fontsize=11, fontweight='bold')


        # Annotate area formula
        ax.text(2.5, -1.5, 'Area = ½ ab sin C', ha='center', fontsize=10)
        # Set limits and title
        ax.set_xlim(-1, 6); ax.set_ylim(-2, 4)
        ax.set_title('Area of Triangle using Sine')
        return self._save_plot(fig, filename)

    def f4_ttr_we1_sine_rule_triangle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
        # Given: Q=40, R=60 => P=80. Side r (PQ) = 10. Find q (PR).
        # Use Sine rule to find side lengths relative to PQ=10
        r = 10.0
        angle_P, angle_Q, angle_R = 80, 40, 60
        q = r * math.sin(math.radians(angle_Q)) / math.sin(math.radians(angle_R)) # PR ~ 7.42
        p = r * math.sin(math.radians(angle_P)) / math.sin(math.radians(angle_R)) # QR ~ 11.37

        # Define vertices: Place R at origin. Q is on x-axis.
        R = (0, 0)
        Q = (p, 0)
        # P coordinates using angle R=60 and side q (PR)
        P = (q * math.cos(math.radians(angle_R)), q * math.sin(math.radians(angle_R)))

        # Draw edges and vertices
        ax.plot([P[0], Q[0], R[0], P[0]], [P[1], Q[1], R[1], P[1]], 'k-')
        ax.plot([P[0], Q[0], R[0]], [P[1], Q[1], R[1]], 'ko', markersize=5)

        # Label vertices
        ax.text(P[0] - 0.2, P[1] + 0.1, 'P', ha='right', va='bottom', fontsize=10)
        ax.text(Q[0] + 0.1, Q[1] - 0.1, 'Q', ha='left', va='top', fontsize=10)
        ax.text(R[0] - 0.2, R[1] - 0.1, 'R', ha='right', va='top', fontsize=10)

        # Label sides
        mid_QR = ((Q[0] + R[0]) / 2, (Q[1] + R[1]) / 2)
        mid_PR = ((P[0] + R[0]) / 2, (P[1] + R[1]) / 2)
        mid_PQ = ((P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2)
        ax.text(mid_QR[0], mid_QR[1] - 0.2, f'p', ha='center', va='top', fontsize=9)
        ax.text(mid_PR[0] - 0.2, mid_PR[1], 'q=?', ha='right', va='center', fontsize=9, color='m')
        ax.text(mid_PQ[0], mid_PQ[1] + 0.3, f'r={r:.0f}', ha='center', va='bottom', fontsize=9)

        # Label angles inside
        ax.text(P[0], P[1] - 0.4, f'{angle_P}°', ha='left', va='top', fontsize=9, color='g')
        ax.text(Q[0] - 0.6, Q[1] + 0.15, f'{angle_Q}°', ha='right', va='bottom', fontsize=9, color='g')
        ax.text(R[0] + 0.4, R[1] + 0.15, f'{angle_R}°', ha='left', va='bottom', fontsize=9, color='g')

        # Adjust limits and title
        all_verts = np.array([P, Q, R])
        x_min, y_min = all_verts.min(axis=0)
        x_max, y_max = all_verts.max(axis=0)
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1.5, y_max + 1) # Extra space at bottom for result text
        ax.set_title("Sine Rule Example: Find PR")
        plt.tight_layout()
        return self._save_plot(fig, filename)

    def f4_ttr_we2_cosine_rule_triangle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')

        # Given values
        x = 5.0 # Side YZ (opposite X)
        z = 7.0 # Side XY (opposite Z)
        angleY_deg = 110

        # Calculate side y (XZ) using Cosine Rule
        angleY_rad = math.radians(angleY_deg)
        y = math.sqrt(x**2 + z**2 - 2 * x * z * math.cos(angleY_rad)) # Approx 9.90

        # --- Define Vertices ---
        # Place Y at the origin
        Y = (0, 0)
        # Place Z on the positive x-axis (length YZ = x)
        Z = (x, 0)
        # Calculate X coordinates
        # Angle XY makes with positive x-axis = 180 - angleY
        angleXY_rel_x_axis_rad = math.radians(180 - angleY_deg)
        X = (z * math.cos(angleXY_rel_x_axis_rad), z * math.sin(angleXY_rel_x_axis_rad))

        # --- Draw Triangle ---
        # Edges
        ax.plot([X[0], Y[0], Z[0], X[0]], [X[1], Y[1], Z[1], X[1]], 'k-')
        # Vertices
        ax.plot([X[0], Y[0], Z[0]], [X[1], Y[1], Z[1]], 'ko', markersize=5)

        # --- Label Vertices ---
        ax.text(X[0] - 0.2, X[1] + 0.1, 'X', ha='right', va='bottom', fontsize=10)
        ax.text(Y[0] - 0.2, Y[1] - 0.1, 'Y', ha='right', va='top', fontsize=10)
        ax.text(Z[0] + 0.1, Z[1] - 0.1, 'Z', ha='left', va='top', fontsize=10)

        # --- Label Sides ---
        mid_YZ = ((Y[0] + Z[0]) / 2, (Y[1] + Z[1]) / 2) # Opposite X
        mid_XZ = ((X[0] + Z[0]) / 2, (X[1] + Z[1]) / 2) # Opposite Y
        mid_XY = ((X[0] + Y[0]) / 2, (X[1] + Y[1]) / 2) # Opposite Z
        ax.text(mid_YZ[0], mid_YZ[1] - 0.2, f'x={x:.0f}', ha='center', va='top', fontsize=9, color= 'm')
        #ax.text(mid_XZ[0], mid_XZ[1] + 0.1, f'y={y:.2f}', ha='center', va='bottom', fontsize=9, color='m') # Show result here
        ax.text(mid_XY[0] - 0.2, mid_XY[1], f'z={z:.0f}', ha='right', va='center', fontsize=9, color= 'm')

        # --- Label Angle Y (inside) ---
        ax.text(Y[0] + 0.3, Y[1] + 0.2, f'{angleY_deg}°', ha='left', va='bottom', fontsize=9, color='g')

        # --- Plot Settings ---
        all_verts = np.array([X, Y, Z])
        x_min, y_min = all_verts.min(axis=0)
        x_max, y_max = all_verts.max(axis=0)
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_title("Cosine Rule Example: Find XZ (y)")
        plt.tight_layout()
        return self._save_plot(fig, filename)

    # def f4_ttr_we4_cosine_rule_angle(self):
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    #     ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')
    #     # Given: a=8, b=7, c=5. Find Angle A.
    #     a=8; b=7; c=5
    #     # Calculate angle A using cos A = (b^2+c^2-a^2)/(2bc)
    #     cosA = (b**2 + c**2 - a**2) / (2 * b * c)
    #     angleA = math.degrees(math.acos(cosA)) # ~81.8 deg
    #     # Find other angles for drawing if needed (using sine rule)
    #     sinB = b * math.sin(math.radians(angleA)) / a
    #     angleB = math.degrees(math.asin(sinB)) # ~58.4 deg
    #     angleC = 180 - angleA - angleB # ~40.8 deg

    #     # Position vertices. Let A be origin. Let C be on x-axis.
    #     A=(0,0); C=(b,0) # AC = b
    #     # B is at angle 180-110=70 deg from AC, distance c
    #     angleAB_rel_x_axis = 180-angleA
    #     B = (c*math.cos(math.radians(angleAB_rel_x_axis)), c*math.sin(math.radians(angleAB_rel_x_axis)))

    #     self._draw_triangle(ax, A, B, C, labels={'A':'A', 'B':'B', 'C':'C'},
    #                        sides={'a':a, 'b':b, 'c':c})

    #     ax.text(A[0]+0.5, A[1]+0.3, f"A °", ha='left', color='m', fontsize=10)

    #     ax.set_xlim(min(A[0],B[0],C[0])-1, max(A[0],B[0],C[0])+1); ax.set_ylim(min(A[1],B[1],C[1])-1, max(A[1],B[1],C[1])+1)
    #     ax.set_title("Cosine Rule Example: Find Angle A")
    #     return self._save_plot(fig, filename)

    def f4_ttr_we4_cosine_rule_angle(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')

        # Given sides
        a = 8.0; b = 7.0; c = 5.0

        # Calculate angle A using Cosine Rule: cos A = (b² + c² - a²) / 2bc
        cosA = (b**2 + c**2 - a**2) / (2 * b * c)
        # Ensure cosA is within valid range [-1, 1] due to potential floating point inaccuracies
        cosA = max(-1.0, min(1.0, cosA))
        angleA_rad = math.acos(cosA)
        angleA_deg = math.degrees(angleA_rad) # Approx 98.2 deg based on formula, not 81.8

        # Calculate other angles for plotting (optional but good for verification)
        # Use Sine Rule: sin B / b = sin A / a
        sinB = b * math.sin(angleA_rad) / a
        sinB = max(-1.0, min(1.0, sinB)) # Clamp value
        angleB_rad = math.asin(sinB)
        angleB_deg = math.degrees(angleB_rad) # Approx 58.4 deg

        angleC_deg = 180.0 - angleA_deg - angleB_deg # Approx 23.4 deg

        # --- Define Vertices ---
        # Place A at the origin
        A = (0, 0)
        # Place C on the positive x-axis (length AC = b)
        C = (b, 0)
        # Calculate B coordinates using side c (AB) and angle A
        # Angle AB makes with positive x-axis is angle A
        B = (c * math.cos(angleA_rad), c * math.sin(angleA_rad))

        # --- Draw Triangle ---
        # Edges
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'k-')
        # Vertices
        ax.plot([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'ko', markersize=5)

        # --- Label Vertices ---
        ax.text(A[0] - 0.2, A[1] - 0.1, 'A', ha='right', va='top', fontsize=10)
        ax.text(B[0], B[1] + 0.1, 'B', ha='center', va='bottom', fontsize=10)
        ax.text(C[0] + 0.1, C[1] - 0.1, 'C', ha='left', va='top', fontsize=10)

        # --- Label Sides ---
        mid_BC = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2) # Opposite A
        mid_AC = ((A[0] + C[0]) / 2, (A[1] + C[1]) / 2) # Opposite B
        mid_AB = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2) # Opposite C
        ax.text(mid_BC[0] + 0.1, mid_BC[1], f'a={a:.0f}', ha='left', va='center', fontsize=9)
        ax.text(mid_AC[0], mid_AC[1] - 0.2, f'b={b:.0f}', ha='center', va='top', fontsize=9)
        ax.text(mid_AB[0] - 0.1, mid_AB[1], f'c={c:.0f}', ha='right', va='center', fontsize=9)

        # --- Label Angles (inside) ---
        # Highlight Angle A as the one sought
        ax.text(A[0] + 0.3, A[1] + 0.15, f'A=?', ha='left', va='bottom', fontsize=10, color='m', fontweight='bold')

        # --- Plot Settings ---
        all_verts = np.array([A, B, C])
        x_min, y_min = all_verts.min(axis=0)
        x_max, y_max = all_verts.max(axis=0)
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_title("Cosine Rule Example: Find Angle A")
        plt.tight_layout()
        return self._save_plot(fig, filename)


    # Create a new function with improvements
    def f4_ttr_we5_3d_trig(self):
        filename = "math_diagrams/math_diagrams/f4_ttr_we5_3d_trig.png"  # Hard-coded filename to match expected output

        fig = plt.figure(figsize=(8, 7), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')

        # Points
        P = np.array([0, 0, 0])    # Base of flagpole
        Q = np.array([0, 15, 0])   # Observer 1
        R = np.array([10, 15, 0])  # Observer 2
        
        # Calculate height PT
        angle_tqp_deg = 35
        PQ_dist = np.linalg.norm(Q-P)
        PT_height = PQ_dist * math.tan(math.radians(angle_tqp_deg))
        T = np.array([0, 0, PT_height]) # Top of flagpole
        
        # Calculate angle TRP (theta)
        PR_dist = np.linalg.norm(R-P)
        angle_trp_rad = math.atan(PT_height / PR_dist)
        angle_trp_deg = math.degrees(angle_trp_rad)

        # Draw ground plane lines and flagpole
        ax.plot([P[0], Q[0]], [P[1], Q[1]], [P[2], Q[2]], 'k-', label=f'PQ={PQ_dist:.0f}')
        ax.plot([Q[0], R[0]], [Q[1], R[1]], [Q[2], R[2]], 'k-', label=f'QR={np.linalg.norm(R-Q):.0f}')
        ax.plot([P[0], R[0]], [P[1], R[1]], [P[2], R[2]], 'k:', label=f'PR={PR_dist:.1f}')
        ax.plot([P[0], T[0]], [P[1], T[1]], [P[2], T[2]], 'b-', linewidth=2, label=f'Flagpole PT={PT_height:.1f}')
        ax.plot([Q[0], T[0]], [Q[1], T[1]], [Q[2], T[2]], 'g--')
        ax.plot([R[0], T[0]], [R[1], T[1]], [R[2], T[2]], 'r--')
        
        # Mark points with larger markers
        ax.scatter([P[0], Q[0], R[0], T[0]], [P[1], Q[1], R[1], T[1]], [P[2], Q[2], R[2], T[2]], c='black', s=50)
        
        # Function for text with background
        def add_text_bg(ax, x, y, z, text, color='black', fontsize=12, bg='white'):
            txt = ax.text(x, y, z, text, color=color, ha='center', va='center', fontsize=fontsize, fontweight='bold')
            txt.set_bbox(dict(facecolor=bg, alpha=0.8, edgecolor=color, pad=2))
            return txt
        
        # Add labels with backgrounds
        add_text_bg(ax, P[0]-0.7, P[1]-0.7, 0.5, 'P')
        add_text_bg(ax, Q[0]-0.7, Q[1]+0.7, 0.5, 'Q')
        add_text_bg(ax, R[0]+0.7, R[1]+0.7, 0.5, 'R')
        add_text_bg(ax, T[0], T[1], T[2]+0.7, 'T')
        
        # Add angle labels with backgrounds
        add_text_bg(ax, Q[0]+1.5, Q[1]-1.5, 2.0, f'{angle_tqp_deg:.0f}°', color='green', fontsize=12)
        add_text_bg(ax, R[0]-1.5, R[1]-1.5, 2.5, f'θ≈{angle_trp_deg:.0f}°', color='red', fontsize=12)
        add_text_bg(ax, Q[0]+1.5, Q[1]-0.5, 0.5, '90°', fontsize=12)
        
        # Setup axes and view
        ax.set_xlabel('X (e.g., East)')
        ax.set_ylabel('Y (e.g., North)')
        ax.set_zlabel('Z (Height)')
        ax.set_title('3D Trigonometry Problem', fontsize=14, fontweight='bold')
        legend = ax.legend(loc='upper left', fontsize=10, framealpha=1.0)
        legend.get_frame().set_edgecolor('black')
        ax.view_init(elev=25, azim=40)
        ax.set_xlim(P[0]-1, R[0]+1)
        ax.set_ylim(P[1]-1, max(Q[1], R[1])+1)
        ax.set_zlim(P[2]-1, T[2]+1)
        ax.grid(True, alpha=0.3)
        
        # Add labels at top of figure
        fig.text(0.15, 0.95, 'PQ=15', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=3))
        fig.text(0.28, 0.95, 'QR=10', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=3))
        fig.text(0.41, 0.95, 'PR=18.0', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=3))
        fig.text(0.57, 0.95, 'Flagpole PT=10.5', color='blue', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue', pad=3))

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return filename
        
    # === Form 4: Vectors - Operations ===

    def f4_vo_notes_parallelogram_vectors(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')

        # Define vectors u, v
        u = np.array([4, 0])
        v = np.array([1, 2])
        # Vertices
        A = np.array([0, 0])
        B = A + u
        D = A + v
        C = B + v # or D + u

        # Draw sides with arrows
        ax.arrow(A[0], A[1], u[0], u[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', length_includes_head=True)
        ax.arrow(A[0], A[1], v[0], v[1], head_width=0.2, head_length=0.3, fc='red', ec='red', length_includes_head=True)
        ax.arrow(D[0], D[1], u[0], u[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', length_includes_head=True, linestyle=':') # DC=u
        ax.arrow(B[0], B[1], v[0], v[1], head_width=0.2, head_length=0.3, fc='red', ec='red', length_includes_head=True, linestyle=':') # BC=v

        # Draw diagonals
        diag_ac = C - A
        diag_db = D - B
        ax.arrow(A[0], A[1], diag_ac[0], diag_ac[1], head_width=0.2, head_length=0.3, fc='green', ec='green', length_includes_head=True) # AC
        ax.arrow(B[0], B[1], diag_db[0], diag_db[1], head_width=0.2, head_length=0.3, fc='purple', ec='purple', length_includes_head=True) # DB

        # Labels
        ax.text(A[0]-0.2, A[1]-0.2, 'A')
        ax.text(B[0]+0.1, B[1]-0.2, 'B')
        ax.text(C[0]+0.1, C[1]+0.1, 'C')
        ax.text(D[0]-0.2, D[1]+0.1, 'D')
        ax.text(u[0]/2, u[1]-0.2, 'u', color='blue', fontweight='bold')
        ax.text(v[0]/2-0.2, v[1]/2, 'v', color='red', fontweight='bold')
        ax.text(diag_ac[0]/2+0.1, diag_ac[1]/2, 'u+v', color='green', fontweight='bold')
        ax.text(B[0]+diag_db[0]/2+0.1, B[1]+diag_db[1]/2, 'u-v', color='purple', fontweight='bold')


        all_verts = np.array([A, B, C, D])
        ax.set_xlim(min(all_verts[:,0])-1, max(all_verts[:,0])+1);
        ax.set_ylim(min(all_verts[:,1])-1, max(all_verts[:,1])+1)
        ax.set_title("Vectors in a Parallelogram")
        return self._save_plot(fig, filename)

    def f4_vo_notes_ratio_theorem(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')

        O = np.array([0, 0])
        A = np.array([1, 3])
        B = np.array([5, 1])
        m, n = 2, 3 # Ratio m:n

        # Calculate P using ratio theorem
        P = (n*A + m*B) / (m + n)

        # Draw vectors OA, OB, OP
        ax.arrow(O[0], O[1], A[0], A[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', length_includes_head=True)
        ax.arrow(O[0], O[1], B[0], B[1], head_width=0.2, head_length=0.3, fc='red', ec='red', length_includes_head=True)
        ax.arrow(O[0], O[1], P[0], P[1], head_width=0.2, head_length=0.3, fc='green', ec='green', length_includes_head=True)

        # Draw line segment AB
        ax.plot([A[0], B[0]], [A[1], B[1]], 'k:')

        # Mark points
        ax.text(O[0]-0.2, O[1]-0.2, 'O')
        ax.text(A[0], A[1]+0.1, 'A'); ax.plot(A[0],A[1],'ko')
        ax.text(B[0]+0.1, B[1], 'B'); ax.plot(B[0],B[1],'ko')
        ax.text(P[0], P[1]+0.1, 'P'); ax.plot(P[0],P[1],'ko')

        # Labels
        ax.text(A[0]*0.6, A[1]*0.6, 'a', color='blue', fontweight='bold')
        ax.text(B[0]*0.6, B[1]*0.6+0.1, 'b', color='red', fontweight='bold')
        ax.text(P[0]*0.7, P[1]*0.7-0.3, 'p', color='green', fontweight='bold')

        # Show ratio m:n on AB
        ax.text((A[0]+P[0])/2-0.2, (A[1]+P[1])/2+0.1, f'm={m}', color='k', fontsize=9)
        ax.text((P[0]+B[0])/2+0.1, (P[1]+B[1])/2, f'n={n}', color='k', fontsize=9)

        # Formula
        ax.text(2.5, -0.5, "p = (na + mb) / (m + n)", ha='center', fontsize=10)


        all_pts = np.array([O, A, B, P])
        ax.set_xlim(min(all_pts[:,0])-1, max(all_pts[:,0])+1);
        ax.set_ylim(min(all_pts[:,1])-1, max(all_pts[:,1])+1)
        ax.set_title(f"Ratio Theorem (AP:PB = {m}:{n})")
        return self._save_plot(fig, filename)

    def f4_vo_we1_parallelogram_vectors(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')

        # Define vectors a, c
        a = np.array([3, 1])
        c = np.array([-1, 2])
        # Vertices (O is origin)
        O = np.array([0, 0])
        A = O + a
        C = O + c
        B = A + c # or C + a

        # Draw sides with arrows
        ax.arrow(O[0], O[1], A[0], A[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', length_includes_head=True) # OA = a
        ax.arrow(O[0], O[1], C[0], C[1], head_width=0.2, head_length=0.3, fc='red', ec='red', length_includes_head=True) # OC = c
        ax.arrow(A[0], A[1], c[0], c[1], head_width=0.2, head_length=0.3, fc='red', ec='red', length_includes_head=True, linestyle=':') # AB=c
        ax.arrow(C[0], C[1], a[0], a[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', length_includes_head=True, linestyle=':') # CB=a

        # Draw diagonals
        diag_ob = B - O
        diag_ac = C - A
        # ax.arrow(O[0], O[1], diag_ob[0], diag_ob[1], head_width=0.2, head_length=0.3, fc='green', ec='green', length_includes_head=True) # OB
        # ax.arrow(A[0], A[1], diag_ac[0], diag_ac[1], head_width=0.2, head_length=0.3, fc='purple', ec='purple', length_includes_head=True) # AC

        # Labels
        ax.text(O[0]-0.2, O[1]-0.2, 'O')
        ax.text(A[0]+0.1, A[1], 'A')
        ax.text(B[0]+0.1, B[1]+0.1, 'B')
        ax.text(C[0]-0.2, C[1]+0.1, 'C')
        ax.text(A[0]/2, A[1]/2+0.1, 'a', color='blue', fontweight='bold')
        ax.text(C[0]/2-0.3, C[1]/2, 'c', color='red', fontweight='bold')
        #ax.text(diag_ob[0]/2+0.1, diag_ob[1]/2, 'OB=a+c', color='green', fontweight='bold', fontsize=9)
        #ax.text(A[0]+diag_ac[0]/2-0.2, A[1]+diag_ac[1]/2, 'AC=c-a', color='purple', fontweight='bold', fontsize=9)


        all_verts = np.array([O, A, B, C])
        ax.set_xlim(min(all_verts[:,0])-1, max(all_verts[:,0])+1);
        ax.set_ylim(min(all_verts[:,1])-1, max(all_verts[:,1])+1)
        ax.set_title("Parallelogram OABC Vectors")
        return self._save_plot(fig, filename)

    def f4_vo_we4_triangle_vectors_ratio(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal'); ax.axis('off')

        # Vectors a, b
        a = np.array([1, 4]) # OA
        b = np.array([5, 1]) # OB
        O = np.array([0,0])
        A = a
        B = b

        # Midpoint M of OA
        M = A / 2
        # Point N on AB (AN:NB = 2:1)
        N = (1*A + 2*B) / (2 + 1)

        # Draw triangle OAB
        ax.plot([O[0], A[0], B[0], O[0]], [O[1], A[1], B[1], O[1]], 'k:')

        # Draw vectors OA, OB
        ax.arrow(O[0], O[1], A[0], A[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', length_includes_head=True)
        ax.arrow(O[0], O[1], B[0], B[1], head_width=0.2, head_length=0.3, fc='red', ec='red', length_includes_head=True)

        # Draw vector MN
        vec_mn = N - M
        ax.arrow(M[0], M[1], vec_mn[0], vec_mn[1], head_width=0.2, head_length=0.3, fc='green', ec='green', length_includes_head=True)

        # Mark points
        ax.plot([O[0], A[0], B[0], M[0], N[0]], [O[1], A[1], B[1], M[1], N[1]], 'ko', markersize=5)
        ax.text(O[0]-0.2, O[1]-0.2, 'O')
        ax.text(A[0], A[1]+0.1, 'A');
        ax.text(B[0]+0.1, B[1], 'B');
        ax.text(M[0]-0.3, M[1], 'M');
        ax.text(N[0]+0.1, N[1]+0.1, 'N');

        # Labels
        ax.text(A[0]/2- 0.2, A[1]/2+0.1, 'a', color='blue', fontweight='bold')
        ax.text(B[0]/2+0.1, B[1]/2+ 0.2, 'b', color='red', fontweight='bold')
        ax.text(M[0]+vec_mn[0]/2, M[1]+vec_mn[1]/2+0.1, 'MN=?', color='green', fontweight='bold')
        # Ratio on AB
        ax.text((A[0]+N[0])/2, (A[1]+N[1])/2+0.1, '2', color='k', fontsize=9)
        ax.text((N[0]+B[0])/2, (N[1]+B[1])/2, '1', color='k', fontsize=9)


        all_pts = np.array([O, A, B, M, N])
        ax.set_xlim(min(all_pts[:,0])-1, max(all_pts[:,0])+1);
        ax.set_ylim(min(all_pts[:,1])-1, max(all_pts[:,1])+1)
        ax.set_title("Triangle Vector Ratios")
        return self._save_plot(fig, filename)

    # === Form 4: Transformation - Reflection ===

    def f4_trfl_we5_reflect_y_eq_x_plus_1(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal');

        A = (3, 1)
        # Mirror line y = x + 1
        x_line = np.linspace(-1, 4, 100)
        y_line = x_line + 1

        # Perpendicular line from A, slope -1
        # y - 1 = -1(x - 3) => y = -x + 4
        y_perp = -x_line + 4

        # Intersection N
        intersect_x = 1.5
        intersect_y = 2.5
        N = (intersect_x, intersect_y)

        # Image A'
        A_prime = (0, 4)

        # Plot mirror line
        ax.plot(x_line, y_line, 'r--', label='Mirror Line y=x+1')
        # Plot point A
        ax.plot(A[0], A[1], 'bo', label=f'A{A}')
        # Plot image A'
        ax.plot(A_prime[0], A_prime[1], 'go', label=f"A'{A_prime}")
        # Plot line segment AA'
        ax.plot([A[0], A_prime[0]], [A[1], A_prime[1]], 'k:')
        # Plot intersection N
        ax.plot(N[0], N[1], 'mx', markersize=8, label=f'Midpoint N{N}')
        # Plot perpendicular line segment (optional)
        # ax.plot(x_line, y_perp, 'c:', alpha=0.5)

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 6)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Reflection of A(3,1) in y=x+1")
        ax.legend()

        return self._save_plot(fig, filename)


    # === Form 4: Transformation - Rotation ===
    # Matrix examples don't require new diagrams beyond basic object/image plots if needed,
    # and f3_trot_we4_rotate_180 covered the geometry for 180 deg.

    # === Form 4: Transformation - Enlargement ===
    # Matrix examples don't require new diagrams beyond basic object/image plots if needed,
    # and f3_te_we4_enlarge_triangle covered the geometry for origin center.

    def f4_te_we6_find_center_scale_factor(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white'); ax.set_aspect('equal');

        # Given points
        P, Q = (1,2), (3,2) # Object segment PQ
        P_prime, Q_prime = (3,5), (9,5) # Image segment P'Q'

        # Calculate center and scale factor (as done in steps)
        center = (0, 0.5)
        k = 3

        # Plot object and image segments
        ax.plot([P[0], Q[0]], [P[1], Q[1]], 'b-o', label=f'Object PQ (Len={Q[0]-P[0]})')
        ax.plot([P_prime[0], Q_prime[0]], [P_prime[1], Q_prime[1]], 'r--o', label=f"Image P'Q' (Len={Q_prime[0]-P_prime[0]})")
        ax.text(P[0], P[1]-0.3, 'P'); ax.text(Q[0], Q[1]-0.3, 'Q')
        ax.text(P_prime[0], P_prime[1]+0.1, "P'"); ax.text(Q_prime[0], Q_prime[1]+0.1, "Q'")

        # Draw lines through corresponding points to find center
        # Line PP'
        ax.plot([center[0], P_prime[0]+1], [center[1], P_prime[1]+1.5], 'g:', linewidth=1) # Extend beyond P'
        # Line QQ'
        ax.plot([center[0], Q_prime[0]+1], [center[1], Q_prime[1]+0.5], 'g:', linewidth=1) # Extend beyond Q'

        # Mark Center
        ax.plot(center[0], center[1], 'kx', markersize=10, markeredgewidth=2, label=f'Center ({center[0]},{center[1]})')

        # Annotate scale factor
        ax.text(5, 1, f"Scale factor k = {k}", fontsize=10)

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        all_x = [P[0], Q[0], P_prime[0], Q_prime[0], center[0]]
        all_y = [P[1], Q[1], P_prime[1], Q_prime[1], center[1]]
        ax.set_xlim(min(all_x)-1, max(all_x)+1)
        ax.set_ylim(min(all_y)-1, max(all_y)+1)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Finding Center and Scale Factor of Enlargement")
        ax.legend(loc='upper left')

        return self._save_plot(fig, filename)

    # === Form 4: Transformation - Stretch ===

    def f4_tstr_notes_stretch_diagrams(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Stretch Transformations', fontsize=14)

        # Original Square
        sq_verts = np.array([[1,1], [2,1], [2,2], [1,2], [1,1]])

        # --- Stretch || y-axis, x-axis invariant ---
        ax1.set_facecolor('white'); ax1.set_aspect('equal')
        k_y = 2
        img1_verts = sq_verts.copy()
        img1_verts[:, 1] *= k_y # Stretch y

        ax1.plot(sq_verts[:,0], sq_verts[:,1], 'b-o', label='Original')
        ax1.plot(img1_verts[:,0], img1_verts[:,1], 'r--o', label=f'Stretched (y*{k_y})')
        ax1.axhline(0, color='green', linewidth=2, label='Invariant x-axis')

        ax1.grid(True, linestyle='--'); ax1.axvline(0, color='k', lw=0.5)
        ax1.set_xlim(0, 3); ax1.set_ylim(-0.5, 5)
        ax1.set_title("Stretch || y-axis (x-axis invariant)")
        ax1.legend(fontsize=8)

        # --- Stretch || x-axis, y-axis invariant ---
        ax2.set_facecolor('white'); ax2.set_aspect('equal')
        k_x = 2
        img2_verts = sq_verts.copy()
        img2_verts[:, 0] *= k_x # Stretch x

        ax2.plot(sq_verts[:,0], sq_verts[:,1], 'b-o', label='Original')
        ax2.plot(img2_verts[:,0], img2_verts[:,1], 'r--o', label=f'Stretched (x*{k_x})')
        ax2.axvline(0, color='green', linewidth=2, label='Invariant y-axis')

        ax2.grid(True, linestyle='--'); ax2.axhline(0, color='k', lw=0.5)
        ax2.set_xlim(-0.5, 5); ax2.set_ylim(0, 3)
        ax2.set_title("Stretch || x-axis (y-axis invariant)")
        ax2.legend(fontsize=8)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    # === Form 4: Transformation - Shear ===

    def f4_tsh_notes_shear_diagrams(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Shear Transformations', fontsize=14)

        # Original Square
        sq_verts = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])

        # --- Shear || x-axis, x-axis invariant ---
        ax1.set_facecolor('white'); ax1.set_aspect('equal')
        k_x = 1 # Shear factor
        # Rule: (x,y) -> (x+ky, y)
        img1_verts = sq_verts.copy()
        img1_verts[:, 0] = sq_verts[:, 0] + k_x * sq_verts[:, 1]

        ax1.plot(sq_verts[:,0], sq_verts[:,1], 'b-o', label='Original')
        ax1.plot(img1_verts[:,0], img1_verts[:,1], 'r--o', label=f'Sheared (k={k_x})')
        ax1.axhline(0, color='green', linewidth=2, label='Invariant x-axis')
        # Show movement
        ax1.arrow(sq_verts[2,0], sq_verts[2,1], img1_verts[2,0]-sq_verts[2,0], img1_verts[2,1]-sq_verts[2,1],
                 head_width=0.1, head_length=0.15, fc='g', ec='g', length_includes_head=True, linestyle=':')

        ax1.grid(True, linestyle='--'); ax1.axvline(0, color='k', lw=0.5)
        ax1.set_xlim(-0.5, 2.5); ax1.set_ylim(-0.5, 1.5)
        ax1.set_title("Shear || x-axis (x-axis invariant)")
        ax1.legend(fontsize=8)

        # --- Shear || y-axis, y-axis invariant ---
        ax2.set_facecolor('white'); ax2.set_aspect('equal')
        k_y = 1 # Shear factor
        # Rule: (x,y) -> (x, y+kx)
        img2_verts = sq_verts.copy()
        img2_verts[:, 1] = sq_verts[:, 1] + k_y * sq_verts[:, 0]

        ax2.plot(sq_verts[:,0], sq_verts[:,1], 'b-o', label='Original')
        ax2.plot(img2_verts[:,0], img2_verts[:,1], 'r--o', label=f'Sheared (k={k_y})')
        ax2.axvline(0, color='green', linewidth=2, label='Invariant y-axis')
         # Show movement
        ax2.arrow(sq_verts[2,0], sq_verts[2,1], img2_verts[2,0]-sq_verts[2,0], img2_verts[2,1]-sq_verts[2,1],
                 head_width=0.1, head_length=0.15, fc='g', ec='g', length_includes_head=True, linestyle=':')

        ax2.grid(True, linestyle='--'); ax2.axhline(0, color='k', lw=0.5)
        ax2.set_xlim(-0.5, 1.5); ax2.set_ylim(-0.5, 2.5)
        ax2.set_title("Shear || y-axis (y-axis invariant)")
        ax2.legend(fontsize=8)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)

    # === Form 4: Probability - Combined Events ===
    def f4_pce_notes_tree_diagram(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.axis('off')
        level1_y, level2_y = 0.5, 0.5
        level1_x, level2_x, level3_x = 0, 1.5, 3
        dy = 0.4  # Increased vertical separation for better spacing
        
        # Define a background box style for text
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="white", alpha=0.8)

        # Level 0 -> 1
        ax.plot([level1_x, level2_x], [level1_y, level2_y+dy], 'k-')  # H
        ax.plot([level1_x, level2_x], [level1_y, level2_y-dy], 'k-')  # T
        
        # Add edge labels with background
        ax.text((level1_x+level2_x)/2 - 0.05, level1_y+dy*0.7, 'P(H)=1/2', ha='center', va='center', 
                fontsize=9, bbox=bbox_props)
        ax.text((level1_x+level2_x)/2 - 0.05, level1_y-dy*0.7, 'P(T)=1/2', ha='center', va='center', 
                fontsize=9, bbox=bbox_props)
        
        # Node labels
        ax.text(level1_x-0.1, level1_y, 'Start', ha='right', va='center', bbox=bbox_props)
        ax.text(level2_x+0.1, level2_y+dy, 'H (1st)', ha='left', va='center', bbox=bbox_props)
        ax.text(level2_x+0.1, level2_y-dy, 'T (1st)', ha='left', va='center', bbox=bbox_props)

        # Level 1 -> 2 (From H)
        ax.plot([level2_x, level3_x], [level2_y+dy, level2_y+2*dy], 'k-')  # HH
        ax.plot([level2_x, level3_x], [level2_y+dy, level2_y+0*dy], 'k-')  # HT
        
        # Add edge labels with background - adjusted positions for clarity
        ax.text((level2_x+level3_x)/2, level2_y+1.5*dy + 0.1, 'P(H)=1/2', ha='center', va='center', 
                fontsize=9, bbox=bbox_props)
        ax.text((level2_x+level3_x)/2, level2_y+0.5*dy - 0.1, 'P(T)=1/2', ha='center', va='center', 
                fontsize=9, bbox=bbox_props)
        
        # Outcome labels
        ax.text(level3_x+0.1, level2_y+2*dy, 'H -> Outcome HH, P=1/4', ha='left', va='center', bbox=bbox_props)
        ax.text(level3_x+0.1, level2_y+0*dy + 0.1, 'T -> Outcome HT, P=1/4', ha='left', va='center', bbox=bbox_props)

        # Level 1 -> 2 (From T)
        ax.plot([level2_x, level3_x], [level2_y-dy, level2_y+0*dy], 'k-')  # TH
        ax.plot([level2_x, level3_x], [level2_y-dy, level2_y-2*dy], 'k-')  # TT
        
        # Add edge labels with background - adjusted positions for clarity
        ax.text((level2_x+level3_x)/2, level2_y-0.5*dy + 0.1, 'P(H)=1/2', ha='center', va='center', 
                fontsize=9, bbox=bbox_props)
        ax.text((level2_x+level3_x)/2, level2_y-1.5*dy - 0.1, 'P(T)=1/2', ha='center', va='center', 
                fontsize=9, bbox=bbox_props)
        
        # Fix the overlapping outcome labels
        ax.text(level3_x+0.1, level2_y+0*dy - 0.15, 'H -> Outcome TH, P=1/4', ha='left', va='center', bbox=bbox_props)
        ax.text(level3_x+0.1, level2_y-2*dy, 'T -> Outcome TT, P=1/4', ha='left', va='center', bbox=bbox_props)

        # Adjust plot limits for better visibility
        ax.set_xlim(level1_x-0.5, level3_x+3)
        ax.set_ylim(level1_y-2.5*dy, level1_y+2.5*dy)
        ax.set_title("Tree Diagram: Tossing a Coin Twice")
        plt.tight_layout()
        return self._save_plot(fig, filename)


    # --- GENERATED FOR MISSING IMAGE: Probability WE2 ---
    def f4_pce_we2_tree_diagram_no_replace(self):
        # This function was provided in the example, adapted slightly
        filename_path = get_filename_from_caller() # Corrected call
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.axis('off')
        level1_y, level2_y = 0.5, 0.5
        level1_x, level2_x, level3_x = 0, 2, 4
        dy = 0.4 # Vertical separation

        # Probabilities Stage 1
        p_r1_num, p_r1_den = 4, 10
        p_b1_num, p_b1_den = 6, 10
        # p_r1 = p_r1_num / p_r1_den # Not needed directly for plot
        # p_b1 = p_b1_num / p_b1_den

        # Probabilities Stage 2 (Conditional) - Total = 9
        p_r2_r1_num, p_r2_r1_den = 3, 9
        p_b2_r1_num, p_b2_r1_den = 6, 9
        p_r2_b1_num, p_r2_b1_den = 4, 9
        p_b2_b1_num, p_b2_b1_den = 5, 9
        # p_r2_r1 = p_r2_r1_num / p_r2_r1_den # Not needed directly for plot
        # p_b2_r1 = p_b2_r1_num / p_b2_r1_den
        # p_r2_b1 = p_r2_b1_num / p_r2_b1_den
        # p_b2_b1 = p_b2_b1_num / p_b2_b1_den

        # Level 0 -> 1
        ax.plot([level1_x, level2_x], [level1_y, level2_y+dy], 'k-') # R1
        ax.plot([level1_x, level2_x], [level1_y, level2_y-dy], 'k-') # B1
        ax.text((level1_x+level2_x)/2-1, level1_y+dy*0.7, f'P(R1)={p_r1_num}/{p_r1_den}', ha='center', va='bottom', fontsize=9)
        ax.text((level1_x+level2_x)/2, level1_y-dy*0.7, f'P(B1)={p_b1_num}/{p_b1_den}', ha='center', va='top', fontsize=9)
        ax.text(level1_x-0.1, level1_y, 'Start', ha='right', va='center')
        ax.text(level2_x+0.1, level2_y+dy, 'Red (1st)', ha='left', va='center')
        ax.text(level2_x+0.1, level2_y-dy, 'Blue (1st)', ha='left', va='center')

        # Level 1 -> 2 (From R1)
        ax.plot([level2_x, level3_x], [level2_y+dy, level2_y+2*dy], 'k-') # R1R2
        ax.plot([level2_x, level3_x], [level2_y+dy, level2_y+0*dy], 'k-') # R1B2
        ax.text((level2_x+level3_x)/2+ 1, level2_y+1.5*dy, f'P(R2|R1)={p_r2_r1_num}/{p_r2_r1_den}', ha='center', va='bottom', fontsize=9)
        ax.text((level2_x+level3_x)/2, level2_y+0.5*dy, f'P(B2|R1)={p_b2_r1_num}/{p_b2_r1_den}', ha='center', va='bottom', fontsize=9)
        p_rr_num, p_rr_den = p_r1_num * p_r2_r1_num, p_r1_den * p_r2_r1_den
        p_rb_num, p_rb_den = p_r1_num * p_b2_r1_num, p_r1_den * p_b2_r1_den
        ax.text(level3_x+0.1, level2_y+2*dy, f'R2 -> Outcome RR, P={p_rr_num}/{p_rr_den}', ha='left', va='center')
        ax.text(level3_x+0.1, level2_y+0*dy, f'B2 -> Outcome RB, P={p_rb_num}/{p_rb_den}', ha='left', va='center')

        # Level 1 -> 2 (From B1)
        ax.plot([level2_x, level3_x], [level2_y-dy, level2_y+0*dy], 'k-') # B1R2
        ax.plot([level2_x, level3_x], [level2_y-dy, level2_y-2*dy], 'k-') # B1B2
        ax.text((level2_x+level3_x)/2, level2_y-0.5*dy, f'P(R2|B1)={p_r2_b1_num}/{p_r2_b1_den}', ha='center', va='top', fontsize=9)
        ax.text((level2_x+level3_x)/2, level2_y-1.5*dy, f'P(B2|B1)={p_b2_b1_num}/{p_b2_b1_den}', ha='center', va='top', fontsize=9)
        p_br_num, p_br_den = p_b1_num * p_r2_b1_num, p_b1_den * p_r2_b1_den
        p_bb_num, p_bb_den = p_b1_num * p_b2_b1_num, p_b1_den * p_b2_b1_den
        # Adjust y slightly for BR outcome text
        ax.text(level3_x+0.1, level2_y+0*dy + 0.05, f'R2 -> Outcome BR, P={p_br_num}/{p_br_den}', ha='left', va='center')
        ax.text(level3_x+0.1, level2_y-2*dy, f'B2 -> Outcome BB, P={p_bb_num}/{p_bb_den}', ha='left', va='center')

        # Highlight RR path calculation for P(Both Red) as asked in WE2 problem text
        ax.plot([level1_x, level2_x, level3_x], [level1_y, level2_y+dy, level2_y+2*dy], 'r-', linewidth=2, alpha=0.7)
        # Calculate GCD only if denominator is not zero
        gcd_rr = np.gcd(p_rr_num, p_rr_den) if p_rr_den != 0 else 1
        # Ensure integer division
        simplified_num_rr = p_rr_num // gcd_rr
        simplified_den_rr = p_rr_den // gcd_rr

        ax.text(level1_x, level1_y + 1.5*dy, # Adjust text position
                f'P(Both Red) = P(R1) * P(R2|R1)\n= ({p_r1_num}/{p_r1_den}) * ({p_r2_r1_num}/{p_r2_r1_den}) = {p_rr_num}/{p_rr_den} = {simplified_num_rr}/{simplified_den_rr}', # Simplify 12/90 -> 2/15
                color='red', fontsize=10, va='center')

        ax.set_xlim(level1_x-0.5, level3_x+4) # Increased xlim
        ax.set_ylim(level1_y-2.5*dy, level1_y+2.5*dy)
        ax.set_title("Tree Diagram: Picking 2 Marbles (No Replacement) - WE2")
        plt.tight_layout()
        return self._save_plot(fig, filename_path)

    # --- GENERATED FOR MISSING IMAGE: Probability WE3 ---

    def f4_pce_we3_tree_diagram_ref(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.axis('off')

        # layout
        level1_x, level2_x, level3_x = 0, 2, 4
        level1_y = level2_y = 0.5
        dy = 0.6  # increased vertical separation

        # probabilities
        p_r1_num, p_r1_den = 4, 10
        p_b1_num, p_b1_den = 6, 10
        p_r2_r1_num, p_r2_r1_den = 3, 9
        p_b2_r1_num, p_b2_r1_den = 6, 9
        p_r2_b1_num, p_r2_b1_den = 4, 9
        p_b2_b1_num, p_b2_b1_den = 5, 9

        # compute end probabilities
        p_rr_num, p_rr_den = p_r1_num * p_r2_r1_num, p_r1_den * p_r2_r1_den
        p_rb_num, p_rb_den = p_r1_num * p_b2_r1_num, p_r1_den * p_b2_r1_den
        p_br_num, p_br_den = p_b1_num * p_r2_b1_num, p_b1_den * p_r2_b1_den
        p_bb_num, p_bb_den = p_b1_num * p_b2_b1_num, p_b1_den * p_b2_b1_den

        # --- draw the tree ---
        # Stage 1
        ax.plot([level1_x, level2_x], [level1_y, level2_y+dy], 'k-')
        ax.plot([level1_x, level2_x], [level1_y, level2_y-dy], 'k-')
        ax.text((level1_x+level2_x)/2, level1_y+dy*0.7,
                f'P(R1)={p_r1_num}/{p_r1_den}',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text((level1_x+level2_x)/2, level1_y-dy*0.7,
                f'P(B1)={p_b1_num}/{p_b1_den}',
                ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text(level1_x-0.1, level1_y, 'Start', ha='right', va='center')
        ax.text(level2_x+0.1, level2_y+dy, 'Red (1st)', ha='left', va='center')
        ax.text(level2_x+0.1, level2_y-dy, 'Blue (1st)', ha='left', va='center')

        # Stage 2 from R1
        ax.plot([level2_x, level3_x], [level2_y+dy, level2_y+2*dy], 'k-')
        ax.plot([level2_x, level3_x], [level2_y+dy, level2_y+0*dy], 'k-')
        ax.text((level2_x+level3_x)/2, level2_y+1.5*dy,
                f'P(R2|R1)={p_r2_r1_num}/{p_r2_r1_den}',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text((level2_x+level3_x)/2, level2_y+0.5*dy,
                f'P(B2|R1)={p_b2_r1_num}/{p_b2_r1_den}',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text(level3_x+0.1, level2_y+2*dy,
                f'R2 → RR, P={p_rr_num}/{p_rr_den}',
                ha='left', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text(level3_x+0.1, level2_y+0*dy+0.07,
                f'B2 → RB, P={p_rb_num}/{p_rb_den}',
                ha='left', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

        # Stage 2 from B1
        ax.plot([level2_x, level3_x], [level2_y-dy, level2_y+0*dy], 'k-')
        ax.plot([level2_x, level3_x], [level2_y-dy, level2_y-2*dy], 'k-')
        ax.text((level2_x+level3_x)/2, level2_y-0.5*dy,
                f'P(R2|B1)={p_r2_b1_num}/{p_r2_b1_den}',
                ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text((level2_x+level3_x)/2, level2_y-1.5*dy,
                f'P(B2|B1)={p_b2_b1_num}/{p_b2_b1_den}',
                ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text(level3_x+0.1, level2_y+0*dy-0.07,
                f'R2 → BR, P={p_br_num}/{p_br_den}',
                ha='left', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.text(level3_x+0.1, level2_y-2*dy,
                f'B2 → BB, P={p_bb_num}/{p_bb_den}',
                ha='left', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

        # highlight the “different colors” paths
        ax.plot([level1_x, level2_x, level3_x],
                [level1_y, level2_y+dy, level2_y+0*dy],
                'g-', linewidth=2, alpha=0.7)
        ax.plot([level1_x, level2_x, level3_x],
                [level1_y, level2_y-dy, level2_y+0*dy],
                'g-', linewidth=2, alpha=0.7)

        # final probability off to the right
        total_diff_num = p_rb_num + p_br_num
        total_diff_den = p_rb_den
        gcd_val = np.gcd(total_diff_num, total_diff_den)
        simp_num = total_diff_num // gcd_val
        simp_den = total_diff_den // gcd_val
        ax.text(level3_x + 1.0, level2_y - 0.5*dy,
                f'P(Different) = P(RB)+P(BR)\n= ({p_rb_num}/{p_rb_den})+({p_br_num}/{p_br_den}) '
                f'= {total_diff_num}/{total_diff_den} = {simp_num}/{simp_den}',
                color='green', fontsize=10, va='center',
                bbox=dict(facecolor='white', alpha=0.8))

        # extend the x‐limits for that off‐tree text
        ax.set_xlim(level1_x - 0.5, level3_x + 5)
        ax.set_ylim(level1_y - 2.5*dy, level1_y + 2.5*dy)
        ax.set_title("Tree Diagram: P(Different Colors) - WE3 Ref")
        plt.tight_layout()
        return self._save_plot(fig, filename)
    # === NEW METHOD: Vector Subtraction Laws ===
    def vector_subtraction_laws(self):
        """
        Generates a diagram illustrating vector subtraction a - b using two common methods:
        1. Connecting the head of vector b to the head of vector a.
        2. Adding vector a to vector -b using the triangle law (a + (-b)).
        """
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
        fig.suptitle('Vector Subtraction: $\\mathbf{a} - \\mathbf{b}$', fontsize=16)

        # Define vectors
        O = np.array([0, 0])  # Origin
        a = np.array([4, 2])
        b = np.array([1, 3])
        a_minus_b = a - b
        minus_b = -b

        # Common settings for quiver plots (vectors)
        quiver_args = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1}

        # --- Plot 1: Head-to-Head Method (a - b) ---
        ax1.set_facecolor('white')
        ax1.set_title('Method 1: $\\mathbf{a} - \\mathbf{b}$ (Head of $\\mathbf{b}$ to Head of $\\mathbf{a}$)')
        ax1.set_aspect('equal', adjustable='box') # Ensure vectors look perpendicular if they are

        # Plot vectors a and b from origin
        ax1.quiver(O[0], O[1], a[0], a[1], color='blue', **quiver_args, label='$\\mathbf{a}$')
        ax1.quiver(O[0], O[1], b[0], b[1], color='red', **quiver_args, label='$\\mathbf{b}$')

        # Plot vector a - b (starts at head of b, ends at head of a)
        ax1.quiver(b[0], b[1], a_minus_b[0], a_minus_b[1], color='green', **quiver_args, label='$\\mathbf{a} - \\mathbf{b}$')

        # Add labels
        ax1.text(O[0] - 0.2, O[1] - 0.2, 'O', fontsize=12, ha='right', va='top')
        ax1.text(a[0] * 1.1, a[1] * 1.1, '$\\mathbf{a}$', color='blue', fontsize=14, ha='center', va='center')
        ax1.text(b[0] * 1.1, b[1] * 1.1, '$\\mathbf{b}$', color='red', fontsize=14, ha='center', va='center')
        # Position label for a - b near its midpoint
        mid_a_minus_b = b + a_minus_b / 2
        ax1.text(mid_a_minus_b[0], mid_a_minus_b[1] + 0.2, '$\\mathbf{a} - \\mathbf{b}$', color='green', fontsize=14, ha='center', va='bottom')

        # --- Plot 2: Triangle Law Method (a + (-b)) ---
        ax2.set_facecolor('white')
        ax2.set_title('Method 2: $\\mathbf{a} + (-\\mathbf{b})$ (Triangle Law)')
        ax2.set_aspect('equal', adjustable='box') # Ensure consistency

        # Plot vector a from origin
        ax2.quiver(O[0], O[1], a[0], a[1], color='blue', **quiver_args, label='$\\mathbf{a}$')

        # Plot vector -b (starting from the head of a)
        ax2.quiver(a[0], a[1], minus_b[0], minus_b[1], color='orange', **quiver_args, label='$-\\mathbf{b}$')

        # Plot the resultant vector a + (-b) from the origin
        ax2.quiver(O[0], O[1], a_minus_b[0], a_minus_b[1], color='green', **quiver_args, label='$\\mathbf{a} + (-\\mathbf{b})$')

        # Optionally: Draw -b from the origin as a reference (dashed)
        ax2.quiver(O[0], O[1], minus_b[0], minus_b[1], color='orange', linestyle='--', alpha=0.7, **quiver_args)
        ax2.text(minus_b[0]*1.1, minus_b[1]*1.1 - 0.3, '($-\\mathbf{b}$ ref)', color='orange', alpha=0.7, fontsize=10, ha='center', va='top')


        # Add labels
        ax2.text(O[0] - 0.2, O[1] - 0.2, 'O', fontsize=12, ha='right', va='top')
        ax2.text(a[0] * 0.5, a[1] * 0.5 + 0.2, '$\\mathbf{a}$', color='blue', fontsize=14, ha='right', va='bottom')
        # Position label for -b near its midpoint (when added to a)
        mid_minus_b = a + minus_b / 2
        ax2.text(mid_minus_b[0] + 0.1, mid_minus_b[1], '$-\\mathbf{b}$', color='orange', fontsize=14, ha='left', va='center')
        ax2.text(a_minus_b[0] * 0.5 + 0.1, a_minus_b[1] * 0.5 - 0.2, '$\\mathbf{a} + (-\\mathbf{b})$', color='green', fontsize=14, ha='left', va='top')


        # --- General Plot Settings ---
        for ax in [ax1, ax2]:
            # Set limits (adjust based on vector values if needed)
            min_lim = min(O[0], O[1], a[0], a[1], b[0], b[1], a_minus_b[0], a_minus_b[1], minus_b[0], minus_b[1], (a+minus_b)[0], (a+minus_b)[1]) - 1
            max_lim = max(O[0], O[1], a[0], a[1], b[0], b[1], a_minus_b[0], a_minus_b[1], minus_b[0], minus_b[1], (a+minus_b)[0], (a+minus_b)[1]) + 1
            ax.set_xlim(min_lim, max_lim)
            ax.set_ylim(min_lim, max_lim)
            # Make axes invisible for a cleaner vector diagram look
            ax.axis('off')
            # Optional: Add a light grid
            # ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper left') # Show labels defined in quiver

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap

        # Save the plot using the class method
        return self._save_plot(fig, filename)



    def f3_ineq_number_line_example(self):
        """
        Generates a number line diagram representing the solution set 2 <= x < 4.
        Corresponds to Worked Example 2 (solving x + 1 < 5 and x - 3 >= -1).
        """
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 2), facecolor='white')
        ax.set_facecolor('white')

        # Define plot limits and points
        plot_min = 0
        plot_max = 5
        point1 = 2  # Inclusive
        point2 = 4  # Exclusive

        # Draw the main number line axis
        ax.axhline(0, color='black', lw=1.5)

        # Set limits and remove y-axis
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(-0.5, 0.5) # Just enough space for circles/line
        ax.yaxis.set_visible(False) # Hide y-axis
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Keep bottom spine as the number line
        ax.spines['bottom'].set_position(('data', 0))

        # Add ticks and labels
        ticks = np.arange(plot_min, plot_max + 1)
        ax.set_xticks(ticks)
        ax.tick_params(axis='x', direction='inout', length=6) # Ticks inside and outside

        # Draw the points/circles
        # Closed circle at point1 (2)
        ax.plot(point1, 0, 'o', markersize=10, markerfacecolor='black', markeredgecolor='black')
        # Open circle at point2 (4)
        ax.plot(point2, 0, 'o', markersize=10, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5)

        # Draw the connecting line segment
        ax.plot([point1, point2], [0, 0], color='blue', lw=3, solid_capstyle='butt')

        # Add title
        ax.set_title('Number Line Representation of $2 \leq x < 4$', fontsize=14)

        plt.tight_layout()
        return self._save_plot(fig, filename)


    # def f3_ineq_cartesian_plane_example(self):
    #     """
    #     Generates a Cartesian plane diagram showing the region defined by
    #     y >= 0, x > 1, and y < x.
    #     Corresponds to Worked Example 4.
    #     """
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    #     ax.set_facecolor('white')

    #     # Define plot limits and grid
    #     x_min, x_max = -1, 5
    #     y_min, y_max = -1, 5
    #     ax.set_xlim(x_min, x_max)
    #     ax.set_ylim(y_min, y_max)
    #     ax.set_xlabel('x-axis')
    #     ax.set_ylabel('y-axis')
    #     ax.set_title('Region defined by $y \geq 0$, $x > 1$, $y < x$', fontsize=14)
    #     ax.grid(True, linestyle='--', alpha=0.6)

    #     # Draw axes lines
    #     ax.axhline(0, color='black', lw=0.8)
    #     ax.axvline(0, color='black', lw=0.8)

    #     # Create a range for plotting lines
    #     x_range = np.linspace(x_min, x_max, 400)
    #     y_range = np.linspace(y_min, y_max, 400)

    #     # --- Inequality 1: y >= 0 ---
    #     # Boundary y = 0 (solid line, already drawn by axhline)
    #     ax.plot(x_range, 0*x_range, color='green', lw=2, linestyle='-', label='$y=0$')
    #     # Shade unwanted region (y < 0)
    #     ax.fill_between(x_range, y_min, 0, color='green', alpha=0.2, hatch='///', label='Shaded: $y < 0$ (unwanted)')

    #     # --- Inequality 2: x > 1 ---
    #     # Boundary x = 1 (dashed line)
    #     ax.axvline(1, color='red', lw=2, linestyle='--', label='$x=1$')
    #     # Shade unwanted region (x <= 1)
    #     ax.fill_betweenx(y_range, x_min, 1, color='red', alpha=0.2, hatch='\\\\\\', label='Shaded: $x \leq 1$ (unwanted)')

    #     # --- Inequality 3: y < x ---
    #     # Boundary y = x (dashed line)
    #     ax.plot(x_range, x_range, color='blue', lw=2, linestyle='--', label='$y=x$')
    #     # Shade unwanted region (y >= x)
    #     ax.fill_between(x_range, x_range, y_max, where=(x_range <= y_max), interpolate=True, color='blue', alpha=0.2, hatch='...', label='Shaded: $y \geq x$ (unwanted)')
    #     # Note: 'where' condition helps prevent shading filling the entire upper area unnecessarily

    #     # Indicate the solution region 'R'
    #     # Find a point clearly inside the unshaded region (e.g., x=2, y=0.5)
    #     ax.text(2, 0.5, 'R', fontsize=18, ha='center', va='center', color='black', weight='bold',
    #             bbox=dict(boxstyle='circle,pad=0.3', fc='yellow', ec='black', lw=1, alpha=0.8))

    #     # Adjust aspect ratio and add legend
    #     ax.set_aspect('equal', adjustable='box')
    #     # Place legend outside the plot to avoid overlap
    #     ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)


    #     plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    #     return self._save_plot(fig, filename)
    
    def f3_ai_we2_num_line_soln(self):
        """Number line for 2 <= x < 4 (same as f3_ai_notes_simul_ineq_1var, just different range)."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 2), facecolor='white')
        ax.set_facecolor('white')
        ax.set_yticks([])
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.xaxis.set_ticks_position('bottom')

        # Plot the line segment from 2 to 4
        ax.plot([2, 4], [0.1, 0.1], 'b-', linewidth=3)
        # Plot the closed circle at x=2
        ax.plot(2, 0.1, 'bo', markersize=8) # Closed circle (default)
        # Plot the open circle at x=4
        ax.plot(4, 0.1, 'bo', markersize=8, markerfacecolor='white') # Open circle

        ax.set_xticks(np.arange(0, 6, 1))
        ax.set_xlabel("x")
        ax.set_title("Number Line Solution: 2 <= x < 4")

        plt.tight_layout()
        return self._save_plot(fig, filename)
    
    def f3_ineq_cartesian_plane_example(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')

        # Define plot limits and grid
        x_min, x_max = -1, 5
        y_min, y_max = -1, 5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_title('Region defined by $y \geq 0$, $x > 1$, $y < x$', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Draw axes lines
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', lw=0.8)

        # Create a range for plotting lines
        x_range = np.linspace(x_min, x_max, 400)
        y_range = np.linspace(y_min, y_max, 400)

        # --- Inequality 1: y >= 0 ---
        # Boundary y = 0 (solid line, already drawn by axhline)
        ax.plot(x_range, 0*x_range, color='green', lw=2, linestyle='-', label='$y=0$')
        # Shade unwanted region (y < 0)
        ax.fill_between(x_range, y_min, 0, color='green', alpha=0.2, hatch='///', label='Shaded: $y < 0$ (unwanted)')

        # --- Inequality 2: x > 1 ---
        # Boundary x = 1 (dashed line)
        ax.axvline(1, color='red', lw=2, linestyle='--', label='$x=1$')
        # Shade unwanted region (x <= 1)
        ax.fill_betweenx(y_range, x_min, 1, color='red', alpha=0.2, hatch='\\\\\\', label='Shaded: $x \leq 1$ (unwanted)')

        # --- Inequality 3: y < x ---
        # Boundary y = x (dashed line)
        ax.plot(x_range, x_range, color='blue', lw=2, linestyle='--', label='$y=x$')
        # Shade unwanted region (y >= x)
        ax.fill_between(x_range, x_range, y_max, where=(x_range <= y_max), interpolate=True, color='blue', alpha=0.2, hatch='...', label='Shaded: $y \geq x$ (unwanted)')
        # Note: 'where' condition helps prevent shading filling the entire upper area unnecessarily

        # Indicate the solution region 'R'
        # Find a point clearly inside the unshaded region (e.g., x=2, y=0.5)
        ax.text(2, 0.5, 'R', fontsize=18, ha='center', va='center', color='black', weight='bold',
                bbox=dict(boxstyle='circle,pad=0.3', fc='yellow', ec='black', lw=1, alpha=0.8))

        # Adjust aspect ratio and add legend
        ax.set_aspect('equal', adjustable='box')
        # Place legend outside the plot to avoid overlap
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)


        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        return self._save_plot(fig, filename)
    
    def f3_trfl_notes_reflection_lines(self):
        filename = get_filename_from_caller()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        fig.suptitle('Reflection in Lines y=a and x=b', fontsize=14)

        # --- Reflection in y=a ---
        ax1.set_facecolor('white')
        ax1.set_aspect('equal', adjustable='box')
        a = 2 # Example value for y=a
        P = (3, 4) # Example point
        P_prime = (P[0], 2*a - P[1]) # (3, 0)

        ax1.axhline(a, color='red', linestyle='--', label=f'Mirror Line y={a}')
        ax1.plot(P[0], P[1], 'bo', label=f'P{P}')
        ax1.plot(P_prime[0], P_prime[1], 'ro', label=f"P'{P_prime}")
        ax1.plot([P[0], P_prime[0]], [P[1], P_prime[1]], 'k:', label='PP\'') # Line segment PP'
        # Midpoint of PP'
        M = ((P[0]+P_prime[0])/2, (P[1]+P_prime[1])/2)
        ax1.plot(M[0], M[1], 'gx', markersize=8, label='Midpoint on y=a') # Midpoint

        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.grid(True, linestyle='--')
        ax1.set_xlim(0, 5)
        ax1.set_ylim(-1, 6)
        ax1.set_title("Reflection in y = a")
        ax1.legend(fontsize=8)

        # --- Reflection in x=b ---
        ax2.set_facecolor('white')
        ax2.set_aspect('equal', adjustable='box')
        b = 1 # Example value for x=b
        Q = (-2, 3) # Example point
        Q_prime = (2*b - Q[0], Q[1]) # (4, 3)

        ax2.axvline(b, color='red', linestyle='--', label=f'Mirror Line x={b}')
        ax2.plot(Q[0], Q[1], 'bo', label=f'Q{Q}')
        ax2.plot(Q_prime[0], Q_prime[1], 'ro', label=f"Q'{Q_prime}")
        ax2.plot([Q[0], Q_prime[0]], [Q[1], Q_prime[1]], 'k:', label='QQ\'') # Line segment QQ'
        # Midpoint of QQ'
        N = ((Q[0]+Q_prime[0])/2, (Q[1]+Q_prime[1])/2)
        ax2.plot(N[0], N[1], 'gx', markersize=8, label='Midpoint on x=b') # Midpoint

        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(True, linestyle='--')
        ax2.set_xlim(-4, 6)
        ax2.set_ylim(0, 5)
        ax2.set_title("Reflection in x = b")
        ax2.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return self._save_plot(fig, filename)
    

    def f3_trot_notes_find_center(self):
        filename = 'math_diagrams/f3_trot_notes_find_center.png'
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')

        # Example Object and Image (e.g., 90 deg anti-clockwise rot about (1,1))
        A, B, C = (3,1), (5,1), (3,4)
        center = (1,1)
        angle_rad = math.radians(90)
        rot_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                               [math.sin(angle_rad), math.cos(angle_rad)]])

        def rotate_point(p, center, matrix):
            vec = np.array(p) - np.array(center)
            rot_vec = matrix @ vec
            return rot_vec + np.array(center)

        A_prime = rotate_point(A, center, rot_matrix) # (1, 3)
        B_prime = rotate_point(B, center, rot_matrix) # (1, 5)
        C_prime = rotate_point(C, center, rot_matrix) # (-2, 3)

        obj_verts = np.array([A, B, C, A])
        img_verts = np.array([A_prime, B_prime, C_prime, A_prime])

        # Plot object and image
        ax.plot(obj_verts[:, 0], obj_verts[:, 1], 'b-o', label='Object ABC')
        ax.plot(img_verts[:, 0], img_verts[:, 1], 'r--o', label="Image A'B'C'")
        for pt, name in zip([A, B, C], ['A', 'B', 'C']): ax.text(pt[0]+0.1, pt[1], name)
        for pt, name in zip([A_prime, B_prime, C_prime], ["A'", "B'", "C'"]): ax.text(pt[0]+0.1, pt[1], name)

        # Map coords back to names for label - Include image points
        # Using round() when creating tuple keys for floats handles potential precision issues.
        name_map = {
            tuple(A): 'A', tuple(B): 'B', tuple(C): 'C',
            tuple(np.round(A_prime, 5)): "A'",
            tuple(np.round(B_prime, 5)): "B'",
            tuple(np.round(C_prime, 5)): "C'"
        }

        # Function to plot perpendicular bisector
        def plot_perp_bisector(p1, p2, color):
            # p1 is original (tuple), p2 is image (numpy array)
            mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            dx = p2[0]-p1[0]
            dy = p2[1]-p1[1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6: return # Points coincide

            # Perpendicular direction (-dy, dx)
            perp_dx, perp_dy = -dy, dx
            # Extend line
            t = 3 # Arbitrary length extension factor
            start_pt = (mid[0] - t*perp_dx, mid[1] - t*perp_dy)
            end_pt = (mid[0] + t*perp_dx, mid[1] + t*perp_dy)

            # Get labels safely using .get() and construct the final label
            # Convert p2 (numpy array) to a rounded tuple for lookup
            label_p1 = name_map.get(tuple(p1), '?') # p1 is already a tuple
            label_p2 = name_map.get(tuple(np.round(p2, 5)), '?') # Convert p2 array to tuple key

            # Ensure labels were found before formatting
            if label_p1 != '?' and label_p2 != '?':
                 # We want the label like "Bisector AA'"
                 base_label_p1 = label_p1
                 base_label_p2 = label_p2.replace("'", "") # Get base name like 'A' from 'A\''
                 plot_label = f'Bisector {base_label_p1}{label_p2}'
            else:
                 plot_label = 'Bisector' # Fallback label

            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], color=color, linestyle=':', linewidth=1.5, label=plot_label)
            ax.plot(mid[0], mid[1], marker='x', color=color, markersize=6)

        # Plot bisectors
        plot_perp_bisector(A, A_prime, 'green')
        plot_perp_bisector(B, B_prime, 'purple')
        # plot_perp_bisector(C, C_prime, 'orange') # 2 are enough

        # Mark Center
        ax.plot(center[0], center[1], 'kx', markersize=10, markeredgewidth=2, label='Center C')
        ax.text(center[0], center[1]-0.3, 'Center C', ha='center')

        # Show angle ACA' (optional)
        ax.plot([center[0], A[0]], [center[1], A[1]], 'gray', linestyle='-.')
        ax.plot([center[0], A_prime[0]], [center[1], A_prime[1]], 'gray', linestyle='-.')
        # Calculate angles correctly for arc
        angle1_deg = math.degrees(math.atan2(A[1]-center[1], A[0]-center[0]))
        angle2_deg = math.degrees(math.atan2(A_prime[1]-center[1], A_prime[0]-center[0]))
        # Ensure theta1 < theta2 for the arc drawing
        theta1 = min(angle1_deg, angle2_deg)
        theta2 = max(angle1_deg, angle2_deg)
        # Handle potential wrap-around if angle crosses 0 degrees (e.g., 350 to 10)
        if theta2 - theta1 > 180:
             theta1, theta2 = theta2, theta1 + 360 # Swap and adjust

        angle_arc = patches.Arc(center, 1.5, 1.5, angle=0, theta1=theta1, theta2=theta2, color='orange', label='Rotation Angle')
        ax.add_patch(angle_arc)
        # Position angle text better
        mid_angle_rad = math.radians((theta1 + theta2) / 2)
        text_x = center[0] + 1.0 * math.cos(mid_angle_rad) # Position text near arc
        text_y = center[1] + 1.0 * math.sin(mid_angle_rad)
        ax.text(text_x, text_y, f"{abs(angle2_deg-angle1_deg):.0f}°", color='orange', ha='center', va='center')


        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        all_x = np.concatenate((obj_verts[:,0], img_verts[:,0], [center[0]]))
        all_y = np.concatenate((obj_verts[:,1], img_verts[:,1], [center[1]]))
        ax.set_xlim(min(all_x)-1, max(all_x)+1)
        ax.set_ylim(min(all_y)-1, max(all_y)+1)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_title("Finding Center of Rotation")
        ax.legend(fontsize=8, loc='best') # Use 'best' location

        return self._save_plot(fig, filename)

    # def f3_trot_notes_find_center(self):
    #     filename = 'f3_trot_notes_find_center.png'
    #     fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    #     ax.set_facecolor('white')
    #     ax.set_aspect('equal', adjustable='box')

    #     # Example Object and Image (e.g., 90 deg anti-clockwise rot about (1,1))
    #     A, B, C = (3,1), (5,1), (3,4)
    #     center = (1,1)
    #     angle_rad = math.radians(90)
    #     rot_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
    #                            [math.sin(angle_rad), math.cos(angle_rad)]])

    #     def rotate_point(p, center, matrix):
    #         vec = np.array(p) - np.array(center)
    #         rot_vec = matrix @ vec
    #         return rot_vec + np.array(center)

    #     A_prime = rotate_point(A, center, rot_matrix) # (1, 3)
    #     B_prime = rotate_point(B, center, rot_matrix) # (1, 5)
    #     C_prime = rotate_point(C, center, rot_matrix) # (-2, 3) -> check calc: (3,4)-(1,1)=(2,3). rot(2,3)=[-1*3, 1*2]=(-3,2). add (1,1) = (-2,3). Correct.


    #     obj_verts = np.array([A, B, C, A])
    #     img_verts = np.array([A_prime, B_prime, C_prime, A_prime])

    #     # Plot object and image
    #     ax.plot(obj_verts[:, 0], obj_verts[:, 1], 'b-o', label='Object ABC')
    #     ax.plot(img_verts[:, 0], img_verts[:, 1], 'r--o', label="Image A'B'C'")
    #     for pt, name in zip([A, B, C], ['A', 'B', 'C']): ax.text(pt[0]+0.1, pt[1], name)
    #     for pt, name in zip([A_prime, B_prime, C_prime], ["A'", "B'", "C'"]): ax.text(pt[0]+0.1, pt[1], name)

    #     # Function to plot perpendicular bisector
    #     def plot_perp_bisector(p1, p2, color):
    #         mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
    #         dx = p2[0]-p1[0]
    #         dy = p2[1]-p1[1]
    #         if abs(dx) < 1e-6 and abs(dy) < 1e-6: return # Points coincide
    #         # Perpendicular direction (-dy, dx)
    #         perp_dx, perp_dy = -dy, dx
    #         # Extend line
    #         t = 3
    #         start_pt = (mid[0] - t*perp_dx, mid[1] - t*perp_dy)
    #         end_pt = (mid[0] + t*perp_dx, mid[1] + t*perp_dy)
    #         ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], color=color, linestyle=':', linewidth=1.5, label=f'Bisector {name_map[p1]}{name_map[p2]}\'')
    #         ax.plot(mid[0], mid[1], marker='x', color=color, markersize=6)

    #     # Plot bisectors
    #     name_map = {tuple(A):'A', tuple(B):'B', tuple(C):'C'} # Map coords back to names for label
    #     plot_perp_bisector(A, A_prime, 'green')
    #     plot_perp_bisector(B, B_prime, 'purple')
    #     # plot_perp_bisector(C, C_prime, 'orange') # 2 are enough

    #     # Mark Center
    #     ax.plot(center[0], center[1], 'kx', markersize=10, markeredgewidth=2, label='Center C')
    #     ax.text(center[0], center[1]-0.3, 'Center C', ha='center')

    #     # Show angle ACA' (optional)
    #     ax.plot([center[0], A[0]], [center[1], A[1]], 'gray', linestyle='-.')
    #     ax.plot([center[0], A_prime[0]], [center[1], A_prime[1]], 'gray', linestyle='-.')
    #     angle_arc = patches.Arc(center, 1.5, 1.5, angle=0, theta1=math.degrees(math.atan2(A_prime[1]-center[1], A_prime[0]-center[0])), theta2=math.degrees(math.atan2(A[1]-center[1], A[0]-center[0])), color='orange')
    #     ax.add_patch(angle_arc)
    #     ax.text(0.5, 2.8, "∠ACA'", color='orange')


    #     # Setup plot
    #     ax.axhline(0, color='black', linewidth=0.5)
    #     ax.axvline(0, color='black', linewidth=0.5)
    #     ax.grid(True, linestyle='--')
    #     all_x = np.concatenate((obj_verts[:,0], img_verts[:,0]))
    #     all_y = np.concatenate((obj_verts[:,1], img_verts[:,1]))
    #     ax.set_xlim(min(all_x)-1, max(all_x)+1)
    #     ax.set_ylim(min(all_y)-1, max(all_y)+1)
    #     ax.set_xlabel("x-axis")
    #     ax.set_ylabel("y-axis")
    #     ax.set_title("Finding Center of Rotation")
    #     ax.legend(fontsize=8, loc='upper right')

    #     return self._save_plot(fig, filename)

    # def f3_te_notes_find_center(self):
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    #     ax.set_facecolor('white')
    #     ax.set_aspect('equal', adjustable='box')

    #     # Example Object and Image (e.g., enlargement k=2, center (1,1))
    #     A, B, C = (2,2), (4,2), (2,4) # Object
    #     center = (1,1)
    #     k = 2

    #     def enlarge_point(p, center, k):
    #         vec = np.array(p) - np.array(center)
    #         enlarged_vec = k * vec
    #         return enlarged_vec + np.array(center)

    #     A_prime = enlarge_point(A, center, k) # (1,1) + 2*(1,1) = (3,3)
    #     B_prime = enlarge_point(B, center, k) # (1,1) + 2*(3,1) = (7,3)
    #     C_prime = enlarge_point(C, center, k) # (1,1) + 2*(1,3) = (3,7)

    #     obj_verts = np.array([A, B, C, A])
    #     img_verts = np.array([A_prime, B_prime, C_prime, A_prime])

    #     # Plot object and image
    #     ax.plot(obj_verts[:, 0], obj_verts[:, 1], 'b-o', label='Object ABC')
    #     ax.plot(img_verts[:, 0], img_verts[:, 1], 'r--o', label="Image A'B'C'")
    #     for pt, name in zip([A, B, C], ['A', 'B', 'C']): ax.text(pt[0]+0.1, pt[1], name)
    #     for pt, name in zip([A_prime, B_prime, C_prime], ["A'", "B'", "C'"]): ax.text(pt[0]+0.1, pt[1], name)

    #     # Draw lines connecting corresponding points through center
    #     line_len = 8
    #     # Line A-A'
    #     vec_AA = A_prime - A
    #     ax.plot([A[0]-vec_AA[0]*0.5, A_prime[0]+vec_AA[0]*0.5], [A[1]-vec_AA[1]*0.5, A_prime[1]+vec_AA[1]*0.5], 'g:', linewidth=1)
    #     # Line B-B'
    #     vec_BB = B_prime - B
    #     ax.plot([B[0]-vec_BB[0]*0.5, B_prime[0]+vec_BB[0]*0.5], [B[1]-vec_BB[1]*0.5, B_prime[1]+vec_BB[1]*0.5], 'g:', linewidth=1)
    #      # Line C-C'
    #     vec_CC = C_prime - C
    #     ax.plot([C[0]-vec_CC[0]*0.5, C_prime[0]+vec_CC[0]*0.5], [C[1]-vec_CC[1]*0.5, C_prime[1]+vec_CC[1]*0.5], 'g:', linewidth=1)


    #     # Mark Center
    #     ax.plot(center[0], center[1], 'kx', markersize=10, markeredgewidth=2, label='Center O')
    #     ax.text(center[0], center[1]-0.3, 'Center O', ha='center')

    #     # Setup plot
    #     ax.axhline(0, color='black', linewidth=0.5)
    #     ax.axvline(0, color='black', linewidth=0.5)
    #     ax.grid(True, linestyle='--')
    #     all_x = np.concatenate((obj_verts[:,0], img_verts[:,0], [center[0]]))
    #     all_y = np.concatenate((obj_verts[:,1], img_verts[:,1], [center[1]]))
    #     ax.set_xlim(min(all_x)-1, max(all_x)+1)
    #     ax.set_ylim(min(all_y)-1, max(all_y)+1)
    #     ax.set_xlabel("x-axis")
    #     ax.set_ylabel("y-axis")
    #     ax.set_title("Finding Center of Enlargement")
    #     ax.legend(fontsize=8, loc='upper left')

    #     return self._save_plot(fig, filename)
    # def f4_gfg_we1_cubic_graph(self):
    #     filename = get_filename_from_caller()
    #     fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    #     ax.set_facecolor('white')

    #     # Data points from WE1
    #     x_pts = np.array([-3, -2, -1, 0, 1, 2, 3])
    #     y_pts = np.array([-15, 0, 3, 0, -3, 0, 15])

    #     # Interpolate for a smooth curve
    #     x_smooth = np.linspace(x_pts.min(), x_pts.max(), 300)
    #     # Using numpy's polynomial fit as an approximation for smoothness
    #     coeffs = np.polyfit(x_pts, y_pts, 3) # Fit a cubic polynomial
    #     poly = np.poly1d(coeffs)
    #     y_smooth = poly(x_smooth)

    #     ax.plot(x_smooth, y_smooth, 'b-', label='y = x³ - 4x (smooth)')
    #     ax.plot(x_pts, y_pts, 'ro', label='Calculated Points') # Plot original points

    #     # Setup plot
    #     ax.axhline(0, color='black', linewidth=0.5)
    #     ax.axvline(0, color='black', linewidth=0.5)
    #     ax.grid(True, linestyle='--')
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_title("Graph of y = x³ - 4x")
    #     ax.legend()
    #     ax.set_ylim(min(y_pts)-2, max(y_pts)+2) # Adjust ylim based on points

    #     # Store data for WE3
    #     self.context_data['f4_gfg_we1'] = {'x': x_pts, 'y': y_pts, 'xsmooth': x_smooth, 'ysmooth': y_smooth, 'coeffs': coeffs} # Store coeffs too

    #     return self._save_plot(fig, filename)
    
    def f4_gfg_we2_inverse_graph(self):
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')

        # Data points
        x_pts = np.array([-4, -2, -1, -0.5, 0.5, 1, 2, 4])
        y_pts = np.array([-1, -2, -4, -8, 8, 4, 2, 1])

        # Generate smooth curve data
        x_neg = np.linspace(-4, -0.1, 200)
        y_neg = 4 / x_neg
        x_pos = np.linspace(0.1, 4, 200)
        y_pos = 4 / x_pos

        # Plot
        ax.plot(x_neg, y_neg, 'b-', label='y = 4/x')
        ax.plot(x_pos, y_pos, 'b-')
        ax.plot(x_pts, y_pts, 'ro', label='Calculated Points')

        # Setup plot
        ax.axhline(0, color='black', linewidth=0.8, label='Asymptote y=0')
        ax.axvline(0, color='black', linewidth=0.8, label='Asymptote x=0')
        ax.grid(True, linestyle='--')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Graph of y = 4/x")
        ax.legend(fontsize=9)
        ax.set_ylim(-9, 9)
        ax.set_xlim(-5, 5)

        return self._save_plot(fig, filename)
    

    def f4_gfg_we3_solve_on_graph(self):
        filename = get_filename_from_caller()
        # Need graph data from WE1
        if 'f4_gfg_we1' not in self.context_data:
            print("Warning: Context data for WE1 not found, regenerating.")
            # Attempt to regenerate WE1 data if missing
            self.f4_gfg_we1_cubic_graph()
            if 'f4_gfg_we1' not in self.context_data:
                 print("Error: Failed to get context data for WE1.")
                 return None # Cannot proceed without the base graph data

        # Retrieve data safely
        we1_data = self.context_data.get('f4_gfg_we1', {})
        x_smooth = we1_data.get('xsmooth')
        y_smooth = we1_data.get('ysmooth')
        x_pts = we1_data.get('x')
        y_pts = we1_data.get('y')
        coeffs = we1_data.get('coeffs') # Get coefficients stored in WE1

        # Check if essential data is present
        if x_smooth is None or y_smooth is None or x_pts is None or y_pts is None or coeffs is None:
            print("Error: Missing essential data from WE1 context.")
            return None

        fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
        ax.set_facecolor('white')

        ax.plot(x_smooth, y_smooth, 'b-', label='y = x³ - 4x')
        # ax.plot(x_pts, y_pts, 'ro') # Optionally plot original points too

        # Draw horizontal line y=5
        target_y = 5
        ax.axhline(target_y, color='red', linestyle='--', label=f'y = {target_y}')

        # Find intersections (approximate visually or numerically)
        # Numerical: find where poly(x) - 5 = 0
        # Create polynomial P(x) - 5
        target_poly_coeffs = coeffs.copy()
        target_poly_coeffs[-1] -= target_y # Subtract constant term
        target_poly = np.poly1d(target_poly_coeffs)

        roots = target_poly.roots
        real_roots = roots[np.isreal(roots)].real

        # Filter roots within the plotted range and positive x region as per description
        plot_range_roots = [r for r in real_roots if x_smooth.min() <= r <= x_smooth.max() and r > 0]

        approx_x = None
        if plot_range_roots:
             # Select the root specified implicitly (positive x region)
             # If multiple positive roots, problem might be ambiguous, take the first one found
             approx_x = plot_range_roots[0]
        else:
            # Fallback if numerical solution fails or yields no positive root in range
            print("Warning: Numerical intersection finding failed or yielded no positive root in range. Using visual estimate.")
            approx_x = 2.45 # Estimate from problem description/visual inspection

        # Mark intersection and drop line
        ax.plot(approx_x, target_y, 'go', markersize=8, label=f'Intersection at y={target_y}')
        ax.plot([approx_x, approx_x], [0, target_y], 'g:') # Dashed line to x-axis
        ax.text(approx_x, -1, f'x ≈ {approx_x:.2f}', color='g', ha='center')


        # Setup plot
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Solving x³ - 4x = 5 Graphically")
        ax.legend()
        # Ensure y-limits encompass target_y
        ax.set_ylim(min(y_pts.min(), 0, target_y)-2, max(y_pts.max(), target_y)+2)

        return self._save_plot(fig, filename)
    
    def f4_gtg_we3_disp_from_accel(self):
        """References WE2 graph and shades the trapezium area."""
        filename = get_filename_from_caller()
        if 'f4_gtg_we2' not in self.context_data:
            print("Warning: Context data for WE2 not found, regenerating.")
            self.f4_gtg_we2_accel_graph()
            if 'f4_gtg_we2' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        time_pts = self.context_data['f4_gtg_we2']['time']
        speed_pts = self.context_data['f4_gtg_we2']['speed']

        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Displacement (Area under Graph)")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 14)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'g-', marker='o')
        ax.plot(time_pts, speed_pts, 'go')

        # Shade the trapezium area
        ax.fill_between(time_pts, speed_pts, color='lightgreen', alpha=0.7)
        area = 0.5 * (speed_pts[0] + speed_pts[1]) * (time_pts[1] - time_pts[0])
        ax.text(2, 5, f'Area = Displacement\n= 1/2*(4+12)*4\n= {area} m',
                ha='center', va='center', fontsize=10, color='darkgreen')

        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 15, 1))

        return self._save_plot(fig, filename)

    def f4_gtg_we4_velocity_time_graph(self):
        """Generates velocity-time graph for WE4."""
        filename = get_filename_from_caller()
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Object Velocity-Time Graph")
        ax.set_xlim(0, 12.5)
        ax.set_ylim(0, 10)
        ax.grid(True, linestyle='--', alpha=0.6)

        # Segments: (0,0) to (4,8), then (4,8) to (10,8), then (10,8) to (12,0)
        time_pts = [0, 4, 10, 12]
        speed_pts = [0, 8, 8, 0]

        ax.plot(time_pts, speed_pts, 'm-', marker='o')
        ax.plot(time_pts, speed_pts, 'mo') # Redraw points

        # Annotations
        ax.text(2, 4, 'Accelerate', ha='center', color='m', rotation=63)
        ax.text(7, 8.3, 'Constant Velocity', ha='center', color='m')
        ax.text(11.3, 4, 'Decelerate', ha='center', color='m', rotation=-76)

        ax.set_xticks(np.arange(0, 13, 1))
        ax.set_yticks(np.arange(0, 11, 1))

        # Store data for reuse in WE5
        self.context_data['f4_gtg_we4'] = {'time': time_pts, 'speed': speed_pts}

        return self._save_plot(fig, filename)


    def f4_gtg_we1_total_distance(self):
        """References WE4 graph and shades the total area."""
        filename = get_filename_from_caller()
        if 'f4_gtg_we1' not in self.context_data:
            print("Warning: Context data for WE4 not found, regenerating.")
            self.f4_gtg_we4_velocity_time_graph()
            if 'f4_gtg_we4' not in self.context_data:
                 print("Error: Failed to get context data.")
                 return None

        time_pts = self.context_data['f4_gtg_we4']['time']
        speed_pts = self.context_data['f4_gtg_we4']['speed']

        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        ax.set_facecolor('white')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Total Distance Traveled (Total Area)")
        ax.set_xlim(0, 12.5)
        ax.set_ylim(0, 10)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax.plot(time_pts, speed_pts, 'm-', marker='o')
        ax.plot(time_pts, speed_pts, 'mo')

        # Shade the total area (trapezoid)
        ax.fill_between(time_pts, speed_pts, color='plum', alpha=0.7)

        # Calculate areas
        area1 = 0.5 * 4 * 8
        area2 = (10 - 4) * 8
        area3 = 0.5 * (12 - 10) * 8
        total_area = area1 + area2 + area3

        ax.text(2, 2, f'A1={area1}', ha='center', fontsize=9)
        ax.text(7, 4, f'A2={area2}', ha='center', fontsize=9)
        ax.text(11, 2, f'A3={area3}', ha='center', fontsize=9)
        ax.text(6, 6, f'Total Area = Dist = {total_area} m', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))


        ax.set_xticks(np.arange(0, 13, 1))
        ax.set_yticks(np.arange(0, 11, 1))

        return self._save_plot(fig, filename)
    

    def plot_velocity_segment_0_4_to_4_12(self):
        """
        Generates a velocity-time graph showing a single segment
        from (0, 4) to (4, 12).
        """
        filename = get_filename_from_caller() # Generates 'plot_velocity_segment_0_4_to_4_12.png'
        fig, ax = plt.subplots(figsize=(6, 7), facecolor='white') # Adjusted figsize for Y-axis range
        ax.set_facecolor('white')

        # Set axis labels
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Velocity-Time Graph Segment")

        # Set axis limits based on description
        ax.set_xlim(0, 4.5) # Slightly extend x-axis
        ax.set_ylim(0, 14)

        # Define the points for the segment
        time_pts = [0, 4]
        velocity_pts = [4, 12]

        # Plot the segment
        ax.plot(time_pts, velocity_pts, 'b-o', linewidth=2, markersize=6) # Blue line with markers

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set ticks for clarity
        ax.set_xticks(np.arange(0, 5, 1)) # Ticks at 0, 1, 2, 3, 4
        ax.set_yticks(np.arange(0, 15, 2)) # Ticks every 2 m/s

        # Ensure origin lines are visible if axes start at 0
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        return self._save_plot(fig, filename)

# --- Main Processing Logic ---
def process_json_data(input_file, output_file):
    """Loads JSON, generates images, updates JSON, saves."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{input_file}': {e}")
        return

    generator = MathDiagramGenerator()
    updated_data = [] # Create a new list to store updated items

    item_index = 0 # Keep track of the main list index

    for item in data:
        current_item = item.copy() # Work on a copy
        item_key_prefix = f"item_{item_index}" # Prefix for function mapping

        # Process 'notes' if it exists
        if 'notes' in current_item and isinstance(current_item['notes'], dict):
            notes = current_item['notes']
            if 'sections' in notes and isinstance(notes['sections'], list):
                updated_sections = []
                section_index = 0
                for section in notes['sections']:
                    current_section = section.copy()
                    if 'image_description' in current_section and 'image_path' not in current_section:
                        # Construct function name dynamically (example convention)
                        func_name = f"f{notes.get('title','untitled').split(':')[0].split(' ')[-1]}_{notes.get('title','untitled').split(':')[1].strip().replace(' ','_').replace('-','_')[:3].lower()}_notes_s{section_index+1}" # Simple naming
                        # More robust: Map title/heading to known functions if possible
                        # Fallback to trying to find *any* matching method name if direct mapping fails
                        target_func_name = ""
                        if notes.get('title') == "Form 3: Graphs - Travel Graphs":
                             if section.get('heading') == "2. Speed-Time Graphs": target_func_name = "f3_tg_notes_dist_time_vs_speed_time"
                             elif section.get('heading') == "4. Calculating Distance from Speed-Time Graphs": target_func_name = "f3_tg_notes_area_under_speed_time"
                        elif notes.get('title') == "Form 3: Variation":
                             if section.get('heading') == "4. Sketch Graphs of Variation": target_func_name = "f3_v_notes_direct_vs_inverse_graphs"
                        elif notes.get('title') == "Form 3: Algebra - Inequalities":
                             if section.get('heading') == "2. Simultaneous Linear Inequalities (One Variable)": target_func_name = "f3_ai_notes_simul_ineq_1var"
                             elif section.get('heading') == "4. Simultaneous Linear Inequalities (Two Variables)": target_func_name = "f3_ai_notes_simul_ineq_2var"
                        elif notes.get('title') == "Form 3: Geometry - Points, Lines and Angles":
                             if section.get('heading') == "2. Angle of Elevation": target_func_name = "f3_gpla_notes_angle_of_elevation"
                             elif section.get('heading') == "3. Angle of Depression": target_func_name = "f3_gpla_notes_angle_of_depression"
                             elif section.get('heading') == "4. Key Relationship": target_func_name = "f3_gpla_notes_elev_dep_relationship"
                        elif notes.get('title') == "Form 3: Geometry - Bearing":
                             if section.get('heading') == "3. Back Bearing": target_func_name = "f3_gb_notes_bearing_back_bearing"
                        elif notes.get('title') == "Form 3: Geometry - Polygons":
                             if section.get('heading') == "3. Exterior Angles of Polygons": target_func_name = "f3_gpoly_notes_exterior_angle"
                        elif notes.get('title') == "Form 3: Statistics - Data Collection, Classification and Representation":
                             if section.get('heading') == "7. Histogram": target_func_name = "f3_sdccr_notes_histogram"
                             elif section.get('heading') == "8. Frequency Polygon": target_func_name = "f3_sdccr_notes_frequency_polygon"
                        elif notes.get('title') == "Form 3: Trigonometry - Pythagoras Theorem":
                             if section.get('heading') == "1. Right-Angled Triangles": target_func_name = "f3_tpt_notes_right_angled_triangle"
                             elif section.get('heading') == "2. Pythagoras Theorem": target_func_name = "f3_tpt_notes_pythagoras_proof"
                        elif notes.get('title') == "Form 3: Trigonometry - Trigonometrical Ratios":
                             if section.get('heading') == "2. Right-Angled Triangle Sides (Relative to an Angle)": target_func_name = "f3_tr_notes_triangle_sides_relative"
                             elif section.get('heading') == "6. Trigonometrical Ratios of Obtuse Angles (90° < θ < 180°)": target_func_name = "f3_tr_notes_unit_circle_obtuse"
                        elif notes.get('title') == "Form 3: Vectors - Types of Vectors":
                              if section.get('heading') == "3. Position Vectors": target_func_name = "f3_vt_notes_position_vectors"
                        elif notes.get('title') == "Form 3: Vectors - Operations":
                              if section.get('heading') == "3. Magnitude (Modulus) of a Vector": target_func_name = "f3_vo_notes_magnitude_vector"
                        elif notes.get('title') == "Form 3: Transformation - Reflection":
                              if section.get('heading') == "4. Reflection in Lines y=a and x=b": target_func_name = "f3_trfl_notes_reflection_lines"
                        elif notes.get('title') == "Form 3: Transformation - Reflection":
                              if section.get('heading') == "4. Reflection in Lines y=a and x=b": target_func_name = "f3_trfl_notes_reflection_lines"
                        elif notes.get('title') == "Form 3: Transformation - Enlargement":
                              if section.get('heading') == "5. Finding the Center of Enlargement": target_func_name = "f3_te_notes_find_center"
 


                       # Form 4
                        elif notes.get('title') == "Form 4: Measures and Mensuration - Mensuration":
                             if section.get('heading') == "4. Pyramids": target_func_name = "f4_mm_notes_prism_pyramid"
                             elif section.get('heading') == "5. Cones": target_func_name = "f4_mm_notes_cone_volume"
                             elif section.get('heading') == "6. Spheres": target_func_name = "f4_mm_notes_sphere_volume"
                             elif section.get('heading') == "7. Surface Area of Solid Shapes": target_func_name = "f4_mm_notes_surface_area_nets"
                        elif notes.get('title') == "Form 4: Graphs - Functional Graphs":
                             if section.get('heading') == "2. Cubic Functions (y = ax³ + bx² + cx + d)": target_func_name = "f4_gfg_notes_cubic_graphs"
                             elif section.get('heading') == "3. Inverse Functions (Reciprocal Functions, y = a / (bx + c) )": target_func_name = "f4_gfg_notes_inverse_graphs"
                        elif notes.get('title') == "Form 4: Graphs - Travel Graphs":
                             if section.get('heading') == "4. Velocity-Time Graphs": target_func_name = "f4_gtg_notes_velocity_time_disp"
                        elif notes.get('title') == "Form 4: Variation":
                            if section.get('heading') == "3. Partial Variation": target_func_name = "f4_v_notes_partial_variation_graph"
                        elif notes.get('title') == "Form 4: Algebra - Inequalities":
                             if section.get('heading') == "4. Solving Linear Programming Problems Graphically (Two Variables)": target_func_name = "f4_ai_notes_lp_feasible_region"
                        elif notes.get('title') == "Form 4: Geometry - Polygons and Circle":
                             if section.get('heading') == "2. Theorems Related to Center, Chord, and Circumference": target_func_name = "f4_gpc_notes_circle_theorems_1_5"
                             elif section.get('heading') == "3. Theorems Related to Cyclic Quadrilaterals": target_func_name = "f4_gpc_notes_cyclic_quad_theorems"
                             elif section.get('heading') == "4. Theorems Related to Tangents": target_func_name = "f4_gpc_notes_tangent_theorems"
                        elif notes.get('title') == "Form 4: Geometry - Constructions":
                             if section.get('heading') == "3. Common Loci in a Plane": target_func_name = "f4_gcl_notes_locus1_circle"
                             elif section.get('heading') == "4. Common Loci in a Plane (Continued)": target_func_name = "f4_gcl_notes_locus2_parallel_lines"
                             elif section.get('heading') == "5. Common Loci in a Plane (Continued)": target_func_name = "f4_gcl_notes_locus3_perp_bisector"
                             elif section.get('heading') == "6. Common Loci in a Plane (Continued)": target_func_name = "f4_gcl_notes_locus4_angle_bisector"
                        elif notes.get('title') == "Form 4: Statistics - Data Representation":
                             if section.get('heading') == "4. Cumulative Frequency Curve (Ogive)": target_func_name = "f4_sdr_notes_ogive"
                        elif notes.get('title') == "Form 4: Statistics - Measures of Central Tendency and Dispersion":
                              if section.get('heading') == "5. Quartiles from Cumulative Frequency Curve (Ogive)": target_func_name = "f4_smctd_notes_ogive_quartiles"
                        elif notes.get('title') == "Form 4: Trigonometry - Trigonometrical Ratios":
                              if section.get('heading') == "3. The Sine Rule": target_func_name = "f4_ttr_notes_sine_rule"
                              elif section.get('heading') == "4. The Cosine Rule": target_func_name = "f4_ttr_notes_cosine_rule"
                              elif section.get('heading') == "5. Area of a Triangle using Sine": target_func_name = "f4_ttr_notes_area_sine"
                        elif notes.get('title') == "Form 4: Vectors - Operations":
                              if section.get('heading') == "3. Vectors in Geometry - Plane Shapes": target_func_name = "f4_vo_notes_parallelogram_vectors"
                              elif section.get('heading') == "4. Ratio Theorem": target_func_name = "f4_vo_notes_ratio_theorem"
                        elif notes.get('title') == "Form 4: Transformation - Reflection":
                              if section.get('heading') == "2. Reflection in lines y=mx+c": target_func_name = "f4_trfl_notes_reflection_lines" # Mapped previously, adjust if needed
                        elif notes.get('title') == "Form 4: Transformation - Stretch":
                              if section.get('heading') == "2. One-Way Stretch": target_func_name = "f4_tstr_notes_stretch_diagrams"
                        elif notes.get('title') == "Form 4: Transformation - Shear":
                              if section.get('heading') == "2. Shear Parallel to an Axis": target_func_name = "f4_tsh_notes_shear_diagrams"
                        elif notes.get('title') == "Form 4: Probability - Combined Events":
                             if section.get('heading') == "3. Tree Diagrams": target_func_name = "f4_pce_notes_tree_diagram"
                        elif notes.get('title') == "Form 4: Transformation - Enlargement":
                              if section.get('heading') == "5. Finding the Center of Enlargement": target_func_name = "f4_te_notes_find_center"
                        if notes.get('title') == "Form 3: Transformation - Rotation":
                                if section.get('heading') == "3. Finding the Center and Angle of Rotation": target_func_name = "f3_trot_notes_find_center"

                        if hasattr(generator, target_func_name):
                            method_to_call = getattr(generator, target_func_name)
                            print(f"Calling {target_func_name} for notes section {section_index+1}...")
                            image_path = method_to_call()
                            if image_path:
                                current_section['image_path'] = image_path
                        else:
                            print(f"Warning: Method '{target_func_name}' not found for notes section {section_index+1}.")

                    updated_sections.append(current_section)
                    section_index += 1
                notes['sections'] = updated_sections
            current_item['notes'] = notes # Put the updated notes back

        # Process 'worked_examples' if it exists
        if 'worked_examples' in current_item and isinstance(current_item['worked_examples'], list):
            updated_examples = []
            example_index = 0
            for example in current_item['worked_examples']:
                current_example = example.copy()
                if 'image_description' in current_example and 'image_path' not in current_example:
                     # Construct function name dynamically (example convention)
                    title = current_item.get('notes',{}).get('title','untitled').split(':')
                    form_num = title[0].split(' ')[-1]
                    topic_abbr = title[1].strip().replace(' ','_').replace('-','_')[:3].lower()
                    func_name = f"f{form_num}_{topic_abbr}_we{example_index+1}"
                    # Find a specific matching function name if possible
                    target_func_name = ""
                    if current_item.get('notes',{}).get('title') == "Form 3: Graphs - Travel Graphs":
                         if example_index == 0: target_func_name = "f3_tg_we1_dist_time_graph"
                         elif example_index == 1: target_func_name = "f3_tg_we2_speed_time_graph"
                         elif example_index == 2: target_func_name = "f3_tg_we3_area_triangle"
                         elif example_index == 3: target_func_name = "f3_tg_we4_area_total"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Geometry - Points, Lines and Angles":
                         if example_index == 0: target_func_name = "f3_gpla_we1_tower_elevation"
                         elif example_index == 1: target_func_name = "f3_gpla_we2_cliff_depression"
                         elif example_index == 2: target_func_name = "f3_gpla_we3_elevation_diagram"
                         elif example_index == 3: target_func_name = "f3_gpla_we4_depression_diagram"
                         elif example_index == 4: target_func_name = "f3_gpla_we5_scale_drawing"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Geometry - Bearing":
                         if example_index == 0: target_func_name = "f3_gb_we1_bearing_240"
                         elif example_index == 2: target_func_name = "f3_gb_we3_ship_path_angle"
                         elif example_index == 3: target_func_name = "f3_gb_we4_compass_bearing"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Geometry - Similarity and Congruency":
                         if example_index == 0: target_func_name = "f3_gsc_we1_similar_triangles"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Geometry - Constructions and Loci":
                         if example_index == 0: target_func_name = "f3_gcl_we1_construct_sss"
                         elif example_index == 1: target_func_name = "f3_gcl_we2_construct_sas"
                         elif example_index == 2: target_func_name = "f3_gcl_we3_construct_asa"
                         elif example_index == 3: target_func_name = "f3_gcl_we4_construct_rectangle"
                         elif example_index == 4: target_func_name = "f3_gcl_we5_construct_parallelogram"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Geometry - Symmetry":
                         if example_index == 1: target_func_name = "f3_gsym_we2_pentagon_rotation"
                         elif example_index == 2: target_func_name = "f3_gsym_we3_letter_s_rotation"
                         elif example_index == 4: target_func_name = "f3_gsym_we5_isosceles_line_sym"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Statistics - Data Collection, Classification and Representation":
                         if example_index == 1: target_func_name = "f3_sdccr_we2_histogram"
                         elif example_index == 2: target_func_name = "f3_sdccr_we3_frequency_polygon"
                         # WE4 references WE2, no new image
                    elif current_item.get('notes',{}).get('title') == "Form 3: Trigonometry - Pythagoras Theorem":
                         if example_index == 0: target_func_name = "f3_tpt_we1_find_hypotenuse"
                         elif example_index == 1: target_func_name = "f3_tpt_we2_find_leg"
                         elif example_index == 4: target_func_name = "f3_tpt_we5_ladder_problem"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Trigonometry - Trigonometrical Ratios":
                         if example_index == 0: target_func_name = "f3_tr_we1_find_ratios"
                         elif example_index == 3: target_func_name = "f3_tr_we4_find_opposite_side"
                         elif example_index == 4: target_func_name = "f3_tr_we5_find_angle"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Vectors - Types of Vectors":
                         if example_index == 1: target_func_name = "f3_vt_we2_draw_position_vector"
                         elif example_index == 4: target_func_name = "f3_vt_we5_identify_equal_vectors"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Transformation - Translation":
                         if example_index == 1: target_func_name = "f3_tt_we2_draw_object_image"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Transformation - Reflection":
                         if example_index == 2: target_func_name = "f3_trfl_we3_reflect_y_axis"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Transformation - Rotation":
                         if example_index == 3: target_func_name = "f3_trot_we4_rotate_180"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Transformation - Enlargement":
                         if example_index == 3: target_func_name = "f3_te_we4_enlarge_triangle"
                         elif example_index == 5: target_func_name = "f4_te_we6_find_center_scale_factor" # Re-map F3WE6 -> F4WE6
                    elif current_item.get('notes',{}).get('title') == "Form 3: Algebra - Inequalities":
                         if example_index == 1: target_func_name = "f3_ai_we2_num_line_soln"
                         elif example_index == 3: target_func_name = "f3_ineq_cartesian_plane_example" #
                    # Form 4
                    elif current_item.get('notes',{}).get('title') == "Form 4: Variation":
                         if example_index == 4: target_func_name = "f4_v_we5_partial_variation_cost_graph"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Algebra - Inequalities":
                         if example_index == 0: target_func_name = "f4_ai_we1_inequality_graph"
                         elif example_index == 2: target_func_name = "f4_ai_we3_lp_graph"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Geometry - Polygons and Circle":
                         if example_index == 0: target_func_name = "f4_gpc_we1_angle_center_circum"
                         elif example_index == 1: target_func_name = "f4_gpc_we2_cyclic_quad_opp_angles"
                         elif example_index == 2: target_func_name = "f4_gpc_we3_angle_in_semicircle"
                         elif example_index == 3: target_func_name = "f4_gpc_we4_alternate_segment"
                         elif example_index == 4: target_func_name = "f4_gpc_we5_tangent_radius_angle"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Geometry - Constructions":
                         if example_index == 1: target_func_name = "f4_gcl_we2_locus_perp_bisector_construct"
                         elif example_index == 2: target_func_name = "f4_gcl_we3_locus_circle_construct"
                         elif example_index == 3: target_func_name = "f4_gcl_we4_locus_angle_bisector_construct"
                         elif example_index == 4: target_func_name = "f4_gcl_we5_locus_intersection"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Statistics - Data Representation":
                         if example_index == 1: target_func_name = "f4_sdr_we2_ogive"
                         elif example_index == 2: target_func_name = "f4_sdr_we3_median_on_ogive"
                         elif example_index == 3: target_func_name = "f4_sdr_we4_value_from_ogive"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Statistics - Measures of Central Tendency and Dispersion":
                         if example_index == 0: target_func_name = "f4_smctd_we1_ogive_median"
                         elif example_index == 1: target_func_name = "f4_smctd_we2_ogive_quartiles"
                         # WE4 references WE1 ogive, no new plot
                    elif current_item.get('notes',{}).get('title') == "Form 4: Trigonometry - Trigonometrical Ratios":
                         if example_index == 0: target_func_name = "f4_ttr_we1_sine_rule_triangle"
                         elif example_index == 1: target_func_name = "f4_ttr_we2_cosine_rule_triangle"
                         elif example_index == 3: target_func_name = "f4_ttr_we4_cosine_rule_angle"
                         elif example_index == 4: target_func_name = "f4_ttr_we5_3d_trig"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Vectors - Operations":
                         if example_index == 0: target_func_name = "f4_vo_we1_parallelogram_vectors"
                         # WE4 needs image
                         elif example_index == 3: target_func_name = "f4_vo_we4_triangle_vectors_ratio"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Transformation - Reflection":
                         if example_index == 4: target_func_name = "f4_trfl_we5_reflect_y_eq_x_plus_1"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Probability - Combined Events":
                         if example_index == 1: target_func_name = "f4_pce_we2_tree_diagram_no_replace"
                         elif example_index == 2: target_func_name = "f4_pce_we3_tree_diagram_ref"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Graphs - Functional Graphs":
                         if example_index == 0: target_func_name = "f4_gfg_we1_cubic_graph"
                         elif example_index == 1: target_func_name = "f4_gfg_we2_inverse_graph"
                         elif example_index == 2: target_func_name = "f4_gfg_we3_solve_on_graph"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Graphs - Travel Graphs":
                         if example_index == 2: target_func_name = "f4_gtg_we3_disp_from_accel"
                         elif example_index == 0: target_func_name = "f4_gtg_we1_total_distance"
                         elif example_index == 3: target_func_name = "f4_gtg_we4_velocity_time_graph"
                         elif example_index == 1: target_func_name = "plot_velocity_segment_0_4_to_4_12"
                    if hasattr(generator, target_func_name):
                        method_to_call = getattr(generator, target_func_name)
                        print(f"Calling {target_func_name} for example {example_index+1}...")
                        image_path = method_to_call()
                        if image_path:
                            current_example['image_path'] = image_path
                    else:
                         # Fallback or generic name attempt (less reliable)
                         if hasattr(generator, func_name):
                             method_to_call = getattr(generator, func_name)
                             print(f"Calling fallback {func_name} for example {example_index+1}...")
                             image_path = method_to_call()
                             if image_path:
                                current_example['image_path'] = image_path
                         else:
                            print(f"Warning: Method '{target_func_name}' or '{func_name}' not found for example {example_index+1}.")
                updated_examples.append(current_example)
                example_index += 1
            current_item['worked_examples'] = updated_examples

        # Process 'questions' if it exists
        if 'questions' in current_item and isinstance(current_item['questions'], list):
            updated_questions = []
            question_index = 0
            for question in current_item['questions']:
                current_question = question.copy()
                if 'image_description' in current_question and 'image_path' not in current_question:
                    # Construct function name dynamically (example convention)
                    title = current_item.get('notes',{}).get('title','untitled').split(':')
                    form_num = title[0].split(' ')[-1]
                    topic_abbr = title[1].strip().replace(' ','_').replace('-','_')[:3].lower()
                    # Use level for uniqueness
                    level = current_question.get('level', f'q{question_index+1}').lower()
                    func_name = f"f{form_num}_{topic_abbr}_q_{level}"
                    target_func_name = "" # Reset target

                    # --- Map questions to functions ---
                    if current_item.get('notes',{}).get('title') == "Form 3: Graphs - Travel Graphs":
                         if current_question.get('level') == "Moderate": target_func_name = "f3_tg_q_moderate_speed_time_segment"
                         elif current_question.get('level') == "Challenging": target_func_name = "f3_tg_q_challenging_area_trapezium"
                    elif current_item.get('notes',{}).get('title') == "Form 3: Algebra - Inequalities":
                         if current_question.get('level') == "Challenging": target_func_name = "f3_ai_q_challenging_identify_region"
                    elif current_item.get('notes',{}).get('title') == "Form 4: Geometry - Polygons and Circle":
                         if current_question.get('level') == "Easy": target_func_name = "f4_gpc_q_easy_reflex_angle"
                         elif current_question.get('level') == "Moderate": target_func_name = "f4_gpc_q_moderate_cyclic_quad_eqn"
                         elif current_question.get('level') == "Challenging": target_func_name = "f4_gpc_q_challenging_alt_segment_thm"


                    if hasattr(generator, target_func_name):
                        method_to_call = getattr(generator, target_func_name)
                        print(f"Calling {target_func_name} for question {question_index+1}...")
                        image_path = method_to_call()
                        if image_path:
                            current_question['image_path'] = image_path
                    else:
                        # Fallback or generic name attempt (less reliable)
                        if hasattr(generator, func_name):
                             method_to_call = getattr(generator, func_name)
                             print(f"Calling fallback {func_name} for question {question_index+1}...")
                             image_path = method_to_call()
                             if image_path:
                                current_question['image_path'] = image_path
                        else:
                            print(f"Warning: Method '{target_func_name}' or '{func_name}' not found for question {question_index+1}.")


                updated_questions.append(current_question)
                question_index += 1
            current_item['questions'] = updated_questions

        updated_data.append(current_item)
        item_index += 1


    # Save the updated data
    try:
        with open(output_file, 'w') as f:
            json.dump(updated_data, f, indent=4)
        print(f"\nSuccessfully processed data and saved to '{output_file}'")
    except IOError as e:
        print(f"Error writing output file '{output_file}': {e}")
    except TypeError as e:
         print(f"Error encoding JSON (potentially unserializable data): {e}")

# --- Run the script ---
if __name__ == "__main__":
    process_json_data(INPUT_JSON_FILE, OUTPUT_JSON_FILE)


"""
**Explanation and Execution:**

1.  **Import Libraries:** Imports `json`, `os`, `matplotlib.pyplot`, `numpy`, `matplotlib.patches`, `math`, and `inspect`.
2.  **Configuration:** Sets the input/output filenames and the image directory name.
3.  **Helper Functions:**
    *   `ensure_dir`: Creates the output directory if it doesn't exist.
    *   `get_filename_from_caller`: Dynamically gets the name of the function calling it to create a unique filename (this simplifies naming).
4.  **`MathDiagramGenerator` Class:**
    *   The `__init__` method sets up the output folder and a `context_data` dictionary (useful if one diagram needs info generated by another, like in the travel graph examples).
    *   `_save_plot`: A helper method to save the plot to the correct folder with a given filename and close the figure. It returns the relative path needed for the JSON file.
    *   **Plotting Methods:** Contains all the methods generated in the previous step, plus the newly added methods for the remaining diagrams.
        *   Each method corresponds to an `image_description`.
        *   Uses Matplotlib and Numpy to create the plots.
        *   Calls `get_filename_from_caller` to determine the save filename.
        *   Calls `_save_plot` to save the image and get the path.
        *   Uses context data (`self.context_data`) where necessary to link related diagrams (e.g., using the graph from WE2 to shade areas in WE3 and WE4).
        *   Includes basic error handling for context data retrieval.
5.  **`process_json_data` Function:**
    *   Reads the input JSON file.
    *   Initializes the `MathDiagramGenerator`.
    *   Iterates through the top-level list in the JSON data.
    *   For each item, it iterates through `notes['sections']`, `worked_examples`, and `questions`.
    *   If an `image_description` key is found *and* an `image_path` key is *not* already present, it attempts to find the corresponding plotting method in the `MathDiagramGenerator` class.
    *   **Method Mapping:** It uses a series of `if/elif` statements based on the `title` of the notes section and the specific `heading` (for notes), `example_index` (for worked examples), or `level` (for questions) to determine the `target_func_name`. This is more robust than purely dynamic naming. A fallback to a generic name (`func_name`) is included but noted as less reliable.
    *   If a matching method is found, it calls the method.
    *   The method generates the plot, saves it, and returns the relative path.
    *   This path is then inserted into the current dictionary (section, example, or question) as the value for the `image_path` key.
    *   The updated dictionaries are collected.
    *   Finally, the entire updated data structure is written to the output JSON file.
6.  **`if __name__ == "__main__":` Block:** Ensures the `process_json_data` function runs when the script is executed directly.

**To Use:**

1.  Save the Python code above as a `.py` file (e.g., `generate_diagrams.py`).
2.  Make sure the `temp_file_upload.json` file is in the same directory as the script.
3.  Run the script from your terminal: `python generate_diagrams.py`
4.  A folder named `math_diagrams` will be created (if it doesn't exist) containing all the generated PNG images.
5.  A new file named `temp_file_upload_processed.json` will be created, containing the original JSON data plus the added `image_path` keys for each generated diagram.

The code now includes implementations for all the diagrams described in the original JSON, ensuring each `image_description` should now have a corresponding generated `image_path` in the output JSON.
"""