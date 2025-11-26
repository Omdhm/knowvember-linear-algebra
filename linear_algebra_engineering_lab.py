# %% [markdown]
# # Linear Algebra: Engineering Workbench
# 
# **Interactive Simulations for Developers & Engineers**
# 
# This notebook provides an interactive environment to build intuition for the linear algebra concepts used in:
# *   **Geospatial Analysis** (Coordinate transformations)
# *   **Drilling & Navigation** (Vector path planning)
# *   **Structural Analysis** (Eigenvalues/Eigenvectors)
# *   **Data Science** (Dimensionality Reduction/PCA)
# 
# **Workflow:** Run the cells to launch the interactive simulation widgets.

# %%
# 🛠️ Engineering Setup
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, FloatSlider, VBox, HBox, Dropdown, Button, Output
import ipywidgets as widgets
import time

# Enable inline animations
%matplotlib widget

print("✅ Engineering Workbench Initialized.")

# %% [markdown]
# ---
# ## 1. Vector Navigation Systems
# 
# **Context:** In drilling and subsea navigation, we deal with displacement vectors (offset from origin).
# 
# **Simulation:** Adjust the `East` (x) and `North` (y) components to align the vessel/drillbit with the target coordinates.

# %%
# 📡 SIMULATION 1: Navigation Control

target_pos = np.array([3.5, 2.0])
tolerance = 0.2

plt.ioff()
fig1, ax1 = plt.subplots(figsize=(7, 7))
plt.ion()

def update_navigation(east, north):
    ax1.clear()
    ax1.set_xlim(-1, 6)
    ax1.set_ylim(-1, 6)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlabel('East Offset (m)')
    ax1.set_ylabel('North Offset (m)')
    
    # Draw Target
    target_circle = plt.Circle(target_pos, tolerance, color='green', alpha=0.2, label='Target Zone')
    ax1.add_patch(target_circle)
    ax1.plot(*target_pos, 'g+', markersize=15, markeredgewidth=2, label='Target Coordinate')
    
    # Draw Current Vector
    ax1.arrow(0, 0, east, north, head_width=0.2, head_length=0.2, fc='#007079', ec='#007079', linewidth=2, label='Current Heading')
    
    # Calculate Deviation
    current_pos = np.array([east, north])
    deviation = np.linalg.norm(current_pos - target_pos)
    
    if deviation < tolerance:
        status_color = 'green'
        status_text = "✅ ON TARGET"
        ax1.set_facecolor('#f0fff0')
    else:
        status_color = '#d10a10' # Equinor Red-ish
        status_text = f"⚠️ DEVIATION: {deviation:.2f} m"
    
    ax1.set_title(f"Navigation Status: {status_text}", color=status_color, fontweight='bold')
    ax1.legend(loc='upper left')
    fig1.canvas.draw_idle()

# Control Panel
s_east = FloatSlider(min=0, max=5, step=0.1, value=1.0, description='East (x):')
s_north = FloatSlider(min=0, max=5, step=0.1, value=1.0, description='North (y):')

interactive(update_navigation, east=s_east, north=s_north)
display(VBox([HBox([s_east, s_north]), fig1.canvas]))
update_navigation(1.0, 1.0)

# %% [markdown]
# ---
# ## 2. Linear Transformations & Stress Analysis
# 
# **Context:** Matrices represent linear transformations. In geophysics and mechanics, these can represent:
# *   **Stress/Strain:** Deformation of a material block.
# *   **Coordinate Mapping:** Transforming between local and global grid systems.
# 
# **Simulation:** Apply Shear and Rotation matrices to visualize grid deformation.

# %%
# 📡 SIMULATION 2: Grid Deformation Analysis

plt.ioff()
fig2, ax2 = plt.subplots(figsize=(7, 7))
plt.ion()

def visualize_deformation(scale_x, scale_y, shear_factor, rotation_rad):
    ax2.clear()
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.grid(False) # We draw our own grid
    
    # Construct Transformation Matrix M = R * S
    # 1. Shear & Scale
    S = np.array([[scale_x, shear_factor], 
                  [0,       scale_y]])
    # 2. Rotation
    c, s = np.cos(rotation_rad), np.sin(rotation_rad)
    R = np.array([[c, -s], 
                  [s,  c]])
    
    M = R @ S
    
    # Calculate Determinant (Area Scaling Factor)
    det = np.linalg.det(M)
    
    # Draw Deformed Grid
    grid_range = np.linspace(-3, 3, 7)
    
    for i in grid_range:
        # Horizontal lines
        line_h = np.array([[x, i] for x in np.linspace(-3, 3, 20)])
        transformed_h = (M @ line_h.T).T
        ax2.plot(transformed_h[:, 0], transformed_h[:, 1], color='#b0b0b0', linewidth=1, alpha=0.5)
        
        # Vertical lines
        line_v = np.array([[i, y] for y in np.linspace(-3, 3, 20)])
        transformed_v = (M @ line_v.T).T
        ax2.plot(transformed_v[:, 0], transformed_v[:, 1], color='#b0b0b0', linewidth=1, alpha=0.5)

    # Draw Unit Square (Material Block)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    t_square = M @ square
    
    # Color based on state
    fill_color = '#007079' # Equinor Primary
    if det < 0: fill_color = '#ff9e3d' # Warning (Flipped)
    if abs(det) < 0.01: fill_color = '#d10a10' # Critical (Singular)
    
    ax2.fill(t_square[0], t_square[1], color=fill_color, alpha=0.6, label='Material Element')
    
    # Basis Vectors
    i_hat = M @ np.array([1, 0])
    j_hat = M @ np.array([0, 1])
    ax2.arrow(0, 0, i_hat[0], i_hat[1], head_width=0.15, fc='black', ec='black', linewidth=2)
    ax2.arrow(0, 0, j_hat[0], j_hat[1], head_width=0.15, fc='black', ec='black', linewidth=2)
    
    # Annotations
    ax2.set_title(f"Transformation Matrix:\n[[{M[0,0]:.2f}, {M[0,1]:.2f}], [{M[1,0]:.2f}, {M[1,1]:.2f}]]\nDeterminant (Area Scale): {det:.2f}", fontsize=11)
    
    if abs(det) < 0.01:
        ax2.text(0, -3.5, "⚠️ SINGULARITY DETECTED", color='red', ha='center', fontweight='bold')
    
    fig2.canvas.draw_idle()

# Controls
style = {'description_width': 'initial'}
w_sx = FloatSlider(min=0.5, max=2.0, value=1.0, step=0.1, description='Scale X (Stretch):', style=style)
w_sy = FloatSlider(min=0.5, max=2.0, value=1.0, step=0.1, description='Scale Y (Stretch):', style=style)
w_sh = FloatSlider(min=-1.0, max=1.0, value=0.0, step=0.1, description='Shear (Strain):', style=style)
w_rot = FloatSlider(min=0, max=2*np.pi, value=0.0, step=0.1, description='Rotation (rad):', style=style)

interactive(visualize_deformation, scale_x=w_sx, scale_y=w_sy, shear_factor=w_sh, rotation_rad=w_rot)
display(VBox([HBox([w_sx, w_sy]), HBox([w_sh, w_rot]), fig2.canvas]))
visualize_deformation(1, 1, 0, 0)

# %% [markdown]
# ---
# ## 3. Principal Component Analysis (Eigenvectors)
# 
# **Context:** In Data Science and Geology, we often need to find the "Principal Directions" of a dataset or a physical system.
# *   **Eigenvectors:** The directions where the transformation acts only as scaling (no rotation).
# *   **Eigenvalues:** The magnitude of that scaling.
# 
# **Simulation:** Rotate the test vector. When it aligns with an Eigenvector, it will point in the same direction as the transformed vector (blue and red arrows align).

# %%
# 📡 SIMULATION 3: Eigenvector Identification

# Define a Symmetric Matrix (common in covariance/stress matrices)
# This guarantees real, orthogonal eigenvectors
A = np.array([[2.0, 1.0], 
              [1.0, 2.0]])

vals, vecs = np.linalg.eig(A)

plt.ioff()
fig3, ax3 = plt.subplots(figsize=(7, 7))
plt.ion()

def analyze_eigenvectors(angle_rad):
    ax3.clear()
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Input Vector v
    v = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    # Transformed Vector Av
    Av = A @ v
    
    # Draw Vectors
    ax3.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.02, label='Input Vector (v)')
    ax3.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1, 
               color='#d10a10', width=0.02, alpha=0.8, label='Transformed (Av)')
    
    # Draw True Eigenvectors (Reference)
    for i in range(2):
        ev = vecs[:, i] * 2.5 # Scale for visibility
        ax3.plot([0, ev[0]], [0, ev[1]], 'g--', alpha=0.4, linewidth=1)
        ax3.text(ev[0], ev[1], f'λ={vals[i]:.1f}', color='green')

    # Check Alignment
    # Cross product magnitude is 0 if parallel
    cross_prod = abs(v[0]*Av[1] - v[1]*Av[0])
    
    title_text = "Rotate v to find the Principal Directions..."
    title_color = 'black'
    
    if cross_prod < 0.1:
        title_text = "✅ EIGENVECTOR ALIGNMENT DETECTED"
        title_color = 'green'
        ax3.set_facecolor('#f0fff0')
        
    ax3.set_title(title_text, color=title_color, fontweight='bold')
    ax3.legend(loc='lower right')
    fig3.canvas.draw_idle()

s_angle = FloatSlider(min=0, max=2*np.pi, step=0.05, value=0.0, description='Vector Angle:')

interactive(analyze_eigenvectors, angle_rad=s_angle)
display(VBox([s_angle, fig3.canvas]))
analyze_eigenvectors(0)

# %% [markdown]
# ---
# ## 4. Dot Product & Similarity
# 
# **Context:** The dot product measures alignment between vectors. In Recommendation Systems and Seismic Analysis, it's used to quantify similarity between signals or feature vectors.
# 
# **Simulation:** Adjust the "Query Vector" to find which "Database Vector" it is most similar to.

# %%
# 📡 SIMULATION 4: Similarity Search

# Database of "Signals" (fixed vectors)
signals = {
    "Signal A": np.array([1.0, 0.5]),
    "Signal B": np.array([-0.5, 1.0]),
    "Signal C": np.array([0.5, -0.5])
}

plt.ioff()
fig4, ax4 = plt.subplots(figsize=(7, 7))
plt.ion()

def similarity_scanner(query_x, query_y):
    ax4.clear()
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    query = np.array([query_x, query_y])
    
    # Draw Query
    ax4.arrow(0, 0, query[0], query[1], head_width=0.1, fc='blue', ec='blue', linewidth=3, label='Query Vector')
    
    # Draw Signals and Compute Dot Products
    max_score = -np.inf
    best_match = None
    
    for name, sig in signals.items():
        # Draw Signal
        ax4.arrow(0, 0, sig[0], sig[1], head_width=0.1, fc='gray', ec='gray', alpha=0.5, linewidth=2)
        ax4.text(sig[0], sig[1], name, color='gray')
        
        # Compute Dot Product (Similarity)
        score = np.dot(query, sig)
        
        if score > max_score:
            max_score = score
            best_match = name
            
    # Highlight Best Match
    if best_match:
        best_sig = signals[best_match]
        ax4.arrow(0, 0, best_sig[0], best_sig[1], head_width=0.12, fc='#007079', ec='#007079', linewidth=3)
        ax4.set_title(f"Best Match: {best_match} (Score: {max_score:.2f})", fontsize=12, fontweight='bold', color='#007079')
        
        # Visualizing Projection (optional, dashed line)
        # Project query onto best match
        proj = (np.dot(query, best_sig) / np.dot(best_sig, best_sig)) * best_sig
        ax4.plot([query[0], proj[0]], [query[1], proj[1]], 'k:', alpha=0.5)

    ax4.legend(loc='upper left')
    fig4.canvas.draw_idle()

s_qx = FloatSlider(min=-1.5, max=1.5, step=0.1, value=1.0, description='Query X:')
s_qy = FloatSlider(min=-1.5, max=1.5, step=0.1, value=0.0, description='Query Y:')

interactive(similarity_scanner, query_x=s_qx, query_y=s_qy)
display(VBox([HBox([s_qx, s_qy]), fig4.canvas]))
similarity_scanner(1.0, 0.0)
