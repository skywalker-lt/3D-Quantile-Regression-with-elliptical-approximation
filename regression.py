# pip3/conda install numpy pandas plotly statsmodels sympy pyyaml
# this is a plotly implementation of extended quantile regression for 3D scatter plot modeling

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from sympy import symbols, Eq, solve
import yaml  # For loading the YAML configuration

# Load the YAML config file
parser = argparse.ArgumentParser(description='3D Quantile Regression')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file')
args = parser.parse_args()

with open(args.config, 'r') as stream:
    config = yaml.safe_load(stream)

# Extract hyperparameters from YAML config file
quantiles = config['quantiles']
data_input = config['data_input']
n_segments = config['n_segments']
ellipse_color = config['ellipse']['color']
ellipse_opacity = config['ellipse']['opacity']
mesh_color = config['mesh']['color']
mesh_opacity = config['mesh']['opacity']

# Extract column names
IV = config['columns']['IV'] # independent Variable (x-axis)
DV1 = config['columns']['DV1'] # Dependent Variable 1 (y-axis)
DV2 = config['columns']['DV2'] # Dependent Variable 2 (z-axis)

# Load the scatter plot data from the CSV file
data = pd.read_csv(data_input, encoding='UTF-8')

# Check if the columns exist in the CSV file
if not {IV, DV1, DV2}.issubset(data.columns):
    raise ValueError(f"The specified columns ({IV}, {DV1}, {DV2}) do not exist in the CSV file.")

# --------------------------------------
# Elliptical Cross-section Approximation
# --------------------------------------

def fit_ellipse(subset, x_plane):
    # linear quantile regression on the scatter plot projected onto the yz-plane (DV2 againsrt DV1)
    mod = QuantReg(subset[DV2], sm.add_constant(subset[DV1]))
    quantile_models = [mod.fit(q=q) for q in quantiles]

    # linear quantile regression on the scatter plot projected onto the transposed zy-plane (DV1 vs DV2)
    mod_transposed = QuantReg(subset[DV1], sm.add_constant(subset[DV2]))
    quantile_models_transposed = [mod_transposed.fit(q=q) for q in quantiles]

    # Define variables for solving the the intersection points
    y_sym, z_sym = symbols('y z')

    # The quantile boundaries generated on the original plane
    eq1 = Eq(z_sym, quantile_models[0].params['const'] + quantile_models[0].params[DV1] * y_sym)
    eq2 = Eq(z_sym, quantile_models[1].params['const'] + quantile_models[1].params[DV1] * y_sym)

    # The quantile boundaries generated on the transposed plane
    eq3 = Eq(y_sym, quantile_models_transposed[0].params['const'] + quantile_models_transposed[0].params[DV2] * z_sym)
    eq4 = Eq(y_sym, quantile_models_transposed[1].params['const'] + quantile_models_transposed[1].params[DV2] * z_sym)

    # solve for four intersection points
    intersection_points_1 = solve((eq1, eq3), (y_sym, z_sym))  # Intersection 1
    intersection_points_2 = solve((eq1, eq4), (y_sym, z_sym))  # Intersection 2
    intersection_points_3 = solve((eq2, eq3), (y_sym, z_sym))  # Intersection 3
    intersection_points_4 = solve((eq2, eq4), (y_sym, z_sym))  # Intersection 4

    # Extract coordinates of the intersections
    y_intersect_1, z_intersect_1 = float(intersection_points_1[y_sym]), float(intersection_points_1[z_sym])
    y_intersect_2, z_intersect_2 = float(intersection_points_2[y_sym]), float(intersection_points_2[z_sym])
    y_intersect_3, z_intersect_3 = float(intersection_points_3[y_sym]), float(intersection_points_3[z_sym])
    y_intersect_4, z_intersect_4 = float(intersection_points_4[y_sym]), float(intersection_points_4[z_sym])

    # Collect the intersections into an array (matrix mathematically) 
    intersection_points = np.array([
        [y_intersect_1, z_intersect_1],
        [y_intersect_2, z_intersect_2],
        [y_intersect_3, z_intersect_3],
        [y_intersect_4, z_intersect_4]
    ])

    # ---------------------------------------------------
    # Ellipse generation via Singular Value Decomposition
    # ---------------------------------------------------

    # Deduce the centroid of the points
    centroid = np.mean(intersection_points, axis=0)

    # Translate the points to the origin
    translated_points = intersection_points - centroid

    # Perform SVD to the major and minor radius and the rotation matrix
    U, S, Vt = np.linalg.svd(translated_points)

    major_radius = S[0]  # Major radius from the s11 element of the singlar matrix
    minor_radius = S[1]  # Minor radius from the s22 element of the singlar matrix

    # Generate points for the ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_points = np.column_stack((major_radius * np.cos(theta), minor_radius * np.sin(theta)))

    # Rotate the ellipse with the rotation matrix V_transpose and translate the ellipse back
    ellipse_rotated = ellipse_points @ Vt
    ellipse_rotated += centroid

    # Project the ellipse to the yz-plane placed the given x vlaue
    ellipse_3d = np.vstack([np.full(ellipse_rotated.shape[0], x_plane), ellipse_rotated.T]).T

    return ellipse_3d

# ------------------
# Dataset processing
# ------------------

time_segments = np.linspace(data[IV].min(), data[IV].max(), n_segments + 1)
ellipses = []

# Fit an ellipse for each segment
for i in range(n_segments):
    
    # Define the subset
    subset = data[(data[IV] >= time_segments[i]) & (data[IV] < time_segments[i + 1])]
    
    # midpoint of the subset's x-interval
    x_plane = (time_segments[i] + time_segments[i + 1]) / 2
    
    # Fit the ellipse for each subset and place it at the x-interval midpoint
    ellipse_3d = fit_ellipse(subset, x_plane)
    ellipses.append(ellipse_3d)

# ---------------------
# Plot all the elements
# ---------------------

fig = go.Figure()

# Plot the original scatter plots
fig.add_trace(go.Scatter3d(
    x=data[IV], y=data[DV1], z=data[DV2], 
    mode='markers', marker=dict(size=2, color='black'), name='Data points'
))

# Plot the ellipses as a filled surface using triangulated mesh
for i in range(n_segments):
    ellipse = ellipses[i]
    x_vals = ellipse[:, 0]
    y_vals = ellipse[:, 1]
    z_vals = ellipse[:, 2]

    # Triangulate the ellipse
    n_points = len(x_vals)
    faces = [[j, (j + 1) % n_points, 0] for j in range(1, n_points)]  # Triangulation

    i_vals = [f[0] for f in faces]
    j_vals = [f[1] for f in faces]
    k_vals = [f[2] for f in faces]

    # Plot the ellipse
    fig.add_trace(go.Mesh3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        i=i_vals,
        j=j_vals,
        k=k_vals,
        color=ellipse_color,
        opacity=ellipse_opacity,
        name=f'Ellipse {i+1}'
    ))

# Plot the mesh connecting consecutive ellipses
for i in range(n_segments - 1):
    ellipse_1 = ellipses[i]
    ellipse_2 = ellipses[i + 1]

    # Vertices of the ellipses
    vertices_x = np.concatenate([ellipse_1[:, 0], ellipse_2[:, 0]])
    vertices_y = np.concatenate([ellipse_1[:, 1], ellipse_2[:, 1]])
    vertices_z = np.concatenate([ellipse_1[:, 2], ellipse_2[:, 2]])

    # Define triangles to connect two consecutive ellipses
    faces = []
    n_points = len(ellipse_1)
    for j in range(n_points - 1):
        faces.append([j, j + 1, j + n_points])
        faces.append([j + 1, j + n_points + 1, j + n_points])

    faces = np.array(faces)

    # Plot the mesh with the color and opacity defined in the yaml file
    fig.add_trace(go.Mesh3d(
        x=vertices_x, 
        y=vertices_y, 
        z=vertices_z, 
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=mesh_color,  # color
        opacity=mesh_opacity,  # opacity
        name=f'Mesh Layer {i+1}'
    ))

# Update the layout
fig.update_layout(scene=dict(
                    xaxis_title='Independent Variable (X)',
                    yaxis_title='Dependent Variable 1 (Y)',
                    zaxis_title='Dependent Variable 2 (Z)'),
                  title="3D Scatter Plot with 20 Connected Ellipses (Filled Purple and Blue Mesh)")

# Display the plot
fig.show()
