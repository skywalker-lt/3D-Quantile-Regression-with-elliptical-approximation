# pip3/conda install numpy pandas plotly statsmodels sympy pyyaml

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from sympy import symbols, Eq, solve
import yaml  # For loading the YAML configuration

# Load the YAML config file
with open("3d-line1.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Extract hyperparameters from config
quantiles = config['quantiles']
data_input = config['data_input']
n_segments = config['n_segments']
ellipse_color = config['ellipse']['color']
ellipse_opacity = config['ellipse']['opacity']
mesh_color = config['mesh']['color']
mesh_opacity = config['mesh']['opacity']

# Extract column names from config
IV = config['columns']['IV']
DV1 = config['columns']['DV1']
DV2 = config['columns']['DV2']

# Load the dataset from the CSV file specified in the config
data = pd.read_csv(data_input, encoding='UTF-8')

# Check if the specified columns exist in the CSV file
if not {IV, DV1, DV2}.issubset(data.columns):
    raise IndexError(f"The specified columns ({IV}, {DV1}, {DV2}) do not exist in the CSV file.")

# --------------------------
# Function to fit the ellipse using quantile regression
# --------------------------

def fit_ellipse(subset, x_plane):
    # Perform quantile regression on the original dataset (Concentration vs Wavelength)
    mod = QuantReg(subset[DV2], sm.add_constant(subset[DV1]))
    quantile_models = [mod.fit(q=q) for q in quantiles]

    # Perform quantile regression on transposed data (Wavelength vs Concentration)
    mod_transposed = QuantReg(subset[DV1], sm.add_constant(subset[DV2]))
    quantile_models_transposed = [mod_transposed.fit(q=q) for q in quantiles]

    # Define symbolic variables for solving the intersections
    y_sym, z_sym = symbols('y z')

    # Define the regression lines based on quantile regression results (original dataset)
    eq1 = Eq(z_sym, quantile_models[0].params['const'] + quantile_models[0].params[DV1] * y_sym)
    eq2 = Eq(z_sym, quantile_models[1].params['const'] + quantile_models[1].params[DV1] * y_sym)

    # Define the regression lines based on quantile regression results (transposed dataset)
    eq3 = Eq(y_sym, quantile_models_transposed[0].params['const'] + quantile_models_transposed[0].params[DV2] * z_sym)
    eq4 = Eq(y_sym, quantile_models_transposed[1].params['const'] + quantile_models_transposed[1].params[DV2] * z_sym)

    # Solve for four intersection points between quantile regression boundaries
    intersection_points_1 = solve((eq1, eq3), (y_sym, z_sym))  # Intersection 1
    intersection_points_2 = solve((eq1, eq4), (y_sym, z_sym))  # Intersection 2
    intersection_points_3 = solve((eq2, eq3), (y_sym, z_sym))  # Intersection 3
    intersection_points_4 = solve((eq2, eq4), (y_sym, z_sym))  # Intersection 4

    # Extract coordinates of the intersections
    y_intersect_1, z_intersect_1 = float(intersection_points_1[y_sym]), float(intersection_points_1[z_sym])
    y_intersect_2, z_intersect_2 = float(intersection_points_2[y_sym]), float(intersection_points_2[z_sym])
    y_intersect_3, z_intersect_3 = float(intersection_points_3[y_sym]), float(intersection_points_3[z_sym])
    y_intersect_4, z_intersect_4 = float(intersection_points_4[y_sym]), float(intersection_points_4[z_sym])

    # Collect the four intersection points
    intersection_points = np.array([
        [y_intersect_1, z_intersect_1],
        [y_intersect_2, z_intersect_2],
        [y_intersect_3, z_intersect_3],
        [y_intersect_4, z_intersect_4]
    ])

    # --------------------------
    # SVD for Ellipse Fitting
    # --------------------------

    # Calculate the centroid (mean of the points)
    centroid = np.mean(intersection_points, axis=0)

    # Translate the points to the centroid
    translated_points = intersection_points - centroid

    # Perform SVD to obtain major and minor axes for the ellipse
    U, S, Vt = np.linalg.svd(translated_points)

    major_radius = S[0] * 0.75   # Major axis
    minor_radius = S[1] * 0.75   # Minor axis

    # Generate points for the ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_points = np.column_stack((major_radius * np.cos(theta), minor_radius * np.sin(theta)))

    # Rotate and translate the ellipse
    ellipse_rotated = ellipse_points @ Vt
    ellipse_rotated += centroid

    # Project the ellipse to the yz-plane at the given x_plane
    ellipse_3d = np.vstack([np.full(ellipse_rotated.shape[0], x_plane), ellipse_rotated.T]).T

    return ellipse_3d

# --------------------------
# Segment the dataset into parts along the x-axis (Time)
# --------------------------

time_segments = np.linspace(data[IV].min(), data[IV].max(), n_segments + 1)
ellipses = []

# Fit an ellipse for each segment
for i in range(n_segments):
    # Subset the data for the current segment
    subset = data[(data[IV] >= time_segments[i]) & (data[IV] < time_segments[i + 1])]
    
    # Calculate the midpoint of the segment
    x_plane = (time_segments[i] + time_segments[i + 1]) / 2
    
    # Fit the ellipse for the subset and place it at the midpoint
    ellipse_3d = fit_ellipse(subset, x_plane)
    ellipses.append(ellipse_3d)

# --------------------------
# Plot the original data, ellipses, and the connecting mesh layers
# --------------------------

fig = go.Figure()

# Plot original data points with smaller dots
fig.add_trace(go.Scatter3d(
    x=data[IV], y=data[DV1], z=data[DV2], 
    mode='markers', marker=dict(size=2, color='black'), name='Data points'
))

# Plot the ellipses as filled surfaces using triangulated mesh
for i in range(n_segments):
    ellipse = ellipses[i]
    x_vals = ellipse[:, 0]
    y_vals = ellipse[:, 1]
    z_vals = ellipse[:, 2]

    # Triangulate the ellipse (create faces for a filled surface)
    n_points = len(x_vals)
    faces = [[j, (j + 1) % n_points, 0] for j in range(1, n_points)]  # Triangulation

    i_vals = [f[0] for f in faces]
    j_vals = [f[1] for f in faces]
    k_vals = [f[2] for f in faces]

    # Plot the ellipse as a filled mesh
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

    # Vertices of the two ellipses
    vertices_x = np.concatenate([ellipse_1[:, 0], ellipse_2[:, 0]])
    vertices_y = np.concatenate([ellipse_1[:, 1], ellipse_2[:, 1]])
    vertices_z = np.concatenate([ellipse_1[:, 2], ellipse_2[:, 2]])

    # Define triangles to connect corresponding points between the two ellipses
    faces = []
    n_points = len(ellipse_1)
    for j in range(n_points - 1):
        faces.append([j, j + 1, j + n_points])
        faces.append([j + 1, j + n_points + 1, j + n_points])

    faces = np.array(faces)

    # Plot the connecting mesh with blue color and 20% opacity
    fig.add_trace(go.Mesh3d(
        x=vertices_x, 
        y=vertices_y, 
        z=vertices_z, 
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=mesh_color,  # Fixed color for the mesh
        opacity=mesh_opacity,  # 20% opacity
        name=f'Mesh Layer {i+1}'
    ))

# Update the layout
fig.update_layout(scene=dict(
                    xaxis_title='Independent Variable (X)',
                    yaxis_title='Dependent Variable 1 (Y)',
                    zaxis_title='Dependent Variable 2 (Z)'),
                  title="3D Scatter Plot with 20 Connected Ellipses (Filled Purple and Blue Mesh)")

# Show the plot
fig.show()
