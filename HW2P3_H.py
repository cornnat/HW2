import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

## LITERALLY just copied my code for a-g and changing the lifting map equation
# and derivative eqns

#read the data file
data = np.loadtxt('/root/Desktop/host/HW2/mesh.dat', skiprows = 1)
x = data[:, 0]  # First column is x
y = data[:, 1]  # Second column is y

#visualizing the data - plotting data and saving fig
def plot_hull_triang(coords, hull=None, triang=None):
    xp,yp = zip(*coords)
    plt.scatter(xp, yp, s=1)  # 's' controls the size of the points

    if hull is not None:
        for i in range(len(hull)):
            c0 = hull[i]
            c1 = hull[(i+1) % len(hull)]
            plt.plot([c0[0], c1[0]], [c0[1], c1[1]], 'b-', lw=2, label='Convex Hull' if i == 0 else "")

    if triang is not None:
         plt.triplot(coords[:,0], coords[:,1], triang.simplices, color='b', lw=0.5, linestyle='-', marker='', label='Delaunay Triangulation')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_A_plot_hull_triangulation.png', bbox_inches='tight', format='png') 
    plt.show()

# Function to calculate the convex hull using monotone chain
def convex_hull_monotone_chain(data):
    # Sort the points by x (and by y if necessary)
    data = data[np.argsort(data[:, 0])]
    
    # Function to check the orientation (cross product)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build the lower hull
    lower = []
    for point in data:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    
    # Build the upper hull
    upper = []
    for point in reversed(data):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    
    # Remove the last point of each half because it is repeated at the beginning of the other half
    return np.array(lower[:-1] + upper[:-1])

#read the data file
points = np.loadtxt('/root/Desktop/host/SEC2/mesh.dat', skiprows = 1)
x = points[:, 0]  # First column is x
y = points[:, 1]  # Second column is y

coords = np.array(list(zip(x,y)))

triang = Delaunay(coords)

hull = convex_hull_monotone_chain(coords)

plot_hull_triang(coords, hull=hull, triang=triang)

#
#
# Part b
def z(x, y):
    return x**2 + x*y + y**2

z_plane = z(coords[:, 0], coords[:,1])
print("z_flat:", z_plane)
coords_3d = np.column_stack((coords, z_plane))

def tri_area_2d(pts):
    """computing area of 2d tri via cross product"""
    a, b, c = pts
    return 0.5 *np.abs(np.cross(b-a, c-a))

def tri_area_3d(pts):
    """area of 3d tri via cross product"""
    a, b, c = pts
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross)

area_ratios = []
for simplex in triang.simplices:
    tri_2d = coords[simplex]
    tri_3d = coords_3d[simplex]

    area_2d = tri_area_2d(tri_2d)
    area_3d = tri_area_3d(tri_3d)

    area_ratios.append(area_3d / area_2d)

area_ratios = np.array(area_ratios)

print("Area ratios (3D/2D):", area_ratios)

area_ratios_capped = np.clip(area_ratios, 0, 100)

# add small epsilon to avoid log(0)
area_ratios_log = np.log10(area_ratios_capped + 1e-6)
area_ratios_normalized = (area_ratios_log - np.min(area_ratios_log)) / (np.max(area_ratios_log) - np.min(area_ratios_log))
print("Area Ratios Normalized (3D / 2D):", area_ratios_normalized)

# plot the heatmap using tripcolor
plt.figure(figsize=(10, 8))
trip = plt.tripcolor(
    coords[:, 0], 
    coords[:, 1], 
    triang.simplices, 
    facecolors=area_ratios_normalized, 
    edgecolors='none',  
    cmap='inferno',     
    shading='flat'    
)

plt.colorbar(trip, label='Area Ratio (3D / 2D), log scale')
plt.title('Change in Triangle Area After Parabolic Lifting')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_B_area_change', bbox_inches='tight')

#
#
#
# Part c OF PART H

def induced_metric(x, y):
    z_x = 2*x + y
    z_y = x + 2*y
    
    E = 1 + z_x**2
    F = z_x * z_y
    G = 1 + z_y**2
    return np.array([[E, F], [F, G]])

def metric_determinant(x, y):
    g = induced_metric(x, y)
    return g[0][0]*g[1][1] - g[0][1]**2

x = coords[:, 0]
y = coords[:, 1]

g = np.array([induced_metric(xi, yi) for xi, yi in zip(x, y)])

print("induced metric is", g)

det_g = metric_determinant(coords[:, 0], coords[:, 1])
print("det_g:",det_g)

plt.figure(figsize=(10, 10))
plt.scatter(coords[:, 0], coords[:, 1], c=det_g, cmap='plasma')
plt.colorbar(label='Determinant of Metric Tensor (Area Distortion)')
plt.title('Determinant of Induced Metric Tensor')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_C_metric_det', bbox_inches='tight')
plt.show()

#
#
#
# part D
from mpl_toolkits.mplot3d import Axes3D

# Function to compute normalized surface normal at (x, y)
def surface_normal(x, y):
    normal = np.array([(-2*x - y), (-2*y - x), 1])
    return normal / np.linalg.norm(normal)  # Normalize the vector

# Compute surface normals for all points
normals = np.array([surface_normal(xi, yi) for xi, yi in zip(coords[:, 0], coords[:, 1])])

# Extract x, y, z for plotting
X, Y, Z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
U, V, W = normals[:, 0], normals[:, 1], normals[:, 2]

# Plot the mesh with normal vectors
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the triangulated mesh
ax.plot_trisurf(X, Y, Z, triangles=triang.simplices, cmap='viridis', alpha=0.6, edgecolor='gray')

# Plot normal vectors
ax.quiver(X, Y, Z, U, V, W, color='r', length=0.2, normalize=True)

# Formatting
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Surface Normals of Lifted Mesh')
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_D_surface_normals.png', bbox_inches='tight')
plt.show()

#
#
#
#
# Part E
# Part E: Compute vertex normals and plot them

def triangle_normal(pts):
    """Computes the normal of a 3D triangle via cross product"""
    a, b, c = pts
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    return normal / np.linalg.norm(normal)  # Normalize

# Initialize normal accumulator
vertex_normals = np.zeros_like(coords_3d)

# Compute triangle normals and accumulate
for simplex in triang.simplices:
    tri_pts = coords_3d[simplex]
    normal = triangle_normal(tri_pts)
    
    # Add normal to each vertex of the triangle
    for i in range(3):
        vertex_normals[simplex[i]] += normal

# Normalize vertex normals
vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)

# Extract components for plotting
U_v, V_v, W_v = vertex_normals[:, 0], vertex_normals[:, 1], vertex_normals[:, 2]

# Plot the lifted mesh with vertex normals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the triangulated mesh
ax.plot_trisurf(X, Y, Z, triangles=triang.simplices, cmap='viridis', alpha=0.6, edgecolor='gray')

# Plot vertex normal vectors
ax.quiver(X, Y, Z, U_v, V_v, W_v, color='r', length=0.2, normalize=True)

# Formatting
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Vertex Normals of Lifted Mesh')
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_E_vertex_normals.png', bbox_inches='tight')
plt.show()

#
#
#
# Part F
# first derivs
## o_x = do_dx = [1, 0, 2x+y]
## o_y = do_dy = [0, 1, 2y+x]
#
# second derivs
## o_xx = do_dxx = [0, 0, 2]
## o_xy = do_dxy = [0, 0, 1]
## o_yy = do_dyy = [0, 0, 2]

r_xx = np.array([0.0, 0.0, 2.0])
r_xy = np.array([0.0, 0.0, 1.0])
r_yy = np.array([0.0, 0.0, 2.0])

L = np.array([np.dot(n, r_xx) for n in vertex_normals])
M = np.array([np.dot(n, r_xy) for n in vertex_normals])
N = np.array([np.dot(n, r_yy) for n in vertex_normals])

plt.figure(figsize=(10, 8))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=L, cmap='viridis', 
                s=50, edgecolor='none')
plt.colorbar(sc, label='Second Fundamental Form Coefficient L')
plt.title('Second Fundamental Form Coefficient L\n(Computed with Vertex Normals)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_F_second_fundamental_form.png', bbox_inches='tight')

#
#
#
# Part G
# # Compute the shape operator S = II * I^-1 for each vertex
principal_curvatures = []
gaussian_curvatures = []
mean_curvatures = []

for i in range(len(coords)):
    # First fundamental form (induced metric)
    E, F, G = g[i].flatten()[:3]

    # Second fundamental form
    L_i, M_i, N_i = L[i], M[i], N[i]

    # Construct matrices
    I_matrix = np.array([[E, F], [F, G]])
    II_matrix = np.array([[L_i, M_i], [M_i, N_i]])

    # Compute shape operator S = I^-1 * II
    I_inv = np.linalg.inv(I_matrix)
    S = I_inv @ II_matrix

    # Compute eigenvalues (principal curvatures)
    k1, k2 = np.linalg.eigvals(S)

    # Store values
    principal_curvatures.append((k1, k2))
    gaussian_curvatures.append(k1 * k2)
    mean_curvatures.append((k1 + k2) / 2)

# Convert lists to arrays
principal_curvatures = np.array(principal_curvatures)
k1_values = principal_curvatures[:, 0]
k2_values = principal_curvatures[:, 1]
gaussian_curvatures = np.array(gaussian_curvatures)
mean_curvatures = np.array(mean_curvatures)

# Visualization
# Visualization
plt.figure(figsize=(10, 8))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=k1_values, cmap='viridis', s=50, edgecolor='none')
plt.colorbar(sc, label='Principal Curvature (K1)')
plt.title('Principal Curvature K2 Visualization')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_G_Principal_Curvature_K1.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=k2_values, cmap='viridis', s=50, edgecolor='none')
plt.colorbar(sc, label='Principal Curvature (K2)')
plt.title('Principal Curvature K2 Visualization')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_G_Principal_Curvature_K2.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=gaussian_curvatures, cmap='inferno', s=50, edgecolor='none')
plt.colorbar(sc, label='Gaussian Curvature (K)')
plt.title('Gaussian Curvature Visualization')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_G_Gaussian_Curvature.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=mean_curvatures, cmap='coolwarm', s=50, edgecolor='none')
plt.colorbar(sc, label='Mean Curvature (H)')
plt.title('Mean Curvature Visualization')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig('/root/Desktop/host/HW2/P3H_plots/P3H_G_Mean_Curvature.png', bbox_inches='tight')
plt.show()