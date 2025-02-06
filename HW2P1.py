import numpy as np

#Problem 1a:
#   Write down the spherical and cylindrical basis in terms of the cartesian basis (x,y,z). 
#   Write a python function or multiple python functions that convert coordinates and basis between the three

#   Cartesian (x, y, z) to Spherical (r, phi, theta)
#       r = np.sqrt(x**2 + y**2 + z**2)
#       theta = np.arccos(z/r)
#       phi = np.arctan(y/x)

#   Cartesian (x, y, z) to Cylindrical (rho, phi, z)
#       r = np.sqrt(x**2 + y**2)
#       phi = np.arctan (y/x)
#       z = z

#   Cylindrical (rho, phi, z) to Cartesian (x, y, z)
#       x = r*(np.cos(phi))
#       y = r*(np.sin(phi))
#       z = z

#   Cylindrical (rho, phi, z) to Spherical (r, phi, theta)
#       r = np.sqrt(rho**2 + z**2)
#       phi = phi
#       theta = np.arctan(rho/z)

#   Spherical (r, phi, theta) to Cartesian (x, y, z)
#       x = r*(np.sin(theta))*(np.cos(phi))
#       y = r*(np.sin(theta))*(np.sin(phi))
#       z = r*(np.cos(theta))

#   Spherical (r, phi, theta) to Cylindrical (rho, phi, z)
#       rho = r*(np.sin(theta))
#       phi = phi
#       z = r*(np.cos(theta))

def cartesian_to_spherical(x, y, z):
    #   Cartesian (x, y, z) to Spherical (r, phi, theta)
    #   angles are in radians
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan(y/x)
    return np.array([r, theta, phi])

def cartesian_to_cylindrical(x, y, z):
#       Cartesian (x, y, z) to Cylindrical (rho, phi, z)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan (y/x)
    z = z
    return np.array([r, phi, z])

def cylindrical_to_cartesian(rho, phi, z):
#       Cylindrical (rho, phi, z) to Cartesian (x, y, z)
#       angles are in radians
    x = r*(np.cos(phi))
    y = r*(np.sin(phi))
    z = z
    return np.array([x, y, z])

def cylindrical_to_spherical(rho, phi, z):
#       Cylindrical (rho, phi, z) to Spherical (r, phi, theta)
#       angles are in radians
    r = np.sqrt(rho**2 + z**2)
    phi = phi
    theta = np.arctan(rho/z)
    return np.array([r, phi, theta])

def spherical_to_cartesian(r, phi, theta):
#       Spherical (r, phi, theta) to Cartesian (x, y, z)
#       angles are in radians
    x = r*(np.sin(theta))*(np.cos(phi))
    y = r*(np.sin(theta))*(np.sin(phi))
    z = r*(np.cos(theta))
    return np.array([x, y, z])

def spherical_to_cylindrical(r, phi, theta):
#       Spherical (r, phi, theta) to Cylindrical (rho, phi, z)
#       angles are in radians
    rho = r*(np.sin(theta))
    phi = phi
    z = r*(np.cos(theta))
    return np.array(rho, phi, z)

#Problem 1b: A position vector in the unit sphere is r = e_r such that the coordinate (1, theta, phi)
#   create local orthonormal coordinate systems on the unit sphere and represent them as vectors in cartesian coordinate system.
#   you should reproduce something like the plot below

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def spherical_basis_to_cartesian(theta, phi):
    """
    Convert spherical basis vectors to Cartesian basis vectors.
    Returns the spherical basis vectors {e_r, e_theta, e_phi} in Cartesian coordinates.
    """
    e_r = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    e_theta = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])
    e_phi = np.array([
        -np.sin(phi),
        np.cos(phi),
        0
    ])
    return e_r, e_theta, e_phi

def plot_top_hemisphere_with_basis_vectors():
    """
    Plot the top hemisphere of the unit sphere and local Cartesian basis vectors at various points.
    Save the plot as a PNG file.
    """
    # Create a grid of theta and phi values for the top hemisphere
    theta = np.linspace(0, np.pi/2, 6)  # Polar angle (0 to pi/2)
    phi = np.linspace(0, 2 * np.pi, 12)  # Azimuthal angle (0 to 2pi)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the top hemisphere of the unit sphere
    u = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
    v = np.linspace(0, np.pi/2, 50)     # Polar angle (0 to pi/2 for top hemisphere)
    x = np.outer(np.cos(u), np.sin(v))  # Parametric equation for x
    y = np.outer(np.sin(u), np.sin(v))  # Parametric equation for y
    z = np.outer(np.ones(np.size(u)), np.cos(v))  # Parametric equation for z
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Plot local basis vectors at selected points
    scale = 0.2  # Scale factor for the basis vectors
    for t in theta:
        for p in phi:
            # Convert spherical coordinates to Cartesian
            r = 1  # Unit sphere
            x, y, z = spherical_to_cartesian(r, t, p)

            # Get the local basis vectors in Cartesian coordinates
            e_r, e_theta, e_phi = spherical_basis_to_cartesian(t, p)

            # Plot the basis vectors (scaled down for better visualization)
            ax.quiver(x, y, z, e_r[0], e_r[1], e_r[2], color='r', length=scale, normalize=True, label='e_r' if t == 0 and p == 0 else "")
            ax.quiver(x, y, z, e_theta[0], e_theta[1], e_theta[2], color='g', length=scale, normalize=True, label='e_theta' if t == 0 and p == 0 else "")
            ax.quiver(x, y, z, e_phi[0], e_phi[1], e_phi[2], color='b', length=scale, normalize=True, label='e_phi' if t == 0 and p == 0 else "")

    # Set plot limits and labels
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([0, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Local Orthonormal Basis Vectors on the Top Hemisphere of a Unit Sphere')

    # Add a legend
    ax.legend()

    # Save the plot as a PNG file
    plt.savefig('/root/Desktop/host/HW2/P1_plots/hemisphere_with_local_orth_basis_vectors.png', dpi=300)
    plt.show()

# Run the plotting function
plot_top_hemisphere_with_basis_vectors()

# Problem 1c: can you plot the unit sphere in spherical basis. If so plot it. If not explain why
# it should be very simple, and don't over think it
#   
# No you cannot plot a unit sphere in a spherical basis. The spherical coordinate basis vectors
# depend on the specific position in space meaning that they are locally defined. The cartesian
# coordinate system and its basis vectors are globally defined and are independent of the location.
# For example, for various vectors, the directionality of the spherical basis vectors change. 

# Problem 1d: Create a function that generates the local coordinate system on a given mesh, parametrized 
# by a general surface z = f(x, y). 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def local_coordinate_system(f, x, y, dx=1e-5, dy=1e-5):
    """
    Generates the local coordinate system on a surface z = f(x, y).
    
    Parameters:
    - f: A function that takes x and y and returns z.
    - x, y: Arrays of x and y coordinates.
    - dx, dy: Step sizes for the gradient computation.
    
    Returns:
    - T_x, T_y: Tangent vectors in the x and y directions.
    - N: Normal vector.
    - e_r: Unit normal vector.
    """
    # Compute the partial derivatives
    df_dx, df_dy = np.gradient(f(x, y), dx, dy)
    
    # Compute the tangent vectors
    T_x = np.stack([np.ones_like(x), np.zeros_like(x), df_dx], axis=-1)  # T_x = (1, 0, df/dx)
    T_y = np.stack([np.zeros_like(x), np.ones_like(x), df_dy], axis=-1)  # T_y = (0, 1, df/dy)
    
    # Compute the normal vector
    N = np.cross(T_x, T_y)
    
    # Normalize the normal vector
    e_r = N / np.linalg.norm(N, axis=-1, keepdims=True)
    
    return T_x, T_y, N, e_r

# Example surface function
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))  # Example surface: z = sin(sqrt(x^2 + y^2))

# Define a grid of x and y values
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Compute the local coordinate system
T_x, T_y, N, e_r = local_coordinate_system(f, X, Y)

# Plot the surface and the local coordinate system
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, color='blue')

# Plot tangent and normal vectors at selected points
step = 2  # Plot every 2nd point to avoid clutter
for i in range(0, X.shape[0], step):
    for j in range(0, X.shape[1], step):
        # Point on the surface
        point = np.array([X[i, j], Y[i, j], Z[i, j]])
        
        # Tangent vectors
        ax.quiver(*point, *T_x[i, j], color='red', length=0.5, normalize=True, label='T_x' if (i == 0 and j == 0) else "")
        ax.quiver(*point, *T_y[i, j], color='green', length=0.5, normalize=True, label='T_y' if (i == 0 and j == 0) else "")
        
        # Normal vector
        ax.quiver(*point, *e_r[i, j], color='purple', length=0.5, normalize=True, label='N' if (i == 0 and j == 0) else "")

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Local Coordinate System on Surface z = f(x, y)')

# Add a legend
ax.legend()

# Save the plot as a PNG file
plt.savefig('/root/Desktop/host/HW2/P1_plots/test_local_coordinate_system_function.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

#
#
#
# Problem 1e
# Define the spherical coordinates
def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Define the basis vectors in spherical coordinates
def spherical_basis_vectors(theta, phi):
    """Compute the spherical basis vectors e_theta and e_phi."""
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_theta, e_phi

# Parallel transport of a vector on a sphere
def parallel_transport(theta_0, alpha, beta, num_points=100):
    """
    Perform parallel transport of a vector n from theta = theta_0 to theta = pi/2.
    
    Parameters:
    - theta_0: Initial polar angle (near the north pole).
    - alpha, beta: Components of the initial vector n.
    - num_points: Number of points along the curve.
    
    Returns:
    - theta_values: Array of theta values along the curve.
    - n_vectors: Array of transported vectors n at each theta.
    """
    # Define the curve gamma(t) = theta(t), phi = 0
    theta_values = np.linspace(theta_0, np.pi / 2, num_points)
    phi_values = np.zeros_like(theta_values)

    # Initialize the vector n
    n_vectors = np.zeros((num_points, 3))
    n_vectors[0] = alpha * spherical_basis_vectors(theta_0, 0)[0] + beta * spherical_basis_vectors(theta_0, 0)[1]

    # Perform parallel transport
    for i in range(1, num_points):
        theta = theta_values[i]
        phi = phi_values[i]

        # Compute the change in the vector n
        delta_theta = theta_values[i] - theta_values[i - 1]
        n_vectors[i] = n_vectors[i - 1] - np.dot(n_vectors[i - 1], spherical_basis_vectors(theta, phi)[0]) * delta_theta * spherical_basis_vectors(theta, phi)[0]

    return theta_values, n_vectors

# Parameters
theta_0 = 0  # Initial polar angle
alpha = 1.0  # Component along e_theta
beta = 0.5  # Component along e_phi
num_points = 100  # Number of points along the curve

# Perform parallel transport
theta_values, n_vectors = parallel_transport(theta_0, alpha, beta, num_points)

# Plot the sphere and the transported vector
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u = np.linspace(0, np.pi, 50)
v = np.linspace(0, 2 * np.pi, 50)
x_sphere = np.outer(np.sin(u), np.cos(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.cos(u), np.ones_like(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.2)

# Plot the curve gamma(t)
x_curve, y_curve, z_curve = spherical_to_cartesian(1, theta_values, 0)
ax.plot(x_curve, y_curve, z_curve, color='red', label='Curve γ(t)')

# Plot the transported vector n at selected points
step = 10  # Plot every 10th point
for i in range(0, num_points, step):
    point = np.array([x_curve[i], y_curve[i], z_curve[i]])
    n_vector = n_vectors[i]
    ax.quiver(*point, *n_vector, color='green', length=0.2, normalize=True, label='n(θ)' if i == 0 else "")

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parallel Transport on a Sphere')

# Add a legend
ax.legend()

# Save the plot as a PNG file
plt.savefig('/root/Desktop/host/HW2/P1_plots/parallel_transport_sphere.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

#
#
#
# Problem 1F

# Define the spherical coordinates
def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)."""
    theta = np.atleast_1d(theta)  # Convert scalars to arrays if needed
    phi = np.atleast_1d(phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Define the basis vectors in spherical coordinates
def spherical_basis_vectors(theta, phi):
    """Compute the spherical basis vectors e_theta and e_phi."""
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_theta, e_phi

# Parallel transport of a vector on a sphere along a circle of constant latitude
def parallel_transport_circle(theta_0, alpha, beta, num_points=100):
    """
    Perform parallel transport of a vector n along a circle of constant latitude.
    
    Parameters:
    - theta_0: Polar angle (constant latitude).
    - alpha, beta: Components of the initial vector n.
    - num_points: Number of points along the curve.
    
    Returns:
    - phi_values: Array of phi values along the curve.
    - n_vectors: Array of transported vectors n at each phi.
    """
    # Define the curve gamma(t) = phi(t), theta = theta_0
    phi_values = np.linspace(0, 2 * np.pi, num_points)
    theta_values = theta_0 * np.ones_like(phi_values)

    # Initialize the vector n
    n_vectors = np.zeros((num_points, 3))
    n_vectors[0] = alpha * spherical_basis_vectors(theta_0, 0)[0] + beta * spherical_basis_vectors(theta_0, 0)[1]

    # Perform parallel transport
    for i in range(1, num_points):
        theta = theta_values[i]
        phi = phi_values[i]

        # Compute the change in the vector n
        delta_phi = phi_values[i] - phi_values[i - 1]
        n_vectors[i] = n_vectors[i - 1] - np.dot(n_vectors[i - 1], spherical_basis_vectors(theta, phi)[1]) * delta_phi * spherical_basis_vectors(theta, phi)[1]

    return phi_values, n_vectors

# Parameters
theta_0 = np.pi / 5  # Constant polar angle (latitude)
alpha = 1.0  # Component along e_theta
beta = 0.5  # Component along e_phi
num_points = 100  # Number of points along the curve

# Perform parallel transport
phi_values, n_vectors = parallel_transport_circle(theta_0, alpha, beta, num_points)

# Plot the sphere and the transported vector
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u = np.linspace(0, np.pi, 50)
v = np.linspace(0, 2 * np.pi, 50)
x_sphere = np.outer(np.sin(u), np.cos(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.cos(u), np.ones_like(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.2)

# Plot the curve gamma(t)
theta_values = np.full_like(phi_values, theta_0)  # Ensure theta is an array
x_curve, y_curve, z_curve = spherical_to_cartesian(1, theta_values, phi_values)
ax.plot(x_curve, y_curve, z_curve, color='red', label='Curve γ(t)')
print("x_curve shape:", np.shape(x_curve))
print("y_curve shape:", np.shape(y_curve))
print("z_curve shape:", np.shape(z_curve))
# Plot the transported vector n at selected points
step = 10  # Plot every 10th point
for i in range(0, num_points, step):
    point = np.array([x_curve[i], y_curve[i], z_curve[i]])
    n_vector = n_vectors[i]
    ax.quiver(*point, *n_vector, color='green', length=0.2, normalize=True, label='n(ϕ)' if i == 0 else "")

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parallel Transport Along a Circle of Constant Latitude')

# Add a legend
ax.legend()

# Save the plot as a PNG file
plt.savefig('/root/Desktop/host/HW2/P1_plots/parallel_transport_circle.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

##
#
#
#
#
# Problem 1g

# Compute the holonomy strength for different θ0 values
theta_0_values = np.linspace(0.1, np.pi / 2 - 0.1, 50)  # Avoid exactly 0 and π/2
inner_products = []

alpha = 1.0  # Initial vector component along e_theta
beta = 0.5  # Initial vector component along e_phi
num_points = 100  # Points along the loop

for theta_0 in theta_0_values:
    phi_values, n_vectors = parallel_transport_circle(theta_0, alpha, beta, num_points)
    
    # Extract initial and final vectors
    n_init = n_vectors[0]  
    n_final = n_vectors[-1]  

    # Compute the inner product
    inner_product = np.dot(n_init, n_final)
    inner_products.append(inner_product)

# Plot holonomy strength vs. θ0
plt.figure(figsize=(8, 6))
plt.plot(theta_0_values, inner_products, marker='o', linestyle='-')
plt.xlabel("theta_0 (Initial Latitude)")
plt.ylabel("Inner Product of Initial and Final Vector")
plt.title('Holonomy Strength vs. Latitude')
plt.grid(True)
plt.savefig('/root/Desktop/host/HW2/P1_plots/holonomy_strength_vs_latitude.png', dpi=300, bbox_inches='tight')
plt.show()
