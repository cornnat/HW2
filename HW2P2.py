import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Problem 2
## Part A
# Crediting shane tobin and deepseek for help

# Define the stereographic projection
def stereographic_projection(x, y, z):
    """
    Compute the stereographic projection of a point (x, y, z) on the unit sphere.
    
    Parameters:
        x (float or np.ndarray): x-coordinate(s) of the point(s) on the sphere.
        y (float or np.ndarray): y-coordinate(s) of the point(s) on the sphere.
        z (float or np.ndarray): z-coordinate(s) of the point(s) on the sphere.
        
    Returns:
        np.ndarray: The projected 2D coordinates as an array of shape (2, ...).
    """
    denom = 1 - z
    return x / denom, y / denom

def generate_curves_on_sphere(theta_p, phi_p):
    """
    Generate two orthogonal curves on the unit sphere at a given point P.

    Parameters:
    theta_p (float): Polar angle (0 ≤ θ ≤ π)
    phi_p (float): Azimuthal angle (0 ≤ φ < 2π)

    Returns:
    tuple: Two curves on the sphere and their tangent vectors.
    """
    # Define the point P on the sphere
    x_p = np.sin(theta_p) * np.cos(phi_p)
    y_p = np.sin(theta_p) * np.sin(phi_p)
    z_p = np.cos(theta_p)
    P = np.array([x_p, y_p, z_p])

    # Compute orthonormal tangent vectors at P
    e_theta = np.array([
        np.cos(theta_p) * np.cos(phi_p),
        np.cos(theta_p) * np.sin(phi_p),
        -np.sin(theta_p)
    ])
    e_phi = np.array([-np.sin(phi_p), np.cos(phi_p), 0])
    e_phi_unit = e_phi / np.sin(theta_p)

    # Parameter t for generating curves
    t = np.linspace(-0.2, 0.2, 100)

    # Generate curve along e_theta
    gamma1 = np.array([P + ti * e_theta for ti in t])
    gamma1 = gamma1 / np.linalg.norm(gamma1, axis=1)[:, np.newaxis]

    # Generate curve along e_phi_unit
    gamma2 = np.array([P + ti * e_phi_unit for ti in t])
    gamma2 = gamma2 / np.linalg.norm(gamma2, axis=1)[:, np.newaxis]

    return gamma1, gamma2, e_theta, e_phi_unit, P

def plot_curves_and_projection(gamma1, gamma2, e_theta, e_phi_unit, P):
    """
    Plot the curves on the unit sphere and their stereographic projection.

    Parameters:
    gamma1 (np.ndarray): First curve on the sphere.
    gamma2 (np.ndarray): Second curve on the sphere.
    e_theta (np.ndarray): Tangent vector along the θ-direction.
    e_phi_unit (np.ndarray): Tangent vector along the φ-direction.
    P (np.ndarray): Point of intersection on the sphere.
    """
    # Project the curves
    gamma1_proj = stereographic_projection(gamma1[:, 0], gamma1[:, 1], gamma1[:, 2])
    gamma2_proj = stereographic_projection(gamma2[:, 0], gamma2[:, 1], gamma2[:, 2])

    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 6))

    # 3D plot of the sphere
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gamma1[:, 0], gamma1[:, 1], gamma1[:, 2], color='purple', label='Curve 1 (θ-direction)')
    ax1.plot(gamma2[:, 0], gamma2[:, 1], gamma2[:, 2], color='orange', label='Curve 2 (φ-direction)')
    ax1.scatter(P[0], P[1], P[2], color='red', s=100, label='Point P')
    ax1.quiver(P[0], P[1], P[2], e_theta[0], e_theta[1], e_theta[2], color='purple', length=0.2, label='Tangent 1 (θ)')
    ax1.quiver(P[0], P[1], P[2], e_phi_unit[0], e_phi_unit[1], e_phi_unit[2], color='orange', length=0.2, label='Tangent 2 (φ)')
    ax1.set_title('Unit Sphere with Curves and Tangent Vectors')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=45)  # Adjust viewing angle

    # 2D plot of the projection
    ax2 = fig.add_subplot(122)
    ax2.plot(gamma1_proj[0], gamma1_proj[1], color='purple', label='Projected Curve 1')
    ax2.plot(gamma2_proj[0], gamma2_proj[1], color='orange', label='Projected Curve 2')
    ax2.scatter(*stereographic_projection(P[0], P[1], P[2]), color='red', s=100, label='Projected P')

    # Compute projected tangent vectors
    dS_e_theta = np.array([gamma1_proj[0][1] - gamma1_proj[0][0], gamma1_proj[1][1] - gamma1_proj[1][0]])
    dS_e_phi = np.array([gamma2_proj[0][1] - gamma2_proj[0][0], gamma2_proj[1][1] - gamma2_proj[1][0]])
    dS_e_theta = dS_e_theta / np.linalg.norm(dS_e_theta) * 0.5  # Scale for visualization
    dS_e_phi = dS_e_phi / np.linalg.norm(dS_e_phi) * 0.5  # Scale for visualization

    ax2.quiver(*stereographic_projection(P[0], P[1], P[2]), dS_e_theta[0], dS_e_theta[1], color='purple', scale=1, scale_units='xy', angles='xy', label='Projected Tangent 1')
    ax2.quiver(*stereographic_projection(P[0], P[1], P[2]), dS_e_phi[0], dS_e_phi[1], color='orange', scale=1, scale_units='xy', angles='xy', label='Projected Tangent 2')
    ax2.set_title('Stereographic Projection of Curves and Tangent Vectors')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.axis('equal')

    # Annotate angles
    ax1.text(P[0], P[1], P[2], '90°', color='black', fontsize=12, ha='center', va='center')
    ax2.text(stereographic_projection(P[0], P[1], P[2])[0], stereographic_projection(P[0], P[1], P[2])[1], '90°', color='black', fontsize=12, ha='center', va='center')

    plt.tight_layout()
    plt.savefig("/root/Desktop/host/HW2/P2_plots/stereographic_projection.png")
    plt.show()

# Example usage with angles
theta_p = np.pi / 4  # Polar angle
phi_p = np.pi / 3    # Azimuthal angle
gamma1, gamma2, e_theta, e_phi_unit, P = generate_curves_on_sphere(theta_p, phi_p)
plot_curves_and_projection(gamma1, gamma2, e_theta, e_phi_unit, P)


## Part B
def generate_great_circles():
    """
    Generate great circles on the unit sphere.

    Returns:
    tuple: Two great circles on the sphere.
    """
    # Parameter t for generating curves
    t = np.linspace(0, 2 * np.pi, 100)

    # Great circle 1: In the x-y plane
    gamma1_x = np.cos(t)
    gamma1_y = np.sin(t)
    gamma1_z = np.zeros_like(t)

    # Great circle 2: In the x-z plane
    gamma2_x = np.cos(t)
    gamma2_y = np.zeros_like(t)
    gamma2_z = np.sin(t)

    # Great circle 3: In the y-z plane
    gamma3_x = np.zeros_like(t)
    gamma3_y = np.cos(t)
    gamma3_z = np.sin(t)

    return (gamma1_x, gamma1_y, gamma1_z), (gamma2_x, gamma2_y, gamma2_z), (gamma3_x, gamma3_y, gamma3_z)

def plot_great_circles_and_projection(gamma1, gamma2, gamma3):
    """
    Plot great circles on the unit sphere and their stereographic projection.

    Parameters:
    gamma1 (tuple): First great circle on the sphere.
    gamma2 (tuple): Second great circle on the sphere.
    gamma3 (tuple): Third great circle on the sphere.
    """
    # Project the great circles
    gamma1_proj = stereographic_projection(gamma1[0], gamma1[1], gamma1[2])
    gamma2_proj = stereographic_projection(gamma2[0], gamma2[1], gamma2[2])
    gamma3_proj = stereographic_projection(gamma3[0], gamma3[1], gamma3[2])

    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 6))

    # 3D plot of the sphere
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gamma1[0], gamma1[1], gamma1[2], color='blue', label='Great Circle 1 (x-y plane)')
    ax1.plot(gamma2[0], gamma2[1], gamma2[2], color='green', label='Great Circle 2 (x-z plane)')
    ax1.plot(gamma3[0], gamma3[1], gamma3[2], color='red', label='Great Circle 3 (y-z plane)')
    ax1.set_title('Great Circles on the Unit Sphere')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=45)  # Adjust viewing angle

    # 2D plot of the projection
    ax2 = fig.add_subplot(122)
    ax2.plot(gamma1_proj[0], gamma1_proj[1], color='blue', label='Projected Great Circle 1')
    ax2.plot(gamma2_proj[0], gamma2_proj[1], color='green', label='Projected Great Circle 2')
    ax2.plot(gamma3_proj[0], gamma3_proj[1], color='red', label='Projected Great Circle 3')
    ax2.set_title('Stereographic Projection of Great Circles')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig("/root/Desktop/host/HW2/P2_plots/great_circles_projection.png")
    plt.show()

# Generate and plot great circles
gamma1, gamma2, gamma3 = generate_great_circles()
plot_great_circles_and_projection(gamma1, gamma2, gamma3)

## Part C
# Problem 2: Parallel Transport Trajectories under Stereographic Projection

from HW2P1 import parallel_transport_circle, spherical_to_cartesian

def plot_parallel_transport_trajectories(theta_0, alpha, beta, num_points=100):
    """
    Plot parallel transport trajectories of a closed loop for various initial vectors
    under the stereographic projection.

    Parameters:
    - theta_0: Polar angle (constant latitude).
    - alpha, beta: Components of the initial vector n.
    - num_points: Number of points along the curve.
    """
    # Perform parallel transport
    phi_values, n_vectors = parallel_transport_circle(theta_0, alpha, beta, num_points)

    # Convert the curve and transported vectors to Cartesian coordinates
    theta_values = np.full_like(phi_values, theta_0)  # Ensure theta is an array
    x_curve, y_curve, z_curve = spherical_to_cartesian(1, theta_values, phi_values)

    # Project the curve and transported vectors using stereographic projection
    curve_proj = stereographic_projection(x_curve, y_curve, z_curve)
    n_vectors_proj = np.array([stereographic_projection(*n_vectors[i]) for i in range(num_points)])

    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 6))

    # 3D plot of the sphere
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_curve, y_curve, z_curve, color='red', label='Curve γ(t)')

    # Plot the transported vectors on the sphere
    step = 5  # Plot every 10th vector
    for i in range(0, num_points, step):
        point = np.array([x_curve[i], y_curve[i], z_curve[i]])
        n_vector = n_vectors[i]
        ax1.quiver(*point, *n_vector, color='green', length=0.05, normalize=True, label='n(ϕ)' if i == 0 else "")

    ax1.set_title('Parallel Transport on the Unit Sphere')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=45)  # Adjust viewing angle

    # 2D plot of the stereographic projection
    ax2 = fig.add_subplot(122)
    ax2.plot(curve_proj[0], curve_proj[1], color='red', label='Projected Curve γ(t)')

    # Plot the projected transported vectors
    for i in range(0, num_points, step):
        ax2.quiver(curve_proj[0][i], curve_proj[1][i], n_vectors_proj[i][0], n_vectors_proj[i][1], color='green', scale=1, scale_units='xy', angles='xy', label='Projected n(ϕ)' if i == 0 else "")

    ax2.set_title('Stereographic Projection of Parallel Transport')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig("/root/Desktop/host/HW2/P2_plots/parallel_transport_trajectories.png")
    plt.show()

# Example usage
theta_0 = np.pi / 5  # Constant polar angle (latitude)
alpha = 1.0  # Component along e_theta
beta = 0.5  # Component along e_phi
plot_parallel_transport_trajectories(theta_0, alpha, beta)

#
#
# Part D
def compute_inner_products(e_theta, e_phi_unit, P):
    """
    Compute inner products of tangent vectors before and after stereographic projection.

    Parameters:
    e_theta (np.ndarray): Tangent vector along θ.
    e_phi_unit (np.ndarray): Tangent vector along φ.
    P (np.ndarray): Point of intersection on the sphere.

    Returns:
    tuple: Inner product on the sphere, inner product after projection.
    """
    # Inner product on the sphere (should be close to zero for orthogonal vectors)
    inner_product_sphere = np.dot(e_theta, e_phi_unit)

    # Project tangent vectors using stereographic projection
    P_proj = np.array(stereographic_projection(*P))
    dS_e_theta = np.array(stereographic_projection(*(P + e_theta))) - P_proj
    dS_e_phi = np.array(stereographic_projection(*(P + e_phi_unit))) - P_proj

    # Compute the inner product in the stereographic projection plane
    inner_product_proj = np.dot(dS_e_theta, dS_e_phi)

    return inner_product_sphere, inner_product_proj

def plot_inner_product_comparison(theta_p, phi_p):
    """
    Plot the inner products before and after stereographic projection.
    
    Parameters:
    theta_p (float): Polar angle of point P on the sphere.
    phi_p (float): Azimuthal angle of point P on the sphere.
    """
    # Generate curves and tangent vectors
    _, _, e_theta, e_phi_unit, P = generate_curves_on_sphere(theta_p, phi_p)

    # Compute inner products
    inner_sphere, inner_proj = compute_inner_products(e_theta, e_phi_unit, P)

    # Create bar plot to compare inner products
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['On Sphere', 'After Projection']
    values = [inner_sphere, inner_proj]
    ax.bar(labels, values, color=['blue', 'red'])

    ax.set_title("Inner Product Before and After Projection")
    ax.set_ylabel("Inner Product Value")
    plt.ylim(-0.1, 0.1)  # Set limits for better visualization
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig("/root/Desktop/host/HW2/P2_plots/inner_product_comparison.png")
    plt.show()

# Example usage
plot_inner_product_comparison(np.pi / 4, np.pi / 3)

## the stereographic projection preserves the inner product between two vectors
## It is a conformal transformation so the lengths and shapes of the curves may change, but the local geometry/angles are preserved

#
#
# Part F
# Can the stereographic projection alter the holonomy on the unit sphere
# when parallel transport (use one loop, 0 → 2π)

# this question is worded weirdly. Holonomy around the closed loop on the unit sphere is trivial,
# i.e. the vector is not changed after completing the 0 to 2pi transport. The holonomy of stereographically
# projected parallel transported vector is also trivial, not only because the vector returns to its
# original coordinate, but there is zero holonomy/no rotation because the plane is flat.

