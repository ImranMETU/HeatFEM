import numpy as np
import matplotlib.pyplot as plt

def compute_ke_rect(kx, ky, a, b, t):
    ke = np.zeros((4, 4))
    
    ke[0, 0] = t * (kx * b / (3 * a) + ky * a / (3 * b))
    ke[0, 1] = t * (-kx * b / (3 * a) + ky * a / (6 * b))
    ke[0, 2] = t * (-kx * b / (6 * a) - ky * a / (6 * b))
    ke[0, 3] = t * (-kx * b / (6 * a) - ky * a / (3 * b))

    ke[1, 0] = ke[0, 1]
    ke[1, 1] = ke[0, 0]
    ke[1, 2] = t * (kx * b / (6 * a) - ky * a / (3 * b))
    ke[1, 3] = ke[0, 2]

    ke[2, 0] = ke[0, 2]
    ke[2, 1] = ke[1, 2]
    ke[2, 2] = ke[0, 0]
    ke[2, 3] = ke[0, 1]

    ke[3, 0] = ke[0, 3]
    ke[3, 1] = ke[1, 3]
    ke[3, 2] = ke[2, 3]
    ke[3, 3] = ke[0, 0]

    return ke

def compute_element_matrices(element_nodes, k, Q, t):
    # For rectangular elements, compute dimensions
    x_coords = element_nodes[:, 0]
    y_coords = element_nodes[:, 1]
    
    a = (np.max(x_coords) - np.min(x_coords)) / 2  # Half-width
    b = (np.max(y_coords) - np.min(y_coords)) / 2  # Half-height
    
    # Compute element stiffness matrix
    ke = compute_ke_rect(k, k, a, b, t)  # Assuming isotropic material (kx = ky = k)
    
    # Compute element force vector (for uniform heat generation)
    area = 4 * a * b  # Element area
    fe = np.ones(4) * Q * area / 4  # Distribute heat generation equally to nodes
    
    return ke, fe

# Define geometry
Lx = 4
Ly = 2
t = 0.1
k = 40.0
Q = 50.0

# Mesh
nx = 2
ny = 2
dx = Lx / nx
dy = Ly / ny

# Generate nodal coordinates
nodes = []
for j in range(ny+1):
    for i in range(nx+1):
        nodes.append([i * dx, j * dy])
nodes = np.array(nodes)

# Element connectivity
elements = [
    [1, 2, 5, 4],
    [2, 3, 6, 5],
    [4, 5, 8, 7],
    [5, 6, 9, 8]
]
elements = np.array(elements) - 1  # Zero-based Python indexing

# Initialize global K and F
ndof = nodes.shape[0]
K = np.zeros((ndof, ndof))
F = np.zeros(ndof)

# Loop over elements
for e in elements:
    ke, fe = compute_element_matrices(nodes[e, :], k, Q, t)
    for i in range(4):
        for j in range(4):
            K[e[i], e[j]] += ke[i, j]
        F[e[i]] += fe[i]

# Apply Dirichlet BCs
fixed_nodes = [0, 3, 6, 7, 8]
T = np.zeros(ndof)

free_nodes = list(set(range(ndof)) - set(fixed_nodes))
K_ff = K[np.ix_(free_nodes, free_nodes)]
F_f = F[free_nodes]

# Solve for free DOFs
T_f = np.linalg.solve(K_ff, F_f)
T[free_nodes] = T_f

# Print nodal temperatures
for i, Ti in enumerate(T):
    print(f"Node {i+1}: T = {Ti:.4f} K")

# Optional: contour plot
plt.scatter(nodes[:,0], nodes[:,1], c=T, cmap='hot')
plt.colorbar(label='Temperature (K)')
plt.show()
