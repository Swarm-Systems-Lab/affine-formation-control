import numpy as np
from numpy.linalg import lstsq

def adyacencia(B):
    """
    For each row in the matrix, find other rows that share a column with non-zero entries.
    Parameters:
    matrix (np.ndarray): A 2D NumPy array where each column has exactly two non-zero entries.
    Returns:
    dict: A dictionary where each key is a row index, and the value is a set of row indices
          that share at least one non-zero column with the key row.
    """
    shared_rows = {}
    # Find the non-zero column indices for each row
    non_zero_columns = [set(np.where(row != 0)[0]) for row in B]
    # For each row, find other rows with shared non-zero columns
    for i, cols in enumerate(non_zero_columns):
        shared_rows[i] = set()
        # Compare with all other rows
        for j, other_cols in enumerate(non_zero_columns):
            if i != j and cols & other_cols:  # Check for intersection in non-zero columns
                shared_rows[i].add(j) 
    return shared_rows

def compute_M(B, T_delta, p_star):
    """
    Computes the matrix M based on the incidence matrix B, target vector T_delta, 
    and the vector of desired positions p_star.
    Parameters:
    B (np.ndarray): Incidence matrix of size (n, m), where n is the number of vertices 
                    and m is the number of edges.
    T_delta (np.ndarray): T_delta = [b , A]; donde     
      b es el vector que determina la traslación y                                        
      A una matriz dXd
    p_star (np.ndarray): Vector of reference configuration p_star of size (d*n,). Each stacked.
    Returns:
    np.ndarray: Matrix M of size (n, m) with weights mu_ij.
    """
    n, m = B.shape  # n vertices, m edges
    #print(n,m)
    M = np.zeros((n, m))  # Initialize matrix M with zeros
    d = p_star.shape[0] // n
    #print(d, p_star.shape, n)
    p_star_matrix = np.reshape(p_star, (d, n), order = 'F')
    #print(p_star_matrix)
    neighbors = adyacencia(B)
    T = T_delta[1] @ p_star_matrix + np.reshape(T_delta[0], (d,1)) 
    #print("T es: ", T)
    for i in range(n):
        # A = np.kron(np.eye(d), B[:,np.where(B[0] != 0)]).T @ p_star
        # print(A)
        # Get the neighbors of vertex i by finding non-zero entries in the i-th row of B
        # Construct the target value for vertex i
        # Construct z_ij* values for each neighbor j of i
        z_star = p_star_matrix[:, i][:, np.newaxis] - p_star_matrix[:, list(neighbors[i])]
        #print(list[neighbors[i]])
        print("z_star", z_star, "Ti", T[:,i])
        #print(z_star)
        # Solve the system z_star * mu_i = T_i for weights mu_i
        # We use least squares in case the system is overdetermined or underdetermined
        mu_i, _, _, _ = lstsq(z_star, T[:,i], rcond=None)
        #print(z_star, T[:,i])
        #print(z_star @ mu_i, mu_i)
        #print(mu_i)
        # Fill in the weights for the neighbors in the i-th row of M
        for idx, j in enumerate(neighbors[i]):
            #print('hola', np.where(B[i] != 0)[0])
            idx2 = (set(np.where(B[i] != 0)[0]).intersection(set(np.where(B[j] != 0)[0]))).pop()
            #print(i,idx,j)
            M[i, idx2] = B[i, idx2] * mu_i[idx]
    return M, T

# Example usage:
# Define the incidence matrix B, target vector T_delta, and vector p_star
# These would need to be defined based on your specific graph structure and data.
B = np.array([
    [ 1, -1,  0,  1,  0,  1,  0,  0],
    [ 0,  1, -1,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  1,  0,  0,  0],
    [ 0,  0,  1, -1, -1,  0,  0,  1],
    [ 0,  0,  0,  0,  0,  0,  1, -1],
    [ 0,  0,  0,  0,  0, -1, -1,  0]
])

T_delta = [np.array([1,0]), 
           np.array([[1,1], [0,1]])
           ]

p_star = np.array([
    1,1,1,-1,-1,1,-1,-1,2,0,0,-2
])

# Compute the matrix M
M, T = compute_M(B, T_delta, p_star)
n, m = B.shape
d = p_star.shape[0] // n
#p_star_matrix = np.reshape(p_star, (d, m), order = 'F')
print("Computed matrix M:")
print(M)
# print( B @ B.T)
# print( M @ B.T )
print( (M @ B.T @ p_star[::2]))
print( (M @ B.T @ p_star[1::2]))
print( T )
# print( T_delta[1] @ p_star_matrix + np.reshape(T_delta[0], (d,1))  )