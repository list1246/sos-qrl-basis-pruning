import sys
import numpy as np
from scipy.optimize import linprog

sys.path.append("..")

# Make sure this import correctly points to your project structure
from src.generator import SOSDataGenerator

def is_in_convex_hull(point, hull_points):
    """
    Check if a point is in the convex hull of a given set of points using linear programming.
    """
    n_points = hull_points.shape[0]
    c = np.zeros(n_points)
    A_eq = np.vstack([hull_points.T, np.ones(n_points)])
    b_eq = np.concatenate([point, [1]])
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
    return res.success

def evaluate_custom_poly_newton():
    num_vars = 2
    degree = 6
    gen = SOSDataGenerator(num_vars=num_vars, degree=degree)
    
    # 1. Define the support set based on the non-zero terms of your polynomial:
    # 7.0025*x1**4*x2**2 + 14.005*x1**3*x2 + 24.852*x1**2*x2**2 + 61.0299*x1**2 - 14.005*x1*x2 + 29.1754*x2**2
    support_exponents = [
        (4, 2),  # x1**4 * x2**2
        (3, 1),  # x1**3 * x2
        (2, 2),  # x1**2 * x2**2
        (2, 0),  # x1**2
        (1, 1),  # x1 * x2
        (0, 2)   # x2**2
    ]
    
    S = np.array(support_exponents)
    basis_monomials = np.array(gen.basis_monomials)
    
    print("\n" + "="*50)
    print("Newton Polytope Pruning for Custom Polynomial")
    print("Polynomial Support (Exponents):", support_exponents)
    print("="*50)
    
    # 2. Filter the basis
    retained_basis = []
    
    for beta in basis_monomials:
        target = 2 * beta
        if is_in_convex_hull(target, S):
            retained_basis.append(tuple(beta))
            
    # 3. Output Results
    print(f"Initial Basis Size : {len(basis_monomials)}")
    print(f"Final Basis Size   : {len(retained_basis)}")
    print("Final SOS Basis Elements (Newton Polytope):")
    
    for b in retained_basis:
        # Format exponents back to a readable string
        terms = []
        if b[0] > 0: terms.append(f"x1^{b[0]}" if b[0] > 1 else "x1")
        if b[1] > 0: terms.append(f"x2^{b[1]}" if b[1] > 1 else "x2")
        if not terms: terms.append("1")
        
        print(f"  - {' * '.join(terms):<10} (Tuple: {b})")
    print("="*50)

if __name__ == "__main__":
    evaluate_custom_poly_newton()