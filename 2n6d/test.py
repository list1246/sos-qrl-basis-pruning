import sys

sys.path.append("..")

import torch
import numpy as np

# Import your modules. Make sure the 'src' path is correctly mapped.
from src.generator import SOSDataGenerator
from src.agent import DoubleDQNAgentPER

def evaluate_custom_polynomial(model_path="./model/train.pth"):
    # Set up device to match your training environment
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 1. Initialize Generator (Match train.py settings: 2 variables, degree 6)
    num_vars = 2
    degree = 6
    generator = SOSDataGenerator(num_vars=num_vars, degree=degree)

    # 2. Map the target polynomial to a coefficient vector
    # Polynomial: 7.0025*x1**4*x2**2 + 14.005*x1**3*x2 + 24.852*x1**2*x2**2 + 61.0299*x1**2 - 14.005*x1*x2 + 29.1754*x2**2
    # Assuming x1 corresponds to the 0th index and x2 to the 1st index in exponent tuples.
    target_terms = {
        (4, 2): 7.0025,
        (3, 1): 14.005,
        (2, 2): 24.852,
        (2, 0): 61.0299,
        (1, 1): -14.005,
        (0, 2): 29.1754
    }

    # Initialize a blank array for all P(x) coefficients
    raw_coeffs = np.zeros(generator.coeff_dim, dtype=np.float32)
    
    # Fill in our specific polynomial coefficients
    for i, mon in enumerate(generator.poly_monomials):
        if mon in target_terms:
            raw_coeffs[i] = target_terms[mon]

    # 3. Preprocess State Data (Crucial step: matches env.py reset function)
    coeffs_t = torch.from_numpy(raw_coeffs).to(DEVICE)
    state_coeffs = torch.sign(coeffs_t) * torch.log1p(torch.abs(coeffs_t))
    
    # Initialize a full mask (all bases active)
    state_mask = torch.ones(generator.mask_dim, dtype=torch.long, device=DEVICE)

    # 4. Initialize Agent and Load Weights
    # Dimensions MUST match train.py (base_dim=512, embed_dim=16)
    agent = DoubleDQNAgentPER(generator.coeff_dim, generator.mask_dim, 
                              device=DEVICE, base_dim=512, embed_dim=16)

    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"[Info] Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"[Error] Model file '{model_path}' not found. Check the file name in your ./model/ folder.")
        return

    # Enter strict evaluation mode (0 exploration)
    agent.epsilon = 0.0
    agent.policy_net.eval()

    # 5. Execute Pruning Inference
    print("\n" + "="*50)
    print("Starting Pruning Inference...")
    print("="*50)

    step = 0
    while True:
        # Agent decides next action based on current state
        action = agent.select_action(state_coeffs, state_mask)

        # Action == mask_dim means the agent triggered the STOP condition
        if action == generator.mask_dim:
            print(f"-> Step {step}: Agent chose STOP.")
            break
        else:
            pruned_monomial = generator.basis_monomials[action]
            print(f"-> Step {step}: Pruned basis element {pruned_monomial}")
            state_mask[action] = 0
            step += 1
            
            # Safeguard: if everything is pruned
            if state_mask.sum() == 0:
                print("-> All bases have been pruned!")
                break

    # 6. Parse and display the final results
    # Get indices where the mask is still 1
    active_indices = torch.nonzero(state_mask).squeeze(-1).tolist()
    if isinstance(active_indices, int): 
        active_indices = [active_indices] # Handle single element case

    final_basis = [generator.basis_monomials[i] for i in active_indices]

    print("\n" + "="*50)
    print("Pruning Complete!")
    print(f"Initial Basis Size : {generator.mask_dim}")
    print(f"Final Basis Size   : {len(final_basis)}")
    print("Final SOS Basis Elements:")
    for b in final_basis:
        # Formatting (exp_x1, exp_x2) back to human-readable string
        terms = []
        if b[0] > 0: terms.append(f"x1^{b[0]}" if b[0] > 1 else "x1")
        if b[1] > 0: terms.append(f"x2^{b[1]}" if b[1] > 1 else "x2")
        if not terms: terms.append("1") # The constant term (0, 0)
        
        print(f"  - {' * '.join(terms):<10} (Tuple: {b})")
    print("="*50)


if __name__ == '__main__':
    # Defaulting to train.pth, adjust if your train.py generated a different name
    evaluate_custom_polynomial(model_path="./model/train.pth")