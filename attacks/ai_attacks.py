# attacks/ai_attacks.py

import torch
import torch.nn as nn
import torchvision.models as models
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
import sys

# Add the parent directory to the path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_image_tensor, save_tensor_image, DEVICE
from src.metrics import compute_psnr_ssim
from src.extraction import extract_and_verify_watermark

# ------------------------------
# Model and Loss Setup
# ------------------------------
# Load a pre-trained model for the adversarial attack (e.g., ResNet18)
try:
    MODEL = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
except AttributeError:
    # Fallback for systems with older torchvision/pip versions
    MODEL = models.resnet18(pretrained=True).to(DEVICE)
MODEL.eval()
LOSS_FN = nn.CrossEntropyLoss()

# ------------------------------
# Adversarial Attacks
# ------------------------------

def fgsm_attack(model: nn.Module, x: torch.Tensor, target_label_tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Performs the Fast Gradient Sign Method (FGSM) attack.

    Args:
        model: The target neural network model.
        x: The watermarked image tensor (input).
        target_label_tensor: The target class label for the loss calculation.
        epsilon: The magnitude of the adversarial perturbation.

    Returns:
        The adversarial image tensor.
    """
    x_adv = x.clone().detach().to(DEVICE)
    x_adv.requires_grad = True
    
    out = model(x_adv)
    loss = LOSS_FN(out, target_label_tensor)
    
    model.zero_grad()
    loss.backward()
    
    grad = x_adv.grad.data
    # Apply sign-based perturbation
    x_adv = x_adv + epsilon * grad.sign()
    
    # Clip the result to maintain valid image range [0, 1]
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()

def pgd_attack(model: nn.Module, x: torch.Tensor, target_label_tensor: torch.Tensor, epsilon: float, alpha: float, iters: int) -> torch.Tensor:
    """
    Performs the Projected Gradient Descent (PGD) attack (iterative FGSM).

    Args:
        model: The target neural network model.
        x: The watermarked image tensor (input).
        target_label_tensor: The target class label.
        epsilon: The maximum L_inf norm perturbation boundary.
        alpha: The step size for each iteration.
        iters: The number of iterations.

    Returns:
        The adversarial image tensor.
    """
    x_orig = x.clone().detach().to(DEVICE)
    x_adv = x_orig.clone().detach() # Initialize adversarial example

    for _ in range(iters):
        x_adv.requires_grad = True
        out = model(x_adv)
        loss = LOSS_FN(out, target_label_tensor)
        
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        
        # Take a step
        x_adv = x_adv + alpha * grad.sign()
        
        # Project the perturbation back onto the L_inf ball (epsilon radius)
        # This keeps the adversarial example within [x_orig - epsilon, x_orig + epsilon]
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        
        # Clip to maintain valid image range [0, 1] and detach for next iteration
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        
    return x_adv

# ------------------------------
# Main Evaluation
# ------------------------------

def run_attack_evaluation(watermarked_path: str,
                          original_bits: np.ndarray,
                          output_dir: str = 'adv_out',
                          target_class: int = 0, # Default target class (e.g., first class in ImageNet)
                          fgsm_epsilons: List[float] = [2/255., 4/255., 8/255.],
                          pgd_epsilons: List[float] = [2/255., 4/255., 8/255.],
                          pgd_alpha: float = 1/255.,
                          pgd_iters: int = 20) -> List[Dict[str, Any]]:
    """
    Runs FGSM and PGD attacks on the watermarked image and evaluates watermark robustness.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load watermarked image as tensor (resized to model's input size)
    x = load_image_tensor(watermarked_path, size=(224, 224))
    
    # Define a generic target label (e.g., class 0) for the untargeted attack
    # An untargeted attack maximizes loss on the correct class. Here, we use target_class as the input label.
    label = torch.tensor([target_class], dtype=torch.long).to(DEVICE)

    results: List[Dict[str, Any]] = []

    # --- 1. FGSM Attack ---
    print(f"\n--- Running FGSM Attacks on {watermarked_path} ---")
    for eps in fgsm_epsilons:
        eps_val_int = int(eps * 255)
        print(f" -> FGSM with epsilon = {eps_val_int}/255...")
        
        x_adv = fgsm_attack(MODEL, x, label, epsilon=eps)
        out_file = os.path.join(output_dir, f'fgsm_eps{eps_val_int}.png')
        save_tensor_image(x_adv, out_file)

        # Image Quality Metrics (between watermarked and attacked image)
        psnr_v, ssim_v = compute_psnr_ssim(x, x_adv)
        
        # Watermark Robustness Metrics
        extracted_bits, ber, nc_bin, acc = extract_and_verify_watermark(out_file, original_bits)

        results.append({
            'attack': 'fgsm', 'epsilon': float(eps),
            'eps_255': eps_val_int,
            'psnr': psnr_v, 'ssim': ssim_v,
            'ber': ber, 'nc_binary': nc_bin, 'extraction_acc': acc,
            'out_file': out_file
        })

    # --- 2. PGD Attack ---
    print(f"\n--- Running PGD Attacks on {watermarked_path} ---")
    for eps in pgd_epsilons:
        eps_val_int = int(eps * 255)
        print(f" -> PGD with epsilon = {eps_val_int}/255...")

        x_adv = pgd_attack(MODEL, x, label, epsilon=eps, alpha=pgd_alpha, iters=pgd_iters)
        out_file = os.path.join(output_dir, f'pgd_eps{eps_val_int}_it{pgd_iters}.png')
        save_tensor_image(x_adv, out_file)

        # Image Quality Metrics
        psnr_v, ssim_v = compute_psnr_ssim(x, x_adv)
        
        # Watermark Robustness Metrics
        extracted_bits, ber, nc_bin, acc = extract_and_verify_watermark(out_file, original_bits)

        results.append({
            'attack': 'pgd', 'epsilon': float(eps),
            'eps_255': eps_val_int,
            'psnr': psnr_v, 'ssim': ssim_v,
            'ber': ber, 'nc_binary': nc_bin, 'extraction_acc': acc,
            'out_file': out_file
        })

    return results

# ------------------------------
# Example Run
# ------------------------------
if __name__ == "__main__":
    # --- IMPORTANT ---
    # 1. Ensure 'watermarked_medical.png' and 'watermark_data.npz' exist in the root directory.
    # 2. 'watermark_data.npz' must contain 'watermark_bits' and 'lh2_patterns'/'hl2_patterns'.
    
    try:
        watermarked_image_path = "../watermarked_medical.png" 
        npz_path = "../watermark_data.npz"
        
        if not os.path.exists(npz_path):
            print(f"Error: Required file '{npz_path}' not found. Please run the embedding process first.")
            sys.exit(1)

        wm_data = np.load(npz_path, allow_pickle=True)
        original_bits = wm_data['watermark_bits']
        
        # Define attack parameters in terms of L_inf norm (e.g., 2, 4, 8 out of 255)
        epsilons_255 = [2, 4, 8]
        fgsm_epsilons = [e / 255. for e in epsilons_255]
        pgd_epsilons = [e / 255. for e in epsilons_255]
        pgd_alpha = 1 / 255. # Step size for PGD

        results = run_attack_evaluation(
            watermarked_image_path, 
            original_bits, 
            output_dir='../Results/adv_results',
            fgsm_epsilons=fgsm_epsilons,
            pgd_epsilons=pgd_epsilons,
            pgd_alpha=pgd_alpha
        )
        
        # Display and save results
        df = pd.DataFrame(results)
        print("\n--- Adversarial Attack Evaluation Summary ---")
        print(df.to_string())
        
        output_csv = '../Results/adv_results/summary.csv'
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")