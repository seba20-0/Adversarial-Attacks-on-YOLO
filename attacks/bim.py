import torch
import torch.nn as nn

class BIM():
    def __init__(self , epsilon, alpha, num_iterations , targeted):
        """
        Basic Iterative Method (BIM) for generating adversarial examples.

        Args:
            epsilon (float): Maximum allowed perturbation.
            alpha (float): Step size for each iteration.
            num_iterations (int): Number of iterations to perform.
        """
        self.epsilon = epsilon
        self.alpha = alpha 
        self.num_iterations = num_iterations
        self.targeted = targeted
    def attack(self , model , image , label):
        """
        Generates an adversarial example using BIM.

        Args:
            model (nn.Module): Target model to attack.
            image (torch.Tensor): Input image tensor.
            label (torch.Tensor): True label tensor for the input image.

        Returns:
            torch.Tensor: Adversarial example tensor.
        """
        # Clone the input images to avoid modifying the original data
        perturbed_image = image.clone().detach()
        perturbed_image.requires_grad = True

        for _ in range(self.num_iterations):
            # Zero the gradients
            model.zero_grad()
            
            # Forward pass
            output = model(perturbed_image)
            
            # Compute the loss
            loss = nn.CrossEntropyLoss()(output, label)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Perform a step similar to FGSM
            grad_sign = perturbed_image.grad.data.sign()
            if self.targeted:
                perturbed_image = perturbed_image - self.alpha * grad_sign
            else:
                perturbed_image = perturbed_image + self.alpha * grad_sign
            
            # Clip the adversarial image to maintain the epsilon constraint [x - epsilon , x + epsilon] L infinty norm
            perturbed_image = torch.clamp(perturbed_image, image - self.epsilon, image + self.epsilon)
            
            # Clamp to ensure pixel values are within the valid range [0, 1]
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            
            # Detach gradients to avoid accumulation
            perturbed_image = perturbed_image.clone().detach()
            perturbed_image.requires_grad = True

        return perturbed_image