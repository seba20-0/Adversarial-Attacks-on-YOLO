import torch
import torch.nn as nn

class PGD:
    def __init__(self, epsilon, alpha, num_iterations):
        """
        Projected Gradient Descent (PGD) for generating adversarial examples.

        Args:
            epsilon (float): Maximum allowed perturbation.
            alpha (float): Step size for each attack iteration.
            num_iterations (int): Number of attack steps.
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iterations = num_iterations

    def attack(self, model, image, label):
        """
        Generates adversarial examples using PGD.

        Args:
            model (nn.Module): Target model to attack.
            image (torch.Tensor): Input image tensor.
            label (torch.Tensor): True label (for untargeted attack) or target label (for targeted attack).

        Returns:
            torch.Tensor: Adversarial image.
        """
        model = model.classify_model
        # Clone the input image to create adversarial examples
        perturbed_image = image.clone().detach()

        # random initialization
        perturbed_image += torch.empty_like(image).uniform_(-self.epsilon, self.epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        for _ in range(self.num_iterations):
            # Enable gradient computation for the perturbed image
            perturbed_image.requires_grad = True
            output = model(perturbed_image)

            # Compute the loss
            loss = nn.CrossEntropyLoss()(output, label)


            # Compute gradients with respect to the input
            model.zero_grad()
            loss.backward()
            grad_sign = perturbed_image.grad.data.sign()

            # Update the adversarial image
            perturbed_image = perturbed_image + self.alpha * grad_sign
            perturbed_image = torch.clamp(perturbed_image, image - self.epsilon, image + self.epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()  # Ensure valid pixel range and detach gradients

        return perturbed_image
