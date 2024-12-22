import torch
import torch.nn as nn
class FGSM:
    def __init__(self , epsilon , targeted = False):
        """
        Fast Gradient Sign Method(FGSM) for generating adversarial examples.

        Args:
            epsilon (float): Maximum allowed perturbation.
            targeted (bool): Specifiy if the attack is targeted or not.
        """
        self.epsilon = epsilon
        self.targeted = targeted
    def attack(self , model , image , label):
        """
        Generates an adversarial example using

        Args:
            model (nn.Module): Target model to attack.
            image (torch.Tensor): Input image tensor.
            label (torch.Tensor): is the true label in the case of untargeted attack and the target label in the case of targeted attacks
        """
        image.requires_grad = True
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()  # Compute gradients with respect to input images
        grad_sign = image.grad.data.sign()
        if (self.targeted):
            perturbed_image = image - self.epsilon * grad_sign # Apply targeted attack

        else:
            perturbed_image = image + self.epsilon * grad_sign  # Apply untargeted attack
        
        # Clip to ensure valid pixel range (0-1 for normalized images)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
