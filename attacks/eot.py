import torch
import torch.nn as nn
from torchvision import transforms
from .fgsm import FGSM

class EOT:
    def __init__(self, num_iterations=100, epsilon=0.01):
        """
        EOT (Expectation over Transformation) attack for generating adversarial examples.

        Args:
            num_iterations (int): Number of iterations to perform.
            epsilon (float): Maximum perturbation allowed for EOT.
        """
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        # Define the transformations here (RandomRotation, RandomResizedCrop, ColorJitter)
        self.transformations = transforms.Compose([
            transforms.RandomRotation(degrees=(-10, 10)),  # Random rotation between -10 and 10 degrees
            transforms.RandomResizedCrop(size=(416, 416), scale=(0.9, 1.1)),  # Random resized crop with scale
            transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Color jitter for brightness and contrast
        ])

        # Initialize the FGSM attack with the provided epsilon 
        self.fgsm_attack = FGSM(epsilon=self.epsilon)

    def attack(self, classify_model, image, labels):
        """
        Performs the EOT attack to generate adversarial examples.

        Args:
            classify_model (nn.Module): The model to attack.
            image (torch.Tensor): The input image tensor.
            labels (torch.Tensor): The true labels of the input image.

        Returns:
            torch.Tensor: The generated adversarial image.
        """
        image_unchanged = image.clone()  # Keep a copy of the original image
        image_unchanged = torch.clamp(image_unchanged, 0, 1)

        for iteration in range(self.num_iterations):
            image.requires_grad = True

            # Apply transformations to the image
            transformed_image = self.transformations(image.squeeze(0))  # Apply the transformation
            transformed_image = transformed_image.unsqueeze(0)

            # Get model outputs
            outputs = classify_model.classify_model(transformed_image)

            # Compute the loss and backpropagate
            classify_model.classify_model.zero_grad()
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            # Apply FGSM to get the adversarial perturbation using the FGSM object
            perturbed_image = self.fgsm_attack.attack(classify_model, transformed_image, labels)

            # Predict the class for the perturbed image
            pred = classify_model.classify_model(perturbed_image)
            _, predicted_classes = torch.max(pred, 1)

            # Check if the attack is successful (if the model misclassifies)
            if labels.item() == predicted_classes.item():
                print("YES")
                self.epsilon = -abs(self.epsilon)  # Flip epsilon to negative if the class is predicted
            else:
                self.epsilon = abs(self.epsilon)

            # Update image with the perturbed delta
            delta = perturbed_image - image
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            image = (image + delta).detach()

            # Print loss every 20 iterations
            if iteration % 20 == 0:
                print(f'Iteration {iteration}, Loss: {loss.item()}')

        return image
