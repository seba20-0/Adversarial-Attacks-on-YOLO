import torch.nn as nn
class FGSM:
    def __init__(self , epsilon , targeted = False):
        self.epsilon = epsilon
        self.targeted = targeted
    def attack(self , model , image , output , label):
        image.requires_grad = True
        outputs = model(image)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()  # Compute gradients with respect to input images
        grad_sign = image.grad.data.sign()
        if (self.targeted):
            perturbed_image = image - self.epsilon * grad_sign # Apply targeted attack

        perturbed_image = image + self.epsilon * grad_sign  # Apply untargeted attack
        
        return perturbed_image
