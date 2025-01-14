# Adversarial Attacks on YOLOv8


This repository provides implementations for testing adversarial attacks on a YOLOv8 model. It includes various attack methods such as **BIM**, **FGSM**, **EOT**, and **PGD**, and allows users to easily apply them to images and visualize their effects.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Available Attacks](#available-attacks)
  - [Example Commands](#example-commands)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

This project allows users to test different adversarial attacks (BIM, FGSM, EOT, and PGD) on a YOLOv8 model. The YOLOv8 model can be used for object detection tasks, and adversarial attacks are applied to check the robustness of the model to perturbations in the input data.

The project also includes the ability to visualize:
- The original image with YOLO predictions.
- The perturbation applied by the adversarial attack.
- The adversarial image with YOLO predictions.

## Installation

To get started with this repository, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/seba20-0/Adversarial-Attacks-on-YOLO.git
   cd Adversarial-Attacks-on-YOLO
   ```
2. **Install the required dependencies**:
   This repository requires torch, Pillow, ultralytics, and other libraries. You can install them using the following:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

### Available-Attacks
`BIM (Basic Iterative Method)`: This attack generates adversarial examples by iteratively applying small perturbations to the input image, constrained by an epsilon value.

`FGSM (Fast Gradient Sign Method)`: This attack computes the adversarial perturbation by using the gradient of the loss with respect to the input image.

`EOT (Expectation over Transformation)`: EOT performs adversarial attacks by averaging over different image transformations, such as rotation and resizing.

`PGD (Projected Gradient Descent)`: PGD is a more sophisticated version of the iterative attack method, where the perturbation is constrained within a given range and applied iteratively.


## Example Commands
---- working on it ----
## Dependencies
The project requires the following libraries:

`torch`: PyTorch framework for implementing the attacks and running YOLOv8.

`ultralytics`: For YOLOv8 model loading and inference.

`numpy`: Array operations.

`Pillow`: Image processing.

`matplotlib`: Visualization of results.

`roboflow`: for dataset.

You can install all dependencies with:
```bash
pip install -r requirements.txt
```
## License
This repository is licensed under the Apache License. See the [LICENSE](LICENSE) file for more details.
