# Emoji Representation Learning with Convolutional Autoencoders
A Deep Learning Approach to Emoji Reconstruction, Classification, and Latent Space Manipulation

This project implements an autoencoder in PyTorch, trained on a subset of the Huggingface Emoji Dataset. The goal is to learn a compact latent representation of emoji images using reconstruction loss (MSE), and extend the model to support image classification as an auxiliary task. We also explore latent space manipulation through vector arithmetic for image composition.

## Project Overview
Autoencoders are powerful unsupervised learning models that compress and reconstruct input data. I designed and trained a convolutional autoencoder to reconstruct emoji images and later extended it to perform multi-task learning with an auxiliary classification head. Finally, I explored latent space composition to generate novel emoji images by combining features from multiple images (e.g., adding glasses from a "nerd face" to a "smiling face").

## Dataset
- Source: **Huggingface** Emoji Dataset
- Selected subset: Emoji descriptions containing "face" (e.g., happy face, nerd face, angry face, etc.)
- Total processed data:
  - ~600 training images
  - ~200 validation images
  - ~200 test images
- Image size: Rescaled to 64x64x3
- Preprocessing:
  - Data augmentation (flips, rotations, etc.)
  - Normalization
  - Labeling based on emoji descriptions (e.g., happy, angry, neutral)

## Key Learnings
- Implemented convolutional autoencoders in PyTorch
- Balanced reconstruction and classification losses
- Understood image latent representations
- Performed vector arithmetic in latent space
- Explored practical applications of multi-task learning
