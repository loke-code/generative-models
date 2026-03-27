***

# Variational Autoencoder (VAE) for Chest X-Ray Generation

## Overview

This is a PyTorch implementation of a Variational Autoencoder (VAE) designed for the generation of synthetic medical images, specifically Chest X-Rays. Utilizing the publicly available Kaggle "Chest X-Ray Images (Pneumonia)" dataset, the model is evaluated using quantitative metrics such as Fréchet Inception Distance (FID) and Inception Score (IS).



## Architecture Deep-Dive

### The Encoder
* **Deep Convolutional Network**: The encoder consists of six convolutional layers that gracefully downsample the input images. 
* **Feature Extraction**: It utilizes a stride of 2, a kernel size of 3x3, and padding of 1 across 4 layers, progressively increasing the channel dimensions ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 1024$) mapping the input to a $4\times4$ spatial dimension. 
* **Activation & Normalization**: Each convolution is followed by 2D Batch Normalization and a LeakyReLU activation function. The flattened feature map is then passed through fully connected layers to predict the latent space parameters (the mean $\mu$ and log-variance $\log(\sigma^2)$).

### The Latent Space
* **Bottleneck Dimension**: The latent space acts as an information bottleneck with a dimension of $z=512$.
* **Reparameterization Trick**: It employs the standard reparameterization trick ($z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0,I)$) to allow for backpropagation through the stochastic latent variables.

### The Decoder
* **Symmetrical Upsampling**: The decoder mirrors the encoder's topology, using transposed convolutions (`ConvTranspose2d`) to iteratively upsample the continuous latent vectors back to the original $64\times64\times3$ target space.
* **Output Formulation**: A final Sigmoid activation function ensures the output pixel intensities match the normalized continuous range of [0, 1].

## Model Training

* **Structural Similarity (SSIM) Loss**: The model's reconstruction loss is formulated as a weighted combination of SSIM and L1 absolute error (specifically $0.84 \times (1-SSIM) + 0.16 \times L1$). This helps the model prioritize perceptual structural features.
* **KL-Divergence Annealing**: A KL-annealing schedule is implemented to help the model focus on reducing the reconstruction loss early on. The weight of the KL-divergence penalty ($\beta$) is linearly increased from 0 to the target of 1 over the first 30 epochs.
* **Optimization**: The model is trained using the Adam optimizer with a learning rate of $1 \times 10^{-4}$ and a batch size of 64 over a total of 100 epochs.

## Generation Paradigms & Evaluation
The model's generative quality is evaluated across three different sampling paradigms:
1. **Reconstructed Images**: Generating images by decoding the latent distributions explicitly predicted for a set of known input images.
2. **Empirical Sampling**: Generating novel images by sampling from an empirical distribution (the calculated mean and standard deviation of the entire training dataset's latent mappings).
3. **Random Generation**: Synthesizing purely random variations by sampling directly from a Standard Normal distribution ($\mathcal{N}(0, I)$).
