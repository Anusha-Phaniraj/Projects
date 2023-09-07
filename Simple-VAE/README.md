# Variational Autoencoder (VAE) for MNIST Image Generation

This code demonstrates how to create a Variational Autoencoder (VAE) using PyTorch to generate images from the MNIST dataset.

### Data Loading
We download and prepare the MNIST dataset for training and testing. The images are transformed into tensors for processing.

### Model Architecture
The VAE has two main parts:

#### Encoder
- The encoder takes input images (28x28 pixels) and compresses them into a lower-dimensional latent space.
- It has two fully connected layers followed by activation functions.
- `self.fc1` reduces the input dimension to an intermediate representation.
- `self.fc2_mu` outputs the mean of the latent vector.
- `self.fc2_logvar` outputs the log variance of the latent vector.

#### Decoder
- The decoder reconstructs images from the latent vectors.
- It also has two fully connected layers.
- `self.fc3` expands the latent vector into an intermediate representation.
- `self.fc4` generates the final image output.

### Forward Pass
- The `forward` method combines the encoder and decoder.
- Input images are encoded to produce mean and log variance vectors.
- These vectors are then reparameterized into a latent vector.
- The latent vector is decoded to reconstruct an output image.

### Loss Function
The loss function has two components:

#### Reconstruction Loss
- Measures how well the generated image matches the input image.
- Calculated using binary cross-entropy.

#### KL Divergence
- Encourages the latent space to have a standard normal distribution.
- Analytically computed.

### Optimizer
- We use the Adam optimizer to update model parameters during training.

### Training Loop
- The VAE is trained for the specified number of epochs.
- In each epoch, the model is set to training mode.
- Training data is passed through the model, and loss is computed.
- Model parameters are updated through backpropagation.

### Testing Loop
- After training, the model is set to evaluation mode.
- We evaluate the model on a test dataset to measure its reconstruction accuracy.

### Generating Images
- Using the trained VAE, we generate new images by sampling random latent vectors.
- These random vectors are passed through the decoder to produce generated images.
- The generated images are displayed in a 5x5 grid using Matplotlib.

#### References : various sources across the internet
