# Generative Adversarial Network (GAN) for MNIST Image Generation

This code demonstrates how to implement a simple Generative Adversarial Network (GAN) using PyTorch to generate images resembling handwritten digits from the MNIST dataset.

### Data Loading
Download and prepare the MNIST dataset for training and testing. The images are transformed into tensors for processing.

### Generator Model
- The generator's role is to create fake images.
- It consists of fully connected layers with ReLU activation functions and a final layer with a tanh activation to generate images.
- The generator takes random noise as input and produces images as output.

### Discriminator Model
- The discriminator's role is to distinguish real images from fake ones.
- It consists of fully connected layers with ReLU activation functions and a final layer with a sigmoid activation to produce probability scores.
- The discriminator takes images (real or fake) as input and outputs a probability indicating whether the input is real or fake.

### Loss Function
- The loss function for both the generator and discriminator is binary cross-entropy.
- The discriminator aims to maximize the probability of correctly classifying real and fake images.
- The generator aims to minimize the probability of its generated images being classified as fake.

### Optimizers
- Adam optimizer is used for both the generator and discriminator.

### Training Loop
- In each epoch, train the discriminator and generator alternately.
- The discriminator is trained to distinguish real and fake images.
- The generator is trained to create fake images that fool the discriminator.
- Calculate and print the losses for both models during training.

### Generating Images
- After training, use the generator to create new images.
- Sample random noise vectors from a standard normal distribution.
- These noise vectors are input to the generator, which produces fake images.
- The generated images are displayed in a grid using Matplotlib.

#### References : various sources across the internet
