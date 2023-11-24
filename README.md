# ArtisticNeuralStyleTransfer
Assignment
Certainly! Below is a brief overview of each function and section in Markdown format:

```markdown
# Neural Style Transfer Readme

## Overview

This repository contains code for performing Neural Style Transfer using PyTorch. Neural Style Transfer is a technique that blends the content of one image with the style of another, creating visually appealing results.

## Setup

### Requirements

- Python 3.x
- PyTorch
- Matplotlib
- Requests
- NumPy

Install the required dependencies using:

```bash
pip install torch matplotlib requests numpy
```

## Usage

1. **Load Images:**
   - Use the `load_image` function to load and preprocess images from file paths or URLs.

   ```python
   content = load_image('/content/golden_gate.jpg').to(device)
   style = load_image('/content/starry_night.jpg', shape=content.shape[-2:]).to(device)
   ```

2. **Display Images:**
   - Utilize the `show_image` and `display_images` functions to visualize images.

   ```python
   content_image = show_image(content)
   style_image = show_image(style)
   display_images(content_image, style_image)
   ```
# Image Comparison

## Content Image

<img src="./Input/golden_gate.jpg" alt="Content Image" width="50%">

## Style Image

<img src="./Input/starry_night.jpg" alt="Another Image" width="50%">


3. **VGG Model Setup:**
   - Load the VGG19 model with pre-trained weights and set to evaluation mode.

   ```python
   vgg = models.vgg19(pretrained=True).features
   for param in vgg.parameters():
       param.requires_grad_(False)
   vgg.to(device)
   ```

4. **Feature Extraction:**
   - Extract features from content and style images using the VGG model.

   ```python
   content_features = get_features(content, vgg)
   style_features = get_features(style, vgg)
   ```

5. **Style Transfer Optimization:**
   - Configure parameters and set up the Adam optimizer for optimization.

   ```python
   show_every = 800
   steps = 40000
   losses = {'content': [], 'style' : [], 'total' : []}
   optimizer = optim.Adam([target], lr=0.003)
   ```

6. **Optimization Loop:**
   - Perform style transfer optimization in a loop, updating the target image.

   ```python
   for i in range(1, steps+1):
       # ... (Refer to the provided code for the optimization loop)
   ```

7. **Visualization:**
   - Visualize intermediate results at specified intervals.

   ```python
   if i % show_every == 0:
       print('Iteration {}: Total Loss = {:.2f}'.format(i, total_loss.item()))
       plt.imshow(show_image(target))
       plt.show()
   ```

## Result Image

<img src="./Output/Result.jpg" alt="Another Image" width="50%">


## Parameters

- Adjust the parameters such as `show_every`, `steps`, `content_weight`, and `style_weight` based on your specific requirements.

## Notes

- Experiment with different content and style images, and tune parameters to achieve desired artistic effects.
- Ensure that the required dependencies are installed before running the code.

Feel free to modify the provided code to suit your specific use case and explore the exciting possibilities of Neural Style Transfer!
```
