# XAI: Use GradCAM
Just training a CNN or ViT is usually not enough. You will need to improve the model performance by further analyzing 
the model architecture and outputs. And sometimes you just want to understand how your model processes the input data 
and comes with those results.

In this notebook, we will be using GradCAM (Gradient-weighted Class Activation Mapping) to visualize the ViT 
outputs. 

## 1. GradCAM
GradCAM stands for Gradient-weighted class activation mappings. In short, we will weight the layer activations 
by gradients, which will generate a heatmap. Then, we can visualize the parts of the image that has the most impact 
on the model's outputs.


