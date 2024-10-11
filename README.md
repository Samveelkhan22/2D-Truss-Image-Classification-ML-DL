# 2D Truss Image Classification Using ML/DL Techniques

This project presents the implementation of several neural network architectures for the classification of 2D truss images. The networks process truss images and associated geographical data, producing both analytical and classification results. We compare multiple model architectures and analyze their performance across various metrics, including accuracy, error rate, confusion matrices, and training gradients.

## Problem Architecture

### Inputs:
- **Truss Image Pixels:** Digital representations of trusses providing visual data on their structure.
- **Geographical Information:** Structural data such as truss members, nodes, connectivity, and support conditions.

### Outputs:
- **Analytical Results:** Quantitative measures like patterns extracted from global stiffness matrices, internal forces, and displacement measurements.
- **Classification Results:** Truss classification based on structural attributes or conditions.

## Neural Networks Implemented

### 1. Capsule Networks (CapsNets)
Capsule Networks are used to classify trusses, utilizing dynamic routing to group data points into capsules and learning hierarchical patterns:
- **Dynamic Routing Pruning:** Reduces training time by limiting routing iterations.
- **Smaller Capsules:** Optimizes memory usage by reducing capsule size.

### 2. Graph Convolutional Networks (GNNs)
Graph-based architectures are applied to classify trusses:
- **Sparse Graph Representations:** Trusses are represented as sparse adjacency matrices to reduce computation.
- **Graph Sampling:** Techniques like GraphSAGE are used to sample a subset of nodes for efficient training.

### 3. Transformer Networks
Transformer models with efficient self-attention mechanisms for truss classification:
- **Efficient Transformer Variants:** Utilizes Linformer, Reformer, or Performer to reduce attention mechanism overhead.
- **Sparse Attention:** Focuses on a small input subset, reducing computation.

### 4. Autoencoders + MLP/GNN
Autoencoders combined with either MLP or GNN for feature extraction and classification:
- **Denoising Autoencoders:** Learns robust latent representations by training on corrupted inputs.
- **Variational Autoencoders (VAEs):** Captures more compact latent features, improving classification speed.

### 5. Mixture of Experts (MoE)
MoE models selectively activate subsets of experts for efficient classification:
- **Selective Expert Activation:** Only activates a few relevant experts, reducing computation.
- **Regularization:** L2 regularization and dropout to prevent overfitting.

## Key Results
- **Accuracy:** Detailed analysis of the classification accuracy for each network.
- **Error Rate:** Evaluates the error rates across the models.
- **Confusion Matrices:** Provides insights into the classification errors made by the models.
- **Training Gradient:** Visualizes the gradient over training epochs for each network.
- **Comparison:** Comparisons between training, validation, and testing stages to highlight model generalization.

## Project Structure
- `CapsNets/`: Code for Capsule Networks.
- `GNNs/`: Code for Graph Convolutional Networks.
- `Transformers/`: Code for Transformer Networks.
- `Autoencoders_MLP_GNN/`: Code for Autoencoders combined with MLP/GNN.
