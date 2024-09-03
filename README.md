# Graph2Tree Model for Symbolic Mathematics

## Overview

The `graph2tree` model is a Graph Neural Network (GNN) combined with a Tree LSTM encoder designed to solve symbolic mathematics problems, specifically focusing on solving integral equations symbolically. The model has been modified from a base code to improve the accuracy and correctness of the generated symbolic equations in comparison to other existing models.

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)

## Background

Symbolic mathematics, particularly solving integral equations, is a challenging task that requires sophisticated models capable of understanding and manipulating mathematical expressions. Traditional methods often struggle with the complexity and variability of symbolic forms. The `graph2tree` model addresses this by leveraging a GNN to understand the structure of mathematical expressions and a Tree LSTM encoder to efficiently generate correct symbolic equations.

## Features

- **Graph Neural Network (GNN):** Captures the structure of mathematical expressions represented as graphs.
- **Tree LSTM Encoder:** Encodes the graph into a tree structure, enabling the generation of more accurate symbolic equations.

## Installation

To install the necessary dependencies and set up the environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/graph2tree.git

# Navigate to the project directory
cd graph2tree

# Install required dependencies
pip install -r requirements.txt
```
## Usage
To use the graph2tree model for solving symbolic mathematics problems:

## Model Architecture
The graph2tree model consists of the following components:

Graph Neural Network (GNN): Processes the input mathematical expressions represented as graphs.
Tree LSTM Encoder: Encodes the graph into a tree structure to capture hierarchical dependencies.
Decoder: Generates the output symbolic equations based on the encoded tree structure.
The architecture is designed to handle the complexities of symbolic mathematics, particularly the variable structures encountered in integral equations.

## Dataset
The model was trained and tested on a dataset of integral equations. The dataset includes a variety of symbolic forms, ensuring that the model can generalize well to different types of integral problems.

- Dataset Source: https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_fwd.tar.gz

## Results
The graph2tree model achieves state-of-the-art results in generating correct symbolic equations for integral problems. Below are some key metrics:

Accuracy: 78%




