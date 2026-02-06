# Binary Classification with Neural Networks

## Overview
This project implements a simple neural network from scratch using NumPy to solve binary classification problems. The implementation demonstrates fundamental concepts of machine learning including forward propagation, backpropagation, and gradient descent optimization.

## What I Learned

### 1. **Neural Network Fundamentals**
- Building a neural network from scratch without using high-level frameworks
- Understanding the mathematical operations behind neural network computations
- Implementing the complete training pipeline manually

### 2. **Key Components Implemented**

#### Sigmoid Activation Function
- Implemented the sigmoid function: `σ(z) = 1 / (1 + e^(-z))`
- Used for converting linear outputs to probabilities between 0 and 1
- Essential for binary classification problems

#### Forward Propagation
- Calculated predictions using: `A = σ(W·X + b)`
- Where W is the weight matrix, X is input data, and b is the bias term
- Generates probability scores for each training example

#### Cost Function (Log Loss)
- Implemented binary cross-entropy loss function
- Measures how well the model's predictions match the actual labels
- Formula: `-[Y·log(A) + (1-Y)·log(1-A)]`

#### Backpropagation
- Computed gradients of the loss function with respect to parameters
- Calculated partial derivatives: `dW` and `db`
- Essential for updating model parameters during training

#### Gradient Descent Optimization
- Updated parameters using: `W = W - α·dW` and `b = b - α·db`
- Learning rate (α) set to 1.2 for faster convergence
- Iteratively minimized the cost function

### 3. **Problem Solved**

#### Problem 1: XOR-like Classification
- **Dataset**: 30 training examples with binary features (0 or 1)
- **Task**: Classify points based on the logical relationship between two features
- **Challenge**: Non-linearly separable data requiring the network to learn complex patterns
- **Result**: Successfully trained a model to separate two classes with a decision boundary

#### Problem 2: Larger-Scale Binary Classification
- **Dataset**: 1000 samples generated using `make_blobs`
- **Features**: Two continuous features representing cluster centers at (2.5, 3) and (6.7, 7.9)
- **Task**: Classify points into two distinct clusters
- **Result**: Achieved clean separation with final parameters:
  - W = [[1.01048463, 1.12890402]]
  - b = [[-10.58273851]]

### 4. **Visualization and Decision Boundaries**
- Plotted data points color-coded by class (blue for class 0, red for class 1)
- Visualized linear decision boundaries learned by the model
- Demonstrated how the model separates different classes in feature space

### 5. **Model Evaluation**
- Tracked cost function across iterations to monitor training progress
- Observed cost decreasing from ~0.69 to ~0.13 over 50 iterations
- Implemented prediction function with 0.5 threshold for classification

## Technical Skills Gained

1. **NumPy Operations**: Matrix multiplication, broadcasting, element-wise operations
2. **Machine Learning Mathematics**: Derivatives, chain rule, optimization
3. **Data Visualization**: Using Matplotlib to plot decision boundaries and training data
4. **Algorithm Implementation**: Translating mathematical formulas into working code
5. **Model Training**: Understanding hyperparameters like learning rate and iterations

## Code Structure

```
├── Data Generation
│   ├── Small dataset (30 samples, binary features)
│   └── Large dataset (1000 samples, continuous features)
│
├── Model Components
│   ├── sigmoid_function() - Activation function
│   ├── layer_size() - Determine network architecture
│   ├── initialize_parameters() - Random weight initialization
│   ├── forward_propagation() - Compute predictions
│   ├── cost_function() - Calculate log loss
│   ├── back_propagation() - Compute gradients
│   └── update_parameters() - Gradient descent step
│
├── Training Pipeline
│   └── nn_model() - Complete training loop
│
└── Evaluation & Visualization
    ├── predict_data() - Make predictions on new data
    └── plot_decision_boundary() - Visualize classification results
```

## Key Takeaways

- **Understanding beats black-box usage**: Building from scratch provided deep insight into how neural networks actually work
- **Gradient descent is powerful**: Simple iterative updates can solve complex classification problems
- **Visualization matters**: Plotting decision boundaries helps verify model behavior
- **Mathematics is essential**: Every operation has a mathematical foundation in calculus and linear algebra


## Dependencies

- NumPy: Numerical computations
- Matplotlib: Data visualization
- scikit-learn: Dataset generation (make_blobs)

---

*This project demonstrates fundamental machine learning concepts through hands-on implementation, providing a solid foundation for understanding more complex neural network architectures.*
