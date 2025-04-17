"""
Synthetic Data Generation.

This module provides functions to generate synthetic datasets 
with controllable feature interactions for testing interaction detection methods.
"""

import numpy as np
import pandas as pd


def generate_interaction_data(a1=1, a2=1, a12=1, correlation=0, n_samples=1000, noise_power=0.1):
    """
    Generate a synthetic regression dataset based on the equation:
        y = a1*x1 + a2*x2 + a12*x1*x2 + noise
        
    Parameters:
        a1 (float): Coefficient for x1 (default=1)
        a2 (float): Coefficient for x2 (default=1)
        a12 (float): Coefficient for interaction term (x1 * x2) (default=1)
        correlation (float): Correlation between x1 and x2 (default=0, uncorrelated)
        n_samples (int): Number of samples in the dataset (default=1000)
        noise_power (float): Standard deviation of Gaussian noise (default=0.1)

    Returns:
        y (pd.Series): Target variable
        X (pd.DataFrame): Feature matrix containing 'X1' and 'X2'
    """
    
    # Generate correlated x1 and x2 using Cholesky decomposition
    mean = [0, 0]  # Assume mean of X1 and X2 is 0
    cov_matrix = [[1, correlation], [correlation, 1]]  # Define covariance structure
    
    # Generate samples from multivariate normal distribution
    x1, x2 = np.random.multivariate_normal(mean, cov_matrix, size=n_samples).T
    
    # Compute the target variable y with interaction
    noise = np.random.normal(0, noise_power, size=n_samples)  # Add Gaussian noise
    y = a1 * x1 + a2 * x2 + a12 * x1 * x2 + noise
    
    # Create feature DataFrame
    X = pd.DataFrame({'X1': x1, 'X2': x2})
    y = pd.DataFrame({'y': y})
    
    return y, X


def generate_linear_interaction_data(n_samples=1000, noise=0.0,
                                    a1=1.0, a2=1.0, a3=1.0,
                                    a12=1.0, a13=1.0, a23=1.0,
                                    random_state=42):
    """
    Generates a linear regression dataset with three features and interaction terms.

    Model:
        y = a1*x1 + a2*x2 + a3*x3 + a12*x1*x2 + a13*x1*x3 + a23*x2*x3 + noise
    
    Parameters:
        n_samples (int): Number of samples to generate.
        noise (float): Standard deviation of Gaussian noise added to y.
        a1, a2, a3 (float): Coefficients for the main effects.
        a12, a13, a23 (float): Coefficients for the interaction terms.
        random_state (int): Seed for random number generation.
    
    Returns:
        y (np.array): The target variable.
        X (pd.DataFrame): A DataFrame containing features 'X1', 'X2', 'X3'.
    """
    
    # Set the random seed for reproducibility
    np.random.seed(random_state)
    
    # Generate features from a normal distribution
    x1 = np.random.normal(loc=0, scale=1, size=n_samples)
    x2 = np.random.normal(loc=0, scale=1, size=n_samples)
    x3 = np.random.normal(loc=0, scale=1, size=n_samples)
    
    # Compute the response variable using the specified linear model with interactions
    y = (a1 * x1 +
         a2 * x2 +
         a3 * x3 +
         a12 * (x1 * x2) +
         a13 * (x1 * x3) +
         a23 * (x2 * x3))
    
    # Add Gaussian noise if specified
    y += noise * np.random.randn(n_samples)
    
    # Create a DataFrame for features
    X = pd.DataFrame({
        "X1": x1,
        "X2": x2,
        "X3": x3
    })
    
    return y, X
