"""
Feature Interaction Strength Calculation.

This module contains the HStatisticCalculator class for calculating pairwise 
feature interaction strengths using H-statistic.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

class HStatisticCalculator:
    """
    Calculate H-statistics to quantify feature interaction strength.
    
    The H-statistic measures the proportion of variance caused by interaction effects
    between pairs of features. Higher values indicate stronger interaction effects.
    
    Attributes:
        model: Trained machine learning model.
        X: Features (DataFrame or array).
        y: Target variable.
        grid_resolution: Number of equally spaced points for PDP computation.
        pdps: Dictionary to store PDP results.
        h_statistics: DataFrame of computed H-statistics.
        hj_statistics: DataFrame of one-vs-all H-statistics.
    """
    
    def __init__(self, model, X, y, grid_resolution=10,view_pdp=False):
        """
        Initialize the calculator with a trained model and data.

        Parameters:
            model: Trained machine learning model.
            X: Features (DataFrame or array).
            y: Target variable.
            grid_resolution: The number of equally spaced points on the grid for PDP computation.
        """
        self.model = model
        self.X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.y = y
        self.grid_resolution = grid_resolution
        self.pdps = {}  # Dictionary to store PDP results
        self.h_statistics = None  # Store H-statistics
        self.hj_statistics = None  # Store one-vs-all H-statistics
        self.view_pd=view_pdp
    def _compute_pdp(self):
        """
        Compute pairwise PDPs and store them in `self.pdps`.
        
        Returns:
            self.pdps: Dictionary of PDP results for feature pairs.
        """
        features = list(self.X.columns)
        features_remaining = features.copy()
        self.pdps = {}

        for feature in features:
            features_remaining.remove(feature)
            for feature_2 in features_remaining:
                feature_pair = (feature, feature_2)
                if not(self.view_pd):
                    plt.ioff()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                pdp_result = PartialDependenceDisplay.from_estimator(
                    self.model, 
                    self.X, 
                    features=(feature, feature_2, feature_pair), 
                    grid_resolution=self.grid_resolution,
                    ax=ax
                )
                if not(self.view_pd):
                    plt.close()  # Close the figure
                    plt.ion()
                self.pdps[feature_pair] = pdp_result
               
                    
                
        return self.pdps

    def compute_pairwise_h_statistic(self):
        """
        Compute the H-statistic for all feature pairs based on PDPs.
        
        Returns:
            DataFrame: H-statistics for all feature pairs.
        """
        if not self.pdps:
            print("PDPs not found. Computing PDPs first.")
            self._compute_pdp()

        H = {}
        for feature_pair, pdp in self.pdps.items():
            feature_pair_name = f"{feature_pair[0]} {feature_pair[1]}"
            H[feature_pair_name] = self.h_stat(pdp)

        self.h_statistics = pd.DataFrame(H, index=['H-Statistic']).T
        # Take square root of H-statistics for interpretability
        self.h_statistics = np.sqrt(self.h_statistics)

        self.h_statistics=self.h_statistics.sort_values(by='H-Statistic', ascending=False)
        
        
        return self.h_statistics

    def h_stat(self, pdp_result):
        """
        Compute the H-statistic from a given PDP result.
        
        Parameters:
            pdp_result: PartialDependenceDisplay object.
            
        Returns:
            float: H-statistic value.
        """
        hi = pdp_result.pd_results[0]['average']
        hj = pdp_result.pd_results[1]['average']
        hij = pdp_result.pd_results[2]['average'][0]

        # Centered PDP values for each feature
        hi_centered = hi - np.mean(hi)
        hj_centered = hj - np.mean(hj)

        # Construct independent (non-interacting) HI and HJ grids for the interaction model
        HJ = np.tile(hj_centered, (len(hi_centered), 1))  # Repeat hj_centered along the rows
        HI = np.tile(hi_centered.reshape(-1, 1), (1, len(hj_centered)))  # Repeat hi_centered along the columns

        # Expected PDP under no interaction assumption
        H3 = HI + HJ

        # Center joint PDP
        hij_centered = hij - np.mean(hij)

        # Compute mean squared difference
        mse_diff = np.sum((H3 - hij_centered) ** 2)

        # Normalize by the mean squared value of hij_centered
        mse_hij = np.sum(hij_centered ** 2)
        return mse_diff / mse_hij if mse_hij != 0 else 0

    def compute_pd_minus_feature(self, deterministic_grid, feature_name):
        """
        Compute the partial dependence for all features except `feature_name`.
        
        Parameters:
            deterministic_grid: Grid of feature values.
            feature_name: Name of the feature to exclude.
            
        Returns:
            array: Partial dependence values.
        """
        num_grid_points, _ = deterministic_grid.shape
        num_samples = len(self.X)

        grid_copy = deterministic_grid.copy()
        partial_dep_sum = np.zeros(num_grid_points)

        for observed_value in self.X.loc[:, feature_name]:
            grid_copy.loc[:, feature_name] = observed_value
            partial_dep_sum += self.model.predict(grid_copy)

        return partial_dep_sum / num_samples

    def compute_one_vs_all_pdp(self):
        """
        Compute the one-vs-all PDP for each feature and return mean H-statistics.
        
        Returns:
            DataFrame: One-vs-all H-statistics for each feature.
        """
        if not self.pdps:
            print("PDPs not found. Computing PDPs first.")
            self._compute_pdp()

        features = list(self.X.columns)
        first_feature = features[0]
        grid_values = {}
        pdpj = {}

        # Extract grid values from PDPs
        for i, feature in enumerate(features[1:]):
            feature_pair = (first_feature, feature)

            if feature_pair not in self.pdps:
                print(f"Skipping {feature_pair} - PDP not computed.")
                continue

            pdp_result = self.pdps[feature_pair]

            if i == 0:
                grid_values[first_feature] = np.linspace(
                    min(pdp_result.pd_results[0]['grid_values'][0]),
                    max(pdp_result.pd_results[0]['grid_values'][0]),
                    self.grid_resolution
                )
                pdpj[first_feature] = pdp_result.pd_results[0]['grid_values'][0]

            unique_values = np.unique(pdp_result.pd_results[1]['grid_values'][0])

            if len(unique_values) == 2:  # Binary feature case
                grid_values[feature] = np.tile(unique_values, self.grid_resolution // 2)
            else:
                grid_values[feature] = np.linspace(
                    min(pdp_result.pd_results[1]['grid_values'][0]), 
                    max(pdp_result.pd_results[1]['grid_values'][0]), 
                    self.grid_resolution
                )

            pdpj[feature] = pdp_result.pd_results[1]['grid_values'][0]

        # Ensure all features have the same number of grid points
        max_length = max(len(v) for v in grid_values.values())
        for key in grid_values:
            values = np.array(grid_values[key])
            if len(values) < max_length:
                grid_values[key] = np.linspace(values.min(), values.max(), max_length)

        grid_values = pd.DataFrame(grid_values)

        # Compute the one-vs-all H-statistic
        Yhat = self.model.predict(grid_values)

        grid_points, dim = grid_values.shape
        H_jvs_all_distribution = np.zeros((grid_points, dim))

        # Compute H-statistic for each feature
        for j, column in enumerate(self.X.columns):
            H_jvs_all_distribution[:, j] = (
                Yhat 
                - np.interp(np.linspace(0, 1, len(Yhat)), np.linspace(0, 1, len(pdpj[column])), pdpj[column])
                - self.compute_pd_minus_feature(grid_values, column)
            )**2

        # Create DataFrame with one-vs-all H-statistics
        self.hj_statistics = pd.DataFrame(
            H_jvs_all_distribution.mean(axis=0).reshape(1, -1), 
            columns=self.X.columns
        ).T
        # Rename the single column
        self.hj_statistics.columns = ['h one vs all']


        # Normalize by the total variance
        self.hj_statistics = np.sqrt(self.hj_statistics / (Yhat**2).sum())

        self.hj_statistics=self.hj_statistics .sort_values(by='h one vs all', ascending=False)

        return self.hj_statistics