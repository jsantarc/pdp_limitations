from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def compute_pdp_interaction(pdp_result,type_=" No Interaction Assumption"):
    """
    Computes interaction strength by comparing the 2D partial dependence (h3) 
    with the sum of 1D partial dependencies (H3) assuming no interaction.

    Parameters:
        pdp_result : PartialDependenceDisplay
            A computed PDP object from sklearn.inspection.PartialDependenceDisplay.
    
    Returns:
        H3 : Expected 2D PDP assuming no interaction.
        h3 : Actual 2D PDP computed by sklearn.
        normalized_mse_diff : Mean squared difference between H3 and h3, normalized.
    """

    # Extract PDP values
    h1 = pdp_result.pd_results[0]['average']  # PDP for X1
    h2 = pdp_result.pd_results[1]['average']  # PDP for X2
    h3 = pdp_result.pd_results[2]['average'][0]  # Joint PDP for (X1, X2)

    # PDP shapes
    pdp1_shape = h1.shape[0]  # Grid size for X1
    pdp2_shape = h2.shape[0]  # Grid size for X2
    R, C = h3.shape  # Grid size for joint PDP (X1, X2)

    # Center the individual PDPs
    h1_centered = h1 - np.mean(h1)
    h2_centered = h2 - np.mean(h2)

    # Proper broadcasting using np.tile
    H2 = np.tile(h2_centered, (pdp1_shape, 1))  # Repeat over rows
    H1 = np.tile(h1_centered.reshape(-1, 1), (1, pdp2_shape))  # Repeat over columns

    # Compute expected 2D PDP assuming no interaction
    H3 = H1 + H2

    # Center the joint PDP correctly
    h3_centered = h3 - np.mean(h3)

    # Compute mean squared difference
    mse_diff = np.sum((H3 - h3_centered) ** 2)

    # Normalize by the mean squared value of h3_centered
    mse_h3 = np.sum(h3_centered ** 2)
    normalized_mse_diff = mse_diff / mse_h3 if mse_h3 != 0 else 0  # Avoid division by zero

    # Print the normalized difference
    print(f"Normalized MSE difference: {normalized_mse_diff:.6f}")

    # Plot Expected PDP (H3)
    plt.figure(figsize=(6, 5))
    plt.imshow(H3, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Expected PDP (H3)")
    plt.title(type_)
    plt.xlabel("Feature 2 Grid")
    plt.ylabel("Feature 1 Grid")
    plt.show()

    # Plot Actual PDP (h3)
    plt.figure(figsize=(6, 5))
    plt.imshow(h3_centered, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Actual PDP (h3)")
    plt.title("Actual PDP (h3) - Joint Feature Dependence")
    plt.xlabel("Feature 2 Grid")
    plt.ylabel("Feature 1 Grid")
    plt.show()


    return H3, h3_centered



def check_pdp_additivity(pdp_result, model):
    """
    Checks whether the model's predictions are equal to the sum of individual PDPs 
    (i.e., tests for interaction). If no interaction exists, yhat ≈ h1 + h2.

    Parameters:
        pdp_result : PartialDependenceDisplay
            A computed PDP object from sklearn.inspection.PartialDependenceDisplay.
        model : Trained machine learning model
            The fitted model used to compute predictions.

    Returns:
        yhat : Model predictions using the PDP grid values.
        h : Sum of individual PDPs assuming no interaction.
    """

    # Extract PDP grid values
    grid_x1 = pdp_result.pd_results[0]['grid_values'][0]  # X1 PDP grid
    grid_x2 = pdp_result.pd_results[1]['grid_values'][0]  # X2 PDP grid

    # Number of grid points
    N1 = len(grid_x1)

    # Construct feature matrix X_ using grid values
    X_ = np.zeros((N1, 2))
    X_[:, 0] = grid_x1  # Assign X1 values
    X_[:, 1] = grid_x2  # Assign X2 values

    # Model predictions at these points
    yhat = model.predict(X_)

    # Extract PDP values for X1 and X2
    h1 = pdp_result.pd_results[0]['average']
    h2 = pdp_result.pd_results[1]['average']

    # Compute sum of individual PDPs (assuming additivity)
    h = h1 + h2
    h = np.squeeze(h, 0)  # Ensure correct shape

    # Plot density curves of yhat (model) vs h (sum of PDPs)
    plt.figure(figsize=(8, 5))
    sns.kdeplot(yhat, color="blue", label="yhat (model)", linewidth=2)
    sns.kdeplot(h, color="red", label="h1 + h2 (expected if no interaction)", linewidth=2)

    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Density Plot: Model Predictions vs Sum of PDPs")
    plt.show()

    return yhat, h


def compute_pdp_ice(model, X, features=None, grid_resolution=50):
    """
    Computes Partial Dependence (PDP) and Individual Conditional Expectation (ICE) plots 
    for all features in the dataset.

    Parameters:
        model : Trained machine learning model
            The fitted model used to compute partial dependence.
        X : DataFrame
            The dataset containing the feature values.
        features : list, optional
            List of features to compute PDP and ICE for. If None, all features in X are used.
        grid_resolution : int, optional
            Number of points in PDP grid (default: 50).
    
    Returns:
        pdp_objects : list
            A list of PartialDependenceDisplay objects for each feature.
    """
    
    # Use all features if none are specified
    if features is None:
        features = X.columns.tolist()

    pdp_objects = []  # Store PDP objects

    for feature in features:
        print(f"Computing PDP & ICE for feature: {feature}...")
        
        # Compute PDP and ICE
        pdp_obj = PartialDependenceDisplay.from_estimator(
            model,
            X,
            [feature],
            kind="both",  # Generates both PDP and ICE plots
            grid_resolution=grid_resolution,
            n_jobs=-1  # Use multiple CPU cores if available
        )
        
        # Store the PDP object
        pdp_objects.append(pdp_obj)

        # Show the plot
        plt.show()

    return pdp_objects