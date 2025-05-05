import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import einops
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import gaussian_kde

def mass_plot(labels, layers, heads=None, streams=None, color_map={0:'red',1:'blue'}, resid={0: 'pre', 1: 'mid', 2: 'post'}):

    if heads:
        heads_per_layer = len(heads[0])  # Number of heads per layer

        # Set up the grid of subplots
        fig, axes = plt.subplots(layers, heads_per_layer, figsize=(heads_per_layer * 4, layers * 4))

        for i, layer in tqdm(enumerate(heads), desc="Processing layers"):
            for j, head in enumerate(layer):
                # Access the correct subplot
                ax = axes[i, j] if layers > 1 and heads_per_layer > 1 else axes[j if layers == 1 else i]

                # Reshape tensor to 2D: (n_samples, -1)
                reshaped_tensor = einops.rearrange(head, 'n_batch batch_size d_head -> (n_batch batch_size) d_head')
                reshaped_labels = einops.rearrange(labels, 'n_batch batch_size -> (n_batch batch_size)')

                # Apply PCA to the current tensor
                pca = PCA(n_components=2)
                pca_transformed = pca.fit_transform(reshaped_tensor)

                # Color coding
                colors = [color_map[int(label)] for label in reshaped_labels]

                # Plot the PCA result for this tensor
                scatter = ax.scatter(
                    pca_transformed[:, 0], pca_transformed[:, 1],
                    c=colors, alpha=0.7
                )

                # Customize the subplot
                ax.set_title(f"Layer {i}, Head {j}")

    elif streams:
        residual_types = len(resid.items())

        # Set up the grid of subplots
        fig, axes = plt.subplots(residual_types, layers, figsize=(layers * 4, residual_types * 4))

        for i, act in tqdm(enumerate(streams.values()), desc="Processing activations"):
            # Determine the subplot's row and column indices
            resid_type = i // layers  # Row: 0, 1, 2, ...
            resid_layer = i % layers  # Column: 0, 1, 2, ...

            ax = axes[resid_type, resid_layer]

            # Reshape tensor to 2D: (n_samples, -1)
            reshaped_tensor = einops.rearrange(act, 'n_batch batch_size d_model -> (n_batch batch_size) d_model')
            reshaped_labels = einops.rearrange(labels, 'n_batch batch_size -> (n_batch batch_size)')

            # Apply PCA to the current tensor
            pca = PCA(n_components=2)
            pca_transformed = pca.fit_transform(reshaped_tensor)

            colors = [color_map[int(label)] for label in reshaped_labels]

            # Plot the PCA result for this tensor
            scatter = ax.scatter(
                pca_transformed[:, 0], pca_transformed[:, 1],
                c=colors, alpha=0.7
            )

            # Customize the subplot
            ax.set_title(f"Residual {resid[resid_type]}, Layer {resid_layer}")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")

    plt.tight_layout()
    plt.show()

def get_direction(data, labels, model):
    
    model = clone(model)
    model.fit(data, labels)
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    theta = np.hstack([intercept, coefficients])

    return theta / np.linalg.norm(theta)

def get_direction_with_constraint(data, labels, model, first_direction):

        # Remove the projection of X onto theta_1
        projection_on_theta1 = np.dot(data, first_direction)
        data_orthogonalized = data - np.outer(projection_on_theta1, first_direction) / np.dot(first_direction, first_direction)

        # Step 3: Train the second logistic regression model on the orthogonalized data
        model = clone(model)
        model.fit(data_orthogonalized[:, 1:], labels) # Exclude the bias term when fitting

        # Step 4: Extract theta_2 (intercept and coefficients)
        theta = np.hstack([model.intercept_[0], model.coef_[0]])
    
        return theta / np.linalg.norm(theta)

def kde(data, labels, model, n_dir=2, zoom_strength=0, adjust=1, kernel=True, scatter=True):

    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples."

    if len(data.shape) > 2:
        data = einops.rearrange(data, 'n_batch batch_size d_model -> (n_batch batch_size) d_model')
    if len(labels.shape) > 1:   
        labels = einops.rearrange(labels, 'n_batch batch_size -> (n_batch batch_size)')

    data = np.array(data.cpu(), dtype=np.float32)
    labels = np.array(labels.cpu(), dtype=np.int32)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    first_direction = get_direction(data, labels, model)
    data_with_bias = np.hstack([np.ones((data.shape[0], 1)), data])
    first_projections = np.dot(data_with_bias, first_direction)

    if n_dir == 1:
        x_min, x_max = np.percentile(first_projections, [0 + zoom_strength, 100 - zoom_strength])
        plt.figure(figsize=(8, 6))

        for class_label in np.unique(labels):
            class_projections = first_projections[labels == class_label]

            # Estimate density using Gaussian KDE
            density = gaussian_kde(class_projections, bw_method='scott')  # You can adjust 'scott' or use 'silverman'
            x_vals = np.linspace(x_min, x_max, 500)  # Use zoomed range for smoother curves
            y_vals = density(x_vals)

            # Plot the KDE curve
            plt.plot(x_vals, y_vals, label=f'Class {class_label}')
            plt.fill_between(x_vals, y_vals, alpha=0.3, label=None)  # Optional: Fill under the curve

        plt.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')
        plt.xlabel('Projection onto LogReg Direction')
        plt.ylabel('Density')
        plt.title('Class Separation Along LogReg Direction')
        plt.xlim(x_min, x_max)
        plt.legend()
        plt.show()

    else:

        second_direction = get_direction_with_constraint(data_with_bias, labels, model, first_direction)
        second_projections = np.dot(data_with_bias, second_direction)
        x_min, x_max = np.percentile(first_projections, [0 + zoom_strength, 100 - zoom_strength])
        y_min, y_max = np.percentile(second_projections, [0 + zoom_strength, 100 - zoom_strength])

        data = pd.DataFrame({
            'First direction': first_projections,
            'Second direction': second_projections,
            'Label': ['False' if label == 0 else 'True' for label in labels]
        })

        # Create jointplot
        g = sns.jointplot(
            data=data,
            x='First direction',
            y='Second direction',
            hue='Label',
            kind='kde',
            palette='tab10',
            linewidths=0.8 if kernel else 0,
            alpha=1,
            bw_adjust=adjust,
            marginal_kws={'fill': False, 'common_norm': False, 'alpha': 1, 'linewidth': 0.8}
        )

        if scatter:
        # Overlay scatterplot on jointplot
          sns.scatterplot(
              data=data,
              x='First direction',
              y='Second direction',
              hue='Label',
              palette='coolwarm',
              marker='o',
              s=10,
              edgecolor='black',
              linewidth=0.2,
              alpha=0.7,
              ax=g.ax_joint
          )

        # Add title
        g.ax_joint.set_xlim(x_min, x_max)
        g.ax_joint.set_ylim(y_min, y_max)
        g.fig.suptitle("KDE with Marginals for Labeled Projections", fontsize=12)
        g.fig.tight_layout()
        plt.show()

    return