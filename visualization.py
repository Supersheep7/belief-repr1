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
from matplotlib import colors 

def mass_plot(labels, heads=None, streams=None, color_map={0:'red', 1:'blue'}, resid={0: 'pre', 1: 'mid', 2: 'post'}):

    layers = len(heads) if heads else len(next(streams.values()))

    if heads is not None:
        heads_per_layer = len(heads[0])  # Number of heads per layer

        # Set up the grid of subplots
        fig, axes = plt.subplots(layers, heads_per_layer, figsize=(heads_per_layer * 4, layers * 4))
        plt.subplots_adjust(hspace=0.5)

        for i, layer in tqdm(enumerate(heads), desc="Processing layers"):
            for j, head in enumerate(layer):
                # Access the correct subplot
                ax = axes[i, j] if layers > 1 and heads_per_layer > 1 else axes[j if layers == 1 else i]

                # Reshape tensor to 2D: (n_samples, -1)
                reshaped_tensor = einops.rearrange(head, 'n_batch batch_size d_head -> (n_batch batch_size) d_head')
                reshaped_labels = einops.rearrange(labels, 'n_batch batch_size -> (n_batch batch_size)')
                # Apply PCA to the current tensor
                pca = PCA(n_components=2)
                pca_transformed = pca.fit_transform(reshaped_tensor.cpu())
                # Color coding
                colors = [color_map[int(label)] for label in reshaped_labels]

                # Plot the PCA result for this tensor
                scatter = ax.scatter(
                                    pca_transformed[:, 0], 
                                    pca_transformed[:, 1],
                                    c=colors, 
                                    alpha=0.8, 
                                    edgecolor='k', 
                                    linewidths=0.5,
                                    s=50  # Adjust the marker size
                                )

                # Customize the subplot
                ax.set_title(f"Layer {i}, Head {j}", fontsize=14)
                ax.set_xlabel("Principal Component 1", fontsize=12)
                ax.set_ylabel("Principal Component 2", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)


    elif streams is not None:
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
            pca_transformed = pca.fit_transform(reshaped_tensor.cpu())

            colors = [color_map[int(label)] for label in reshaped_labels]

            # Plot the PCA result for this tensor
            scatter = ax.scatter(
                                pca_transformed[:, 0], 
                                pca_transformed[:, 1],
                                c=colors, 
                                alpha=0.8, 
                                edgecolor='k', 
                                linewidths=0.5,
                                s=50  # Adjust the marker size
                            )

            # Customize the subplot
            ax.set_title(f"Residual {resid[resid_type]}, Layer {resid_layer}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Principal Component 1", fontsize=12)
            ax.set_ylabel("Principal Component 2", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
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

def plot_kde_with_scatter(data, x, y, labels, x_label, y_label, title, x_range=None, y_range=None, kernel=True, scatter=True, adjust=1):
    """
    Helper function to create a KDE plot with an optional scatter overlay.
    """
    g = sns.jointplot(
        data=data,
        x=x,
        y=y,
        hue=labels,
        kind='kde',
        palette='tab10',
        linewidths=0.8 if kernel else 0,
        alpha=1,
        bw_adjust=adjust,
        marginal_kws={'fill': True, 'common_norm': False, 'alpha': 0.3, 'linewidth': 0.8}
    )

    if scatter:
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=labels,
            palette='coolwarm',
            marker='o',
            s=10,
            edgecolor='black',
            linewidth=0.2,
            alpha=0.7,
            ax=g.ax_joint
        )

    if x_range:
        g.ax_joint.set_xlim(*x_range)
    if y_range:
        g.ax_joint.set_ylim(*y_range)

    g.ax_joint.grid(True, linestyle='--', alpha=0.6)
    g.fig.suptitle(title, fontsize=12)
    g.fig.tight_layout()
    plt.show()

def kde(data, labels, model, n_dir=2, zoom_strength=0, adjust=1, kernel=True, scatter=True, pca=False):

    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples."

    if len(data.shape) > 2:
        data = einops.rearrange(data, 'n_batch batch_size d_model -> (n_batch batch_size) d_model')
    if len(labels.shape) > 1:   
        labels = einops.rearrange(labels, 'n_batch batch_size -> (n_batch batch_size)')

    data = np.array(data.cpu(), dtype=np.float32)
    labels = np.array(labels.cpu(), dtype=np.int32)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    if pca:
        # Perform PCA
        pca_model = PCA(n_components=2)
        pca_projections = pca_model.fit_transform(data)
        
        x_min, x_max = np.percentile(pca_projections[:, 0], [0 + zoom_strength, 100 - zoom_strength])
        y_min, y_max = np.percentile(pca_projections[:, 1], [0 + zoom_strength, 100 - zoom_strength])

        data_frame = pd.DataFrame({
            'PCA1': pca_projections[:, 0],
            'PCA2': pca_projections[:, 1],
            'Label': ['False' if label == 0 else 'True' for label in labels]
        })

        plot_kde_with_scatter(
            data=data_frame,
            x='PCA1',
            y='PCA2',
            labels='Label',
            x_label='PCA1',
            y_label='PCA2',
            title="PCA with KDE for Labeled Projections",
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            kernel=kernel,
            scatter=scatter,
            adjust=adjust
        )
        return

    # Original KDE logic for LogReg directions
    first_direction = get_direction(data, labels, model)
    data_with_bias = np.hstack([np.ones((data.shape[0], 1)), data])
    first_projections = np.dot(data_with_bias, first_direction)

    if n_dir == 1:
        x_min, x_max = np.percentile(first_projections, [0 + zoom_strength, 100 - zoom_strength])
        plt.figure(figsize=(8, 6))

        for class_label in np.unique(labels):
            class_projections = first_projections[labels == class_label]

            # Estimate density using Gaussian KDE
            density = gaussian_kde(class_projections, bw_method='scott')
            x_vals = np.linspace(x_min, x_max, 500)
            y_vals = density(x_vals)

            # Plot the KDE curve
            plt.plot(x_vals, y_vals, label=f'Class {class_label}')
            plt.fill_between(x_vals, y_vals, alpha=0.3, label=None)

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

        data_frame = pd.DataFrame({
            'First direction': first_projections,
            'Second direction': second_projections,
            'Label': ['False' if label == 0 else 'True' for label in labels]
        })

        plot_kde_with_scatter(
            data=data_frame,
            x='First direction',
            y='Second direction',
            labels='Label',
            x_label='First direction',
            y_label='Second direction',
            title="KDE with Marginals for Labeled Projections",
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            kernel=kernel,
            scatter=scatter,
            adjust=adjust
        )

    return

def pretty_line(x, x1=None, title="DummyTitle", x_axis="DummyXaxis", y_axis="DummyYaxis", x_label="DummyXLabel", y_label="DummyYLabel"):

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='w')
    ax.set_facecolor('#e0e0e0') 
    ax.plot(x, color='#007acc', alpha=0.7, marker='o', markersize=8, linewidth=2.5, label=x_label)
    if x1 is not None:
        ax.plot(x1, color='#d62728', alpha=0.7, marker='o', markersize=8, linewidth=2.5,
                label=y_label)
    ax.set_title(title,
                fontsize=18, pad=20, weight='bold', color='#333333')
    ax.set_xlabel(x_axis, fontsize=14, labelpad=10, color='#333333')
    ax.set_ylabel(y_axis, fontsize=14, labelpad=10, color='#333333')

    ax.grid(visible=True, which='major', color='#f7f7f7', linewidth=1.5, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12, color='#555555')

    # Customize legend
    legend = ax.legend(fontsize=12, loc='upper right', frameon=True)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#e0e0e0')
    legend.get_frame().set_alpha(0.9)

    # Clean up spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Display the plot
    plt.tight_layout()
    plt.show()

def pretty_line_mass(
    *lines,
    labels=None,
    title="Probe Accuracy on Residual Stream",
    x_axis="Layers",
    y_axis="Accuracy"
):
    """
    Create a pretty line plot with any number of lines.

    Parameters:
    - *lines: Any number of sequences (e.g., lists, numpy arrays) to plot.
    - labels: List of labels for each line. If None, default labels will be used.
    - title: Title of the plot.
    - x_axis: Label for the x-axis.
    - y_axis: Label for the y-axis.
    """
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='w')
    ax.set_facecolor('#e0e0e0') 

    colors = ['#007acc', '#007acc', 
              '#d62728', '#d62728', 
              '#2ca02c', '#2ca02c', 
              '#9467bd', '#9467bd', 
              '#8c564b', '#8c564b', 
              '#e377c2', '#e377c2', 
              '#7f7f7f', '#7f7f7f']
    markers = ['o', '^']
    num_lines = len(lines)
    
    if labels is None:
        labels = [f"Line {i+1}" for i in range(num_lines)]
    
    for i, line in enumerate(lines):
        ax.plot(
            line,
            color=colors[i % len(colors)],
            alpha=0.5,
            marker=markers[i % len(markers)],
            markersize=5,
            linewidth=2.5,
            label=labels[i]
        )
    
    ax.set_title(title, fontsize=18, pad=20, weight='bold', color='#333333')
    ax.set_xlabel(x_axis, fontsize=14, labelpad=10, color='#333333')
    ax.set_ylabel(y_axis, fontsize=14, labelpad=10, color='#333333')

    ax.grid(visible=True, which='major', color='#f7f7f7', linewidth=1.5, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12, color='#555555')

    # Customize legend
    legend = ax.legend(fontsize=10.5, loc='upper left', frameon=True)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#e0e0e0')
    legend.get_frame().set_alpha(0.5)

    # Clean up spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Display the plot
    plt.tight_layout()
    plt.show()

def pretty_heatmap(accuracies, title="DummyTitle", x_axis="DummyXaxis", y_axis="DummyYaxis", model="DummyModel", probe="DummyProbe", dataset="DummyDataset"):

    accuracies = np.array(accuracies)
    # Assuming tot_accuracies_heads is already defined
    sorted_accuracies = np.sort(accuracies, axis=1)[:, ::-1]  # Reverse the order on the X-axis
    sorted_accuracies = sorted_accuracies[::-1, :]  # Reverse the order on the Y-axis (layers)
    norm = colors.Normalize(vmin=sorted_accuracies.min(), vmax=max(sorted_accuracies.max(), 0.75))

    # Set figure aesthetics
    plt.figure(figsize=(10, 8))  # Slightly larger figure for clarity
    sns.set(style="whitegrid")  # Light grid background for better visibility
    ax = sns.heatmap(
        sorted_accuracies,
        annot=False,
        fmt=".2f",
        cmap="cividis", 
        cbar_kws={"shrink": 0.9, "aspect": 22},  # Adjust colorbar size and aspect
        linewidths=0,  
        linecolor="white",
        norm=norm 
    )

    # Adjust Y-axis ticks to reflect the reversed order
    num_layers = sorted_accuracies.shape[0]
    num_heads = sorted_accuracies.shape[1]
    ax.set_yticks(np.arange(num_layers) + 0.5)  # Center ticks
    ax.set_yticklabels(np.arange(num_layers - 1, -1, -1), fontsize=10)  # Reversed layer indices with proper font size

    ax.set_xticks([])

    # Titles and labels
    plt.suptitle(title, fontsize=18)
    plt.title(f"Model: {model} | Probe: {probe} | Dataset: {dataset} ")
    plt.xlabel(x_axis, fontsize=12, labelpad=10)
    plt.ylabel(y_axis, fontsize=12, labelpad=10)

    # Add gridlines to separate elements clearly
    ax.hlines(np.arange(1, num_layers), *ax.get_xlim(), colors="white", linestyles="solid", linewidth=0.2)
    ax.vlines(np.arange(1, num_heads), *ax.get_ylim(), colors="white", linestyles="solid", linewidth=0.2)

    # Show the plot
    plt.tight_layout()
    plt.show()

def pretty_sweep(data, ks, alphas, metric="DummyMetric", custom_subtitle=None):

    # Create the heatmap
    plt.figure(figsize=(10, 8))  # Adjust size
    ax = plt.gca()
    ax.set_aspect('equal')

    sns.heatmap(data, annot=True, fmt=".3f", cmap="Blues", cbar=False, linewidths=0.1, linecolor='grey')
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, ks)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, alphas)
    # Add titles and labels
    if custom_subtitle is not None:
      plt.suptitle(f"Intervention effect | metric: {metric}", fontsize=16)
      plt.title(f"{custom_subtitle}")
    else:
      plt.title(f"Intervention effect | metric: {metric}", fontsize=16, pad=16)
    plt.xlabel("Alpha", labelpad=10)
    plt.ylabel("K", labelpad=10)

    # Show the plot
    plt.show()