import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import einops
from tqdm import tqdm

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

def kde(data, pc1, pc2, color='blue', label=None):
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create KDE plot
    sns.kdeplot(data, ax=ax, color=color, label=label)

    # Customize the plot
    ax.set_title('Kernel Density Estimation')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    if label:
        ax.legend()

    plt.show()