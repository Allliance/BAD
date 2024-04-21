import matplotlib.pyplot as plt
import torchvision

def visualize_samples(dataloader, n, title="Sample", normal_label=1):
    plt.clf()
    normal_samples = []
    abnormal_samples = []

    def to_3_channels(image):
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        return image

    # Collect n x n samples
    for images, labels in dataloader:
        for i, l in enumerate(labels):
            image = to_3_channels(images[i])
            if len(normal_samples) < n * n and l == normal_label:
                normal_samples.append(image)
            elif len(abnormal_samples) < n * n and l != normal_label:
                abnormal_samples.append(image)
            if len(normal_samples) == n * n and len(abnormal_samples) == n * n:
                break
        if len(normal_samples) == n * n and len(abnormal_samples) == n * n:
            break

    
    normal_grid = torchvision.utils.make_grid(normal_samples, nrow=n)
    abnormal_grid = torchvision.utils.make_grid(abnormal_samples, nrow=n)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    fig.patch.set_alpha(0)
    fig.suptitle(title, fontsize=16)

    axs[0].imshow(normal_grid.permute(1, 2, 0))
    axs[0].set_title('Normal', fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(abnormal_grid.permute(1, 2, 0))
    axs[1].set_title('Abnormal', fontsize=14)
    axs[1].axis('off')

    plt.show()
    


def plot_gaps(cleans, bads, dataset, best_eps, verbose=False):
    
    plt.clf()
    # Plot both arrays
    x = np.arange(len(cleans))  # the label locations

    width = 0.35  
    
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    for i in range(len(cleans)):
        ax.text(x[i], -0.1, f"Eps: {best_eps[i]:.3f}", ha='center')
        
    rects1 = ax.bar(x - width/2, cleans, width, label='clean resnet', color='blue')
    rects2 = ax.bar(x + width/2, bads, width, label='bad resnet', color='red')

    ax.set_ylabel('auroc')
    ax.set_title(f'Bar Chart for {dataset} for best eps')
    ax.set_xticks(x)
    ax.legend()

    fig.savefig(f'results/{dataset}.png', bbox_inches='tight')
    if verbose:
        plt.show()


def plot_process(epsilons, gaps, title, verbose=False):
    plt.clf()
    
    plt.plot(epsilons, gaps)  # Plot y1 with blue color
    plt.xlabel('epsilon')
    plt.ylabel('auroc gap')
    plt.title(title)
    plt.savefig(f'results/{title}.png')
    if verbose:
        plt.show()



def plot_tsne(features, labels):
    
    num_classes = len(set(labels))
    tsne = TSNE(n_components=2).fit_transform(features)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='tab20')  # Adjust the colormap as needed
    plt.colorbar(boundaries=np.arange(12)-0.5).set_ticks(np.arange(11))
    plt.title('TSNE Embedding')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.savefig("my_tsne_plot.png")

def plot_umap(features, labels):
    num_classes = len(set(labels))

    umap_emb = umap.UMAP().fit_transform(features)

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=labels, cmap='tab20')  # Adjust the colormap as needed
    plt.colorbar(boundaries=np.arange(12)-0.5).set_ticks(np.arange(11))
    plt.title('UMAP Embedding')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig("my_umap_plot.png")
    
    