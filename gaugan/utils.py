import matplotlib.pyplot as plt


def plot_results(images, titles, figure_size=(12, 12)):
    """Plot results in a row

    Args:
        images: List of images (PIL or numpy arrays)
        titles: List of titles corresponding to images
    """
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()
