import matplotlib.pyplot as plt


def plot_results(images, titles, figure_size=(12, 12), save_path=None):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
