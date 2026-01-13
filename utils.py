from matplotlib import pyplot as plt


def visualize_image(gray_image, title=""):
    fig, ax = plt.subplots()
    ax.imshow(gray_image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.show()