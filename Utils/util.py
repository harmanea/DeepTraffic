import matplotlib.pyplot as plt

def show_images(images):
    n = len(images)
    cols = int(n**0.5)
    rows = n // cols + (1 if n % cols > 0 else 0)

    fig = plt.figure(figsize=(8, 8))
    for i in range(n):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(images[i])

    plt.show()

