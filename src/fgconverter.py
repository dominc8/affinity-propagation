import matplotlib.pyplot as plt
import imageio
from io import BytesIO

class FGConverter:
    def __init__(self):
        self.images = []

    def add_fig(self, figure, close_figure = True):
        output = BytesIO()
        figure.savefig(output)
        if close_figure:
            plt.close(figure)
        output.seek(0)
        self.images.append(imageio.imread(output))

    def make_gif(self, filename, fps = 10, **kwargs):
        imageio.mimsave(filename, self.images, fps=fps, **kwargs)
        self.images.clear()

