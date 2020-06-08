import matplotlib.pyplot as plt
import IPython.display as idisplay

from . import images
from . import var as var


def clear_output(wait=True):
    if not var.colab:
        plt.close('all')
        pass
    else:
        idisplay.clear_output(wait=wait)


def display(objs):
    # if not gv.colab:
    if True:
        plt.close('all')
        plt.ion()
        plt.show()
        plt.subplot(1, 1, 1)
        images.imshow(objs)

        # plt.show(block=False)
        plt.draw()
        plt.pause(0.001)
    else:
        idisplay.display(objs)


