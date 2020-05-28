import platform
import matplotlib.pyplot as plt
import IPython.display as idisplay


def clear_output(wait=True):
    if platform.system() == 'Windows':
        plt.close('all')
        pass
    else:
        idisplay.clear_output(wait=wait)


def display(objs):
    if platform.system() == 'Windows':
        plt.ion()
        plt.show()
        plt.imshow(objs)
        # plt.show(block=False)
        plt.draw()
        plt.pause(0.001)
    else:
        idisplay.display(
            objs,
            metadata={
                'width': objs.width,
                'height': objs.height
            }
        )


