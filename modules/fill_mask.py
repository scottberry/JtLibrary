from scipy import ndimage as ndi
import collections
import pylab as plt
import numpy as np
from tmlib.jterator import jtapi


def fill_mask(mask, **kwargs):
    '''
    Jterator module to fill holes (enclosed pixel regions of connected components)
    in a binary image.

    Parameters
    ----------
    mask: numpy.ndarray
        binary image that should be filled
    **kwargs: dict
        additional arguments provided by Jterator:
        "data_file", "figure_file", "experiment_dir", "plot", "job_id"

    Returns
    -------
    namedtuple[numpy.ndarray[bool]]
        filled binary image: "filled_mask"
    '''
    img = ndi.binary_fill_holes(mask)

    if kwargs['plot']:

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(1, 1, 1)

        img_obj = np.zeros(img.shape)
        img_obj[img > 0] = 1
        img_obj[img == 0] = np.nan

        ax1.imshow(img_obj, cmap=plt.cm.Set1)
        ax1.set_title('filled mask', size=20)

        fig.tight_layout()

        jtapi.save_mpl_figure(fig, kwargs['figure_file'])

    output = collections.namedtuple('Output', 'filled_mask')
    return output(img)
