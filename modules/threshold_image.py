import mahotas as mh
from bokeh.plotting import figure
from bokeh.palettes import Reds3
import collections
import numpy as np
from jtlib import plotting
from tmlib import image_utils


def threshold_image(image, correction_factor=1, min_threshold=None,
                    max_threshold=None,  **kwargs):
    '''
    Jterator module for thresholding an image with Otsu's method.
    For more information see
    `mahotas docs <http://mahotas.readthedocs.org/en/latest/api.html?highlight=otsu#mahotas.otsu>`_.

    Parameters
    ----------
    image: numpy.ndarray
        grayscale image that should be thresholded
    correction_factor: int, optional
        value by which the calculated threshold level will be multiplied
        (default: ``1``)
    min_threshold: int, optional
        minimal threshold level (default: ``numpy.min(image)``)
    max_threshold: int, optional
        maximal threshold level (default: ``numpy.max(image)``)
    **kwargs: dict
        additional arguments provided by Jterator:
        "data_file", "figure_file", "experiment_dir", "plot", "job_id"

    Returns
    -------
    namedtuple[numpy.ndarray[bool]]
        binary thresholded image: "thresholded_image"

    Raises
    ------
    ValueError
        when all pixel values of `image` are zero after rescaling
    '''
    if max_threshold is None:
        max_threshold = np.max(image)
    if min_threshold is None:
        min_threshold = np.min(image)

    # threshold function requires unsigned integer type
    if not str(image.dtype).startswith('uint'):
        raise TypeError('Image must have unsigned integer type')

    thresh = mh.otsu(image)

    thresh = thresh * correction_factor

    if thresh > max_threshold:
        thresh = max_threshold
    elif thresh < min_threshold:
        thresh = min_threshold

    thresh_image = image > thresh

    if kwargs['plot']:

        # Get the contours of the mask
        img_border = mh.labeled.borders(thresh_image)

        # Convert the image to 8-bit for display
        rescaled_image = image_utils.convert_to_uint8(image)

        dims = rescaled_image.shape
        fig = figure(x_range=(0, dims[1]), y_range=(0, dims[0]),
                     tools=["reset, resize, save, pan, box_zoom, wheel_zoom"],
                     webgl=True)

        # Bokeh cannot deal with RGB images in form of 3D numpy arrays.
        # Therefore, we have to work around it by adapting the color palette.
        img_rgb = rescaled_image.copy()
        img_rgb[rescaled_image == 255] = 254
        img_rgb[img_border] = 255
        # border pixels will be colored red, all others get gray colors
        palette = plotting.create_bokeh_palette('greys')
        palette_rgb = np.array(palette)
        palette_rgb[-1] = Reds3[0]
        fig.image(image=[img_rgb[::-1]],
                  x=[0], y=[0], dw=[dims[1]], dh=[dims[0]],
                  palette=palette_rgb)
        fig.grid.grid_line_color = None
        fig.title = 'overlay of mask contours'
        fig.axis.visible = None

        plotting.save_bk_figure(fig, kwargs['figure_file'])

    output = collections.namedtuple('Output', 'thresholded_image')
    return output(thresh_image)