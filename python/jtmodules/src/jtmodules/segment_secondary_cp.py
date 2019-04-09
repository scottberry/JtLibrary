import logging
import numpy as np
import mahotas as mh
import collections
import centrosome.cpmorphology
import centrosome.propagate
import scipy.ndimage
import skimage.morphology

logger = logging.getLogger(__name__)

VERSION = '0.0.3'
M_PROPAGATION = "Propagation"
M_WATERSHED_G = "Watershed - Gradient"
M_WATERSHED_I = "Watershed - Image"
M_DISTANCE_N = "Distance - N"
M_DISTANCE_B = "Distance - B"

Output = collections.namedtuple('Output', ['secondary_label_image', 'figure'])


def main(primary_label_image, intensity_image, method, threshold=None,
         regularization_factor=0.01, distance_to_dilate=3, fill_holes=True,
         plot=False):
    '''Detects secondary objects in an image by expanding the primary objects
    encoded in `primary_label_image`. The outlines of secondary objects are
    determined based on the watershed transform of `intensity_image` using the
    primary objects in `primary_label_image` as seeds.
    '''

    if not np.any(primary_label_image == 0):
        secondary_label_image = primary_label_image
    else:
        n_objects = len(np.unique(primary_label_image[1:]))
        logger.info(
            'primary label image has %d objects',
            n_objects - 1
        )
        if n_objects > 1:

            labels_in = primary_label_image.copy()

            # convert intensity image to float64
            if intensity_image.dtype == np.uint8:
                img = np.float64(intensity_image) / 255
            else if intensity_image.dtype == np.uint16:
                img = np.float64(intensity_image) / 65535

            # threshold intensity image
            thresholded_image = intensity_image > threshold

            if method == M_DISTANCE_N:
                distances, (i, j) = scipy.ndimage.distance_transform_edt(
                    labels_in == 0,return_indices=True)
                labels_out = np.zeros(labels_in.shape, int)
                dilate_mask = distances <= distance_to_dilate
                labels_out[dilate_mask] = labels_in[i[dilate_mask], j[dilate_mask]]

            elif method == M_DISTANCE_B:
                labels_out, distances = centrosome.propagate.propagate(
                    img, labels_in, thresholded_image, 1.0)
                labels_out[distances > distance_to_dilate] = 0
                labels_out[labels_in > 0] = labels_in[labels_in > 0]

            elif method == M_PROPAGATION:
                labels_out, distance = centrosome.propagate.propagate(
                    img, labels_in, thresholded_image, regularization_factor)

            elif method == M_WATERSHED_G:
                # Apply the sobel filter to the image (both horizontal
                # and vertical). The filter measures gradient.
                sobel_image = np.abs(scipy.ndimage.sobel(img))

                # Combine the seeds and thresholded image to mask the watershed
                watershed_mask = np.logical_or(thresholded_image, labels_in > 0)

                # Perform the first watershed
                labels_out = skimage.morphology.watershed(
                    connectivity=np.ones((3, 3), bool),
                    image=sobel_image,
                    markers=labels_in,
                    mask=watershed_mask
                )

            elif method == M_WATERSHED_I:
                # invert the image so that the maxima are filled first
                # and the cells compete over what's close to the threshold
                inverted_img = 1 - img

                # Combine the seeds and thresholded image to mask the watershed
                watershed_mask = np.logical_or(thresholded_image, labels_in > 0)

                # Perform the watershed
                labels_out = skimage.morphology.watershed(
                    connectivity=np.ones((3, 3), bool),
                    image=inverted_img,
                    markers=labels_in,
                    mask=watershed_mask
                )

            if self.fill_holes:
                secondary_label_image = centrosome.cpmorphology.fill_labeled_holes(labels_out)
            else:
                secondary_label_image = labels_out

        # re-implement this
        #        secondary_label_image = self.filter_labels(secondary_label_image,
        #                                       objects, workspace)
        secondary_label_image = secondary_label_image.astype(np.int32)

        else:
            logger.info('skipping secondary segmentation')
            secondary_label_image = np.zeros(
                primary_label_image.shape, dtype=np.int32
            )

    n_objects = len(np.unique(secondary_label_image)[1:])
    logger.info('identified %d objects', n_objects)

    if plot:
        from jtlib import plotting
        colorscale = plotting.create_colorscale(
            'Spectral', n=n_objects, permute=True, add_background=True
        )
        outlines = mh.morph.dilate(mh.labeled.bwperim(secondary_label_image > 0))
        plots = [
            plotting.create_mask_image_plot(
                primary_label_image, 'ul', colorscale=colorscale
                ),
            plotting.create_mask_image_plot(
                secondary_label_image, 'ur', colorscale=colorscale
            ),
            plotting.create_intensity_overlay_image_plot(
                intensity_image, outlines, 'll'
            )
        ]
        figure = plotting.create_figure(plots, title='secondary objects')
    else:
        figure = str()

    return Output(secondary_label_image, figure)




