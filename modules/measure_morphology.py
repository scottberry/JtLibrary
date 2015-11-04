from skimage import measure
from jtlib import utils
from jtlib import features
from tmlib.writers import DatasetWriter


def measure_morphology(labeled_image, objects_name, zernike=False, **kwargs):
    '''
    Jterator module for measuring morhological features of objects
    (connected components) in a labeled image.
    For more details see
    `mahotas docs <http://mahotas.readthedocs.org/en/latest/features.html>`_.

    Parameters
    ----------
    labeled_image: numpy.ndarray[int]
        input label image
    objects_name: str
        name of the objects (labeled connected components) in `labeled_image`
    zernike: bool, optional
        whether Zernike moments should be measured (defaults to False)
    **kwargs: dict
        additional arguments provided by Jterator:
        "data_file", "figure_file", "experiment_dir", "plot", "job_id"

    See also
    --------
    `jtlib.features`
    '''
    measurement_names = []
    measurement_names.append('area_shape')  # measure by default
    if zernike:
        measurement_names.append('zernike')

    data = dict()

    # Get coordinates of region containing individual objects
    regions = measure.regionprops(labeled_image)

    # Calculate threshold across the whole image
    RADIUS = 100

    measurements = dict()
    for m in measurement_names:
        measurements[m] = list()

    for j, r in enumerate(regions):

        # Crop images to region of current object
        mask = utils.crop_image(labeled_image, bbox=r.bbox)
        mask = mask == (j+1)  # only current object

        # basic area/shape features
        feats = features.measure_morphology(r)
        measurements['area_shape'].append(feats)

        # zernike moments
        if 'zernike' in measurement_names:
            feats = features.measure_zernike(mask, radius=RADIUS)
            measurements['zernike'].append(feats)

    for m in measurement_names:
        feature_names = measurements[m][0].keys()
        for f in feature_names:
            feats = [item[f] for item in measurements[m]]
            data['Morphology_%s' % f] = feats

    # TODO: make a nice plot showing the label image and some features
    # in form of a scatterplot with the data points colored the same
    # as the objects in the label image

    with DatasetWriter(kwargs['data_file']) as f:
        for k, v in data.iteritems():
            f.write('%s/features/%s' % (objects_name, k), data=v)