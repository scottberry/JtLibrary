# Copyright 2017 Scott Berry, University of Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Jterator module converting an image to an intensity_image.'''
import collections
import logging
import numpy as np

VERSION = '0.0.2'

Output = collections.namedtuple('Output', ['intensity_image', 'figure'])

logger = logging.getLogger(__name__)


def main(image, output_type='16-bit', plot=False):
    '''Converts an arbitrary Image to an IntensityImage

    Parameters
    ----------
    image: numpy.ndarray
        image to be converted
    output_type: numpy.ndarray
        output data type
    plot: bool, optional
        whether a plot should be generated (default: ``False``)

    Returns
    -------
    jtmodules.convert_to_intensity.Output
    '''
    if output_type == '8-bit':
        bit_depth = np.uint8
        max_value = pow(2, 8)
    elif output_type == '16-bit':
        bit_depth = np.uint16
        max_value = pow(2, 16)
    else:
        logger.warn('unrecognised requested output data-type %s, using 16-bit', output_type)
        bit_depth = np.uint16
        max_value = pow(2, 16)

    if image.dtype == np.int32:
        logger.info('Converting label image to intensity image')
        if (np.amax(image) < max_value):
            intensity_image = image.astype(dtype=bit_depth)
        else:
            logger.warn(
                '%d objects in input label image exceeds maximum (%d)',
                np.amax(image),
                max_value
            )
            intensity_image = image
    else:
        logger.info('Converting non-label image to intensity image')
        intensity_image = image.astype(dtype=bit_depth)

    if plot:
        from jtlib import plotting
        n_objects = len(np.unique(image)[1:])
        colorscale = plotting.create_colorscale(
            'Spectral', n=n_objects, permute=True, add_background=True
        )
        plots = [
            plotting.create_mask_image_plot(
                image, 'ul', colorscale=colorscale
            ),
            plotting.create_intensity_image_plot(
                intensity_image, 'ur'
            )
        ]
        figure = plotting.create_figure(plots, title='convert_to_intensity_image')
    else:
        figure = str()

    return Output(intensity_image, figure)
