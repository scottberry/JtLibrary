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
'''Jterator module for applying a mask to a label image
'''
import logging
import collections
import mahotas as mh
import numpy as np

VERSION = '0.1.0'

logger = logging.getLogger(__name__)

Output = collections.namedtuple('Output', ['masked_image', 'figure'])


def main(image, mask, value = 0, plot=False):

    mask_image = np.copy(image)
    mask_image[mask == 0] = value

    if plot:
        logger.info('create plot')
        from jtlib import plotting
        data = [
            plotting.create_mask_image_plot(
                mask, 'ul'
            ),
            plotting.create_intensity_image_plot(
                mask_image, 'ur'
            )
        ]
        figure = plotting.create_figure(
            data,
            title='Masked intensity image'
        )
    else:
        figure = str()

    return Output(mask_image, figure)
