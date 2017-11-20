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
'''Jterator module for plotting a LabelImage
'''
import logging
import collections

VERSION = '0.1.0'

logger = logging.getLogger(__name__)

Output = collections.namedtuple('Output', ['figure'])


def main(image, plot=False):

    if plot:
        logger.info('create plot')
        from jtlib import plotting
        colorscale = plotting.create_colorscale(
            'Spectral', n=image.max(), permute=True, add_background=True
        )
        data = [
            plotting.create_mask_image_plot(
                image, 'ul', colorscale=colorscale
            )
        ]
        figure = plotting.create_figure(
            data,
            title='LabelImage with "{0}" objects'.format(
                image.max()
            )
        )
    else:
        figure = str()

    return Output(figure)
