from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import h5py
import numpy as np

class SimpleCategorialHDF5Reader(object):
    """
    user should call `openf()` and `closef()` to start/finish.

    assumes stored image shape is [N, depth(=1 greyscale), H, W]
    """

    def __init__(
        self,
        hdf5_file,
        target_field='hadro_data/n_hadmultmeas',
        target_dtype=np.uint32,
        nlabels=6
    ):
        self._file = hdf5_file
        self._f = None
        self._nlabels = nlabels
        self._target_field = target_field
        self._target_dtype = target_dtype

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['event_data/eventids'].shape[0]
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_samples(self, start, stop):
        image_x = self._f['img_data/hitimes-x'][start: stop]
        image_u = self._f['img_data/hitimes-u'][start: stop]
        image_v = self._f['img_data/hitimes-v'][start: stop]
#        image_x = np.moveaxis(image_x, 1, -1)
#        image_u = np.moveaxis(image_u, 1, -1)
#        image_v = np.moveaxis(image_v, 1, -1)
        label = self._f[self._target_field][start: stop].reshape([-1])
        label[label > self._nlabels - 1] = self._nlabels - 1
        oh_label = np.zeros(
            (label.size, self._nlabels), dtype=self._target_dtype
        )
        oh_label[np.arange(label.size), label] = 1
        eventid = self._f['event_data/eventids'][start: stop].reshape([-1])
        
        return eventid, oh_label, image_x, image_u, image_v
    
    def get_key(self, start, stop, key):
        data = self._f[key][start: stop].reshape([-1])
        if self._target_field in key:
            data[data > self._nlabels - 1] = self._nlabels - 1

        return data
