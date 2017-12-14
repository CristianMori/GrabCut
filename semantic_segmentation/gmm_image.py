from sklearn import mixture
import numpy as np
from typing import List


class GMMImage:
    def __init__(self, num_components=5, num_channel=3):
        self.num_components = num_components
        self.num_channels = num_channel
        # 5 is the number of components grab cut uses.
        self.model = mixture.GaussianMixture(n_components=num_components, covariance_type='full')

    def fit(self, data: np.array):
        assert data.shape[1] == self.num_channels
        self.model.fit(data)

    def predict(self, image: np.ndarray):
        height, width, num_channel = image.shape
        assert num_channel == 3  # only BGR, no alpha
        probs = self.model.predict(image.reshape(-1, 3))  # type: np.ndarray
        return probs.reshape((height, width))

    @property
    def get_params_opencv(self):
        ret = np.zeros((1, 65), dtype=np.float64)
        ptr = 0

        data_size = self.num_components
        assert len(self.model.weights_) == data_size
        ret[0, ptr: ptr + self.num_components] = self.model.weights_
        ptr += data_size

        data_size = self.num_components * self.num_channels
        assert len(self.model.means_.flatten()) == data_size
        ret[0, ptr: ptr + data_size] = self.model.means_.flatten()
        ptr += data_size

        data_size = self.num_components * self.num_channels * self.num_channels
        assert len(self.model.covariances_.flatten()) == data_size
        ret[0, ptr: ptr + data_size] = self.model.covariances_.flatten()
        ptr += data_size
        return ret
        # return self.model.weights_, self.model.means_.flatten(), self.model.covariances_.flatten()
