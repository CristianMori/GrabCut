from sklearn import mixture
import numpy as np


class GMMImage:
    def __init__(self, num_components=5, num_channel=3):
        self.num_components = num_components
        self.num_channels = num_channel
        # 5 is the number of components grab cut uses.
        self.model = mixture.GaussianMixture(n_components=num_components, covariance_type='full')

    def fit(self, data: np.array):
        assert data[1] == self.num_channels
        self.model.fit(data)

    def get_params_opencv(self) -> np.array:
        ret = np.zeros(65)
        ptr = 0

        data_size = self.num_components
        assert len(self.model.weights_) == data_size
        ret[ptr: ptr + self.num_components] = self.model.weights_
        ptr += data_size

        data_size = self.num_components * self.num_channels
        assert len(self.model.means_.flatten()) == data_size
        ret[ptr: ptr + data_size] = self.model.means_.flatten()
        ptr += data_size

        data_size = self.num_components * self.num_channels * self.num_channels
        assert len(self.model.covariances_.flatten()) == data_size
        ret[ptr: ptr + data_size] = self.model.covariances_.flatten()
        ptr += data_size

        return ret
