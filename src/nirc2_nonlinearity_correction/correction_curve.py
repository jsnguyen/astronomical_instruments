import importlib.resources
import numpy as np
from numpy.polynomial import Polynomial

class NIRC2:
    def apply_nonlinearity_correction(data, year):
        data_file = importlib.resources.files(f'nirc2_nonlinearity_correction.data_{year}').joinpath(f'correction_curve_{year}.npy')
        coef = np.load(data_file, allow_pickle=True)
        correction_curve = Polynomial(coef)
        correction_function = 1-correction_curve
        corrected_data = correction_function(data)*data
        return corrected_data