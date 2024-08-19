import numpy as np
from astronomical_instruments import NIRC2

def main():
    data = np.ones((1024,1024))
    corrected = NIRC2.apply_nonlinearity_correction(data, '2019')

    print(data)
    print(corrected)

    NIRC2.generate_nonlinearity_correction('2024', make_plots=True)
    NIRC2.generate_nonlinearity_correction('2019', make_plots=True)

    # make the check plots
    NIRC2.generate_nonlinearity_correction('2024', use_correction_curve=True, save_curves=False, make_plots=True)
    NIRC2.generate_nonlinearity_correction('2019', use_correction_curve=True, save_curves=False, make_plots=True)

if __name__ == '__main__':
    main()