# astronomical_instruments

This package includes some helpful functions for reducing instrument-specific data. Right now, it only has functions for NIRC2.

## Installation

``` bash
git clone https://www.github.com/jsnguyen/astronomical_instruments
pip install ./astronomical_instruments
```

## Usage

### Bad Pixel Masks

To generate a bad pixel mask:

``` python
from astronomical_instruments import NIRC2

# just an example, load your data here instead
dark_frames = [np.zeros((1024,1024)) for _ in range(10)]
flat_frames = [np.ones((1024,1024)) for _ in range(10)]

# change the sigma clip parameters and box radius parameters as needed
# default choices are generally good for NIRC2
bad_pixel_mask = NIRC2.make_bad_pixel_mask(dark_frames, flat_frames)
```

To generate the bad pixel mask using calibration data from 2023-01-01 from the KOA:

``` python
from astronomical_instruments import NIRC2

# the arg saves to bad_pixel_mask_20230101.fits
bad_pixel_mask = NIRC2.make_bad_pixel_mask_20230101(save_mask=True)
```

### Non-Linearity

``` python
import numpy as np
from astronomical_instruments import NIRC2

data = np.ones((1024,1024))
corrected = NIRC2.apply_nonlinearity_correction(data, '2019')

# or

corrected = NIRC2.apply_nonlinearity_correction(data, '2023')
```

If you want to regenerate the correction curves, and diagnostic plots:

``` python
from astronomical_instruments import NIRC2

NIRC2.generate_nonlinearity_correction('2024', make_plots=True)
NIRC2.generate_nonlinearity_correction('2019', make_plots=True)

# make the check plots
NIRC2.generate_nonlinearity_correction('2024', use_correction_curve=True, save_curves=False, make_plots=True)
NIRC2.generate_nonlinearity_correction('2019', use_correction_curve=True, save_curves=False, make_plots=True)
```

## Linearity Analysis

Upgrades to NIRC2 in late 2023 slightly changed the behavior of the detector, notably, the gains are different. Changing the gain has resultingly changed the linearity characteristics of the detector. We perform an analysis using reduced data from early 2024 with the new instrument upgrades and characterize the non-linearity in the system. A function to correct for the non-linearity is provided. However, recent changes (~July 2024) require that a linearity test be performed again and new data needs to be taken.

In 2024, linearity data was collected in Kp and Ks band. Data was collected simply by turning on the flat lamps and exposing for increasing amounts of time. While we don't expect any difference between wavelengths, slight variation is observed in the dataset. Both 2019 and 2024 datasets are included in this repository.

Plots showing the non-linearity:

<div>
    <img src="plots/linearity_2019.png" alt="linearity 2019 plot" width="300"/>
    <img src="plots/linearity_2024.png" alt="linearity 2024 plot" width="300"/>
</div>

Plots showing the percent deviation from linearity:

$$ \% = \frac{\text{linear\_data}-\text{actual\_data}}{\text{linear\_data}} $$

<div>
    <img src="plots/linearity_diff_2019.png" alt="diff 2019 plot" width="300"/>
    <img src="plots/linearity_diff_2024.png" alt="diff 2024 plot" width="300"/>
</div>

The correction curves were derived from polynomial fits to the percent deviation from linearity. Note that we cut off some points on each end of the curve to fit 100 to 20000 DN in 2019, 100 to 5000 DN in 2024 a little better. In practice, data values on images typically fall somewhere in this range.

Lastly, checking that our correction works:

<div>
    <img src="plots/CHECK_linearity_2019.png" alt="check linearity 2019 plot" width="300"/>
    <img src="plots/CHECK_linearity_2024.png" alt="check linearity 2024 plot" width="300"/>
</div>

And we see that the plots are now much closer to linear across the range! The upper ends aren't so important to fit since the detector typically does not reach those levels.

## Notes

The initial 2019 analysis included in `old_analysis/`.

There's a couple extra plots in `plots` not shown here.

I don't know the exact dates for when linearity changed from the 2019 state to the 2024 state. I also don't know the rule of thumb the Keck SAs use for the new linearity limit. Previously the limit was 10000 in 2019.
