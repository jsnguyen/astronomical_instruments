import importlib.resources
import fnmatch

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt

from astropy.io import fits

# ignoring weird warning from Keck files...
from astropy.io.fits.verify import AstropyUserWarning
import warnings
warnings.simplefilter('ignore', category=AstropyUserWarning)

from correction_curve import NIRC2

def plot_linearity(linearity_data, n_fit, year, use_correction_curve):

    fig, ax = plt.subplots(figsize=(6,6))
    for bandname in linearity_data.keys():
        exp_times, linearity_line, full_data_mean, full_data_std = linearity_data[bandname]

        ax.errorbar(exp_times, full_data_mean, yerr=full_data_std, marker='o', linestyle='none', label=f'Average {bandname.capitalize()}-Band', alpha=0.7)
        #ax.plot(fit_xs, np.vectorize(res)(fit_xs), linestyle='-', color=ax.get_lines()[-1].get_c(), label='Polynomial Fit')
        ax.plot(exp_times[:n_fit], linearity_line(exp_times[:n_fit]), linestyle='--', color='black', alpha=0.7)
        ax.plot(exp_times[n_fit-1:], linearity_line(exp_times[n_fit-1:]), linestyle='--', color='black', alpha=0.3)

        if use_correction_curve:
            ax.set_title(f'CHECK Linearity {year}')
        else:
            ax.set_title(f'Linearity {year}')

        ax.set_xlabel('Exposure Time [s]')
        ax.set_ylabel('Average Pixel Count [DN/px]')
        
        ax.set_xlim(0,125)
        ax.set_ylim(0,1.25*np.max(full_data_mean))
        
    ax.grid()
    ax.legend()   

    if use_correction_curve:
        plt.savefig(f'CHECK_linearity_{year}.png', bbox_inches='tight')
    else:
        plt.savefig(f'linearity_{year}.png', bbox_inches='tight')

def plot_linearity_deviation(linearity_data, correction_curves, diffs, cutoff, year, use_correction_curve):
    
    means = []
    fig, ax = plt.subplots(figsize=(6,6))
    for bandname in correction_curves.keys():
        nonlinearity_fit = correction_curves[bandname]
        diff = diffs[bandname]

        _, _, full_data_mean, full_data_std = linearity_data[bandname]
        weights = full_data_std/full_data_mean # higher number = more weight
        means.append(full_data_mean)

        fit_xs = np.linspace(np.min(full_data_mean), np.max(full_data_mean), 1000)

        ax.errorbar(full_data_mean[cutoff:], diff[cutoff:]*100, yerr=np.abs(weights[cutoff:]*diff[cutoff:]*100), marker='o', alpha=0.7, label=f'Percent Deviation {bandname.capitalize()}-Band')
        ax.plot(fit_xs, np.vectorize(nonlinearity_fit)(fit_xs)*100, linestyle='-', color=ax.get_lines()[-1].get_c(), label='2nd Order Polynomial Fit')

        if use_correction_curve:
            ax.set_title(f'CHECK Deviation From Linearity {year}')
        else:
            ax.set_title(f'Deviation From Linearity {year}')

    ax.set_xlabel('Average Pixel Count [DN/px]')
    ax.set_ylabel('Percent Deviation From Linearity [%]')

    ax.set_xlim(0,1.25*np.max(means))
    ax.set_ylim(-20,10)

    ax.grid()
    ax.legend()

    if use_correction_curve:
        plt.savefig(f'CHECK_linearity_diff_{year}.png', bbox_inches='tight')
    else:
        plt.savefig(f'linearity_diff_{year}.png', bbox_inches='tight')

def fit_linearity(data, headers, cutoff, n_fit, order=1):
    
    data_file = importlib.resources.files('nirc2_nonlinearity_correction.data').joinpath(f'nirc2mask.fits')
    bad_pixel_mask = fits.getdata(data_file)
    
    exp_times = np.array([h['ITIME'] for h in headers])[cutoff:]
    data = data[cutoff:]
    headers = headers[cutoff:]

    full_data_mean = []
    full_data_std = []
    for d in data:
        masked = np.ma.array(d, mask=bad_pixel_mask)
        full_data_mean.append(np.ma.mean(masked))
        full_data_std.append(np.ma.std(masked))

    full_data_mean = np.array(full_data_mean)
    full_data_std = np.array(full_data_std)
    
    weights = np.square(full_data_std/full_data_mean) # higher number = more weight

    # fitting a line to the first n points
    linearity_line = Polynomial.fit(exp_times[:n_fit], full_data_mean[:n_fit], order, w=weights[:n_fit])

    return exp_times, linearity_line, full_data_mean, full_data_std

def calc_correction_curves(linearity_data, correction_curve_slice, order=1):

    dn, up = correction_curve_slice

    diffs = {}
    correction_curves = {}
    for bandname in linearity_data.keys():
        exp_times, linearity_line, full_data_mean, full_data_std = linearity_data[bandname]

        expected = linearity_line(exp_times)
        diff = (full_data_mean - expected) /  expected
        weights = full_data_std/full_data_mean # higher number = more weight

        nonlinearity_fit = Polynomial.fit(full_data_mean[dn:up], diff[dn:up], order, w=weights[dn:up])
        print(f'{bandname.capitalize()}-Band Fit parameters: {nonlinearity_fit.convert().coef}')

        correction_curves[bandname] = nonlinearity_fit
        diffs[bandname] = diff

    return correction_curves, diffs

def linearity(year, cutoff=None, correction_curve_slice=None, use_correction_curve=False, save_curves=True, order=1):

    '''
    read in the datasets, fit the linearity, make plots, and save the final correction curves

    args:
        year: can be '2019' or '2024' only, the two data sets that are available
        cutoff: the lower end cutoff for the fits
        correction_curve_slice: slices the corrected diff data to chop off the beginning or end to do a fit on the linear part of the system
        use_correction_curve: uses the pre-computed correction curve to check if it worked correctly
        save_curves: save .npy files
        order: the order of the final fit for the correction curve

    return:
        None
    '''

    valid_years = ['2019', '2024']
    if year not in valid_years:
        print(f'Year must be in {valid_years}!')
        return

    # cutoffs are needed in the fit to fit a curve better to the linear region in the detector

    if year == '2024':
        bands = ['kp', 'ks'] # h band seems to be bad for this data
        cutoff = 1 # cutoff of the beginning few elements that are bad
        correction_curve_slice = 3,-1 # cutoff the first few, and the last one to get a better fit
        n_fit = 5 # fit to the first n elements

    elif year == '2019':
        bands = ['h']
        cutoff = 0 # cutoff of the beginning few elements that are bad
        correction_curve_slice = 3,-2 # cutoff the first few and last one to get a better fit
        n_fit = 6 # fit to the first n elements

    linearity_data = {}
    for bandname in bands:

        data_folder = importlib.resources.files(f'nirc2_nonlinearity_correction.data_{year}.linearity_{bandname}')
        data_files = []
        for resource in data_folder.iterdir():
            if fnmatch.fnmatch(resource.name, '*.fits'):
                data_files.append(resource)
        data_files = sorted(data_files)

        data = []
        headers = []
        for fn in data_files:
            d = fits.getdata(fn)/4 # divide by 4 to get back into units of DN
            if use_correction_curve:
                d = NIRC2.apply_nonlinearity_correction(d, year)

            data.append(d) 
            headers.append(fits.getheader(fn))


        res = fit_linearity(data, headers, cutoff, n_fit)
        linearity_data[bandname] = res

    correction_curves, diffs = calc_correction_curves(linearity_data, correction_curve_slice, order=order)

    if save_curves:
        filename = f'correction_curve_{year}.npy'
        if len(correction_curves.keys()) > 1:
            for key in correction_curves.keys():
                correction_curves[key] = correction_curves[key].convert(domain=[0, 10000])
            np.save(filename, np.mean(list(correction_curves.values())).convert().coef)
        else:
            np.save(filename, list(correction_curves.values())[0].convert().coef)

    plot_linearity(linearity_data, n_fit, year, use_correction_curve)
    plot_linearity_deviation(linearity_data, correction_curves, diffs, cutoff, year, use_correction_curve)

if __name__=='__main__':
    linearity('2024')
    linearity('2019')
    linearity('2024', use_correction_curve=True, save_curves=False)
    linearity('2019', use_correction_curve=True, save_curves=False)

#linearity('2019')