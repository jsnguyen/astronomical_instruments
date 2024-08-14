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

def plot_linearity(linearity_data, n_fit, bandname, year):

    fig, ax = plt.subplots(figsize=(6,6))
    for bandname in linearity_data.keys():
        exp_times, linearity_line, full_data_mean, full_data_std = linearity_data[bandname]

        ax.errorbar(exp_times, full_data_mean, yerr=full_data_std, marker='o', linestyle='none', label=f'Average {bandname.capitalize()}-Band', alpha=0.7)
        #ax.plot(fit_xs, np.vectorize(res)(fit_xs), linestyle='-', color=ax.get_lines()[-1].get_c(), label='Polynomial Fit')
        ax.plot(exp_times[:n_fit], linearity_line(exp_times[:n_fit]), linestyle='--', color='black', alpha=0.7)
        ax.plot(exp_times[n_fit-1:], linearity_line(exp_times[n_fit-1:]), linestyle='--', color='black', alpha=0.3)

        ax.set_title(f'Linearity {year}')
        ax.set_xlabel('Exposure Time [s]')
        ax.set_ylabel('Average Pixel Count [DN/px]')
        
        ax.set_xlim(0,125)
        ax.set_ylim(0,1.25*np.max(full_data_mean))
        
    ax.grid()
    ax.legend()   

    plt.savefig(f'linearity_{year}.png', bbox_inches='tight')
    plt.savefig(f'linearity_{year}.pdf', bbox_inches='tight')

def plot_linearity_deviation(linearity_data, correction_curves, diffs, cutoff, bandname, year):
    
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

    ax.set_title(f'Deviation From Linearity {year}')

    ax.set_xlabel('Average Pixel Count [DN/px]')
    ax.set_ylabel('Percent Deviation From Linearity [%]')

    ax.set_xlim(0,1.25*np.max(means))
    ax.set_ylim(-20,10)

    ax.grid()
    ax.legend()

    plt.savefig(f'diff_linearity_{year}.png', bbox_inches='tight')
    plt.savefig(f'diff_linearity_{year}.pdf', bbox_inches='tight')

def fit_linearity(data_files, cutoff, n_fit, order=1):
    data = []
    headers = []
    for fn in data_files:
        data.append(fits.getdata(fn)/4) # divide by 4 to get back into units of DN
        headers.append(fits.getheader(fn))
    
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
    print(linearity_line.convert().coef)

    return exp_times, linearity_line, full_data_mean, full_data_std

def make_nonlinearity_function(exp_times, linearity_line, full_data_mean, full_data_std, correction_curve_slice, order=1):
    dn, up = correction_curve_slice

    expected = linearity_line(exp_times)
    diff = (full_data_mean - expected) /  expected
    weights = full_data_std/full_data_mean # higher number = more weight

    nonlinearity_fit = Polynomial.fit(full_data_mean[dn:up], diff[dn:up], order, w=weights[dn:up])
    return nonlinearity_fit, diff

def calc_correction_curves(linearity_data, correction_curve_slice):

    diffs = {}
    correction_curves = {}
    for bandname in linearity_data.keys():
        exp_times, linearity_line, full_data_mean, full_data_std = linearity_data[bandname]

        nonlinearity_fit, d = make_nonlinearity_function(exp_times, linearity_line, full_data_mean, full_data_std, correction_curve_slice, order=1)
        correction_curves[bandname] = nonlinearity_fit
        diffs[bandname] = d

    return correction_curves, diffs

def linearity(year: str):

    valid_years = ['2019', '2024']
    if year not in valid_years:
        print(f'Year must be in {valid_years}!')
        return

    if year == '2024':
        bands = ['kp', 'ks'] # h band seems to be bad for this data
        cutoff = 1 # cutoff of the beginning few elements that are bad
        correction_curve_slice = 3,-1
        n_fit = 5 # fit to the first n elements

    elif year == '2019':
        bands = ['h']
        cutoff = 0 # cutoff of the beginning few elements that are bad
        correction_curve_slice = 3,-2
        n_fit = 6 # fit to the first n elements

    linearity_data = {}
    for bandname in bands:
        linearity_data[bandname] = []

        data_folder = importlib.resources.files(f'nirc2_nonlinearity_correction.data_{year}.linearity_{bandname}')
        data_files = []
        for resource in data_folder.iterdir():
            if fnmatch.fnmatch(resource.name, '*.fits'):
                data_files.append(resource)
        data_files = sorted(data_files)

        res = fit_linearity(data_files, cutoff, n_fit)
        linearity_data[bandname] = res

    correction_curves, diffs = calc_correction_curves(linearity_data, correction_curve_slice)

    filename = f'correction_curve_{year}.npy'
    if len(correction_curves.keys()) > 1:
        for key in correction_curves.keys():
            correction_curves[key] = correction_curves[key].convert(domain=[0, 10000])
        np.save(filename, np.mean(list(correction_curves.values())).convert().coef)
    else:
        np.save(filename, list(correction_curves.values())[0].convert().coef)

    plot_linearity(linearity_data, n_fit, bandname, year)
    plot_linearity_deviation(linearity_data, correction_curves, diffs, cutoff, bandname, year)