import importlib.resources

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt
from astropy.io import fits

import warnings
from astropy.io.fits.verify import AstropyUserWarning
warnings.simplefilter('ignore', category=AstropyUserWarning)

def divide_image_into_sections(image_array, div_shape):
    rows, cols = div_shape
    # Get image dimensions
    height, width = image_array.shape[:2]
    
    # Calculate the size of each section
    section_height = height // rows
    section_width = width // cols
    
    # Create a list to store the image sections
    sections = []
    
    # Loop through the grid and slice the sections
    for row in range(rows):
        for col in range(cols):
            start_y = row * section_height
            end_y = (row + 1) * section_height
            start_x = col * section_width
            end_x = (col + 1) * section_width
            
            # Slice the section
            section = image_array[start_y:end_y, start_x:end_x]
            sections.append(section)
    
    return sections

def median_sigma_clip(img, n_std):
    
    mask = np.zeros(img.shape,dtype=bool)

    median = np.median(img)
    std = np.std(img)
    
    sigma_clip_upper = img > median + n_std*std
    sigma_clip_lower = img < median - n_std*std
    mask += np.logical_or(sigma_clip_upper, sigma_clip_lower)
    
    return mask

def section_median_sigma_clip(img, div_shape, n_std):
    
    mask = np.zeros(img.shape,dtype=bool)
    
    img_sections = divide_image_into_sections(img, div_shape)
    mask_sections = divide_image_into_sections(mask, div_shape)

    for i_s,m_s in zip(img_sections, mask_sections):
        m_s += median_sigma_clip(i_s, n_std)
    
    return mask

def diagonal_line_mask(arr, m, b, width, xlim=[None, None], ylim=[None,None]):
    '''
    Selects the diagonal portion of the image based on the equation y = mx + c.

    Parameters:
        image: np.ndarray, the input image.
        m: float, the slope of the line.
        c: float, the y-intercept of the line.

    Returns:
        masked_image: np.ndarray, the image with only the diagonal portion selected.
    '''

    if xlim[0] == None:
        xlim[0] = 0
    if xlim[1] == None:
        xlim[1] = arr.shape[1]

    if ylim[0] == None:
        ylim[0] = 0
    if ylim[1] == None:
        ylim[1] = arr.shape[0]
    
    mask = np.zeros(arr.shape, dtype=bool)
    
    sy, sx = arr.shape

    
    # Iterate over each pixel in the image
    for y in range(sy):
        for x in range(sx):
            # Calculate the corresponding y value based on the line equation
            y_line = int(m * x + b)
            
            # Check if the current pixel is below the line
            if y-width < y_line < y+width:
                if xlim[0] < x < xlim[1] and ylim[0] < y < ylim[1]:
                    mask[y, x] = True  # Set mask to 255 (white) for the selected portion
    
    return mask


def pixel_box_slice(data, center, box_radius):
    '''
    Creates a slice of an image centered around a given pixel.

    args:
        image (np.ndarray): The input image as a 2D numpy array.
        center (tuple): The (y, x) coordinates of the center pixel.
        half_width (int): The half-width of the slice.

    return:
        np.ndarray: The sliced image centered around the given pixel.
    '''
    
    cy, cx = center
    sy, sx = data.shape
    
    y_start = max(cy - box_radius, 0)
    y_end = min(cy + box_radius + 1, sy)
    x_start = max(cx - box_radius, 0)
    x_end = min(cx + box_radius + 1, sx)
    
    return data[y_start:y_end, x_start:x_end]

def median_compare_neighbors(data, box_radius, n_std):

    sy,sx = data.shape
    
    mask = np.zeros(data.shape, dtype=bool)
    
    ys, xs = np.meshgrid(np.arange(0, sy), np.arange(0, sx))
    indices = np.array((ys.ravel(),xs.ravel())).T
    
    for coord in indices:
        cy, cx = coord
        box = pixel_box_slice(data, coord, box_radius)
        if data[cy,cx] > np.median(box)+n_std*np.std(box):
            mask[cy,cx] = True

    return mask

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
    
    data_file = importlib.resources.files('astronomical_instruments.data').joinpath(f'nirc2mask.fits')
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