import numpy as np

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