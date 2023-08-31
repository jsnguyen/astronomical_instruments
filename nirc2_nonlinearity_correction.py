import numpy as np

def correct_nonlinearity(x):
    '''
    Desc:
    Correction of the NIRC2 non-linearity using polynomial fits
    to average pixel values across the entire frame.

    Args:
    x -> a single value or an array. Works by the power of numpy!
    '''

    
    '''
    Follows the numpy Polynomial convention of mapping the
    parameters into a new space then applying the coefficients
    to the degree specified
    IE:
    x' = offset + scale*x
    y = polynomial(x')
    '''

    # this comes from a second order polynomial fit to the percent deviation from linearity to the full frame of data
    mapparms = (-1.0211077219276714, 8.696749359728794e-05)
    coef = (-0.05588699, -0.09272797, -0.03780352)
    
    # map the x values to the new space, offset and scale
    # the mapping should always be linear
    map_x = mapparms[0] + mapparms[1]*x
    
    # apply the polynomial coefficients
    # should work with arbitrary degree
    # res gives the PERCENT deviation from linearity
    # IE: -0.05 is a 5% negative deviation from linearity
    res = 0
    for i in range(len(coef)):
        res += coef[i]*np.power(map_x,i)

    # apply the "opposite" of the deviation from linearity to correct it
    return (1-res) * x
    
def quad_correct_quad_nonlinearity(data):
    '''
    Desc:
    Correction for the NIRC2 nonlinearity on a quadrant basis.
    2nd order polynomials were fit to linearity data from each quadrant separately.
    This procedure works as long as the image can be divided into 2x2 quadrants
    
    Args:
    data -> should be a 2D numpy array, the raw image itself in DN
    '''

    # first, divide the image into a 2x2 squae
    n_div = 2 # 2x2 square

    squares = []
    sy,sx = data.shape
    dy = sy//n_div
    dx = sx//n_div

    for j in range(n_div):
        for i in range(n_div):    
            xlo = i*dx
            xhi = (i+1)*dx
            ylo = j*dy
            yhi = (j+1)*dy
            squares.append(data[ylo:yhi,xlo:xhi])
    
    '''
    Follows the numpy Polynomial convention of mapping the
    parameters into a new space then applying the coefficients
    to the degree specified
    IE:
    x' = offset + scale*x
    y = polynomial(x')
    '''
    
    # this comes from a second order polynomial fit to the percent deviation from linearity
    quad_mapparms = [(-1.020797148632062, 8.654348182322101e-05),
                     (-1.0209661996068107, 8.580344047989143e-05),
                     (-1.0213522022060764, 8.78147384875344e-05),
                     (-1.0213225297554214, 8.774086680214187e-05)]
    
    quad_coef = [[-0.05438358, -0.08831162, -0.04400768],
                 [-0.05824548, -0.09164471, -0.02772357],
                 [-0.05497404, -0.09655265, -0.04087235],
                 [-0.05579917, -0.09289254, -0.0372074 ]]

    # now apply the 4 separate polynomial fits to each quadrant
    new_squares = []
    for sq,mapparms,coef in zip(squares, quad_mapparms, quad_coef):
    
        # map the x values to the new space, offset and scale
        # the mapping should always be linear
        map_x = mapparms[0] + mapparms[1]*sq
        
        # apply the polynomial coefficients
        # should work with arbitrary degree
        # res gives the PERCENT deviation from linearity
        # IE: -0.05 is a 5% negative deviation from linearity
        res = 0
        for i in range(len(coef)):
            res += coef[i]*np.power(map_x,i)

        # so to get the correction, apply the opposite of the deviation from linearity (1 - deviation)
        new_squares.append((1-res) * sq)

    # reassemble our image
    bot = np.concatenate((new_squares[0], new_squares[1]), axis=1)
    top = np.concatenate((new_squares[2], new_squares[3]), axis=1)
    new_data = np.concatenate((bot, top), axis=0)

    return new_data
