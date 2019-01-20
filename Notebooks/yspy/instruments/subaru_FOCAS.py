import os
import numpy as np
from astropy.convolution import convolve, Box1DKernel

__all__ = ['suboverscan', 'separate_chips',
           'check_chip1', 'group_summary']


def suboverscan(data, xbin, chip=2, ybox=5):
    ''' Subtracts overscan region from the input image
    TODO: Resolve -- currently only chip 2 is available & xbin=1 is not tested.
    TODO: Change so that it uses ccdproc after the release of 1.3

    Note
    ----
    Using overscan is not much beneficial than bias subtraction, since there
    does exist some 2D pattern. See "Bias pattern" in the link:
        https://www.naoj.org/Observing/Instruments/FOCAS/ccdinfo.html

    Following is from
        https://www.naoj.org/Observing/Instruments/FOCAS/ccdinfo.html
    updated on 2010-09-28.

    |                      | Chip2           |                    |                     |                     | Chip1           |                    |                     |                         |
    | -------------------- | --------------- | ------------------ | ------------------- | ------------------- | --------------- | ------------------ | ------------------- | ----------------------- |
    |                      | ch1             | ch2                | ch3                 | ch4                 | ch1             | ch2                | ch3                 | ch4                     |
    | gain (e/ADU)         | 2.105           | 1.968              | 1.999               | 1.918               | 2.081           | 2.047              | 2.111               | 2.087                   |
    | readout noise (e)    | 4.3(*1)         | 3.7                | 3.4                 | 3.6                 | 4.2(*1)         | 3.8                | 3.6                 | 4.0                     |
    | active area(*2)      | [9:520,49:4224] | [553:1064,49:4224] | [1081:1592,49:4224] | [1625:2136,49:4224] | [9:520,49:4224] | [553:1064,49:4224] | [1081:1592,49:4224] | [1626:2137,49:4224](*3) |
    | over-scan region(*2) | [521:536,*]     | [537:552,*]        | [1593:1608,*]       | [1609:1624,*]       | [521:536,*]     | [537:552,*]        | [1593:1608,*]       | [1610:1625,*](*3)       |

    (*1) Modification of grounding cables has reduced the readout noise of ch1.

    (*2) These values are for images without binning.

    (*3) There is an extra column at x=1609 of Chip1 which causes a shift in the
        position of ch4.
       if chip == 2:
            gain = [2.105, 1.968, 1.999, 1.918]
            ronoise = [4.3, 3.7, 3.4, 3.6]

    Parameters
    ----------
    ybox: int
        Binning factor to smooth the image along the Y-axis.

    Return
    ------
    ccdout: astropy.nddata.CCDData
        The overscan subtracted image (overscan subtracted but not trimmed).
    '''
#    data = fits.getdata(fname)
#    hdr = fits.getheader(fname)
#    ylen = int(hdr['naxis2'])
#    xbin = int(hdr['bin-fct1'])
#    chip = int(hdr['det-id'])
#    print('{:s} -- chip {:d}'.format(fname, chip))
#    if fname[:-5] != hdr['frameid']:
#        print('\tThe header name (FRAMEID) and file name not coherent.')

    ylen = np.shape(data)[0]

    if chip == 1:
        raise ValueError('Chip 1 is not yet implemented...')

    # In FITS, L[a:b] means L[a], L[a+1], ..., L[b], so total (b-a+1) elements.
    # Also the indexing starts from 1.
    # Finally, FITS uses XY order, while the np.ndarray is using row-column
    # order, i.e., YX order, as most programming languages do.
    # Thus, to port to Python, L[a:b] should become L[b:a-1].
    if xbin == 1:
        if chip == 2:
            print('x Binning 1: I have not checked it yet')
            overscan = [data[:, 520:536],    # in FITS, 521:536
                        data[:, 536:552],    # in FITS, 537:552
                        data[:, 1592:1608],  # in FITS, 1593:1608
                        data[:, 1608:1624]]  # in FITS, 1609:1624
#        else:

    elif xbin == 2:
        if chip == 2:
            overscan = [data[:, 260:276],   # in FITS, 261:276
                        data[:, 276:292],   # in FITS, 277:292
                        data[:, 812:828],   # in FITS, 813:828
                        data[:, 828:844]]   # in FITS, 829:844
            # length in x-direction, including overscan region
            ch_xlen = [276, 276, 276, 276]
#        else:

    overscan_ch = []
    for ch in range(4):
        overscan_pattern = np.average(overscan[ch], axis=1)
        # The original code ``ovsub.cl`` uses ``IMAGES.IMFILTER.BOXCAR`` task for
        # smoothing the overscan region. Same is implemented in astropy.convolution.
        # convolve with option ``boundary='extend'``.
        overscan_smoothed = convolve(overscan_pattern,
                                     Box1DKernel(ybox),
                                     boundary='extend')
        overscan_map = np.repeat(overscan_smoothed, ch_xlen[ch])
        overscan_map = overscan_map.reshape(ylen, ch_xlen[ch])
        overscan_ch.append(overscan_map)

    overscan_map = np.hstack(overscan_ch)
    overscan_subtracted = data - overscan_map
#    ccdout = CCDData(overscan_subtracted, header = hdr, unit='adu')

#    if outputdir != '':
#        ccdout.write(outputdir, overwrite=True)

    return overscan_subtracted


def separate_chips(directory_prefix='', fname_prefix='FCSA', verbose=True):
    ''' Separates files into chips 1 and 2.

    FOCAS has convention that the filename ends with odd number is from
    chip 1 and the even number from chip 2. Based on this convention, this
    function automatically separates images into two chips (make
    subdirectories).

    Parameters
    ----------
    directory_prefix: str, optional
        The directory where the images are stored. Either absolute or relative.
        Set to default ('') if you are currently at the directory where the
        images are stored.

    fname_prefix: str, optional
        Only the files that start with this string will be affected.

    verbose: bool, optional
        Whether to print out the name of the files. This is not necessarily
        in alphabetical order, since the file movement can be done by
        multithreading.

    '''
    if directory_prefix == '':
        directory_prefix = os.getcwd()

    if not os.path.isabs(directory_prefix):  # if relative directory,
        directory_prefix = os.path.join(os.getcwd(), directory_prefix)

    if not os.path.isdir(directory_prefix):  # if data directory does not exist,
        raise NameError('The directory {:s} does not exist'.format(directory_prefix))

    os.makedirs(os.path.join(directory_prefix, 'chip1'), exist_ok=True)
    os.makedirs(os.path.join(directory_prefix, 'chip2'), exist_ok=True)

    for file in os.listdir(directory_prefix):
        oldpath = os.path.join(directory_prefix, file)

        if file.startswith(fname_prefix):
            print(oldpath)
            if file.endswith(("1.fits", "3.fits", "5.fits", "7.fits", "9.fits")):
                    newpath = os.path.join(directory_prefix, 'chip1', file)
                    os.rename(oldpath, newpath)

            elif file.endswith(("0.fits", "2.fits", "4.fits", "6.fits", "8.fits")):
                newpath = os.path.join(directory_prefix, 'chip2', file)
                os.rename(oldpath, newpath)


def group_summary(table, save=False, output='summary.csv'):
    ''' Make a summary csv file.

    Parameters
    ----------
    table: atropy.table.Table
    '''
    grouped = table.group_by(['det-id', 'obs-mod', 'data-typ',
                              'bin-fct1', 'bin-fct2',
                              'exptime'])
    if save:
        grouped.write(output, format='ascii.csv', overwrite=True)

    return grouped


def check_chip1(fits_tab):
    ''' Check whether there is chip1 data in chip2 directory.

    This is made because this tutorial uses chip 2 data only.

    Parameters
    ----------
    fits_tab: astropy.table.Table
    '''

    chip1 = fits_tab['det-id'] == 1
    if not np.count_nonzero(chip1) == 0:
        raise ValueError('There exists some file(s) from chip 1!!\n')