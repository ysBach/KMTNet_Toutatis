from matplotlib import pyplot as plt
from astropy.modeling.models import Gaussian1D
import numpy as np
import os
from astropy.table import Table
from astropy import units as u


def fake_lamp(linelist_table, name_wave, name_amp, fwhm=4,
              wvlen_min=3000, wvlen_max=9000, show_plot=True):
    ''' Make a fake Gaussian shaped comparison lamp image from given line list.

    Example
    -------
    # Obtain the data table from here:
    #   http://www.ls.eso.org/lasilla/Telescopes/2p2/E1p5M/memos_notes/hearfene_gt15.dat
    COORDLIST = os.path.join('.', 'hearfene.dat')
    HeArFeNe = Table.read(COORDLIST, format='ascii', guess=True)
    graphics.plot_lines(HeArFeNe, 'WAVE', 'INTENS', fwidth=4,
                        wavelength_min=WAVELENGTH_MIN,
                        wavelength_max=WAVELENGTH_MAX)

    '''
    gaussian_stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))
    table = linelist_table.copy()
    lineplot = Gaussian1D(amplitude=0)
    plotted_wavelengths = []
    
    for i in range(len(table)):        
        if ((table[name_amp][i] > 20) 
            and (table[name_wave][i] > wvlen_min)
            and (table[name_wave][i] < wvlen_max)):
            # put as you wish: and not (table[name_ion][i].startswith('Fe'))
            plotted_wavelengths.append(table[name_wave][i])
            lineplot += Gaussian1D(amplitude = table[name_amp][i],
                                   mean = table[name_wave][i],
                                   stddev = gaussian_stddev)

    if show_plot:
        x = np.arange(wvlen_min, wvlen_max)
        plt.plot(x, lineplot(x), lw=1)
        plt.xlabel("Wavelengths (Angstrom)")
        plt.ylabel("Intensity (arbitrary unit)")
        plt.show()
    
    return lineplot


def read_IRAF_linelist(path):
    ''' Read the line list file given by IRAF and return astropy.table.Table
    
    Example
    -------
    FeAr = read_IRAF_linelist('fear.dat')
    HeNeAr = read_IRAF_linelist('henear.dat')
    '''
    datname = os.path.split(path)[-1]
    
    if datname.lower() == 'FeAr.dat'.lower():
        linelist = Table.read(path, format='ascii', guess=True,
                              names=["wavelength", "source"])
    
    elif datname.lower() == 'HeNeAr.dat'.lower():
        linelist = Table.read(path, format='ascii.fixed_width_no_header',
                              col_starts=(0, 12),
                              names=["wavelength", "source"])
    
    linelist["wavelength"].unit = u.angstrom
    
    return linelist