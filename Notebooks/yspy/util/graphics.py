import warnings
from matplotlib import rc
from astropy.visualization import ZScaleInterval, ImageNormalize
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from ..math.stats import give_stats

__all__ = ['disable_mplkeymaps', 'znorm', 'zimshow', 'save_thumbnail',
           'CCDData_summary_plot']


def disable_mplkeymaps():
    rc('keymap',
       fullscreen='',
       home='',
       back='',
       forward='',
       pan='',
       zoom='',
       save='',
       quit='',
       grid='',
       yscale='',
       xscale='',
       all_axes=''
       )


def znorm(image):
    return ImageNormalize(image, interval=ZScaleInterval())


def zimshow(ax, image, **kwargs):
    return ax.imshow(image, norm=znorm(image), origin='lower', **kwargs)


def save_thumbnail(data, savename, parameters='zscale', overwrite=False):
    ''' Saves the thumbnail of the data.
    Parameters
    ----------
    data: ndarray
        The data to be used for thumbnail generation
    savename: path-like
        The save path
    scale: 'zscale' or dict
        If ``'zscale'``, ``zimshow`` will be used. If not, it must be a dict
        for kwargs for ``plt.imshow``.
    '''
    savepath = Path(savename)
    if overwrite or (not savepath.exists()):
        plt.close('all')
        fig, ax = plt.subplots(1, 1)
        ax.set_title(savepath.name)
        if parameters == 'zscale':
            im = zimshow(ax, data)
        else:
            im = ax.imshow(data, **parameters)
        fig.colorbar(im, label='ADU')
        plt.show()
        plt.savefig(str(savepath))
    else:
        print(f"{savename} already exists!\n\tPass")
    plt.close('all')


def CCDData_summary_plot(ccd, xscale='linear', yscale='log', output=None):

    # TODO: Use MAD for median uncertainty
    # TODO: Put errorbar for both avg and median
    # TODO: Put min/max, 1/99% percentiles & zscale min/max
    def _put_errorbar(ax, height, stat):
        ax.errorbar(stat['med'], height, xerr=stat['std'], lw=5, capsize=5)
        ax.set_title(
            f"{stat['med']:.3f} +- {stat['std']:.3f} (avg = {stat['avg']:.3f})")

    plt.close('all')

    f, axs = plt.subplots(2, 2, figsize=(16, 9))
    ax_d = axs[0, 0]
    ax_dh = axs[0, 1]
    ax_u = axs[1, 0]
    ax_uh = axs[1, 1]

    data = ccd.data
    try:
        uncert = ccd.uncertainty.array
    except AttributeError:
        uncert = np.ones_like(data)
        warnings.warn('There is no uncertainty in the CCDData, '
                      + 'so uniform (1.0) map is made.')

    s_d = give_stats(data)
    s_u = give_stats(uncert)

    zimshow(ax_d, data)
    ax_d.set_title(f'Pixel Values')

    dh_width = max(3, 10 * s_d['std'])  # at least 3 (ADU)
    dh_xlab = np.arange(int(s_d['min']), int(s_d['max']))
    dh, edges_d = np.histogram(data, bins=dh_xlab)
    ax_dh.bar(edges_d[:-1], dh, width=1, fill=False)
    ax_dh.set_xlabel("Pixel Value (ADU)")
    ax_dh.set_ylabel("# of Pixels")
    ax_dh.set_xlim(s_d['med'] - dh_width / 2, s_d['med'] + dh_width / 2)
    _put_errorbar(ax_dh, 0.0, s_d)

    zimshow(ax_u, uncert)
    ax_u.set_title('Uncertainty')

    uh_width = max(1, 10 * s_u['std'])  # at least 1 (ADU)
    uh_step = max(0.05, uh_width / 20)  # at least 0.05 (ADU)
    uh_xlab = np.arange(s_u['med'] - uh_width / 2,
                        s_u['med'] + uh_width / 2, uh_step)
    # +- 5 sigma with 0.5 sigma interval
    uh, edges_u = np.histogram(uncert, bins=uh_xlab)
    ax_uh.bar(edges_u[:-1], uh, width=uh_step, fill=False)
    ax_uh.set_xlabel("Pixel Uncertainty (ADU)")
    ax_uh.set_ylabel("# of Pixels")
    ax_uh.set_xlim(s_u['med'] - uh_width / 2, s_u['med'] + uh_width / 2)
    _put_errorbar(ax_uh, 0.0, s_u)

    for a in [ax_dh, ax_uh]:
        a.set_xscale(xscale)
        a.set_yscale(yscale)

    plt.suptitle(f"{output}")
    if output is not None:
        plt.savefig(str(output), dpi=80, bbox_inches='tight')
    plt.close('all')
