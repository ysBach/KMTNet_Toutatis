import numpy as np
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.stats import sigma_clip
from scipy import stats

__all__ = ['give_stats']


def give_stats(data, percentiles=[1, 99], N_extrema=None):
    ''' Calculates simple statistics.
    Parameters
    ----------
    data: array-like
        The data to be analyzed.
    percentiles: list-like, optional
        The percentiles to be calculated.
    N_extrema: int, optinoal
        The number of low and high elements to be returned when the whole data
        are sorted. If ``None``, it will not be calculated. If ``1``, it is
        identical to min/max values.

    Example
    -------
    >>> bias = CCDData.read("bias_bin11.fits")
    >>> dark = CCDData.read("pdark_300s_27C_bin11.fits")
    >>> percentiles = [0.1, 1, 5, 95, 99, 99.9]
    >>> stats.give_stats(bias, percentiles=percentiles, N_extrema=5)
    >>> stats.give_stats(dark, percentiles=percentiles, N_extrema=5)
    '''
    data = np.atleast_1d(data)

    result = {}

    d_num = np.size(data)
    d_min = np.min(data)
    d_pct = np.percentile(data, percentiles)
    d_max = np.max(data)
    d_avg = np.mean(data)
    d_med = np.median(data)
    d_std = np.std(data, ddof=1)

    zs = ImageNormalize(data, interval=ZScaleInterval())
    d_zmin = zs.vmin
    d_zmax = zs.vmax

    result["N"] = d_num
    result["min"] = d_min
    result["max"] = d_max
    result["avg"] = d_avg
    result["med"] = d_med
    result["std"] = d_std
    result["percentiles"] = d_pct
    result["zmin"] = d_zmin
    result["zmax"] = d_zmax

    if N_extrema is not None:
        data_flatten = np.sort(data, axis=None)  # axis=None will do flatten.
        d_los = data_flatten[:N_extrema]
        d_his = data_flatten[-1 * N_extrema:]
        result["ext_lo"] = d_los
        result["ext_hi"] = d_his

    return result


def gaussian_pdf(mean, stddev, normed=True):
    '''
    Parameters
    ----------
    mean, stddev: float
        The mu and sigma parameters for the Gaussian normal distribution.
    normed: bool, optional
        If true, the pdf will be returned. If False, the maximum value will be
        normalized to 1, i.e., the sum of the pdf will not be unity.
    '''
    rv = stats.norm(mean, stddev)
    pdf = rv.pdf
    if normed:
        return pdf
    else:
        return lambda x: np.sqrt(2 * np.pi * stddev**2) * pdf(x)


def plot_stats(ax, datas, percentiles=[1, 99], N_extrema=None, scale='log',
               **kwargs):
    ''' Plot statistical summary similar but not identical to Box plots.
    If you do Boxplot from matplotlib for 10 MB data, say, there can be too many
    outliers to plot on the graph which makes the plotting to take a lot of
    time.

    Example
    -------
    >>> bias = CCDData.read("bias_bin11.fits")
    >>> dark_1s = CCDData.read("pdark_300s_27C_bin11.fits").data / 300
    >>> percentiles = [0.1, 1, 5, 95, 99, 99.9]
    >>> f, ax = plt.subplots(1, 1, figsize=(8,8))
    >>> ax, clipped_stats = stats.plot_stats(ax, [bias, dark_1s], percentiles=percentiles, N_extrema=5)
    >>> ax.set_ylabel("ADU")
    >>> ax.set_xticks(np.arange(4))
    >>> ax.set_xticklabels(['', 'Bias', 'Dark_1s', ''])
    >>> plt.tight_layout()
    >>> plt.savefig('bias_dark_analysis.png')
    '''

    datas = np.atleast_1d(datas)
    clipped_stats = []
    for i, data in enumerate(datas):
        x = i + 1
        s = give_stats(data=data, percentiles=percentiles, N_extrema=N_extrema)

        data_clipped = sigma_clip(data, **kwargs)
        n_rej = np.count_nonzero(data_clipped.mask)
        c_avg = np.mean(data_clipped)
        c_med = np.median(data_clipped)
        c_std = np.std(data_clipped, ddof=1)
        clipped_stats.append([c_avg, c_med, c_std])

        # avg, med, std from sigma-clipped data
        ax.errorbar(x, c_avg, yerr=c_std, marker='_',
                    capsize=10, markersize=10,
                    label=f"sig-clipped:\nN = {s['N'] - n_rej}")
        ax.scatter(x, c_med, color='r', marker='x')

        # percentiles, extrema, and zscale from original data
        ax.scatter([x - 0.1] * len(percentiles),
                   s["percentiles"], color='k', marker='>')
        ax.scatter([x + 0.1] * 2, [s["zmin"], s["zmax"]], color='r', marker='<')
        ax.plot(x, s["max"], color='k', marker='_', ms=20)
        ax.plot(x, s["min"], color='k', marker='_', ms=20)

        if N_extrema is not None:
            ax.scatter([x] * N_extrema, s["ext_lo"], color='k', marker='x')
            ax.scatter([x] * N_extrema, s["ext_hi"], color='k', marker='x')

    ax.set_xlim(0, len(datas) + 1)
    ax.set_yscale(scale)
    ax.legend()

    return ax, clipped_stats
