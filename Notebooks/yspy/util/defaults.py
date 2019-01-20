"""
Collection of useful default values
"""

import numpy as np

__all__ = []

MEDCOMB_KEYS_INT = dict(dtype='int',
                        combine_method='median',
                        reject_method=None,
                        unit=None,
                        combine_uncertainty_function=None)

SUMCOMB_KEYS_INT = dict(dtype='int',
                        combine_method='sum',
                        reject_method=None,
                        unit=None,
                        combine_uncertainty_function=None)

MEDCOMB_KEYS_FLT32 = dict(dtype='float32',
                          combine_method='median',
                          reject_method=None,
                          unit=None,
                          combine_uncertainty_function=None)


LACOSMIC = dict(sigclip=4.5, sigfrac=0.3,
                objlim=5.0, gain=1.0, readnoise=6.5,
                satlevel=np.inf, pssl=0.0, niter=4,
                sepmed=False, cleantype='medmask', fsmode='median',
                psfmodel='gauss', psffwhm=2.5, psfsize=7,
                psfk=None, psfbeta=4.765, verbose=False)