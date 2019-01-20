# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
IRSA Moving Object Query Tool
=============================

.. topic:: Revision History

    Work supported by Korea Astronomy and Space science Institute (KASI)
    KMTNet/DEEP-South project in 2018.

    :Originally contributed by: Yoonsoo Bach (dbstn95@gmail.com)
"""
from astropy import config as _config


class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `astroquery.irsa_dust`.
    """
    # maintain a list of URLs in case the user wants to append a mirror
    server = _config.ConfigItem(
        ['http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-query?searchForm=MO'],
        'Name of the irsa_dust server to use.')
    timeout = _config.ConfigItem(
        30,
        'Default timeout for connecting to server.')


conf = Conf()


from .core import IrsaMO, IrsaMOClass

__all__ = ['IrsaMO', 'IrsaMOClass',
           'Conf', 'conf',
           ]
