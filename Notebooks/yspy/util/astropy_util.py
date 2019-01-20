from astropy import units as u
import numpy as np


def change_to_quantity(x, desired=None):
    ''' Change the non-Quantity object to astropy Quantity.
    Parameters
    ----------
    x: object changable to astropy Quantity
        The input to be changed to a Quantity. If a Quantity is given, ``x`` is
        changed to the ``desired``, i.e., ``x.to(desired)``.
    desired: astropy Unit, optional
        The desired unit for ``x``.
    Returns
    -------
    ux: Quantity
    Note
    ----
    If Quantity, transform to ``desired``. If ``desired = None``, return it as
    is. If not Quantity, multiply the ``desired``. ``desired = None``, return
    ``x`` with dimensionless unscaled unit.
    '''
    if not isinstance(x, u.quantity.Quantity):
        if desired is None:
            ux = x * u.dimensionless_unscaled
        else:
            ux = x * desired
    else:
        if desired is None:
            ux = x
        else:
            ux = x.to(desired)
    return ux


def col2arr(column):
    ''' Converts the values in an astropy column to numpy array.
    '''
    return np.array(column.tolist())


