# -*- coding: utf-8 -*-

from ..postproc import background
import numpy as np

#    data1 = np.array([[1, 1, 1, 1, 1],
#                      [1, 2, 2, 2, 1],
#                      [1, 2, 3, 2, 1],
#                      [1, 2, 2, 2, 1],
#                      [1, 1, 1, 1, 1]])


def test_sky_fit_mean_iter0():
    fake_sky = np.array([1, 1, 2, 2, 3, 3, 100])
    values = background.sky_fit(fake_sky, method='mean', sky_iter=0)
    expect = [16.0, 34.301186984876026, 7, 0]
    assert values == expect

def test_sky_fit_mean_nsigma1_iter1():
    fake_sky = np.array([1, 1, 2, 2, 3, 3, 100])
    values = background.sky_fit(fake_sky, method='mean', sky_nsigma=1, 
                                sky_iter=1)
    expect = [2.0, 0.81649658092772603, 6, 1]
    assert values == expect
    
def test_sky_fit_med_iter0():
    fake_sky = np.array([1, 1, 2, 2, 3, 3, 100])
    values = background.sky_fit(fake_sky, method='median', sky_iter=0)
    expect = [2.0, 34.301186984876026, 7, 0]
    assert values == expect
    
def test_sky_fit_med_nsigma1_iter1():
    fake_sky = np.array([1, 1, 2, 2, 3, 3, 100])
    values = background.sky_fit(fake_sky, method='median', sky_nsigma=1, 
                                sky_iter=1)
    expect = [2.0, 0.81649658092772603, 6, 1]
    assert values == expect
    
def test_sky_fit_sex_nsigma1_iter1():
    fake_sky = np.array([1, 1, 2, 2, 3, 3, 100])
    values = background.sky_fit(fake_sky, method='mode', sky_nsigma=1, 
                                sky_iter=1)
    expect = [2.0, 0.81649658092772603, 6, 1]
    assert values == expect
    
