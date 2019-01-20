import inputs
import numpy as np

# For testing
import importlib
importlib.reload(inputs)


#%%
# Load data
#image_skysub = fits.open(inputs.filename[:-4]+'cs.fits')[0].data
image = inputs.hdu[0].data

data_targ = np.loadtxt(inputs.filename+'.targ', dtype=bytes, skiprows=1).astype(str)
data_star = np.loadtxt(inputs.filename+'.star', dtype=bytes).astype(str)

pos_targ  = data_targ[4:6].astype(float)
x_targ    = int(pos_targ[0])
y_targ    = int(pos_targ[1])
trail_len = data_targ[9].astype(float)
trail_ang = data_targ[11].astype(float)
pos_star  = data_star[:,[0,1]].astype(float)

if len(data_star) < 5:
    print('\tLESS THAN 5 STARS LEFT: TERMINATING!')
    pass

#%%
# Cosmic ray erase

from astroscrappy import detect_cosmics

print('\tCosmic-ray removal...')

half_cbox_targ = inputs.cbox_size_targ // 2
inmask = np.zeros_like(image).astype(bool)
inmask[y_targ-half_cbox_targ:y_targ+half_cbox_targ, \
       x_targ-half_cbox_targ:x_targ+half_cbox_targ] = True
# detect_cosmics erroneously erases the target because it is kind of
# tiny bright point. Hence, I used the inmask to protect the target region.

crmask, image_cr = detect_cosmics(image, \
                                  inmask=inmask, \
                                  psfmodel='moffat', \
                                  satlevel=np.infty,\
                                  readnoise=inputs.RONOISE,\
#                                  sepmed=False,\
                                  cleantype='medmask',\
#                                  fsmode='median',\
                                  verbose=False)
# print(detect_cosmics.__doc__)
#   To reproduce the most similar behavior to the original LA Cosmic
#   (written in IRAF), set  inmask = None, satlevel = np.inf, sepmed=False,
#   cleantype='medmask', and fsmode='median'.
# cleantype='meanmask' (default) seems to make 'holes' in CR-detected region..
#   * inmask[y,x] near the target should be set, especially when the observation
#     was made in non-sidereal tracking mode. The CR-rejection may misunderstand
#     the target as a point-like cosmic-ray.

image_reduc = image_cr.copy()

#import matplotlib.pyplot as plt
#plt.figure(figsize=(15,15))
#plt.imshow(image_cr, vmin=20, vmax=150, origin='lower')
#diff_cr = (image_cr-image)/image_cr
#plt.imshow(diff_cr, vmin=-.2, vmax=diff_cr.max(), origin='lower', cmap='jet',alpha=0.5)
#plt.colorbar()
#
#plt.figure(figsize=(15,15))
#plt.imshow(image_cr, origin='lower', cmap='gray_r', vmax=900)
#plt.colorbar()
print('\t\tDone!\n')

#%%
## Subtract sky (background subtraction)
#from photutils import Background2D
#
#print('\tBackground estimation...')
#
#bkg_sex = Background2D(image_cr, \
#                       box_size=inputs.box, \
#                       filter_size=inputs.filt)
## By default, Background2D uses sigma_clip of 3-sigma 10-iters,
## and the background estimator as SExtractorBackground.
#
#image_skysub = image_cr - bkg_sex.background
##import matplotlib.pyplot as plt
##img_sex_diff = (image_cr - bkg_sex.background)/image_cr * 100
##plt.figure(figsize=(15,15))
##plt.imshow(image_cr, origin='lower', vmin=300, vmax=500)
##plt.imshow(img_sex_diff, origin='lower', vmin=0, vmax=100)
##plt.imshow(image_skysub, origin='lower', vmin=0, vmax=100)
##plt.colorbar()
#print('\t\tDone!\n')
#
## Save the CR-rejected & sky-subtracted image as a new file.
## The new file will have 'cs', which means 'CR & Sky'.
#
#if inputs.save_skysub:
#    from astropy.io import fits
#    from astropy.time import Time
#    newfits = fits.PrimaryHDU(data=image_reduc, header=inputs.header)
#    newhdul = fits.HDUList([newfits])
#    newhdul[0].header['history'] = "Cosmic-ray rejection and sky-subtracted"
#    newhdul[0].header['history'] = "at time {0}".format(Time.now())
#    newhdul.writeto(inputs.filename[:-4]+'cs.fits', output_verify='ignore')
#    # Some of IAO/OAO fits files do not have astropy standard HDU headers.
#    # output_verify should be set 'ignore' for code to run correctly.

#image_reduc = image_skysub.copy()

#%%
# Get FWHM from the target by 2-D Moffat fitting
from astropy.stats import sigma_clipped_stats
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import functional_models
from photutils import find_peaks

print('\tTarget Moffat fitting...')

# Crop the image and primitively subtract sky near the target for fitting.
image_targ = image_reduc[y_targ-half_cbox_targ:y_targ+half_cbox_targ, \
                          x_targ-half_cbox_targ:x_targ+half_cbox_targ].copy()

mean, med, std = sigma_clipped_stats(image_targ, sigma=3.0, iters=10)
image_targ    -= med  #primitive sky subtraction
peaks          = find_peaks(image_targ, threshold=3*std, border_width=2)
fit_y, fit_x   = np.mgrid[:2*half_cbox_targ, :2*half_cbox_targ]
fitter         = LevMarLSQFitter()

print('\t\t{0} peak(s) detected... Finding best one...'.format(len(peaks)))


def give_moffat(x_peak, y_peak):
    moffat_init  = functional_models.Moffat2D(amplitude=image_targ.max(),\
                                          x_0=x_peak, \
                                          y_0=y_peak, \
                                          bounds={'gamma':(1., 10.), \
                                                  'amplitude':(np.max(image_targ)/1.5,\
                                                               2*np.max(image_targ) )} )
    #If the amplitude is not bounded, it sometimes finds "negative" amplitude fitting...
    return fitter(moffat_init, fit_x, fit_y, image_targ)

def give_var(fit, x_peak, y_peak):
    peak_targ  = fit.amplitude.value
    residual   = image_targ - fit(fit_x, fit_y)
    test_image = residual[x_peak[0]-2:x_peak[0]+3,
                          y_peak[0]-2:y_peak[0]+3].copy()
    # make 5 by 5 test image and get max_min variation near the target
    # Since photutils gives 'x_peak' as 'row' of numpy, one must use
    # x_peak as row as above.
    variation  = test_image.max() - test_image.min()
    return variation/peak_targ


if len(peaks) == 0:
    x_peak = half_cbox_targ * np.ones(1).astype(int)
    y_peak = half_cbox_targ * np.ones(1).astype(int)
    moffat = give_moffat(x_peak=x_peak, y_peak=y_peak)
    var    = give_var(moffat, x_peak=x_peak, y_peak=y_peak)
else: # Use the one with highest peak as the first guess
    peaks.sort('peak_value')
    x_peak = peaks[-1]['x_peak'] * np.ones(1).astype(int)
    y_peak = peaks[-1]['y_peak'] * np.ones(1).astype(int)
    moffat = give_moffat(x_peak=x_peak, y_peak=y_peak)
    var    = give_var(moffat, x_peak=x_peak, y_peak=y_peak)
    if(var > 1.):
        i = -1
        while(var > 1.):
            i += 1
            if i == len(peaks)-1:
                break
            # N.B. index [len(peaks)-1] has been used as first guess.
            x_peak = peaks[i]['x_peak'] * np.ones(1).astype(int)
            y_peak = peaks[i]['y_peak'] * np.ones(1).astype(int)
            moffat = give_moffat(x_peak=x_peak, y_peak=y_peak)
            var    = give_var(moffat, x_peak=x_peak, y_peak=y_peak)


if (var > 1.):
    raise Warning('Residual image has too large variation near the target! Not a good fit..')

# Get FWHM
x_targ_fit   = moffat.x_0.value - half_cbox_targ + x_targ
y_targ_fit   = moffat.y_0.value - half_cbox_targ + y_targ
pos_targ_fit = np.array([x_targ_fit, y_targ_fit]).reshape((2,))
width        = moffat.gamma.value
power        = moffat.alpha.value
peak_targ    = moffat.amplitude.value
FWHM_moffat  = 2 * width * np.sqrt( 2**(1/power) - 1)

print('\t\tDone!')
print('\t\tMoffat FWHM = {0:.3f}\n'.format(FWHM_moffat))

#from astropy.modeling.models import custom_model
## Define model
#@custom_model
#def Elliptical_Moffat2D(x, y, \
#                        N_sky = 1., amplitude = 1., phi=0.1, power = 1.,\
#                        x_0 = inputs.cbox_size_targ, y_0 = inputs.cbox_size_targ,\
#                        width_x = 1., width_y = 1.):
#    c = np.cos(phi)
#    s = np.sin(phi)
#    A = (c / width_x) ** 2 + (s / width_y)**2
#    B = (s / width_x) ** 2 + (c / width_y)**2
#    C = 2 * s * c * (1/width_x**2 - 1/width_y**2)
#    denom = (1 + A * (x-x_0)**2 + B * (y-y_0)**2 + C*(x-x_0)*(y-y_0))**power
#    return N_sky + amplitude / denom
#
#fit_y, fit_x = np.mgrid[:2*cbox_size, :2*cbox_size]
#moffat_init  = Elliptical_Moffat2D(amplitude=image_targ.max(), \
#                                   x_0=cbox_size, y_0=cbox_size, \
#                                   bounds={'phi':(-np.pi/2,np.pi/2),\
#                                           'N_sky':(0., 0.)})
#fit_moffat = LevMarLSQFitter()
#moffat     = fit_moffat(moffat_init, fit_x, fit_y, image_targ)
#x_targ_fit = moffat.x_0.value
#y_targ_fit = moffat.y_0.value
#phi_moffat = moffat.phi.value
#FWHM_moffat = np.array([2 * moffat.width_x.value * np.sqrt( 2**(1/moffat.power.value) - 1),\
#                        2 * moffat.width_y.value * np.sqrt( 2**(1/moffat.power.value) - 1)])
#%%
# Show
if inputs.verify_peak:
    if len(peaks) > 1 or len(peaks) == 0:
        if len(peaks)>1:
            print('\tFitting, image, and residual plots.')
            print('\tThe x-markers are found peaks.')
            print('\tThe star-marker is final fit.')
            print('\tIn residual, parenthesis: mean, median, and stdev.')
        else:
            print('\tNo peaks detected. Check image carefully.')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,8))
        ax1 = plt.subplot(1,3,1)
        im1 = ax1.imshow(image_targ, origin='lower')
        plt.colorbar(im1, orientation='horizontal')
        plt.title('cropped image')
        ax2 = plt.subplot(1,3,2)
        im2 = ax2.imshow(moffat(fit_x,fit_y), origin='lower') #target
        plt.colorbar(im2, orientation='horizontal')
        plt.title('2D circ Moffat fitting')
        ax3  = plt.subplot(1,3,3)
        residual = image_targ - moffat(fit_x, fit_y)
        mean = np.mean(residual)
        med  = np.median(residual)
        std  = np.std(residual)
        im3  = ax3.imshow(residual, origin='lower')
        plt.colorbar(im3, orientation='horizontal')
        for i in range(0, len(peaks)):
            ax1.plot(peaks['x_peak'][i], peaks['y_peak'][i], marker='x')
            ax1.plot(moffat.x_0.value, moffat.y_0.value, marker='*', alpha=0.7)
            ax2.plot(peaks['x_peak'][i], peaks['y_peak'][i], marker='x')
            ax2.plot(moffat.x_0.value, moffat.y_0.value, marker='*', alpha=0.7)
            ax3.plot(peaks['x_peak'][i], peaks['y_peak'][i], marker='x')
            ax3.plot(moffat.x_0.value, moffat.y_0.value, marker='*', alpha=0.7)
        plt.title('residual\n({0:.2f}, {1:.2f}, {2:.2f})'.format(mean, med, std))
        plt.suptitle('Moffat: FWHM={0:.2f}, width={1:.2f}, amplit={2:.2f}'.format(FWHM_moffat, width, peak_targ))
        plt.show()



#%% Get star_fit
# To each of the star positions, find the center of the star.
# This should be done because the catalog center may not be exactly the same
# as that of image.
from photutils import centroid_2dg
from scipy.spatial.distance import cdist

def find_star_fit(pos_star, \
                  peak_bezel=0, \
                  peak_max_dist=10, \
                  half_cbox=inputs.cbox_size_star//2, \
                  peak_threshold=5.):
    '''
    Calculates the plausible stellar positions in the image.

    From the input 'positions', which contains the star's catalog positions in
    image XY coordinate, this function calculates the fitted star position.
    It crops the image centered at the catalog center +- cbox_size.
    Then do the 'find_peak', and rejects the image if the farthest peak
    distance is larger than peak_max_dist. This may mean that there are
    more than two stars in the field.
    '''
    output = np.zeros_like(pos_star)
    for i in range(0, len(pos_star)):
        x_star     = pos_star[i,0].copy().astype(int)
        y_star     = pos_star[i,1].copy().astype(int)
        image_cbox = image_reduc[y_star-half_cbox:y_star+half_cbox,\
                                 x_star-half_cbox:x_star+half_cbox].copy()
        mean, med, std = sigma_clipped_stats(image_cbox, sigma=3.0, iters=10)
        image_cbox    -= med  # primitive sky subtraction
        threshold      = peak_threshold * std
        peaks          = find_peaks(image_cbox, threshold=threshold)
        if len(peaks) == 0:
            output[i, :] = -1, -1
            continue
        pos_peak = np.zeros((len(peaks),2))
        if len(peaks) > 1:
            for j in range(0, len(peaks)):
                pos_peak[j, 0] = peaks[j]['x_peak']
                pos_peak[j, 1] = peaks[j]['y_peak']
            dist = cdist(pos_peak, pos_peak)
            if dist.max() > peak_max_dist:
                output[i, :] = -1, -1
                continue
#         plt.imshow(image_cbox, origin='lower')
#
#x_star = pos_star[i,0].astype(int)
#y_star = pos_star[i,1].astype(int)
#image_cbox = image_reduc[y_star-inputs.cbox_size_star//2:y_star+inputs.cbox_size_star//2,\
#                          x_star-inputs.cbox_size_star//2:x_star+inputs.cbox_size_star//2]
#plt.imshow(image_cbox, origin='lower')
#plt.plot(np.mean(pos_peak[:,0]), np.mean(pos_peak[:,1]), '*')
#fitted_x, fitted_y    = centroid_2dg(image_cbox)
#plt.plot(fitted_x, fitted_y, 'X')
#
#
#
#mean, med, std = sigma_clipped_stats(image_cbox, sigma=3.0, iters=10)
#threshold      = med + (5 * std)
#peaks          = find_peaks(image_cbox, threshold=threshold)
#fitted_x, fitted_y    = centroid_2dg(image_cbox)
#fitted_x + pos_star[i,0] - cbox_size
#fitted_y + pos_star[i,1] - cbox_size
#       from photutils import centroid_com, centroid_1dg, centroid_2dg
#       plt.imshow(image_cbox)
#       x1, y1 = centroid_com(image_cbox)
#       x2, y2 = centroid_2dg(image_cbox)
#       x3, y3 = centroid_1dg(image_cbox)
#       plt.plot(x1, y1, markersize=15, marker='*', color='blue')
#       plt.plot(x2, y2, markersize=15, marker='*', color='red')
#       plt.plot(x3, y3, markersize=15, marker='*', color='white')
#      I would conclude that 2dg gives more appropriate centers in most cases!
        mask_lo = image_cbox < -3*std
        mask_hi = image_cbox > peaks['peak_value'].max()
        # change pixels outside the above range to median (i.e., 0)
        image_cbox[np.logical_or(mask_lo, mask_hi)] = 0
        fitted_x, fitted_y = centroid_2dg(image_cbox)
        output[i, 0] = fitted_x + pos_star[i,0] - half_cbox
        output[i, 1] = fitted_y + pos_star[i,1] - half_cbox
#        plt.imshow(image_cbox)
#        plt.colorbar()
#        plt.plot(fitted_x, fitted_y, '*')
#        plt.show()
#        plt.cla()
#        print('{0} {1:5.2f} {2:5.2f}'.format(i, fitted_x, fitted_y))
    return output

print('\tStar position fitting...')

max_dist     = 1.*data_targ[9].astype(float) + 3.
pos_star_fit = find_star_fit(pos_star, image_reduc, peak_max_dist=max_dist)

if len(pos_star_fit) < 5:
    print('\tLESS THAN 5 STARS LEFT: TERMINATING!')
    pass

mask_fit     = (pos_star_fit[:,0] >= 0)
nrej         = np.size(mask_fit) - np.count_nonzero(mask_fit)

if nrej > 0:
    print('\t\t{0} stars rejected'.format(nrej))

pos_star     = pos_star[mask_fit]
pos_star_fit = pos_star_fit[mask_fit]
data_star    = data_star[mask_fit, :]
data_star[:,[0,1]] = pos_star_fit

np.savetxt(inputs.filename + '.star', data_star, fmt='%s')

print('\t\tDone!\n')
# TODO: Check whether DAOFIND and then match with 'closest' will work better.




#%%
# Do aperture photometry

from photutils import CircularAperture as CircAp
from photutils import CircularAnnulus as CircAn
from astropy.stats import sigma_clip
from photutils import RectangularAperture as RectAp
from photutils import RectangularAnnulus as RectAn

print('\tAperture photometry...')

ronoise = inputs.RONOISE
gain    = inputs.GAIN



def sky_fit(all_sky, method='Mode', sky_nsigma=3, sky_iter=5, \
            mode_option='sex', med_factor=2.5, mean_factor=1.5):
    '''
    method = mean, median, Mode
    Mode : analogous to Mode Estimator Background of photutils.
    mode_option is needed only when method='Mode':
        sex  == (med_factor, mean_factor) = (2.5, 1.5)
        IRAF == (med_factor, mean_factor) = (3, 2)
        MMM  == (med_factor, mean_factor) = (3, 2)
    '''
    if method == 'mean':
        return np.mean(all_sky), np.std(all_sky)
    elif method == 'median':
        return np.median(all_sky), np.std(all_sky)
    elif method == 'Mode':
        sky_clip   = sigma_clip(all_sky, sigma=sky_nsigma, iters=sky_iter)
        sky_clipped= all_sky[np.invert(sky_clip.mask)]
        nsky       = np.count_nonzero(sky_clipped)
        mean       = np.mean(sky_clipped)
        med        = np.median(sky_clipped)
        std        = np.std(sky_clipped)
        nrej       = len(all_sky) - len(sky_clipped)
        if mode_option == 'IRAF':
            if (mean < med):
                sky = mean
            else:
                sky = 3 * med - 2 * mean

        elif mode_option == 'MMM':
            sky = 3 * med - 2 * mean

        elif mode_option == 'sex':
            if (mean - med) / std > 0.3:
                sky = med
            else:
                sky = (2.5 * med) - (1.5 * mean)
        else:
            raise ValueError('mode_option not understood')

        return sky, std, nsky, nrej


def give_gauss(image_in, x_peak, y_peak):
    gauss_init  = functional_models.Gaussian2D(amplitude=image_in.max(),\
                                          x_mean=x_peak, \
                                          y_mean=y_peak, \
                                          bounds={'amplitude':(np.max(image_in)/1.5,\
                                                               2*np.max(image_in) )} )
    #If the amplitude is not bounded, it sometimes finds "negative" amplitude fitting...
    return fitter(gauss_init, fit_x, fit_y, image_in)

def give_theta(number_of_stars=5):
    '''
    Select appropriatly bright stars and get 'theta' value.
    This is necessary because the trailing direction is not exactly the
    same as that obtained from ephemeris.
    '''
    # TODO: Will it be better to get 'trail' from here, than ephemeris, like theta?
    mag  = data_star[:,4]
    ind  = len(mag)//2
    half_cbox_star = inputs.cbox_size_star // 2
    fit_y, fit_x   = np.mgrid[:2*half_cbox_star, :2*half_cbox_star]
    thetas = np.zeros((number_of_stars))
    for i in range(0, number_of_stars):
        x_tmp, y_tmp = pos_star_fit[ind-2+i].round().astype(int)
        image_tmp    = image_reduc[y_tmp-half_cbox_star:y_tmp+half_cbox_star, \
                                   x_tmp-half_cbox_star:x_tmp+half_cbox_star].copy()
        mean, med, std = sigma_clipped_stats(image_tmp, sigma=3.0, iters=10)
        image_tmp   -= med  #primitive sky subtraction
        gauss_fit    = give_gauss(image_tmp, half_cbox_star, half_cbox_star)
        sig_x = gauss_fit.x_stddev.value
        sig_y = gauss_fit.y_stddev.value
        th    = gauss_fit.theta.value
#        print(sig_x, sig_y,np.rad2deg(th % (2*np.pi)))
#        plt.imshow(image_tmp)
#        plt.imshow(gauss_fit(fit_x, fit_y), origin='lower')
        thetas[i]    = th % (2*np.pi)
        if thetas[i] > np.pi:
            thetas[i] = thetas[i] - np.pi
        if sig_x < sig_y:
            thetas[i] = thetas[i] - np.pi/2
#        print(thetas[i])
    theta = np.median(thetas)
    return theta

def do_Rect_phot(pos, FWHM, trail, ap_min=3., ap_factor=1.5, \
                 win=None, wout=None, hout=None,\
                 sky_nsigma=3., sky_iter=10 ):
    if win == None:
        win  = 4 * FWHM + trail
    if wout == None:
        wout = 8 * FWHM + trail
    if hout == None:
        hout = 8 * FWHM
    N = len(pos)
    if pos.ndim == 1:
        N = 1
    theta    = give_theta(number_of_stars=5)
    an       = RectAn(pos, w_in=win, w_out=wout, h_out=hout, theta=theta)
    ap_size  = np.max([ap_min, ap_factor*FWHM])
    aperture = RectAp(pos, w=(trail+ap_size), h=ap_size, theta=theta)
    flux     = aperture.do_photometry(image_reduc, method='exact')[0]
    # do phot and get sum from aperture. [0] is sum and [1] is error.
    #For test:
#FWHM = FWHM_moffat.copy()
#trail=trail_len.copy()
#win  = 4 * FWHM + trail
#wout = 8 * FWHM + trail
#hout = 8 * FWHM
#N=len(pos_star_fit)
#an       = RectAn(pos_star_fit, w_in=win, w_out=wout, h_out=hout, theta=(theta+np.pi/2))
#ap_size  = 1.5*FWHM_moffat
#aperture = RectAp(pos_star_fit, w=(trail+ap_size), h=ap_size, theta=(theta+np.pi/2))
#flux     = aperture.do_photometry(image_reduc, method='exact')[0]
#plt.figure(figsize=(12,12))
#plt.imshow(image_reduc, origin='lower', vmin=-10, vmax=1000)
#an.plot(color='white')
#aperture.plot(color='red')
    flux_ss  = np.zeros(N)
    error    = np.zeros(N)
    for i in range(0, N):
        mask_an    = (an.to_mask(method='center'))[i]
        #   cf: test = mask_an.cutout(image_reduc) <-- will make cutout image.
        sky_an     = mask_an.apply(image_reduc)
        all_sky    = sky_an[np.nonzero(sky_an)]
        # only annulus region will be saved as np.ndarray
        msky, stdev, nsky, nrej = sky_fit(all_sky, method='Mode', mode_option='sex')
        area       = aperture.area()
        flux_ss[i] = flux[i] - msky*area  # sky subtracted flux
        error[i]   = np.sqrt( flux_ss[i]/gain \
                           + area * stdev**2 \
                           + area**2 * stdev**2 / nsky )
        if inputs.star_img_save:
            from matplotlib import pyplot as plt
            mask_ap    = (aperture.to_mask(method='exact'))[i]
            star_ap_ss = mask_ap.apply(image_reduc-msky)
            sky_an_ss  = mask_an.apply(image_reduc-msky)
            plt.suptitle('{0}, Star ID={1} ({nsky:3d} {nrej:3d} {msky:7.2f} {stdev:7.2f})'.format(
                    inputs.filename, i, nsky=nsky, nrej=nrej, msky=msky, stdev=stdev ))
            ax1 = plt.subplot(1,2,1)
            im1 = ax1.imshow(sky_an_ss, origin='lower')
            plt.colorbar(im1, orientation='horizontal')
            ax2 = plt.subplot(1,2,2)
            im2 = ax2.imshow(star_ap_ss, origin='lower')
            plt.colorbar(im2, orientation='horizontal')
            plt.savefig('{0}.star{1}.png'.format(inputs.filename, i))
            plt.clf()
        if pos.ndim > 1:
            print('\t[{x:7.2f}, {y:7.2f}], {nsky:3d} {nrej:3d} {msky:7.2f} {stdev:7.2f} {flux:7.1f} {ferr:3.1f}'.format(\
                          x=pos[i][0], y=pos[i][1], \
                          nsky=nsky, nrej=nrej, msky=msky, stdev=stdev,\
                          flux=flux_ss[i], ferr=error[i]))
    return flux_ss, error

def do_Circ_phot(pos, FWHM, ap_min=3., ap_factor=1.5, rin=None, rout=None, \
                   sky_nsigma=3., sky_iter=10):
    '''
    In rigorous manner, we should use
    error = sqrt (flux / epadu + area * stdev**2 + area**2 * stdev**2 / nsky)
    as in http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?phot .
    Here flux == aperture_sum - sky, i.e., flux from image_reduc.
        stdev should include sky error AND ronoise.
    '''
    if rin == None:
        rin  = 4 * FWHM
    if rout == None:
        rout = 6 * FWHM
    N = len(pos)
    if pos.ndim == 1:
        N = 1
    an       = CircAn(pos, r_in=rin, r_out=rout)
    ap_size  = np.max([ap_min, ap_factor*FWHM])
    aperture = CircAp(pos, r=ap_size)
    flux     = aperture.do_photometry(image_reduc, method='exact')[0]
    # do phot and get sum from aperture. [0] is sum and [1] is error.
#For test:
#N=len(pos_star_fit)
#an       = CircAn(pos_star_fit, r_in=4*FWHM_moffat, r_out=6*FWHM_moffat)
#ap_size  = 1.5*FWHM_moffat
#aperture = CircAp(pos_star_fit, r=ap_size)
#flux     = aperture.do_photometry(image_reduc, method='exact')[0]
    flux_ss  = np.zeros(N)
    error    = np.zeros(N)
    for i in range(0, N):
        mask_an    = (an.to_mask(method='center'))[i]
        #   cf: test = mask_an.cutout(image_reduc) <-- will make cutout image.
        sky_an     = mask_an.apply(image_reduc)
        all_sky    = sky_an[np.nonzero(sky_an)]
        # only annulus region will be saved as np.ndarray
        msky, stdev, nsky, nrej = sky_fit(all_sky, method='Mode', mode_option='sex')
        area       = aperture.area()
        flux_ss[i] = flux[i] - msky*area  # sky subtracted flux
        error[i]   = np.sqrt( flux_ss[i]/gain \
                           + area * stdev**2 \
                           + area**2 * stdev**2 / nsky )
# To know rejected number, uncomment the following.
#        plt.imshow(sky_an, cmap='gray_r', vmin=-20)
#        plt.colorbar()
#        plt.show()
#        mask_ap  = (aperture.to_mask(method='exact'))[i]
#        star_ap  = mask_ap.apply(image_reduc)
#        plt.imshow(star_ap)
#        plt.colorbar()
#        plt.show()
#        plt.cla()
        if pos.ndim > 1:
            print('\t[{x:7.2f}, {y:7.2f}], {nsky:3d} {nrej:3d} {msky:7.2f} {stdev:7.2f} {flux:7.1f} {ferr:3.1f}'.format(\
                          x=pos[i][0], y=pos[i][1], \
                          nsky=nsky, nrej=nrej, msky=msky, stdev=stdev,\
                          flux=flux_ss[i], ferr=error[i]))
    return flux_ss, error

print('\tStars sky estimation')
print('\t(image X,  image Y, nsky, nrej, sky, sky_std, flux, flux_err)')
flux_targ, error_targ = do_Circ_phot(pos_targ_fit, FWHM=FWHM_moffat,
                                     ap_factor=1.5)
flux_star, error_star = do_Rect_phot(pos_star_fit, FWHM=FWHM_moffat,
                                     trail=trail_len)


#    Do the histogram centroid as IRAF's FITSKYPARS centroid algorithm.
# Define the fillowing hist_sclip and put the latter into sky_fit.
#
#def hist_sclip(data, nsigma, iters, bin_size):
#    if nsigma/bin_size % 1 != 0:
#        raise ValueError('khist is not integer multiple of binsize.')
#    stdev   = np.std(data)
#    med     = np.median(data)
#    nbin    = int(2. * nsigma / bin_size)
#    sig_lo  = med - nsigma * stdev
#    sig_hi  = med + nsigma * stdev
#    hist, bin_edges = np.histogram(data, bins=nbin,\
#                                   range=(sig_lo, sig_hi))
#    return hist, bin_edges
#
#        method == 'centroid' or 'sex'
#        hist, bin_edges = hist_sclip(sky_data, khist, sky_clip_iter, binsize)
#        plt.hist(hist, bin_edges)
#        bin_mean = np.zeros(len(bin_edges)-1)
#        for i in range(0, len(bin_mean)):
#            bin_mean[i] = np.mean((bin_edges[i], bin_edges[i+1]))
#        sky_cent = np.sum( hist * bin_edges[:-1] ) / np.sum(hist)
#        return sky_cent
#%% TESTING
#
#mean, med, std = sigma_clipped_stats(image_targ, sigma=3.0, iters=10)
#image_targ    -= med  #primitive sky subtraction
#peaks          = find_peaks(image_targ, threshold=3*std, border_width=2)
#
#print('\t\t{0} peak(s) detected... Finding best one...'.format(len(peaks)))
#
#
#an1       = CircAn(pos_targ_fit, r_in=4*FWHM_moffat, r_out=6*FWHM_moffat)
#ap_size1  = np.max([3, 1.5*FWHM_moffat])
#aperture1 = CircAp(pos_targ_fit, r=ap_size1)
#
#an3       = RectAn(pos_star_fit, w_in=20, w_out=30, h_out=20, theta=1.06)
#aperture3 = RectAp(pos_star_fit, w=5, h=3, theta=0.5)
#
#
#plt.figure(figsize=(12,12))
#plt.imshow(image_reduc, origin='lower', vmin=-10, vmax=1000)
#an1.plot(color='white')
#aperture1.plot(color='red')
#an3.plot(color='white', linewidth=1)
#aperture3.plot(color='red')

#plt.colorbar()
#flux     = aperture.do_photometry(image_reduc)[0]
## do phot and get sum from aperture. [0] is sum and [1] is error.
#mask_an  = an.to_mask(method='exact')[0]
## default method (exact) will use 'subpixels', which may not be good.
## But I am using this because it is IRAF default.
##   cf: test = mask_an.cutout(image_reduc) <-- will make cutout image.
#plt.imshow(test)
#sky_an   = mask_an.apply(image_reduc)        # only annulus region will be saved as np.ndarray
#sky_clip = sigma_clip(sky_an, sigma=sky_sigma, iters=sky_iter)
##    nrej     = np.count_nonzero(sky_an) - np.count_nonzero(sky_clip)
#nsky     = np.count_nonzero(sky_clip)
#msky     = np.mean(sky_clip)
#stdev    = np.std(sky_clip)
#area     = aperture.area()
#flux_ss  = flux - msky*area  # sky subtracted flux
#error    = np.sqrt( flux_ss/gain \
#                   + area * stdev**2 \
#                   + area**2 * stdev**2 / nsky )

#sky_targ     = np.mean(sky_targ_clip)
#sky_std_targ = np.std(sky_targ_clip)
#targ_error   = np.sqrt( (flux_targ - sky_targ*aper_targ.area())/gain \
#                       + aper_targ.area() * sky_std_targ**2 \
#                       + aper_targ.area()**2 * sky_std_targ**2 / ann_targ.area() )
## N.B. calc_total_error should get "background subtracted" image.
## N.B. The second input should include "ALL" errors except for Poisson errors.
#
#phot_targ = APPHOT(image_reduc, apertures=aper_targ, error=errors)
#aper_star = CircAp(pos_star_fit, r=aper_size)
#phot_star = APPHOT(image_reduc, apertures=aper_star, error=errors)
#
#
##plt.figure(figsize=(15,15))
##plt.imshow(image_reduc, origin='lower', vmax=200, vmin=10)
##aper_star.plot(color='green', lw=5, alpha=0.8)
##plt.figure(figsize=(15,15))
##plt.imshow(image_reduc, origin='lower', vmax=400)
##aper_targ.plot(color='red')



#%%
# change instrumental flux to instrumental magnitude

def flux2mag(flux, ferr=0, zero=0):
    '''
    return instrumental magnitude from instrumental flux.
    flux = instrumental flux
    ferr = instrumental flux error (same unit with flux)
    zero = instrumental magnitude zero point
    '''
    zeros = flux<=0
    flux[zeros] = 10**(99)
    mag  = zero - 2.5 * np.log10(flux) + np.log10(inputs.EXPTIME)
    merr = 1/(np.log(10)*0.4) * ferr/flux
    return mag, merr

mag_targ, merr_targ = flux2mag(flux_targ, error_targ)
mag_star, merr_star = flux2mag(flux_star, error_star)

with open(inputs.filename+'.targ.mag', 'w+') as file:
    file.write('{0:.4f} {1:.4f} {2:.4f}\n'.format(mag_targ[0], merr_targ[0], FWHM_moffat))

with open(inputs.filename+'.star.mag', 'w+') as file:
    for i in range(0, len(mag_star)):
        file.write('{0:.4f} {1:.4f}\n'.format(mag_star[i], merr_star[i]))
# The first row of .mag file is the target.


print('\t\tDone!\n')








