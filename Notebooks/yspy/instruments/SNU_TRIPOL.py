

useful_header_keys = ["NAXIS1", "NAXIS2", "DATE", "OBJECT",
                      "EXPOS", "FILTER", "XBIN", "YBIN", "CCD_TEMP",
                      "CCD_COOL", "EGAIN"]
''' The useful parameters for TRIPOL
'''

# useless_OBJECT = ['test', 'focus', 'focusin', 'focusout']
# ''' The useless frames have lower-cased header OBJECT values in this list.
# Example
# -------
# >>> from astropy.io import fits
# >>> from pathlib import Path
# >>> from yspy.instruments.SNU_TRIPOL import useless_OBJECT
# >>> paths = Path('.').glob('*.fits')
# >>> for f in paths:
# >>>     try:
# >>>         hdr_object = fits.getval(f, 'OBJECT', extension=0).lower()
# >>>     except:
# >>>         f.unlink()
# >>>     if hdr_object in useless_OBJECT:
# >>>         f.unlink()
# '''