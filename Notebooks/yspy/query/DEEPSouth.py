import warnings
from astropy.table import Table, vstack, Column
from pathlib import Path
import numpy as np
import pysftp


class DEEPSouthDownloader:
    '''
    # initialize
    >>> Apr2018 = DEEPSouthDownloader(start=20180405, end=20180430)

    # Let it find the log files in the remote path
    >>> Apr2018.find_logs()
    >>> print("First 3 log files for example: \n\t", Apr2018.remotelogpaths[:3])

    # Let it download the log files and save it to Path('.')
    >>> Apr2018.get_logs(locallogdir='.', verbose=True)
    # NOTE: You may use UNIX like path as a str OR use path-like
    # (e.g., pathlib.Path or os.path.path).

    # Make a summary of log files with given conditions
    >>> Apr2018.summarize_log(conditions = dict(PROJID = "NEO",
                                            IMGTYPE="OBJECT",
                                            EXP = "60"))
    # NOTE: available column names for ``conditions``:
    #   PROJID, IMGTYPE, OBJECT, RA, DEC, EXP, SECZ, ST, FILT, DATE-OBS, UT, MIDJD,
    #   FILENAME, OBSERVATORY

    # If you want to save it,
    >>> Apr2018.log_summary.write("log2018Apr.csv", format='ascii.csv', overwrite=True)

    >>> Apr2018.get_FITS()
    '''
    def __init__(self, start=20160101, end=20161231,
                 observatory=['CTIO', 'SSO', 'SAAO'], condition=None,
                 locallogdir=None, localfitsdir=None):

        if isinstance(observatory, str):
            observatory = [observatory]

        elif not isinstance(observatory, list):
            observatory = list(observatory)

        if locallogdir is None:
            locallogdir = Path('.')

        if localfitsdir is None:
            localfitsdir = Path('.')

        self.start = start
        self.end = end
        self.observatory = observatory
        self.condition = condition
        self.locallogdir = locallogdir
        self.localfitsdir = localfitsdir
        self.sftp = None
        self.v = None
        self.remotelogpaths = None
        self.locallogpaths = None
        self.localfitspaths = None
        self.got_log = False
        self.log_summary = None

    def __str__(self):
        return (f"Search from {self.start} to {self.end} at {self.observatory} "
                + f"with {self.condition}")

    def set_connect(self):
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        cinfo = dict(host='210.219.33.23',
                     username='neo',
                     password='neo0212',
                     port=7774,
                     cnopts=cnopts)

        sftp = pysftp.Connection(**cinfo)
        self.sftp = sftp

    def find_logs(self, start=None, end=None):
        # if not set_connect, do it
        if self.sftp is None:
            self.set_connect()

        # Overwrite if given
        if start is not None:
            self.start = start

        if end is not None:
            self.end = end

        # Find directories for year & observatories
        years = np.unique([self.start // 10000, self.end // 10000])
        dirs = []
        remotelogpaths = []

        for y in years:
            for obs in self.observatory:
                directory = f"/KMTNet_pMEF/NEO/Y{y}/{obs}"
                dirs.append(directory)

        # Find logs
        for d in dirs:
            for fname in self.sftp.listdir(d):
                if fname.endswith('.log'):
                    date = int(fname.split('_')[0])  # YYYYMMDD
                    if (self.start < date) and (date < self.end):
                        remotelogpath = Path(f"{d}/{fname}")
                        remotelogpaths.append(remotelogpath)

        self.remotelogpaths = remotelogpaths

    def get_logs(self, locallogdir=None, callback=None, preserve_mtime=False,
                 verbose=True):
        # Do if not done
        if self.remotelogpaths is None:
            self.find_logs()

        # Initialize
        if callback is None and verbose:
            def callback(x, y):
                return print("{} / {}".format(x, y))

        self.locallogdir = Path(locallogdir)

        # Download logs
        locallogpaths = []

        for remotelogpath in self.remotelogpaths:
            locallogpath = self.locallogdir / remotelogpath.name
            locallogpaths.append(locallogpath)
            self.sftp.get(str(remotelogpath), str(locallogpath),
                          callback, preserve_mtime)

        self.locallogpaths = locallogpaths
        self.got_log = True

    def summarize_log(self, conditions=None):
        if not self.got_log:
            raise ValueError("You didn't get the log files, "
                             + "so cannot summarize log.")

        def _getFITSpath(row):
            fname = row["FILENAME"]
            obs = row["OBSERVATORY"]
            date = row["FILENAME"].split('.')[1]
            year = date[:4]
            remotefitspath = Path(
                f"/KMTNet_pMEF/NEO/Y{year}/{obs}/{date}/{fname}")
            return remotefitspath

        # Override if given
        if conditions is not None:
            self.conditions = conditions

        log_summary = []

        for locallogpath in self.locallogpaths:
            data = Table.read(locallogpath, format='ascii.fixed_width',
                              header_start=1,
                              col_starts=(0, 9, 19, 31, 45, 59, 66, 73, 84, 91,
                                          112, 136, 154))
            data.remove_rows([0, -1])
            obs = Column([locallogpath.name.split('_')[1]])
            data["OBSERVATORY"] = obs
            rem = []
            for row in data:
                remotefitspath = _getFITSpath(row)
                rem.append(str(remotefitspath))

            rem = Column(rem)
            data["FILEPATH"] = rem

            log_summary.append(data)

        log_summary = vstack(log_summary)

        if self.conditions is not None:
            for k, v in self.conditions.items():
                log_summary = log_summary[log_summary[k] == v]

        self.log_summary = log_summary

    def get_FITS(self, localfitsdir=None, callback=None, preserve_mtime=False,
                 verbose=True):
        if callback is None and verbose:
            def callback(x, y):
                return print("{} / {}".format(x, y))

        # Override if given
        self.localfitsdir = Path(localfitsdir)

        localfitspaths = []
        for row in self.log_summary:
            fname = row["FILENAME"]
            localfitspath = localfitsdir / fname
            remotefitspath = Path(row["FILEPATH"])
            localfitspaths.append(localfitspath)
            self.sftp.get(str(remotefitspath), str(localfitspath),
                          callback, preserve_mtime)

        self.localfitspaths = localfitspaths

    def split_ext(self, localfitsdir=None, keep_original=True):
        # Override if given
        if localfitsdir is not None:
            self.localfitsdir = localfitsdir

        # If self.localfitspaths already exist (by ``get_FITS``), show warn
        if self.localfitspaths is not None:
            warnings.warn("I already have local fits files path. "
                          + f"Override by ({self.localfitsdir}).glob('*.fits')")

        self.localfitspaths = Path(self.localfitsdir).glob("*.fits")


# class DEEPSouthDownloader:
#     '''
#     '''
#     def __init__(self, start=20160101, end=20161231,
#                  observatory=['CTIO', 'SSO', 'SAAO'], condition=None):
#         self.start = start
#         self.end = end
#         if isinstance(observatory, str):
#             observatory = [observatory]
#         elif not isinstance(observatory, list):
#             observatory = list(observatory)
#         self.observatory = observatory
#         self.sftp = None
#         self.condition = condition
#         self.v = None
#         self.remotelogpaths = None
#         self.locallogdir = None
#         self.locallogpaths = None
#         self.localfitsdir = None
#         self.localfitspaths = None
#         self.got_log = False
#         self.log_summary = None

#     def __str__(self):
#         return (f"Search from {self.start} to {self.end} at {self.observatory} "
#                 + f"with {self.condition}")

#     def set_connect(self):
#         cnopts = pysftp.CnOpts()
#         cnopts.hostkeys = None
#         cinfo = dict(host='210.219.33.23',
#                      username='neo',
#                      password='neo0212',
#                      port=7774,
#                      cnopts=cnopts)

#         sftp = pysftp.Connection(**cinfo)
#         self.sftp = sftp

#     def find_logs(self, start=None, end=None):
#         # if not set_connect, do it
#         if self.sftp is None:
#             self.set_connect()

#         # Overwrite if given
#         if start is not None:
#             self.start = start

#         if end is not None:
#             self.end = end

#         # Find directories for year & observatories
#         years = np.unique([self.start // 10000, self.end // 10000])
#         dirs = []
#         remotelogpaths = []

#         for y in years:
#             for obs in self.observatory:
#                 directory = f"/KMTNet_pMEF/NEO/Y{y}/{obs}"
#                 dirs.append(directory)

#         # Find logs
#         for d in dirs:
#             for fname in self.sftp.listdir(d):
#                 if fname.endswith('.log'):
#                     date = int(fname.split('_')[0])  # YYYYMMDD
#                     if (self.start < date) and (date < self.end):
#                         remotelogpath = Path(f"{d}/{fname}")
#                         remotelogpaths.append(remotelogpath)

#         self.remotelogpaths = remotelogpaths

#     def get_logs(self, locallogdir=None, callback=None, preserve_mtime=False,
#                  verbose=True):
#         # Do if not done
#         if self.remotelogpaths is None:
#             self.find_logs()

#         # Initialize
#         if callback is None and verbose:
#             callback = lambda x, y: print("{} / {}".format(x, y))

#         if locallogdir is None:
#             locallogdir = Path('.')

#         self.locallogdir = Path(locallogdir)

#         # Download logs
#         locallogpaths = []

#         for remotelogpath in self.remotelogpaths:
#             locallogpath = self.locallogdir / remotelogpath.name
#             locallogpaths.append(locallogpath)
#             self.sftp.get(str(remotelogpath), str(locallogpath),
#                           callback, preserve_mtime)

#         self.locallogpaths = locallogpaths
#         self.got_log = True

#     def summarize_log(self, conditions=None):
#         if not self.got_log:
#             raise ValueError("You didn't get the log files, "
#                              + "so cannot summarize log.")

#         def _getFITSpath(row):
#             fname = row["FILENAME"]
#             obs = row["OBSERVATORY"]
#             date = row["FILENAME"].split('.')[1]
#             year = date[:4]
#             remotefitspath = Path(f"/KMTNet_pMEF/NEO/Y{year}/{obs}/{date}/{fname}")
#             return remotefitspath

#         # Override if given
#         if conditions is not None:
#             self.conditions = conditions

#         log_summary = []

#         for locallogpath in self.locallogpaths:
#             data = Table.read(locallogpath, format='ascii.fixed_width',
#                               header_start=1,
#                               col_starts=(0, 9, 19, 31, 45, 59, 66, 73, 84, 91,
#                                           112, 136, 154))
#             data.remove_rows([0, -1])
#             obs = Column([locallogpath.name.split('_')[1]])
#             data["OBSERVATORY"] = obs
#             rem = []
#             for row in data:
#                 remotefitspath = _getFITSpath(row)
#                 rem.append(str(remotefitspath))

#             rem = Column(rem)
#             data["FILEPATH"] = rem

#             log_summary.append(data)

#         log_summary = vstack(log_summary)

#         if self.conditions is not None:
#             for k, v in self.conditions.items():
#                 log_summary = log_summary[log_summary[k] == v]

#         self.log_summary = log_summary

#     def get_FITS(self, localfitsdir=None, callback=None, preserve_mtime=False,
#                  verbose=True):
#         if callback is None and verbose:
#             callback = lambda x, y: print("{} / {}".format(x, y))

#         # Override if given
#         if localfitsdir is None:
#             localfitsdir = Path('.')

#         self.localfitsdir = Path(localfitsdir)

#         localfitspaths = []
#         for row in self.log_summary:
#             fname = row["FILENAME"]
#             localfitspath = localfitsdir / fname
#             remotefitspath = Path(row["FILEPATH"])
#             localfitspaths.append(localfitspath)
#             self.sftp.get(str(remotefitspath), str(localfitspath),
#                           callback, preserve_mtime)

#         self.localfitspaths = localfitspaths

