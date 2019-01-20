#!/usr/bin/env python2
'''
Astrometry.net "client.py" with modification.
2018-07-12 ysBach
'''
from __future__ import print_function
import os
import sys
import time
import base64

try:
    # py3
    from urllib.parse import urlencode, quote
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
except ImportError:
    # py2
    from urllib import urlencode, quote
    from urllib2 import urlopen, Request, HTTPError

# from exceptions import Exception
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from email.encoders import encode_noop

import json


def json2python(data):
    try:
        return json.loads(data)
    except (json.decoder.JSONDecodeError, TypeError):
        pass
    return None


python2json = json.dumps


class MalformedResponse(Exception):
    pass


class RequestError(Exception):
    pass


class Client(object):
    default_url = 'http://nova.astrometry.net/api/'

    def __init__(self, apiurl=default_url):
        self.session = None
        self.apiurl = apiurl

    def get_url(self, service):
        return self.apiurl + service

    def send_request(self, service, args={}, file_args=None):
        '''
        service: string
        args: dict
        '''
        if self.session is not None:
            args.update({'session': self.session})
        print('Python:', args)
        json = python2json(args)
        print('Sending json:', json)
        url = self.get_url(service)
        print('Sending to URL:', url)

        # If we're sending a file, format a multipart/form-data
        if file_args is not None:
            # Make a custom generator to format it the way we need.
            from io import BytesIO
            try:
                # py3
                from email.generator import BytesGenerator as TheGenerator
            except ImportError:
                # py2
                from email.generator import Generator as TheGenerator

            m1 = MIMEBase('text', 'plain')
            m1.add_header('Content-disposition',
                          'form-data; name="request-json"')
            m1.set_payload(json)
            m2 = MIMEApplication(file_args[1], 'octet-stream', encode_noop)
            m2.add_header('Content-disposition',
                          'form-data; name="file"; filename="%s"' % file_args[0])
            mp = MIMEMultipart('form-data', None, [m1, m2])

            class MyGenerator(TheGenerator):
                def __init__(self, fp, root=True):
                    # don't try to use super() here; in py2 Generator is not a
                    # new-style class.  Yuck.
                    TheGenerator.__init__(self, fp, mangle_from_=False,
                                          maxheaderlen=0)
                    self.root = root

                def _write_headers(self, msg):
                    # We don't want to write the top-level headers;
                    # they go into Request(headers) instead.
                    if self.root:
                        return
                    # We need to use \r\n line-terminator, but Generator
                    # doesn't provide the flexibility to override, so we
                    # have to copy-n-paste-n-modify.
                    for h, v in msg.items():
                        self._fp.write(('%s: %s\r\n' % (h, v)).encode())
                    # A blank line always separates headers from body
                    self._fp.write('\r\n'.encode())

                # The _write_multipart method calls "clone" for the
                # subparts.  We hijack that, setting root=False
                def clone(self, fp):
                    return MyGenerator(fp, root=False)

            fp = BytesIO()
            g = MyGenerator(fp)
            g.flatten(mp)
            data = fp.getvalue()
            headers = {'Content-type': mp.get('Content-type')}

        else:
            # Else send x-www-form-encoded
            data = {'request-json': json}
            print('Sending form data:', data)
            data = urlencode(data)
            data = data.encode('utf-8')
            print('Sending data:', data)
            headers = {}

        request = Request(url=url, headers=headers, data=data)

        try:
            f = urlopen(request)
            txt = f.read()
            print('Got json:', txt)
            result = json2python(txt)
            print('Got result:', result)
            stat = result.get('status')
            print('Got status:', stat)
            if stat == 'error':
                errstr = result.get('errormessage', '(none)')
                raise RequestError('server error message: ' + errstr)
            return result
        except HTTPError as e:
            print('HTTPError', e)
            txt = e.read()
            open('err.html', 'wb').write(txt)
            print('Wrote error text to err.html')

    def login(self, apikey):
        args = {'apikey': apikey}
        result = self.send_request('login', args)
        sess = result.get('session')
        print('Got session:', sess)
        if not sess:
            raise RequestError('no session in result')
        self.session = sess

    def _get_upload_args(self, **kwargs):
        args = {}
        for key, default, typ in [('allow_commercial_use', 'd', str),
                                  ('allow_modifications', 'd', str),
                                  ('publicly_visible', 'y', str),
                                  ('scale_units', None, str),
                                  ('scale_type', None, str),
                                  ('scale_lower', None, float),
                                  ('scale_upper', None, float),
                                  ('scale_est', None, float),
                                  ('scale_err', None, float),
                                  ('center_ra', None, float),
                                  ('center_dec', None, float),
                                  ('parity', None, int),
                                  ('radius', None, float),
                                  ('downsample_factor', None, int),
                                  ('tweak_order', None, int),
                                  ('crpix_center', None, bool),
                                  ('x', None, list),
                                  ('y', None, list),
                                  # image_width, image_height
                                  ]:
            if key in kwargs:
                val = kwargs.pop(key)
                val = typ(val)
                args.update({key: val})
            elif default is not None:
                args.update({key: default})
        print('Upload args:', args)
        return args

    def url_upload(self, url, **kwargs):
        args = dict(url=url)
        args.update(self._get_upload_args(**kwargs))
        result = self.send_request('url_upload', args)
        return result

    def upload(self, fn=None, **kwargs):
        args = self._get_upload_args(**kwargs)
        file_args = None
        if fn is not None:
            try:
                f = open(fn, 'rb')
                file_args = (fn, f.read())
            except IOError:
                print('File %s does not exist' % fn)
                raise
        return self.send_request('upload', args, file_args)

    def submission_images(self, subid):
        result = self.send_request('submission_images', {'subid': subid})
        return result.get('image_ids')

    def overlay_plot(self, service, outfn, wcsfn, wcsext=0):
        from astrometry.util import util as anutil
        wcs = anutil.Tan(wcsfn, wcsext)
        params = dict(crval1=wcs.crval[0], crval2=wcs.crval[1],
                      crpix1=wcs.crpix[0], crpix2=wcs.crpix[1],
                      cd11=wcs.cd[0], cd12=wcs.cd[1],
                      cd21=wcs.cd[2], cd22=wcs.cd[3],
                      imagew=wcs.imagew, imageh=wcs.imageh)
        result = self.send_request(service, {'wcs': params})
        print('Result status:', result['status'])
        plotdata = result['plot']
        plotdata = base64.b64decode(plotdata)
        open(outfn, 'wb').write(plotdata)
        print('Wrote', outfn)

    def sdss_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('sdss_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def galex_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('galex_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def myjobs(self):
        result = self.send_request('myjobs/')
        return result['jobs']

    def job_status(self, job_id, justdict=False):
        result = self.send_request('jobs/%s' % job_id)
        if justdict:
            return result
        stat = result.get('status')
        if stat == 'success':
            result = self.send_request('jobs/%s/calibration' % job_id)
            print('Calibration:', result)
            result = self.send_request('jobs/%s/tags' % job_id)
            print('Tags:', result)
            result = self.send_request('jobs/%s/machine_tags' % job_id)
            print('Machine Tags:', result)
            result = self.send_request('jobs/%s/objects_in_field' % job_id)
            print('Objects in field:', result)
            result = self.send_request('jobs/%s/annotations' % job_id)
            print('Annotations:', result)
            result = self.send_request('jobs/%s/info' % job_id)
            print('Calibration:', result)

        return stat

    def annotate_data(self, job_id):
        """
        :param job_id: id of job
        :return: return data for annotations
        """
        result = self.send_request('jobs/%s/annotations' % job_id)
        return result

    def sub_status(self, sub_id, justdict=False):
        result = self.send_request('submissions/%s' % sub_id)
        if justdict:
            return result
        return result.get('status')

    def jobs_by_tag(self, tag, exact):
        exact_option = 'exact=yes' if exact else ''
        result = self.send_request(
            'jobs_by_tag?query=%s&%s' % (quote(tag.strip()), exact_option),
            {},
        )
        return result


def query_astrometry(server=Client.default_url, apikey=None, upload=None,
                     upload_xy=None, wait=True, wcs=None, newfits=None,
                     kmz=None, annotate=None, upload_url=None,
                     scale_units=None, scale_lower=None, scale_upper=None,
                     scale_est=None, scale_err=None, center_ra=None,
                     center_dec=None, radius=None, downsample_factor=None,
                     parity=None, tweak_order=None, crpix_center=None,
                     sdss_wcs=None, galex_wcs=None, solved_id=None,
                     sub_id=None, job_id=None, myjobs=True,
                     jobs_by_exact_tag=None, jobs_by_tag=None, public='n',
                     allow_mod='sa', allow_commercial='n', sleep_interval=5):
    '''
    Parameters
    ----------
    server: str
        Set server base URL (e.g., ``Client.default_url``).

    apikey: str
        API key for Astrometry.net web service;
        if not given will check AN_API_KEY environment variable

    upload: path-like
        Upload a file (The destination path)

    upload_xy: path-like
        Upload a FITS x,y table as JSON

    wait: bool
        After submitting, monitor job status

    wcs: path-like
        Download resulting wcs.fits file, saving to given filename.
        Implies ``wait=True`` if ``urlupload`` or ``upload`` is not None.

    newfits: path-like
        Download resulting new-image.fits file, saving to given filename.
        Implies ``wait=True`` if ``urlupload`` or ``upload`` is not None.

    kmz: path-like
        Download resulting kmz file, saving to given filename;
        Implies ``wait=True`` if ``urlupload`` or ``upload`` is not None.

    annotate: path-like
        Store information about annotations in give file, JSON format;
        Implies ``wait=True`` if ``urlupload`` or ``upload`` is not None.

    upload_url: str
        Upload a file at specified url.

    scale_units: str in ['arcsecperpix', 'arcminwidth', 'degwidth', 'focalmm']
        Units for scale estimate.

    scale_lower, scale_upper: float
        Scale lower-bound and upper bound.

    scale_est: float
        Scale estimate.

    scale_err: float
        Scale estimate error (in PERCENT), e.g., 10 if you estimate can be off
        by 10%.

    center_ra, center_dec: float
        RA center and DEC center.

    radius: float
        Search radius around RA, Dec center.

    downsample_factor: int
        Downsample image by this factor.

    parity: str in ['0', '1']
        Parity (flip) of image.

    tweak_order: int
        SIP distortion order (if None, defaults to 2).

    crpix_center: bool
        Set reference point to center of image?

    sdss_wcs: list of two str
        Plot SDSS image for the given WCS file; write plot to given PNG filename

    galex_wcs: list of two str
        Plot GALEX image for the given WCS file; write plot to given PNG filename

    solved_id: int
        retrieve result for jobId instead of submitting new image

    sub_id, job_id: bool
        Get status of a submission or job

    myjobs: bool
        Get all my jobs

    jobs_by_exact_tab: bool
        Get a list of jobs associated with a given tag--exact match

    jobs_by_tag: bool
        Get a list of jobs associated with a given tag

    private: str in ['y', 'n']
        Hide this submission from other users

    allow_mod: str in ['sa', 'd']
        Select license to allow derivative works of submission, but only if
        shared under same conditions of original license

    allow_commercial: str in ['n', 'd']
        Select license to disallow commercial use of submission

    sleep_interval: int, float
        How long to wait for printing the information.
    '''

    if apikey is None:
        # try the environment
        apikey = os.environ.get('AN_API_KEY', None)
    if apikey is None:
        print("API key for Astrometry.net web service;"
              + "if not given will check AN_API_KEY environment variable")
        print()
        print('You must either specify --apikey or set AN_API_KEY')
        sys.exit(-1)

    args = {}
    args['apiurl'] = server
    c = Client(**args)
    c.login(apikey)

    if upload or upload_url or upload_xy:
        if wcs or kmz or newfits or annotate:
            wait = True

        kwargs = dict(allow_commercial_use=allow_commercial,
                      allow_modifications=allow_mod,
                      publicly_visible=public)
        if scale_lower and scale_upper:
            kwargs.update(scale_lower=scale_lower,
                          scale_upper=scale_upper,
                          scale_type='ul')
        elif scale_est and scale_err:
            kwargs.update(scale_est=scale_est,
                          scale_err=scale_err,
                          scale_type='ev')
        elif scale_lower or scale_upper:
            kwargs.update(scale_type='ul')
            if scale_lower:
                kwargs.update(scale_lower=scale_lower)
            if scale_upper:
                kwargs.update(scale_upper=scale_upper)

        for key in ['scale_units', 'center_ra', 'center_dec', 'radius',
                    'downsample_factor', 'tweak_order', 'crpix_center', ]:
            if eval(key) is not None:
                kwargs[key] = eval(key)
        if parity is not None:
            kwargs.update(parity=int(parity))

        if upload:
            upres = c.upload(upload, **kwargs)
        if upload_xy:
            from astrometry.util.fits import fits_table
            T = fits_table(upload_xy)
            kwargs.update(x=[float(x) for x in T.x], y=[float(y) for y in T.y])
            upres = c.upload(**kwargs)
        if upload_url:
            upres = c.url_upload(upload_url, **kwargs)

        stat = upres['status']
        if stat != 'success':
            print('Upload failed: status', stat)
            print(upres)
            sys.exit(-1)

        sub_id = upres['subid']

    if wait:
        if solved_id is None:
            if sub_id is None:
                print("Can't --wait without a submission id or job id!")
                sys.exit(-1)

            while True:
                stat = c.sub_status(sub_id, justdict=True)
                print('Got status:', stat)
                jobs = stat.get('jobs', [])
                if len(jobs):
                    for j in jobs:
                        if j is not None:
                            break
                    if j is not None:
                        print('Selecting job id', j)
                        solved_id = j
                        break
                time.sleep(sleep_interval)

        while True:
            stat = c.job_status(solved_id, justdict=True)
            print('Got job status:', stat)
            if stat.get('status', '') in ['success']:
                success = (stat['status'] == 'success')
                break
            time.sleep(sleep_interval)

    if solved_id:
        # we have a jobId for retrieving results
        retrieveurls = []
        if wcs:
            # We don't need the API for this, just construct URL
            url = server.replace('/api/', '/wcs_file/%i' % solved_id)
            retrieveurls.append((url, wcs))
        if kmz:
            url = server.replace('/api/', '/kml_file/%i/' % solved_id)
            retrieveurls.append((url, kmz))
        if newfits:
            url = server.replace(
                '/api/', '/new_fits_file/%i/' % solved_id)
            retrieveurls.append((url, newfits))

        for url, fn in retrieveurls:
            print('Retrieving file from', url, 'to', fn)
            f = urlopen(url)
            txt = f.read()
            w = open(fn, 'wb')
            w.write(txt)
            w.close()
            print('Wrote to', fn)

        if annotate:
            result = c.annotate_data(solved_id)
            with open(annotate, 'w') as f:
                f.write(python2json(result))

    if wait:
        # behaviour as in old implementation
        sub_id = None

    if sdss_wcs:
        (wcsfn, outfn) = sdss_wcs
        c.sdss_plot(outfn, wcsfn)
    if galex_wcs:
        (wcsfn, outfn) = galex_wcs
        c.galex_plot(outfn, wcsfn)

    if sub_id:
        print(c.sub_status(sub_id))
    if job_id:
        print(c.job_status(job_id))

    if jobs_by_tag:
        tag = jobs_by_tag
        print(c.jobs_by_tag(tag, None))
    if jobs_by_exact_tag:
        tag = jobs_by_exact_tag
        print(c.jobs_by_tag(tag, 'yes'))

    if myjobs:
        jobs = c.myjobs()
        print(jobs)

    return success
