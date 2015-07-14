from __future__ import print_function, division, absolute_import

import copy
import logging
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix

from hexrd.coreutil import initialize_experiment

from hexrd.utils.progressbar import (
    Bar, ETA, Percentage, ProgressBar, ReverseBar
    )

descr = 'Cakes the 2D diffraction data'
example = """
examples:
    hexrd cake-data config.yml
"""
piBy180 = np.pi / 180.0

from hexrd import USE_NUMBA
if USE_NUMBA:
    import numba

logger = logging.getLogger(__name__)

if USE_NUMBA:
    @numba.njit
    def extract_ijv(in_array, threshold, out_i, out_j, out_v):
        n = 0
        w, h = in_array.shape

        for i in range(w):
            for j in range(h):
                v = in_array[i,j]
                if v > threshold:
                    out_v[n] = v
                    out_i[n] = i
                    out_j[n] = j
                    n += 1

        return n

    class CooMatrixBuilder(object):
        def __init__(self):
            self.v_buff = np.empty((2048*2048,), dtype=np.int16)
            self.i_buff = np.empty((2048*2048,), dtype=np.int16)
            self.j_buff = np.empty((2048*2048,), dtype=np.int16)

        def build_matrix(self, frame, threshold):
            count = extract_ijv(frame, threshold,
                                self.i_buff, self.j_buff, self.v_buff)
            return coo_matrix((self.v_buff[0:count].copy(),
                               (self.i_buff[0:count].copy(),
                                self.j_buff[0:count].copy())),
                              shape=frame.shape)

else: # not USE_NUMBA
    class CooMatrixBuilder(object):
        def build_matrix(self, frame, threshold):
            mask = frame > threshold
            return coo_matrix((frame[mask], mask.nonzero()),
                              shape=frame.shape)


def configure_parser(sub_parsers):
    p = sub_parsers.add_parser('cake-data', description = descr, help = descr)

    p.add_argument(
      'yml', type=str,
      help='YAML configuration file'
    )

    p.set_defaults(func=execute)


def process_cake(reader, detector, cfg, show_progress=True):
    # TODO: this should be updated to read only the frames requested in cfg
    # either the images start, step, stop, or based on omega start, step, stop
    start = time.time()

    n_frames = reader.getNFrames()
    logger.info("reading %d frames of data", n_frames)
    # Read all frames from the GE files
    imgOut = reader.read(nframes=n_frames, sumImg=np.maximum)
            
    elapsed = time.time()-start
    logger.info('read %d frames in %g seconds', n_frames, elapsed)
    # Write "max over all" frame
    f = os.path.join(
        cfg.working_dir,        
        '-'.join([cfg.analysis_name, 'max_img.ge2'])
        )
    writer = reader.getWriter(f)
    writer.write(imgOut)
    logger.info('wrote max intensity frame to %s', f)

    # Standard polar rebinning
    logger.info('running standard polar binning with corrected detector params')
    # TODO: move all these to config file
    etaMin = 0.0*piBy180
    etaMax = 360.0*piBy180
    rhoMin = 100.0
    rhoMax = 1000.0
    numEta = 1
    numRho = 1000
    
    correct = True
    
    kwa = {
        'etaRange' : [etaMin, etaMax],
        'numEta'   : numEta,
        'rhoRange' : [rhoMin, rhoMax],
        'numRho'   : numRho,
        'corrected': correct
        }
    # Do the binning
    cake_data = detector.polarRebin(imgOut, log=None, **kwa)


    f = os.path.join(
        cfg.working_dir,        
        '-'.join([cfg.analysis_name, 'intensity-twotheta.data'])
        )

    intensity = cake_data['intensity']
    radialDat = cake_data['radius']
    azimuDat  = cake_data['azimuth']
    corrected = cake_data['corrected']
    
    if corrected:
        rad_str = 'two-theta'
    else:
        rad_str = 'radius'
    # Write data to file    
    if isinstance(f, file):
        fid = f
    elif isinstance(f, str) or isinstance(f, unicode):
        fid = open(f, 'w')
        logger.info('Writing cake data to %s', f)
        pass
    for i in range(len(intensity)):
        #print >> fid, "# AZIMUTHAL BLOCK %d" % (i)
        #print >> fid, "# eta = %1.12e\n# %s\tintensity"% (azimuDat[i], rad_str)
        fid.write("# AZIMUTHAL BLOCK %d\n" % (i))
        fid.write("# eta = %1.12e\n# %s\tintensity\n"% (azimuDat[i], rad_str))
        for j in range(len(intensity[i, :])):
            #print >> fid, "%1.12e\t%1.12e" % (radialDat[j], intensity[i, j])
            fid.write("%1.12e\t%1.12e\n" % (radialDat[j], intensity[i, j]))
    fid.close()
    pass
# Done
    return reader

def execute(args, parser):
    import logging
    import os
    import sys

    import yaml

    from hexrd import config


    # load the configuration settings
    cfgs = config.open(args.yml)

    # configure logging to the console:
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = logging.getLogger('hexrd')
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    cf = logging.Formatter('%(asctime)s - %(message)s', '%y-%m-%d %H:%M:%S')
    ch.setFormatter(cf)
    logger.addHandler(ch)

    logger.info('=== begin cake-data ===')

    for cfg in cfgs:
        logger.info('*** begin caking for analysis "%s" ***', cfg.analysis_name)

        # configure logging to file for this particular analysis
        logfile = os.path.join(
            cfg.working_dir,
            'cake-data.log'
            )
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(log_level)
        ff = logging.Formatter(
                '%(asctime)s - %(name)s - %(message)s',
                '%m-%d %H:%M:%S'
                )
        fh.setFormatter(ff)
        logger.info("logging to %s", logfile)
        logger.addHandler(fh)

        # process the data
        pd, reader, detector = initialize_experiment(cfg)
        # Load frames, generate "max over all" frame and bin the data
        show_progress = True
        reader = process_cake(reader, detector, cfg, show_progress)

        # stop logging for this particular analysis
        fh.flush()
        fh.close()
        logger.removeHandler(fh)

        logger.info('*** end caking for analysis "%s" ***', cfg.analysis_name)

    logger.info('=== end cake-data ===')
    # stop logging to the console
    ch.flush()
    ch.close()
    logger.removeHandler(ch)
