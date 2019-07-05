#!/usr/bin/env python3

import argparse
import logging
import pathlib
import re
from multiprocessing import Pool
import pylab as plt

from skimage.transform import resize
import numpy as np
import pandas as pd
from pysigproc import SigprocFile
from candidate import *
from gpu_utils import gpu_dedisp_and_dmt_crop 
logger = logging.getLogger()

__author__ = 'Devansh Agarwal'
__email__ = 'da0017@mix.wvu.edu'

def _get_chunk(self, tstart=None, tstop=None, offset=None):
    """
    Read the data around the candidate from the filterbank
    :param tstart: Start time in the fiterbank, to read from
    :param tstop: End time in the filterbank, to read till
    :return:
    """
    if tstart is None:
        tstart = self.tcand - (2*self.dispersion_delay()) - self.width * self.tsamp
        if tstart < 0:
            tstart = 0
    if tstop is None:
        tstop = self.tcand + (0*self.dispersion_delay()) + self.width * self.tsamp
        if tstop > self.tend:
            tstop = self.tend
    nstart = int(tstart / self.tsamp)
    nsamp = int((tstop - tstart) / self.tsamp)

    if self.width < 2:
        nchunk = self.min_samp
    else:
        nchunk = self.width * self.min_samp // 2
    if nsamp < nchunk:
        # if number of time samples less than nchunk, make it min_samp.
        nstart -= (nchunk - nsamp) // 2
        nsamp = nchunk
    if nstart < 0:
        self.data = self.get_data(nstart=0, nsamp=nsamp + nstart, offset=offset)[:, 0, :]
        logging.debug('median padding data as nstart < 0')
        self.data = pad_along_axis(self.data, nsamp, loc='start', axis=0, mode='median')
    elif nstart + nsamp > self.nspectra:
        self.data = self.get_data(nstart=nstart, nsamp=self.nspectra - nstart, offset=offset)[:, 0, :]
        logging.debug('median padding data as nstop > nspectra')
        self.data = pad_along_axis(self.data, int(nsamp), loc='end', axis=0, mode='median')
    else:
        self.data = self.get_data(nstart=nstart, nsamp=nsamp, offset=offset)[:, 0, :]


    if self.kill_mask is not None:
        assert len(self.kill_mask) == self.data.shape[1]
        data_copy = self.data.copy()
        data_copy[:, self.kill_mask] = 0
        self.data = data_copy
        del data_copy

    return self

def _fil_name_to_hdr(fil_name, tstart_MJD):
    filename_numbers = re.findall(r'\d+', fil_name)
    beam_number = int(filename_numbers[0])
    fil_file = SigprocFile()
    fil_file.rawdatafile = str(fil_name)
    fil_file.source_name = 'source'
    fil_file.machine_id  = 0
    fil_file.barycentric = 0
    fil_file.pulsarcentric = 1
    fil_file.telescope_id = 1
    fil_file.src_raj = 0
    fil_file.src_dej = 0
    fil_file.az_start = 0
    fil_file.za_start = 0
    fil_file.data_type = 32
    fil_file.fch1 = 148.9 #MHz
    fil_file.foff = -0.003305176 #Mhz
    if beam_number % 2 == 0:
        fil_file.nchans = 1920
    else:
        fil_file.nchans = 1984
    fil_file.nbeams = 20
    fil_file.ibeam = beam_number
    fil_file.nbits = 32
    fil_file.tstart = tstart_MJD
    fil_file.tsamp = 327.68e-6
    fil_file.nifs = 1
    fil_file.nspectra = int(2**15)
    fil_file.tend = fil_file.nspectra*fil_file.tsamp
    return fil_file

def normalise(data):
    """
    Noramlise the data by unit standard deviation and zero median
    :param data: data
    :return:
    """
    data = np.array(data, dtype=np.float32)
    data -= np.median(data)
    data /= np.std(data)
    return data


def cand2h5(cand_val):
    """
    TODO: Add option to use cand.resize for reshaping FT and DMT
    Generates h5 file of candidate with resized frequency-time and DM-time arrays
    :param cand_val: List of candidate parameters (fil_name, snr, width, dm, label, tcand(s))
    :type cand_val: Candidate
    :return: None
    """
    fil_name, snr, width, dm, label, tcand, MJD_file, block_id, kill_mask_path, args = cand_val
    copy_header = _fil_name_to_hdr(fil_name, tstart_MJD=MJD_file)
    if kill_mask_path == kill_mask_path:
        kill_mask_file = pathlib.Path(kill_mask_path)
        if kill_mask_file.is_file():
            logging.info(f'Using mask {kill_mask_path}')
            kill_chans = np.loadtxt(kill_mask_path, dtype=np.int)
            kill_mask = np.zeros(copy_header.nchans, dtype=np.bool)
            kill_mask[kill_chans]= True

    else:
        logging.debug('No Kill Mask')
        kill_mask = None

    cand = Candidate(fil_name, copy_hdr = copy_header, snr=snr, width=width, dm=dm, label=label, tcand=tcand, kill_mask=kill_mask)
    cand.tend = copy_header.tend
    cand.nspectra = copy_header.nspectra
    cand.tstart -= cand.dispersion_delay() #+ (cand.width*cand.tsamp)//2
    cand = _get_chunk(cand, offset = (block_id-1)*copy_header.nspectra)
    cand=gpu_dedisp_and_dmt_crop(cand)
    print(cand.dedispersed.shape)
    print(cand.dmt.shape)

    if cand.dedispersed.shape != (256,256):
        cand.dedispersed=resize(cand.dedispersed, (256,256))
    #cand.fp.close()
    #logging.info('Got Chunk')
    #cand.dmtime()
    #logging.info('Made DMT')
    #if args.opt_dm:
    #    logging.info('Optimising DM')
    #    logging.warning('This feature is experimental!')
    #    cand.optimize_dm()
    #else:
    #    cand.dm_opt = -1
    #    cand.snr_opt = -1
    #cand.dedisperse()
    #logging.info('Made Dedispersed profile')

    #pulse_width = cand.width
    #if pulse_width == 1:
    #    time_decimate_factor = 1
    #else:
    #    time_decimate_factor = pulse_width // 2

    ## Frequency - Time reshaping
    #cand.decimate(key='ft', axis=0, pad=True, decimate_factor=time_decimate_factor, mode='median')
    #crop_start_sample_ft = cand.dedispersed.shape[0] // 2 - args.time_size // 2
    #print(cand.dedispersed.shape)
    #print(crop_start_sample_ft)
    #cand.dedispersed = crop(cand.dedispersed, crop_start_sample_ft, args.time_size, 0)
    #print(cand.dedispersed.shape)
    #logging.info(f'Decimated Time axis of FT to tsize: {cand.dedispersed.shape[0]}')

    #if cand.dedispersed.shape[1] % args.frequency_size == 0:
    #    cand.decimate(key='ft', axis=1, pad=True, decimate_factor=cand.dedispersed.shape[1] // args.frequency_size,
    #                  mode='median')
    #    logging.info(f'Decimated Frequency axis of FT to fsize: {cand.dedispersed.shape[1]}')
    #else:
    #    cand.resize(key='ft', size=args.frequency_size, axis=1, anti_aliasing=True)
    #    logging.info(f'Resized Frequency axis of FT to fsize: {cand.dedispersed.shape[1]}')

    ## DM-time reshaping
    #cand.decimate(key='dmt', axis=1, pad=True, decimate_factor=time_decimate_factor, mode='median')
    #crop_start_sample_dmt = cand.dmt.shape[1] // 2 - args.time_size // 2
    #cand.dmt = crop(cand.dmt, crop_start_sample_dmt, args.time_size, 1)
    #logging.info(f'Decimated DM-Time to dmsize: {cand.dmt.shape[0]} and tsize: {cand.dmt.shape[1]}')

    cand.dmt = normalise(cand.dmt)
    cand.dedispersed = normalise(cand.dedispersed)

    fout = cand.save_h5(out_dir=args.fout)
    logging.info(fout)
    if args.plot:
        logging.info('Displaying the candidate')
        plot_h5(fout, show=False, save=True, detrend=False)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Be verbose', action='store_true')
    parser.add_argument('-fs', '--frequency_size', type=int, help='Frequency size after rebinning', default=256)
    parser.add_argument('-ts', '--time_size', type=int, help='Time length after rebinning', default=256)
    parser.add_argument('-c', '--cand_param_file', help='csv file with candidate parameters', type=str, required=True)
    parser.add_argument('-n', '--nproc', type=int, help='number of processors to use in parallel (default: 2)',
                        default=2)
    parser.add_argument('-p', '--plot', dest='plot', help='To display and save the candidates plots',
                        action='store_true')
    parser.add_argument('-o', '--fout', help='Output file directory for candidate h5', type=str)
    parser.add_argument('-opt', '--opt_dm', dest='opt_dm', help='Optimise DM', action='store_true', default=False)
    values = parser.parse_args()

    logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'

    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    else:
        logging.basicConfig(level=logging.INFO, format=logging_format)

    cand_pars = pd.read_csv(values.cand_param_file,
                            names=['fil_file', 'snr', 'stime', 'dm', 'width',  'MJD_file', 'block_id', 'label', 'kill_mask_path'])
    process_list = []
    for index, row in cand_pars.iterrows():
        process_list.append(
            [row['fil_file'], row['snr'], 2**row['width'], row['dm'], row['label'], row['stime'], row['MJD_file'],
             row['block_id'], row['kill_mask_path'], values])
    with Pool(processes=values.nproc) as pool:
        pool.map(cand2h5, process_list, chunksize=1)
