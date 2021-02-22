#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2016/08/23 Version 34 - parameters24 input file needed
# 2017/10/27 Version 39 - Reformatted PEP8 Code
# 2017/11/05 Version 40 - Corrections to tdifmin, tstda calculations
# 2019/10/15 Version pympa - xcorr substitued with correlate_template from obspy

# First Version August 2014 - Last October 2017 (author: Alessandro Vuan)

# Code for the detection of microseismicity based on cross correlation
# of template events. The code exploits multiple cores to speed up time
#
# Method's references:
# The code is developed and maintained at
# Istituto Nazionale di Oceanografia e Geofisica di Trieste (OGS)
# and was inspired by collaborating with Aitaro Kato and collegues at ERI.
# Kato A, Obara K, Igarashi T, Tsuruoka H, Nakagawa S, Hirata N (2012)
# Propagation of slow slip leading up to the 2011 Mw 9.0 Tohoku-Oki
# earthquake. Science doi:10.1126/science.1215141
#
# For questions comments and suggestions please send an email to avuan@inogs.it
# The kernel function xcorr used from Austin Holland is modified in pympa
# Recommended the use of Obspy v. 1.2.0 with the substitution of xcorr function with
# correlate_template

# Software Requirements: the following dependencies are needed (check import
# and from statements below)
# Python "obspy" package installed via Anaconda with all numpy and scipy
# packages
# Python "math" libraries
# Python "bottleneck" utilities to speed up numpy array operations
#
import itertools
import logging
import os.path
from pathlib import Path
from time import perf_counter as timer

import numpy as np
from obspy import read, Stream
from obspy.core import UTCDateTime
from obspy.core.event import read_events
from obspy.signal.trigger import coincidence_trigger

# from pympa import read_parameters, read_settings, listdays, trim_fill, stack, mad, process_input, mag_detect, reject_moutliers, csc
from pympa import *

if __name__ == '__main__':
    logging.basicConfig(filename='run.log',
                        filemode='w',
                        format='%(levelname)s: %(message)s')
    start_time = timer()
    settings = read_settings("settings.yaml")

    # set time precision for UTCDATETIME
    UTCDateTime.DEFAULT_PRECISION = settings['utc_prec']
    cat = read_events(settings['ev_catalog'])

    for day in listdays(*settings['date_range']):
        # settings to cut exactly 24 hours file from without including
        # previous/next day
        logging.debug(f"Processing {day}")

        for itemp in range(*settings['template_range']):
            # open file containing detections
            logging.debug(f"itemp = {itemp}")
            # open statistics file for each detection
            events_list = []

            mt = cat[itemp].magnitudes[0].mag

            # read ttimes, select the num_ttimes (parameters,
            # last line) channels
            # and read only these templates
            travel_dir = Path(settings['travel_dir'])
            with open(travel_dir / f"{itemp}.ttimes", "r") as ttim:
                travel_times = dict(x.rstrip().split(None, 1) for x in ttim)

            template_stream = get_template_stream(itemp, travel_times, settings)

            nch_min = settings['nch_min']
            nchunk = settings['nchunk']
            if len(template_stream) >= nch_min:
                h24 = 86400
                chunk_start = UTCDateTime(day)
                end_time = chunk_start + h24
                chunks = []
                while chunk_start < end_time:
                    chunk_end = min(chunk_start + h24 / nchunk, end_time)
                    chunks.append((chunk_start, chunk_end))
                    chunk_start += h24 / nchunk

                for t1, t2 in chunks:
                    logging.debug(f"Processing chunk ({t1}, {t2})")
                    chunk_stream = get_chunk_stream(template_stream, day, t1, t2, settings)

                    if len(chunk_stream) >= nch_min:
                        ntl = len(template_stream)
                        # for each template event
                        damaxat = {}
                        # reference time to be used for
                        # retrieving time synchronization
                        reft = min(tr.stats.starttime for tr in template_stream)

                        for il, tr in enumerate(template_stream):
                            sta_t = tr.stats.station
                            cha_t = tr.stats.channel
                            tid_t = "%s.%s" % (sta_t, cha_t)
                            damaxat[tid_t] = max(abs(tr.data))

                        # find minimum time to recover origin time
                        time_values = [float(v) for v in travel_times.values()]
                        min_time_value = min(time_values)
                        min_time_key = [k for k, v in travel_times.items() if v == str(min_time_value)]

                        # clear global_variable
                        stream_cft = Stream()
                        for nn, ss, ich in itertools.product(settings['networks'], settings['stations'], settings['channels']):
                            stream_cft += process_input(settings['temp_dir'], itemp, nn, ss, ich, chunk_stream)

                        # seconds in 24 hours
                        nfile = len(stream_cft)
                        tstart = np.empty(nfile)
                        tend = np.empty(nfile)
                        tdif = np.empty(nfile)
                        stall = Stream()
                        for idx, tc_cft in enumerate(stream_cft):
                            # get station name from trace
                            sta = tc_cft.stats.station
                            chan = tc_cft.stats.channel
                            net = tc_cft.stats.network
                            # get stream starttime
                            # waveforms should have the same number of npts
                            # and should be synchronized to the S-wave travel time
                            tdif[idx] = travel_times[f"{net}.{sta}.{chan}"]
                            tstart[idx] = tc_cft.stats.starttime + tdif[idx]
                            tend[idx] = tstart[idx] + (h24 / nchunk) + 60
                            ts = UTCDateTime(tstart[idx], precision=settings['utc_prec'])
                            te = UTCDateTime(tend[idx], precision=settings['utc_prec'])
                            stall += tc_cft.trim(starttime=ts, endtime=te, nearest_sample=True, pad=True, fill_value=0)

                        # compute mean cross correlation from the stack of
                        # CFTs (see stack function)
                        tstart = min(tr.stats.starttime for tr in stall)
                        df = stall[0].stats.sampling_rate
                        npts = stall[0].stats.npts
                        ccmad, tdifmin = stack(stall, df, tstart, travel_times, npts, settings['stdup'], settings['stddown'], nch_min)
                        logging.debug(f"tdifmin = {tdifmin}")

                        if tdifmin is not None:
                            # compute mean absolute deviation of abs(ccmad)
                            tstda = mad(ccmad.data)

                            # define threshold as 9 times std  and quality index
                            thresholdd = settings['factor_thre'] * tstda

                            # Run coincidence trigger on a single CC trace resulting from the CFTs stack
                            # essential threshold parameters Cross correlation thresholds
                            thr_on = thresholdd
                            thr_off = thresholdd - 0.15 * thresholdd
                            thr_coincidence_sum = 1.0
                            similarity_thresholds = {"BH": thr_on}
                            trigger_type = None
                            stcc = Stream(traces=[ccmad])
                            triglist = coincidence_trigger(trigger_type,
                                                           thr_on,
                                                           thr_off,
                                                           stcc,
                                                           thr_coincidence_sum,
                                                           trace_ids=None,
                                                           similarity_thresholds=similarity_thresholds,
                                                           delete_long_trigger=False,
                                                           trigger_off_extension=3.0,
                                                           details=True)

                            for itrig, trg in enumerate(triglist):
                                # tdifmin is computed for contributing channels within the stack function
                                if tdifmin == min_time_value:
                                    tt = trg["time"] + min_time_value
                                else:
                                    diff_time = min_time_value - tdifmin
                                    tt = trg["time"] + diff_time + min_time_value
                                [nch,
                                 cft_ave,
                                 crt,
                                 cft_ave_trg,
                                 crt_trg,
                                 nch3,
                                 nch5,
                                 nch7,
                                 nch9,
                                 channels_list] = csc(stall, stcc, trg, tstda, settings['sample_tol'], settings['cc_threshold'], nch_min)

                                if int(nch) >= nch_min:
                                    nn = len(chunk_stream)
                                    md = np.zeros(nn)
                                    # for each trigger, detrended, and filtered continuous
                                    # data channels are trimmed and amplitude useful to
                                    # estimate magnitude is measured.
                                    timex = UTCDateTime(tt)
                                    for il, tc in enumerate(chunk_stream):
                                        ss = tc.stats.station
                                        ich = tc.stats.channel
                                        netwk = tc.stats.network
                                        if template_stream.select(station=ss, channel=ich).__nonzero__():
                                            ttt = template_stream.select(station=ss, channel=ich)[0]
                                            uts = UTCDateTime(ttt.stats.starttime).timestamp
                                            utr = UTCDateTime(reft).timestamp
                                            timestart = timex - tdifmin + (uts - utr)
                                            timend = timestart + settings['temp_length']
                                            ta = tc.copy()
                                            ta.trim(starttime=timestart, endtime=timend, pad=True, fill_value=0)
                                            tid_c = f"{ss}.{ich}"
                                            damaxac = max(abs(ta.data))
                                            dct = damaxac
                                            dtt = damaxat[tid_c]
                                            if dct != 0 and dtt != 0:
                                                md[il] = mag_detect(mt, damaxat[tid_c], damaxac)
                                    mdr = reject_moutliers(md, 1)
                                    mm = round(mdr.mean(), 2)
                                    record = (itemp, itrig, UTCDateTime(tt), mm, mt, nch, tstda,
                                             cft_ave, crt, cft_ave_trg, crt_trg,
                                             nch3, nch5, nch7, nch9, channels_list)
                                    events_list.append(record)
                        else:
                            logging.info(f"{day} {itemp} {t1} {t2} num. correlograms lower than nch_min\n")
                    else:
                        logging.info(f"{day} {itemp} {t1} {t2}  num.  24h channels lower than nch_min\n")
            else:
                logging.info(f"{day} {itemp} num.  templates lower than nch_min\n")

            with open(f"{itemp}.{day.strftime('%y%m%d')}.stats", "w+") as file:
                for event in events_list:
                    channels_list = event[-1]
                    for channel in channels_list:
                        str22 = "%s %s %s %s \n" % channel
                        file.write(str22)
                    file.write(" ".join(f"{x}" for x in event[:-1]) + "\n")

    print(" elapsed time ", timer() - start_time, " seconds")
