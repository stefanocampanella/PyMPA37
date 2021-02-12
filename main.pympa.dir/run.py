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
import os.path
from time import perf_counter

import numpy as np
from obspy import read, Stream, Trace
from obspy.core import UTCDateTime
from obspy.core.event import read_events
from obspy.io.quakeml.core import _is_quakeml
from obspy.io.zmap.core import _is_zmap
from obspy.signal.trigger import coincidence_trigger

from pympa import read_parameters, listdays, trim_fill, stack, mad, process_input, mag_detect, reject_moutliers, csc
import logging

if __name__ == '__main__':
    logging.basicConfig(filename='run.log',
                        filemode='w',
                        format='%(levelname)s: %(message)s')
    start_time = perf_counter()
    # read 'parameters24' file to setup useful variables
    [
        stations,
        channels,
        networks,
        lowpassf,
        highpassf,
        sample_tol,
        cc_threshold,
        nch_min,
        temp_length,
        utc_prec,
        cont_dir,
        temp_dir,
        travel_dir,
        dateperiod,
        ev_catalog,
        t_start,
        t_stop,
        factor_thre,
        stdup,
        stddown,
        chan_max,
        nchunk
    ] = read_parameters("parameters24")

    # set time precision for UTCDATETIME
    UTCDateTime.DEFAULT_PRECISION = utc_prec

    # read Catalog of Templates Events
    logging.debug("event catalog should be ZMAP or QUAKEML")
    if _is_zmap(ev_catalog):
        logging.debug("reading ZMAP catalog")
    elif _is_quakeml(ev_catalog):
        logging.debug("reading QUAKEML catalog")
    else:
        logging.warning("Event catalog is neither ZMAP nor QUAKEML")
    cat = read_events(ev_catalog)

    # generate list of days "
    year = int(dateperiod[0])
    month = int(dateperiod[1])
    day = int(dateperiod[2])
    period = int(dateperiod[3])

    for day in listdays(year, month, day, period):
        # settings to cut exactly 24 hours file from without including
        # previous/next day
        logging.debug(f"Processing {day}")

        for itemp in range(t_start, t_stop):
            # open file containing detections
            logging.debug(f"itemp = {itemp}")
            # open statistics file for each detection
            f1 = open(f"{itemp}.{day}.stats", "w+")

            mt = cat[itemp].magnitudes[0].mag

            # read ttimes, select the num_ttimes (parameters,
            # last line) channels
            # and read only these templates
            with open(f"{travel_dir}{itemp}.ttimes", "r") as ttim:
                d = dict(x.rstrip().split(None, 1) for x in ttim)
                v = sorted(d, key=lambda x: float(d[x]))[0:chan_max]

            stt = Stream()
            for vvc in v:
                n_net, n_sta, n_chn = vvc.split(".")
                filename = f"{temp_dir}{itemp}.{n_net}.{n_sta}..{n_chn}.mseed"
                logging.debug(f"Reading {filename}")
                stt += read(filename, dtype="float32")

            if len(stt) >= nch_min:
                tc = Trace()
                chunks = []
                h24 = 86400
                chunk_start = UTCDateTime(day)
                end_time = chunk_start + h24

                while chunk_start < end_time:
                    chunk_end = min(chunk_start + h24 / nchunk, end_time)
                    chunks.append((chunk_start, chunk_end))
                    chunk_start += h24 / nchunk

                for t1, t2 in chunks:
                    logging.debug(f"Processing chunk ({t1}, {t2})")
                    stream_df = Stream()
                    for tr in stt:
                        finpc1 = f"{cont_dir}{day.strftime('%y%m%d')}.{tr.stats.station}.{tr.stats.channel}"
                        if os.path.exists(finpc1) and os.path.getsize(finpc1) > 0:
                            st = read(finpc1, starttime=t1, endtime=t2, dtype="float32")
                            if len(st) != 0:
                                st.merge(method=1, fill_value=0)
                                tc = st[0]
                                stat = tc.stats.station
                                chan = tc.stats.channel
                                tc.detrend("constant")
                                # 24h continuous trace starts 00 h 00 m 00.0s
                                trim_fill(tc, t1, t2)
                                tc.filter("bandpass", freqmin=lowpassf, freqmax=highpassf, zerophase=True)
                                # store detrended and filtered continuous data in a Stream
                                stream_df += Stream(traces=[tc])

                    if len(stream_df) >= nch_min:
                        ntl = len(stt)
                        # for each template event
                        damaxat = {}
                        # reference time to be used for
                        # retrieving time synchronization
                        reft = min([tr.stats.starttime for tr in stt])

                        for il, tr in enumerate(stt):
                            sta_t = tr.stats.station
                            cha_t = tr.stats.channel
                            tid_t = "%s.%s" % (sta_t, cha_t)
                            damaxat[tid_t] = max(abs(tr.data))

                        # find minimum time to recover origin time
                        time_values = [float(v) for v in d.values()]
                        min_time_value = min(time_values)
                        min_time_key = [k for k, v in d.items() if v == str(min_time_value)]

                        # clear global_variable
                        stream_cft = Stream()
                        for nn, ss, ich in itertools.product(networks, stations, channels):
                            stream_cft += process_input(temp_dir, itemp, nn, ss, ich, stream_df)

                        # seconds in 24 hours
                        nfile = len(stream_cft)
                        tstart = np.empty(nfile)
                        tend = np.empty(nfile)
                        tdif = np.empty(nfile)

                        for idx, tc_cft in enumerate(stream_cft):
                            # get station name from trace
                            sta = tc_cft.stats.station
                            chan = tc_cft.stats.channel
                            net = tc_cft.stats.network
                            delta = tc_cft.stats.delta

                            npts = (h24 / nchunk) / delta
                            tdif[idx] = float(d[f"{net}.{sta}.{chan}"])

                        stall = Stream()
                        for idx, tc_cft in enumerate(stream_cft):
                            # get stream starttime
                            tstart[idx] = tc_cft.stats.starttime + tdif[idx]
                            # waveforms should have the same number of npts
                            # and should be synchronized to the S-wave travel time
                            secs = (h24 / nchunk) + 60
                            tend[idx] = tstart[idx] + secs
                            check_npts = (tend[idx] - tstart[idx]) / tc_cft.stats.delta
                            ts = UTCDateTime(tstart[idx], precision=utc_prec)
                            te = UTCDateTime(tend[idx], precision=utc_prec)
                            stall += tc_cft.trim(starttime=ts, endtime=te, nearest_sample=True, pad=True, fill_value=0)
                        tstart = min([tr.stats.starttime for tr in stall])
                        df = stall[0].stats.sampling_rate
                        npts = stall[0].stats.npts

                        # compute mean cross correlation from the stack of
                        # CFTs (see stack function)
                        ccmad, tdifmin = stack(stall, df, tstart, d, npts, stdup, stddown, nch_min)
                        logging.debug(f"tdifmin = {tdifmin}")

                        if tdifmin is not None:
                            # compute mean absolute deviation of abs(ccmad)
                            tstda = mad(ccmad.data)

                            # define threshold as 9 times std  and quality index
                            thresholdd = factor_thre * tstda

                            # Run coincidence trigger on a single CC trace
                            # resulting from the CFTs stack
                            # essential threshold parameters
                            # Cross correlation thresholds
                            thr_on = thresholdd
                            thr_off = thresholdd - 0.15 * thresholdd
                            thr_coincidence_sum = 1.0
                            similarity_thresholds = {"BH": thr_on}
                            trigger_type = None
                            stcc = Stream(traces=[ccmad])
                            triglist = coincidence_trigger(
                                trigger_type,
                                thr_on,
                                thr_off,
                                stcc,
                                thr_coincidence_sum,
                                trace_ids=None,
                                similarity_thresholds=similarity_thresholds,
                                delete_long_trigger=False,
                                trigger_off_extension=3.0,
                                details=True,
                            )
                            tdifmin = min(tdif)

                            for itrig, trg in enumerate(triglist):
                                # tdifmin is computed for contributing channels
                                # within the stack function
                                if tdifmin == min_time_value:
                                    tt = trg["time"] + min_time_value
                                else:
                                    diff_time = min_time_value - tdifmin
                                    tt = trg["time"] + diff_time + min_time_value
                                [
                                    nch,
                                    cft_ave,
                                    crt,
                                    cft_ave_trg,
                                    crt_trg,
                                    nch3,
                                    nch5,
                                    nch7,
                                    nch9
                                ] = csc(stall, stcc, trg, tstda, sample_tol, cc_threshold, nch_min, f1)

                                if int(nch) >= nch_min:
                                    nn = len(stream_df)
                                    md = np.zeros(nn)
                                    # for each trigger, detrended,
                                    # and filtered continuous
                                    # data channels are trimmed and
                                    # amplitude useful to
                                    # estimate magnitude is measured.
                                    timex = UTCDateTime(tt)
                                    for il, tc in enumerate(stream_df):
                                        ss = tc.stats.station
                                        ich = tc.stats.channel
                                        netwk = tc.stats.network
                                        if stt.select(station=ss, channel=ich).__nonzero__():
                                            ttt = stt.select(station=ss, channel=ich)[0]
                                            uts = UTCDateTime(ttt.stats.starttime).timestamp
                                            utr = UTCDateTime(reft).timestamp
                                            timestart = timex - tdifmin + (uts - utr)
                                            timend = timestart + temp_length
                                            ta = tc.copy()
                                            ta.trim(starttime=timestart, endtime=timend, pad=True, fill_value=0)
                                            tid_c = f"{ss}.{ich}"
                                            damaxac = max(abs(ta.data))
                                            dct = damaxac
                                            dtt = damaxat[tid_c]
                                            if dct != 0 and dtt != 0:
                                                md[il] = mag_detect(mt, damaxat[tid_c], damaxac)
                                                # f2.write(f"{tid_c} {md[il]}\n")
                                    mdr = reject_moutliers(md, 1)
                                    mm = round(mdr.mean(), 2)
                                    str33 = (itemp, itrig, UTCDateTime(tt), mm, mt, nch, tstda,
                                             cft_ave, crt, cft_ave_trg, crt_trg,
                                             nch3, nch5, nch7, nch9)
                                    f1.write(" ".join(f"{x}" for x in str33) + "\n")
                        else:
                            logging.info(f"{day} {itemp} {t1} {t2} num. correlograms lower than nch_min\n")
                    else:
                        logging.info(f"{day} {itemp} {t1} {t2}  num.  24h channels lower than nch_min\n")
            else:
                logging.info(f"{day} {itemp} num.  templates lower than nch_min\n")
    print(" elapsed time ", perf_counter() - start_time, " seconds")
