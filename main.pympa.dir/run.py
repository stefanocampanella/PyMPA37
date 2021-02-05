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
# import useful libraries

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
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
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
        start_itemp,
        stop_itemp,
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
    ncat = len(cat)

    # read template from standard input
    # startTemplate = input("INPUT: Enter Starting template ")
    # stopTemplate = input("INPUT: Enter Ending template ")
    # print("OUTPUT: Running from template", startTemplate,  " to ", stopTemplate)
    t_start = start_itemp
    t_stop = stop_itemp

    # loop over days

    # generate list of days "
    year = int(dateperiod[0])
    month = int(dateperiod[1])
    day = int(dateperiod[2])
    period = int(dateperiod[3])
    days = listdays(year, month, day, period)

    # initialise stt as a stream of templates
    # and stream_df as a stream of continuous waveforms
    stt = Stream()
    stream_df = Stream()
    stream_cft = Stream()
    stall = Stream()
    ccmad = Trace()

    for day in days:
        # settings to cut exactly 24 hours file from without including
        # previous/next day
        iday = "%s" % (day[4:6])
        imonth = "%s" % (day[2:4])
        iyear = "20%s" % (day[0:2])
        iiyear = int(iyear)
        logging.debug(f"Processing {iyear}, {imonth}, {iday}")
        iimonth = int(imonth)
        iiday = int(iday)
        iihour = 23
        iimin = 59
        iisec = 0

        for itemp in range(t_start, t_stop):
            stt.clear()
            # open file containing detections
            f = open(f"{itemp}.{day}.cat", "w+")
            logging.debug(f"itemp = {itemp}")
            # open statistics file for each detection
            f1 = open(f"{itemp}.{day}.stats", "w+")
            # open file including magnitude information
            f2 = open(f"{itemp}.{day}.stats.mag", "w+")
            # open file listing exceptions
            f3 = open(f"{itemp}.{day}.except", "w+")

            mt = cat[itemp].magnitudes[0].mag

            # read ttimes, select the num_ttimes (parameters,
            # last line) channels
            # and read only these templates
            with open(f"{travel_dir}{itemp}.ttimes", "r") as ttim:
                d = dict(x.rstrip().split(None, 1) for x in ttim)
                v = sorted(d, key=lambda x: float(d[x]))[0:chan_max]

            for vvc in v:
                n_net, n_sta, n_chn = vvc.split(".")
                filename = f"{temp_dir}{itemp}.{n_net}.{n_sta}..{n_chn}.mseed"
                logging.debug(f"Reading {filename}")
                stt += read(filename, dtype="float32")

            if len(stt) >= nch_min:
                tc = Trace()
                chunks = []
                h24 = 86400
                chunk_start = UTCDateTime(iiyear, iimonth, iiday)
                end_time = chunk_start + h24

                while chunk_start < end_time:
                    chunk_end = min(chunk_start + h24 / nchunk, end_time)
                    chunks.append((chunk_start, chunk_end))
                    chunk_start += h24 / nchunk

                for t1, t2 in chunks:
                    logging.debug(f"Processing chunk ({t1}, {t2})")
                    stream_df.clear()
                    for tr in stt:
                        finpc1 = f"{cont_dir}{day}.{tr.stats.station}.{tr.stats.channel}"
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
                                # store detrended and filtered continuous data
                            # in a Stream
                                stream_df += Stream(traces=[tc])

                    if len(stream_df) >= nch_min:
                        ntl = len(stt)
                        amaxat = np.empty(ntl)
                        # for each template event
                        md = np.zeros(ntl)
                        damaxat = {}
                        # reference time to be used for
                        # retrieving time synchronization
                        reft = min([tr.stats.starttime for tr in stt])

                        for il, tr in enumerate(stt):
                            amaxat[il] = max(abs(tr.data))
                            sta_t = tr.stats.station
                            cha_t = tr.stats.channel
                            tid_t = "%s.%s" % (sta_t, cha_t)
                            damaxat[tid_t] = float(amaxat[il])

                        # find minimum time to recover origin time
                        time_values = [float(v) for v in d.values()]
                        min_time_value = min(time_values)
                        min_time_key = [k for k, v in d.items() if v == str(min_time_value)]

                        # clear global_variable
                        stream_cft.clear()
                        stcc = Stream()
                        for nn in networks:
                            for ss in stations:
                                for ich in channels:
                                    stream_cft += process_input(temp_dir, itemp, nn, ss, ich, stream_df)
                        stall.clear()
                        stcc.clear()
                        stnew = Stream()
                        tr = Trace()

                        tc_cft = Trace()
                        tsnew = UTCDateTime()

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

                            # Trace ccmad is stored in a Stream
                            stcc = Stream(traces=[ccmad])

                            # Run coincidence trigger on a single CC trace
                            # resulting from the CFTs stack
                            # essential threshold parameters
                            # Cross correlation thresholds
                            xcor_cut = thresholdd
                            thr_on = thresholdd
                            thr_off = thresholdd - 0.15 * thresholdd
                            thr_coincidence_sum = 1.0
                            similarity_thresholds = {"BH": thr_on}
                            trigger_type = None
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
                            ntrig = len(triglist)

                            tt = np.empty(ntrig)
                            cs = np.empty(ntrig)
                            nch = np.empty(ntrig)
                            cft_ave = np.empty(ntrig)
                            crt = np.empty(ntrig)
                            cft_ave_trg = np.empty(ntrig)
                            crt_trg = np.empty(ntrig)
                            nch3 = np.empty(ntrig)
                            nch5 = np.empty(ntrig)
                            nch7 = np.empty(ntrig)
                            nch9 = np.empty(ntrig)
                            mm = np.empty(ntrig)
                            timex = UTCDateTime()
                            tdifmin = min(tdif[0:])

                            for itrig, trg in enumerate(triglist):
                                # tdifmin is computed for contributing channels
                                # within the stack function
                                if tdifmin == min_time_value:
                                    tt[itrig] = trg["time"] + min_time_value
                                else:
                                    diff_time = min_time_value - tdifmin
                                    tt[itrig] = trg["time"] + diff_time + min_time_value
                                cs[itrig] = trg["coincidence_sum"]
                                cft_ave[itrig] = trg["cft_peak_wmean"]
                                crt[itrig] = trg["cft_peaks"][0] / tstda
                                # traceID = trg['trace_ids']
                                # check single channel CFT
                                [
                                    nch[itrig],
                                    cft_ave[itrig],
                                    crt[itrig],
                                    cft_ave_trg[itrig],
                                    crt_trg[itrig],
                                    nch3[itrig],
                                    nch5[itrig],
                                    nch7[itrig],
                                    nch9[itrig],
                                ] = csc(stall, stcc, trg, tstda, sample_tol, cc_threshold, nch_min, f1)

                                if int(nch[itrig]) >= nch_min:
                                    nn = len(stream_df)
                                    amaxac = np.zeros(nn)
                                    md = np.zeros(nn)
                                    # for each trigger, detrended,
                                    # and filtered continuous
                                    # data channels are trimmed and
                                    # amplitude useful to
                                    # estimate magnitude is measured.
                                    damaxac = {}
                                    mchan = {}
                                    timestart = UTCDateTime()
                                    timex = UTCDateTime(tt[itrig])
                                    for il, tc in enumerate(stream_df):
                                        ss = tc.stats.station
                                        ich = tc.stats.channel
                                        netwk = tc.stats.network
                                        if stt.select(station=ss, channel=ich).__nonzero__():
                                            ttt = stt.select(station=ss, channel=ich)[0]
                                            s = f"{netwk}.{ss}.{ich}"
                                            uts = UTCDateTime(ttt.stats.starttime).timestamp
                                            utr = UTCDateTime(reft).timestamp
                                            timestart = timex - tdifmin + (uts - utr)
                                            timend = timestart + temp_length
                                            ta = tc.copy()
                                            ta.trim(starttime=timestart, endtime=timend, pad=True, fill_value=0)
                                            amaxac[il] = max(abs(ta.data))
                                            tid_c = f"{ss}.{ich}"
                                            damaxac[tid_c] = float(amaxac[il])
                                            dct = damaxac[tid_c]
                                            dtt = damaxat[tid_c]
                                            if dct != 0 and dtt != 0:
                                                md[il] = mag_detect(mt, damaxat[tid_c], damaxac[tid_c])
                                                mchan[tid_c] = md[il]
                                                f2.write(f"{tid_c}{mchan[tid_c]}\n")
                                    mdr = reject_moutliers(md, 1)
                                    mm[itrig] = round(np.mean(mdr), 2)
                                    cft_ave[itrig] = round(cft_ave[itrig], 3)
                                    crt[itrig] = round(crt[itrig], 3)
                                    cft_ave_trg[itrig] = round(cft_ave_trg[itrig], 3)
                                    crt_trg[itrig] = round(crt_trg[itrig], 3)
                                    str33 = (
                                            "%s %s %s %s %s %s %s %s %s "
                                            "%s %s %s %s %s %s %s\n"
                                            % (
                                                day[0:6],
                                                str(itemp),
                                                str(itrig),
                                                str(UTCDateTime(tt[itrig])),
                                                str(mm[itrig]),
                                                str(mt),
                                                str(nch[itrig]),
                                                str(tstda),
                                                str(cft_ave[itrig]),
                                                str(crt[itrig]),
                                                str(cft_ave_trg[itrig]),
                                                str(crt_trg[itrig]),
                                                str(nch3[itrig]),
                                                str(nch5[itrig]),
                                                str(nch7[itrig]),
                                                str(nch9[itrig]),
                                            )
                                    )
                                    f1.write(str33)
                                    f2.write(str33)
                                    str1 = "%s %s %s %s %s %s %s %s\n" % (
                                        str(itemp),
                                        str(UTCDateTime(tt[itrig])),
                                        str(mm[itrig]),
                                        str(cft_ave[itrig]),
                                        str(crt[itrig]),
                                        str(cft_ave_trg[itrig]),
                                        str(crt_trg[itrig]),
                                        str(int(nch[itrig])),
                                    )
                                    f.write(str1)
                        else:
                            f3.write(f"{day} {itemp} {t1} {t2} num. correlograms lower than nch_min\n")
                    else:
                        f3.write(f"{day} {itemp} {t1} {t2}  num.  24h channels lower than nch_min\n")
            else:
                f3.write(f"{day} {itemp} num.  templates lower than nch_min\n")
            f1.close()
            f2.close()
            f3.close()
            f.close()
    print(" elapsed time ", perf_counter() - start_time, " seconds")
