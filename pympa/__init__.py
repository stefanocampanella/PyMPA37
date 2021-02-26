import itertools
import os
import os.path
import datetime
from math import log10
from pathlib import Path

import numpy as np
import logging
from obspy import read, Stream, Trace, UTCDateTime
from obspy.signal.cross_correlation import correlate_template
from obspy.signal.trigger import coincidence_trigger
from yaml import load, FullLoader


def find_events(day, itemp, template_stream, travel_times, mt, settings):
    h24 = 86400
    nchunk = settings['nchunk']
    nch_min = settings['nch_min']
    chunk_start = UTCDateTime(day)
    end_time = chunk_start + h24
    chunks = []
    while chunk_start < end_time:
        chunk_end = min(chunk_start + h24 / nchunk, end_time)
        chunks.append((chunk_start, chunk_end))
        chunk_start += h24 / nchunk

    events_list = []
    for t1, t2 in chunks:
        logging.debug(f"Processing chunk ({t1}, {t2})")
        chunk_stream = get_chunk_stream(template_stream, day, t1, t2, settings)

        if len(chunk_stream) >= nch_min:
            stall, ccmad, tdifmin = stack_all(itemp, chunk_stream, travel_times, settings)

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

                # find minimum time to recover origin time
                min_time_value = min(float(v) for v in travel_times.values())
                damaxat = {}
                for il, tr in enumerate(template_stream):
                    sta_t = tr.stats.station
                    cha_t = tr.stats.channel
                    tid_t = "%s.%s" % (sta_t, cha_t)
                    damaxat[tid_t] = max(abs(tr.data))
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
                     channels_list] = csc(stall, stcc, trg, tstda, settings['sample_tol'],
                                          settings['cc_threshold'], nch_min)

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
                                # reference time to be used for retrieving time synchronization
                                reft = min(tr.stats.starttime for tr in template_stream)
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
    return events_list

def stack_all(itemp, chunk_stream, travel_times, settings):
    utc_prec = settings['utc_prec']
    h24 = 86400
    nchunk = settings['nchunk']
    correlation_stream = get_correlation_stream(itemp, chunk_stream, settings)
    # seconds in 24 hours
    nfile = len(correlation_stream)
    tstart = np.empty(nfile)
    tend = np.empty(nfile)
    tdif = np.empty(nfile)
    stall = Stream()
    for idx, tc_cft in enumerate(correlation_stream):
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
    # stall = get_stall(itemp, chunk_stream, travel_times, settings)
    tstart = min(tr.stats.starttime for tr in stall)
    df = stall[0].stats.sampling_rate
    npts = stall[0].stats.npts
    ccmad, tdifmin = stack(stall, df, tstart, travel_times, npts, settings)
    logging.debug(f"tdifmin = {tdifmin}")

    return stall, ccmad, tdifmin


def get_stall(itemp, chunk_stream, travel_times, settings):
    h24 = 86400
    nchunk = settings['nchunk']
    stall = Stream()
    for nn, ss, ich in itertools.product(settings['networks'], settings['stations'], settings['channels']):
        if tc_cft := process_input(itemp, nn, ss, ich, chunk_stream, settings):
            # waveforms should have the same number of npts
            # and should be synchronized to the S-wave travel time
            tstart = tc_cft.stats.starttime + float(travel_times[f"{nn}.{ss}.{ich}"])
            tend = tstart + (h24 / nchunk) + 60
            stall += tc_cft.trim(starttime=tstart, endtime=tend, nearest_sample=True, pad=True, fill_value=0)
    return stall


def get_correlation_stream(itemp, chunk_stream, settings):
    stream_cft = Stream()
    for nn, ss, ich in itertools.product(settings['networks'], settings['stations'],
                                         settings['channels']):
        stream_cft += process_input(itemp, nn, ss, ich, chunk_stream, settings)
    return stream_cft


def get_chunk_stream(template_stream, day, t1, t2, settings):
    stream_df = Stream()
    for tr in template_stream:
        cont_dir = Path(settings['cont_dir'])
        filepath = cont_dir / f"{day.strftime('%y%m%d')}.{tr.stats.station}.{tr.stats.channel}"
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with filepath.open('rb') as file:
                st = read(file, starttime=t1, endtime=t2, dtype="float32")
            if len(st) != 0:
                st.merge(method=1, fill_value=0)
                tc = st[0]
                tc.detrend("constant")
                # 24h continuous trace starts 00 h 00 m 00.0s
                trim_fill(tc, t1, t2)
                tc.filter("bandpass", freqmin=settings['lowpassf'], freqmax=settings['highpassf'], zerophase=True)
                # store detrended and filtered continuous data in a Stream
                stream_df += Stream(traces=[tc])
    return stream_df


def get_template_stream(itemp, travel_times, settings):
    template_stream = Stream()
    temp_dir = Path(settings['temp_dir'])
    v = sorted(travel_times, key=lambda x: float(travel_times[x]))[0:settings['chan_max']]
    for vvc in v:
        n_net, n_sta, n_chn = vvc.split(".")
        filepath = temp_dir / f"{itemp}.{n_net}.{n_sta}..{n_chn}.mseed"
        with filepath.open('rb') as file:
            logging.debug(f"Reading {filepath}")
            template_stream += read(file, dtype="float32")
    return template_stream


def listdays(start, stop):
    date = start
    while date < stop:
        yield date
        date += datetime.timedelta(days=1)


def read_settings(par):
    # read 'parameters24' file to setup useful variables
    with open(par) as file:
        settings = load(file, FullLoader)
    return settings


def trim_fill(tc, t1, t2):
    tc.trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
    return tc


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# itemp = template number, nn =  network code, ss = station code,
# ich = channel code, stream_df = Stream() object as defined in obspy library
def process_input(itemp, nn, ss, ich, stream_df, settings):
    template_directory = Path(settings['temp_dir'])
    template_path = template_directory / f"{itemp}.{nn}.{ss}..{ich}.mseed"
    st_cft = Stream()
    if os.path.isfile(template_path):
        if os.path.getsize(template_path) > 0:
            # print "ok template exist and not empty"
            with template_path.open('rb') as template_file:
                st_temp = read(template_file, dtype="float32")
            tt = st_temp[0]
            # continuous data are stored in stream_df
            sc = stream_df.select(station=ss, channel=ich)
            if sc:
                tc = sc[0]
                fct = correlate_template(tc.data, tt.data, normalize="full", method="auto")
                fct = np.nan_to_num(fct)
                stats = {"network": tc.stats.network,
                         "station": tc.stats.station,
                         "location": "",
                         "channel": tc.stats.channel,
                         "starttime": tc.stats.starttime,
                         "npts": len(fct),
                         "sampling_rate": tc.stats.sampling_rate,
                         "mseed": {"dataquality": "D"}}
                st_cft = Stream(Trace(data=fct, header=stats))
            else:
                logging.debug("No stream is found")
        else:
            logging.debug(f"{template_path} is empty")
    return st_cft


def quality_cft(trac):
    return np.nanstd(abs(trac.data))


def stack(stall, df, tstart, d, npts, settings):
    """
    Function to stack traces in a stream with different trace.id and
    different starttime but the same number of datapoints.
    Returns a trace having as starttime
    the earliest startime within the stream
    """
    stdup = settings['stdup']
    stddown = settings['stddown']
    nch_min = settings['nch_min']
    std_trac = np.empty(len(stall))
    td = np.fromiter((quality_cft(tr) for tr in stall), dtype=float)
    avestd = np.nanmean(std_trac)
    avestdup = avestd * stdup
    avestddw = avestd * stddown

    for jtr, tr in enumerate(stall):
        if std_trac[jtr] >= avestdup or std_trac[jtr] <= avestddw:
            stall.remove(tr)
            logging.debug(f"Removed Trace n Stream = {tr} {std_trac[jtr]} {avestd}")
            td[jtr] = 99.99
        else:
            sta = tr.stats.station
            chan = tr.stats.channel
            net = tr.stats.network
            s = f"{net}.{sta}.{chan}"
            td[jtr] = float(d[s])

    header = {"network": "STACK",
              "station": "BH",
              "channel": "XX",
              "starttime": tstart,
              "sampling_rate": df,
              "npts": npts}

    if len(stall) >= nch_min:
        tdifmin = min(td)
        data = np.nanmean([tr.data for tr in stall], axis=0)
    else:
        tdifmin = None
        data = np.zeros(npts)
    return Trace(data=data, header=header), tdifmin


def csc(stall, stcc, trg, tstda, sample_tolerance, single_channelcft, nch_min):
    """
    The function check single channel cft compute the maximum CFT's
    values at each trigger time and counts the number of channels
    having higher cross-correlation
    nch, cft_ave, crt are re-evaluated on the basis of
    +/- 2 sample approximation. Statistics are written in stat files
    """
    # important parameters: a sample_tolerance less than 2 results often
    # in wrong magnitudes
    trigger_time = trg["time"]
    tcft = stcc[0]
    t0_tcft = tcft.stats.starttime
    trigger_shift = trigger_time.timestamp - t0_tcft.timestamp
    trigger_sample = round(trigger_shift / tcft.stats.delta)
    max_sct = np.empty(len(stall))
    max_trg = np.empty(len(stall))
    max_ind = np.empty(len(stall))
    chan_sct = np.chararray((len(stall),), 12)

    for icft, tsc in enumerate(stall):
        # get cft amplitude value at corresponding trigger and store it in
        # check for possible 2 sample shift and eventually change trg['cft_peaks']
        chan_sct[icft] = (tsc.stats.network + "." + tsc.stats.station + " " + tsc.stats.channel)
        tmp0 = max(trigger_sample - sample_tolerance, 0)
        tmp1 = trigger_sample + sample_tolerance + 1
        max_sct[icft] = max(tsc.data[tmp0:tmp1])
        max_ind[icft] = np.nanargmax(tsc.data[tmp0:tmp1])
        max_ind[icft] = sample_tolerance - max_ind[icft]
        max_trg[icft] = tsc.data[trigger_sample: trigger_sample + 1]
    nch = (max_sct > single_channelcft).sum()

    if nch >= nch_min:
        nch09 = (max_sct > 0.9).sum()
        nch07 = (max_sct > 0.7).sum()
        nch05 = (max_sct > 0.5).sum()
        nch03 = (max_sct > 0.3).sum()
        cft_ave = np.nanmean(max_sct)
        crt = cft_ave / tstda
        cft_ave_trg = np.nanmean(max_trg)
        crt_trg = cft_ave_trg / tstda
        max_sct = max_sct.T
        max_trg = max_trg.T
        chan_sct = chan_sct.T
        channels_list = []
        for idchan in range(len(max_sct)):
            record = (chan_sct[idchan].decode(),
                      max_trg[idchan],
                      max_sct[idchan],
                      max_ind[idchan])
            channels_list.append(record)
        return nch, cft_ave, crt, cft_ave_trg, crt_trg, nch03, nch05, nch07, nch09, channels_list
    else:
        return tuple(1 for _ in range(10))


def mag_detect(magt, amaxt, amaxd):
    """
    mag_detect(mag_temp,amax_temp,amax_detect)
    Returns the magnitude of the new detection by using the template/detection
    amplitude trace ratio
    and the magnitude of the template event
    """
    amaxr = amaxt / amaxd
    magd = magt - log10(amaxr)
    return magd


def reject_moutliers(data, m=1.0):
    nonzeroind = np.nonzero(data)[0]
    nzlen = len(nonzeroind)
    data = data[nonzeroind]
    datamed = np.nanmedian(data)
    d = np.abs(data - datamed)
    mdev = 2 * np.median(d)
    if mdev == 0:
        inds = np.arange(nzlen)
        data[inds] = datamed
    else:
        s = d / mdev
        inds = np.where(s <= m)
    return data[inds]


def mad(dmad):
    """
    calculate daily median absolute deviation
    """
    ccm = dmad[dmad != 0]
    med_val = np.nanmedian(ccm)
    return np.nansum(abs(ccm - med_val) / len(ccm))
