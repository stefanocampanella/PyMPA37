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


def listdays(start, stop):
    date = start
    while date < stop:
        yield date
        date += datetime.timedelta(days=1)


def list_chunks(day, nchunk):
    delta = datetime.timedelta(days=1) / nchunk
    end = day + datetime.timedelta(days=1)
    chunk_start = UTCDateTime(day)
    while chunk_start < end:
        yield chunk_start, min(chunk_start + delta, end)
        chunk_start += delta


def find_events(itemp, template_stream, continuous_stream, travel_times, mt, settings):
    events_list = []
    correlation_stream = get_correlation_stream(itemp, continuous_stream, settings)
    stall, ccmad, tdifmin = stack(correlation_stream, travel_times, settings)
    # compute mean absolute deviation of abs(ccmad)
    tstda = mad(ccmad.data)
    # define threshold as 9 times std  and quality index
    threshold = settings['factor_thre'] * tstda
    stcc = Stream(traces=ccmad)
    if tdifmin is not None:
        # Run coincidence trigger on a single CC trace resulting from the CFTs stack
        # essential threshold parameters Cross correlation thresholds
        triglist = coincidence_trigger(None,
                                       threshold,
                                       threshold - 0.15 * threshold,
                                       stcc,
                                       1.0,
                                       trace_ids=None,
                                       similarity_thresholds={"BH": threshold},
                                       delete_long_trigger=False,
                                       trigger_off_extension=3.0,
                                       details=True)
        # find minimum time to recover origin time
        min_time_value = min(float(v) for v in travel_times.values())
        for itrig, trg in enumerate(triglist):
            [nch,
         cft_ave,
         crt,
         cft_ave_trg,
         crt_trg,
         nch3,
         nch5,
         nch7,
         nch9,
         channels_list] = csc(stall, stcc, trg, tstda, settings)
            # for each trigger, detrended, and filtered continuous
            # data channels are trimmed and amplitude useful to
            # estimate magnitude is measured.
            # tdifmin is computed for contributing channels within the stack function
            tt = trg["time"] + 2 * min_time_value - tdifmin
            if nch >= settings['nch_min']:
                md = np.zeros(len(continuous_stream))
                for il, tc in enumerate(continuous_stream):
                    ss = tc.stats.station
                    ich = tc.stats.channel
                    if template_stream.select(station=ss, channel=ich):
                        template_trace, = template_stream.select(station=ss, channel=ich)
                        uts = template_trace.stats.starttime
                        # reference time to be used for retrieving time synchronization
                        reft = min(tr.stats.starttime for tr in template_stream)
                        timestart = tt - tdifmin + (uts - reft)
                        timend = timestart + settings['temp_length']
                        ta = tc.copy()
                        ta.trim(starttime=timestart, endtime=timend, pad=True, fill_value=0)
                        damaxac = max(abs(ta.data))
                        dtt = max(abs(template_trace.data))
                        if damaxac != 0 and dtt != 0:
                            md[il] = mag_detect(mt, dtt, damaxac)
                mdr = reject_moutliers(md)
                mm = round(mdr.mean(), 2)
                record = (itemp, itrig, tt, mm, mt, nch, tstda,
                          cft_ave, crt, cft_ave_trg, crt_trg,
                          nch3, nch5, nch7, nch9, channels_list)
                events_list.append(record)
    return events_list


def get_correlation_stream(itemp, chunk_stream, settings):
    stream_cft = Stream()
    for nn, ss, ich in itertools.product(settings['networks'],
                                         settings['stations'],
                                         settings['channels']):
        stream_cft += process_input(itemp, nn, ss, ich, chunk_stream, settings)
    return stream_cft


def get_continuous_stream(template_stream, day, t1, t2, settings):
    cont_dir = Path(settings['cont_dir'])
    stream_df = Stream()
    for tr in template_stream:
        filepath = cont_dir / f"{day.strftime('%y%m%d')}.{tr.stats.station}.{tr.stats.channel}"
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with filepath.open('rb') as file:
                st = read(file, dtype="float32", starttime=t1, endtime=t2)
                if st:
                    st.merge(method=1, fill_value=0)
                    tc = st[0]
                    tc.detrend("constant")
                    tc.trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
                    tc.filter("bandpass", freqmin=settings['lowpassf'], freqmax=settings['highpassf'], zerophase=True)
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


def stack(correlation_stream, travel_times, settings):
    """
    Function to stack traces in a stream with different trace.id and
    different starttime but the same number of datapoints.
    Returns a trace having as starttime
    the earliest startime within the stream
    """
    nchunk = settings['nchunk']
    stall = Stream()
    for tc_cft in correlation_stream:
        sta = tc_cft.stats.station
        chan = tc_cft.stats.channel
        net = tc_cft.stats.network
        tstart = UTCDateTime(tc_cft.stats.starttime + float(travel_times[f"{net}.{sta}.{chan}"]))
        tend = tstart + datetime.timedelta(days=1) / nchunk
        stall += tc_cft.trim(starttime=tstart, endtime=tend, nearest_sample=True, pad=True, fill_value=0)

    stdup = settings['stdup']
    stddown = settings['stddown']
    nch_min = settings['nch_min']
    std_trac = np.fromiter((np.nanstd(abs(tr.data)) for tr in stall), dtype=float)
    avestd = np.nanmean(std_trac)
    avestdup = avestd * stdup
    avestddw = avestd * stddown

    td = np.empty(len(stall))
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
            td[jtr] = float(travel_times[s])

    if len(stall) >= nch_min:
        tdifmin = min(td)
        data = np.nanmean([tr.data for tr in stall], axis=0)
    else:
        tdifmin = None
        data = np.zeros_like(stall[0].data)

    header = {"network": "STACK",
              "station": "BH",
              "channel": "XX",
              "starttime": min(tr.stats.starttime for tr in stall),
              "sampling_rate": stall[0].stats.sampling_rate,
              "npts": stall[0].stats.npts}
    return stall, Trace(data=data, header=header), tdifmin


def csc(stall, stcc, trg, tstda, settings):
    """
    The function check single channel cft compute the maximum CFT's
    values at each trigger time and counts the number of channels
    having higher cross-correlation
    nch, cft_ave, crt are re-evaluated on the basis of
    +/- 2 sample approximation. Statistics are written in stat files
    """
    # important parameters: a sample_tolerance less than 2 results often
    # in wrong magnitudes
    sample_tolerance = settings['sample_tol']
    single_channelcft = settings['cc_threshold']
    nch_min = settings['nch_min']

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
        nch03, nch05, nch07, nch09 = [(max_sct > threshold).sum() for threshold in (0.3, 0.5, 0.7, 0.9)]
        cft_ave = np.nanmean(max_sct)
        crt = cft_ave / tstda
        cft_ave_trg = np.nanmean(max_trg)
        crt_trg = cft_ave_trg / tstda
        max_sct = max_sct.T
        max_trg = max_trg.T
        chan_sct = chan_sct.T
        channels_list = []
        for idchan in range(len(max_sct)):
            record = (chan_sct[idchan].decode(), max_trg[idchan], max_sct[idchan], max_ind[idchan])
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
