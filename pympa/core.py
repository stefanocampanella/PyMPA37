import datetime
from math import log10
from functools import lru_cache

import numpy as np
import logging
from obspy import read, Stream, Trace, UTCDateTime
from obspy.signal.cross_correlation import correlate_template
from obspy.signal.trigger import coincidence_trigger


def range_days(start, stop):
    date = start
    while date < stop:
        yield date
        date += datetime.timedelta(days=1)


def read_travel_times(travel_times_dir_path, template_number, chan_max=12):
    travel_times = {}
    with open(travel_times_dir_path / f"{template_number}.ttimes", "r") as file:
        while line := file.readline():
            key, value = line.split(' ')
            key = tuple(key.split('.'))
            value = float(value)
            travel_times[key] = value
    keys = sorted(travel_times, key=lambda x: travel_times[x])
    if len(keys) > chan_max:
        keys = keys[:chan_max]
    travel_times = {key: travel_times[key] for key in keys}
    return travel_times


def read_template_stream(templates_dir_path, template_number, channel_list):
    template_stream = Stream()
    for net, sta, chn in channel_list:
        filepath = templates_dir_path / f"{template_number}.{net}.{sta}..{chn}.mseed"
        with filepath.open('rb') as file:
            logging.debug(f"Reading {filepath}")
            template_stream += read(file, dtype="float32")
    return template_stream


def read_continuous_stream(continuous_dir_path, day, channel_list, freqmin=3.0, freqmax=8.0):
    continuous_stream = Stream()
    for _, st, ch in channel_list:
        try:
            filepath = continuous_dir_path / f"{day.strftime('%y%m%d')}.{st}.{ch}"
            continuous_stream += read_continuous_trace(filepath, day, freqmin, freqmax)
        except Exception as err:
            logging.warning(f"{err}")

    return continuous_stream


@lru_cache
def read_continuous_trace(filepath, day, freqmin, freqmax):
    with filepath.open('rb') as file:
        stream = read(file, dtype="float32")
        stream.merge(method=1, fill_value=0)
        trace, = stream
        trace.detrend("constant")
        begin = UTCDateTime(day)
        trace.trim(starttime=begin,
                   endtime=begin + datetime.timedelta(days=1),
                   pad=True,
                   fill_value=0)
        trace.filter("bandpass",
                     freqmin=freqmin,
                     freqmax=freqmax,
                     zerophase=True)
    return trace


def find_events(template_stream, continuous_stream, travel_times, mt, settings):
    events_list = []
    correlation_stream = compute_correlation_stream(template_stream, continuous_stream)
    stall, ccmad = stack(correlation_stream, travel_times, settings)
    if stall and ccmad is not None:
        # compute mean absolute deviation of abs(ccmad)
        tstda = abs(ccmad.data - np.median(ccmad.data)).mean()
        # define threshold as 9 times std  and quality index
        threshold = settings['factor_thre'] * tstda
        # Run coincidence trigger on a single CC trace resulting from the CFTs stack
        # essential threshold parameters Cross correlation thresholds
        triglist = coincidence_trigger(None,
                                       threshold,
                                       0.85 * threshold,
                                       Stream(ccmad),
                                       1.0)
        for trg in triglist:
            nch, stats, channel_list = csc(stall, ccmad, trg, settings)
            if nch >= settings['nch_min']:
                mdr = magnitude(continuous_stream, template_stream, trg['time'], mt, settings)
                tt = trg["time"] + time_diff_min(stall, travel_times)
                record = (tt, mdr, tstda, stats, channel_list)
                events_list.append(record)
    return events_list


def time_diff_min(stall, travel_times):
    time_diff = []
    for tr in stall:
        net = tr.stats.network
        station = tr.stats.station
        channel = tr.stats.channel
        time_diff.append(travel_times[(net, station, channel)])
    return min(time_diff)


def compute_correlation_stream(template_stream, continuous_stream):
    correlation_stream = Stream()
    for tt in template_stream:
        try:
            sc = continuous_stream.select(station=tt.stats.station, channel=tt.stats.channel)
            if sc:
                tc, = sc
                fct = correlate_template(tc.data, tt.data, normalize="full", method="auto")
                fct = np.nan_to_num(fct)
                header = {"network": tc.stats.network,
                          "station": tc.stats.station,
                          "channel": tc.stats.channel,
                          "starttime": tc.stats.starttime,
                          "npts": len(fct),
                          "sampling_rate": tc.stats.sampling_rate,
                          "mseed": {"dataquality": "D"}}
                correlation_stream += Trace(data=fct, header=header)
        except Exception as err:
            logging.debug(f"{err} while processing station {tt.stats.station}, channel {tt.stats.channel}")
    return correlation_stream


def stack(correlation_stream, travel_times, settings):
    """
    Function to stack traces in a stream with different trace.id and
    different starttime but the same number of datapoints.
    Returns a trace having as starttime
    the earliest startime within the stream
    """
    stall = Stream()
    for tc_cft in correlation_stream:
        sta = tc_cft.stats.station
        chan = tc_cft.stats.channel
        net = tc_cft.stats.network
        tstart = tc_cft.stats.starttime + travel_times[(net, sta, chan)]
        tend = tstart + datetime.timedelta(days=1)
        stall += tc_cft.trim(starttime=tstart, endtime=tend, nearest_sample=True, pad=True, fill_value=0)

    if len(stall) >= settings['nch_min']:
        std_trac = np.fromiter((np.nanstd(abs(tr.data)) for tr in stall), dtype=float)
        std_trac = std_trac / np.nanmean(std_trac)
        for std, tr in zip(std_trac, stall):
            if std <= settings['stddown'] or std >= settings['stdup']:
                stall.remove(tr)
                logging.debug(f"Removed Trace n Stream = {tr} {std}")
        header = {"network": "STACK",
                  "station": "BH",
                  "channel": "XX",
                  "starttime": min(tr.stats.starttime for tr in stall),
                  "sampling_rate": stall[0].stats.sampling_rate,
                  "npts": stall[0].stats.npts}
        ccmad = Trace(data=np.nanmean([tr.data for tr in stall], axis=0), header=header)
    else:
        ccmad = None

    return stall, ccmad


def csc(stall, tcft, trg, settings):
    """
    The function check single channel cft compute the maximum CFT's
    values at each trigger time and counts the number of channels
    having higher cross-correlation
    nch, cft_ave, crt are re-evaluated on the basis of
    +/- 2 sample approximation. Statistics are written in stat files
    """
    # important parameters: a sample_tolerance less than 2 results often in wrong magnitudes
    sample_tolerance = settings['sample_tol']
    single_channelcft = settings['cc_threshold']
    nch_min = settings['nch_min']

    trigger_time = trg["time"]
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
    nch03, nch05, nch07, nch09 = [(max_sct > threshold).sum() for threshold in (0.3, 0.5, 0.7, 0.9)]
    cft_ave = np.nanmean(max_sct)
    cft_ave_trg = np.nanmean(max_trg)
    max_sct = max_sct.T
    max_trg = max_trg.T
    chan_sct = chan_sct.T
    channels_list = []
    for idchan in range(len(max_sct)):
        record = (chan_sct[idchan].decode(), max_trg[idchan], max_sct[idchan], max_ind[idchan])
        channels_list.append(record)
    return nch, (cft_ave, cft_ave_trg, nch03, nch05, nch07, nch09), channels_list


def magnitude(continuous_stream, template_stream, trigger_time, mt, settings):
    reft = min(tr.stats.starttime for tr in template_stream)
    md = []
    for continuous_trace in continuous_stream:
        ss = continuous_trace.stats.station
        ich = continuous_trace.stats.channel
        if stream := template_stream.select(station=ss, channel=ich):
            template_trace, = stream
            timestart = trigger_time + (template_trace.stats.starttime - reft)
            timend = timestart + settings['temp_length']
            continuous_trace_view = continuous_trace.slice(starttime=timestart, endtime=timend)
            amaxd = max(abs(continuous_trace_view.data))
            amaxt = max(abs(template_trace.data))
            magd = mt - log10(amaxt / amaxd)
            md.append(magd)
    md = np.array(md)
    md_tail = np.abs(md - np.median(md))
    mdr = md[md_tail <= 2 * np.median(md_tail)]
    return mdr.mean()


def mad(dmad):
    """
    calculate daily median absolute deviation
    """
    return np.mean(abs(dmad - np.median(dmad)))
