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


@lru_cache
def read_travel_times(travel_times_path, chan_max=12):
    travel_times = {}
    with open(travel_times_path, "r") as file:
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


@lru_cache
def read_template_stream(templates_dir_path, template_number, channel_list):
    template_stream = Stream()
    for net, sta, chn in channel_list:
        try:
            filepath = templates_dir_path / f"{template_number}.{net}.{sta}..{chn}.mseed"
            with filepath.open('rb') as file:
                logging.debug(f"Reading {filepath}")
                template_stream += read(file, dtype="float32")
        except Exception as err:
            logging.warning(f"{err}")
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


def correlation_detector(template_stream, continuous_stream, travel_times, template_magnitude, settings):
    events_list = []
    correlation_stream = correlate_streams(template_stream, continuous_stream, std_range=settings['std_range'])
    stacked_stream, mean_correlation_trace = stack(correlation_stream, travel_times)
    mean_absolute_deviation = abs(mean_correlation_trace.data - np.median(mean_correlation_trace.data)).mean()
    threshold = settings['factor_thre'] * mean_absolute_deviation
    # Run coincidence trigger on a single CC trace resulting from the CFTs to_wavefront_time
    # essential threshold parameters Cross correlation thresholds
    triggers = coincidence_trigger(None,
                                   threshold,
                                   0.85 * threshold,
                                   Stream(mean_correlation_trace),
                                   0.0)
    reference_time = list(travel_times.values())[0]
    for trigger in triggers:
        nch, stats, channel_list = csc(stacked_stream, mean_correlation_trace, trigger['time'], settings)
        if nch >= settings['nch_min']:
            event_magnitude = magnitude(continuous_stream, template_stream, trigger['time'], template_magnitude)
            event_time = trigger['time'] + reference_time
            record = (event_time, event_magnitude, mean_absolute_deviation, stats, channel_list)
            events_list.append(record)
    return events_list


def correlate_streams(template_stream, continuous_stream, std_range=(0.25, 1.5)):
    correlation_stream = Stream()
    for tt in template_stream:
        network, station, channel = tt.stats.network, tt.stats.station, tt.stats.channel
        tc, = continuous_stream.select(network=network, station=station, channel=channel)
        fct = correlate_template(tc.data, tt.data)
        fct = np.nan_to_num(fct)
        header = {"network": tc.stats.network,
                  "station": tc.stats.station,
                  "channel": tc.stats.channel,
                  "starttime": tc.stats.starttime,
                  "sampling_rate": tc.stats.sampling_rate}
        correlation_stream += Trace(data=fct, header=header)

    std_trac = np.fromiter((np.std(abs(tr.data)) for tr in correlation_stream), dtype=float)
    std_trac = std_trac / std_trac.mean()
    for std, tr in zip(std_trac, correlation_stream):
        if std < std_range[0] or std > std_range[1]:
            correlation_stream.remove(tr)
            logging.debug(f"Removed trace {tr} with std {std} from correlation stream")

    return correlation_stream


def stack(stream, travel_times):

    stacked_stream = stream.copy()
    for trace in stacked_stream:
        key = trace.stats.network, trace.stats.station, trace.stats.channel
        starttime = trace.stats.starttime + travel_times[key]
        endtime = starttime + datetime.timedelta(days=1)
        trace.trim(starttime=starttime, endtime=endtime, nearest_sample=True, pad=True, fill_value=0)

    header = {"starttime": min(trace.stats.starttime for trace in stacked_stream),
              "sampling_rate": stacked_stream[0].stats.sampling_rate}
    mean_correlation = Trace(data=np.mean([trace.data for trace in stacked_stream], axis=0), header=header)

    return stacked_stream, mean_correlation


def csc(correlation_stream, mean_correlation_trace, trigger_time, settings):
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

    t0_tcft = mean_correlation_trace.stats.starttime
    trigger_shift = trigger_time.timestamp - t0_tcft.timestamp
    trigger_sample = round(trigger_shift / mean_correlation_trace.stats.delta)
    max_sct = np.empty(len(correlation_stream))
    max_trg = np.empty(len(correlation_stream))
    max_ind = np.empty(len(correlation_stream))
    chan_sct = {}

    for icft, tsc in enumerate(correlation_stream):
        # get cft amplitude value at corresponding trigger and store it in
        # check for possible 2 sample shift and eventually change trg['cft_peaks']
        chan_sct[icft] = tsc.stats.network + "." + tsc.stats.station + " " + tsc.stats.channel
        tmp0 = max(trigger_sample - sample_tolerance, 0)
        tmp1 = trigger_sample + sample_tolerance + 1
        max_sct[icft] = max(tsc.data[tmp0:tmp1])
        max_ind[icft] = np.argmax(tsc.data[tmp0:tmp1])
        max_ind[icft] = sample_tolerance - max_ind[icft]
        max_trg[icft] = tsc.data[trigger_sample: trigger_sample + 1]

    nch = (max_sct > single_channelcft).sum()
    nch03, nch05, nch07, nch09 = [(max_sct > threshold).sum() for threshold in (0.3, 0.5, 0.7, 0.9)]
    cft_ave = np.mean(max_sct)
    cft_ave_trg = np.mean(max_trg)
    channels_list = []
    for idchan in range(len(max_sct)):
        record = (chan_sct[idchan], max_trg[idchan], max_sct[idchan], max_ind[idchan])
        channels_list.append(record)
    return nch, (cft_ave, cft_ave_trg, nch03, nch05, nch07, nch09), channels_list


def magnitude(continuous_stream, template_stream, trigger_time, mt):
    reft = min(tr.stats.starttime for tr in template_stream)
    md = []
    for continuous_trace in continuous_stream:
        ss = continuous_trace.stats.station
        ich = continuous_trace.stats.channel
        if stream := template_stream.select(station=ss, channel=ich):
            template_trace, = stream
            timestart = trigger_time + (template_trace.stats.starttime - reft)
            timend = timestart + (template_trace.stats.endtime - template_trace.stats.starttime)
            continuous_trace_view = continuous_trace.slice(starttime=timestart, endtime=timend)
            amaxd = max(abs(continuous_trace_view.data))
            amaxt = max(abs(template_trace.data))
            magd = mt - log10(amaxt / amaxd)
            md.append(magd)
    md = np.array(md)
    md_tail = np.abs(md - np.median(md))
    mdr = md[md_tail <= 2 * np.median(md_tail)]
    return mdr.mean()
