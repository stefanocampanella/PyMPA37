import datetime
import logging
from functools import lru_cache
from math import log10

import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
from obspy.signal.cross_correlation import correlate_template
from scipy.signal import find_peaks


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
    stacked_stream = stack(correlation_stream, travel_times)
    mean_correlation = np.mean([trace.data for trace in stacked_stream], axis=0)

    delta = stacked_stream[0].stats.delta
    starttime = min(trace.stats.starttime for trace in stacked_stream)
    reference_time = min(travel_times[(trace.stats.network,
                                       trace.stats.station,
                                       trace.stats.channel)]
                         for trace in correlation_stream)
    peaks, _ = find_peaks(mean_correlation, distance=template_stream[0].stats.npts)
    for peak in peaks:
        channels = fix_correlations(stacked_stream,
                                    peak,
                                    sample_tolerance=settings['sample_tol'])
        num_channels = sum(1 for _, corr, _ in channels if corr > settings['cc_threshold'])
        if num_channels >= settings['nch_min']:
            trigger_time = starttime + peak * delta
            event_magnitude = magnitude(continuous_stream, template_stream, trigger_time, template_magnitude)
            event_time = trigger_time + reference_time
            event_correlation = sum(corr for _, corr, _ in channels) / len(channels)
            record = (event_time, event_magnitude, event_correlation, channels)
            events_list.append(record)
    return events_list


def correlate_streams(template_stream, continuous_stream, std_range=(0.25, 1.5)):
    correlation_stream = Stream()
    for template_trace in template_stream:
        network = template_trace.stats.network
        station = template_trace.stats.station
        channel = template_trace.stats.channel
        continuous_trace, = continuous_stream.select(network=network, station=station, channel=channel)
        correlation = correlate_template(continuous_trace.data, template_trace.data)
        correlation = np.nan_to_num(correlation)
        header = {"network": continuous_trace.stats.network,
                  "station": continuous_trace.stats.station,
                  "channel": continuous_trace.stats.channel,
                  "starttime": continuous_trace.stats.starttime,
                  "sampling_rate": continuous_trace.stats.sampling_rate}
        correlation_stream += Trace(data=correlation, header=header)

    trace_std = np.fromiter((np.std(abs(trace.data)) for trace in correlation_stream), dtype=float)
    trace_std = trace_std / trace_std.mean()
    for std, tr in zip(trace_std, correlation_stream):
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

    return stacked_stream


def fix_correlations(stacked_stream, trigger_sample, sample_tolerance=6):
    channels = []
    for trace in stacked_stream:
        lower = max(trigger_sample - sample_tolerance, 0)
        upper = min(trigger_sample + sample_tolerance + 1, len(trace.data))
        stats = trace.stats
        id = stats.network + "." + stats.station + ".." + stats.channel
        sample_shift = np.argmax(trace.data[lower:upper]) - sample_tolerance
        correlation = trace.data[trigger_sample + sample_shift]
        channels.append((id, correlation, sample_shift))

    return channels


def magnitude(continuous_stream, template_stream, trigger_time, template_magnitude):
    reft = min(tr.stats.starttime for tr in template_stream)
    channel_magnitudes = []
    for continuous_trace in continuous_stream:
        if stream := template_stream.select(station=continuous_trace.stats.station,
                                            channel=continuous_trace.stats.channel):
            template_trace, = stream
            timestart = trigger_time + (template_trace.stats.starttime - reft)
            timend = timestart + (template_trace.stats.endtime - template_trace.stats.starttime)
            continuous_trace_view = continuous_trace.slice(starttime=timestart, endtime=timend)
            continuous_absolute_max = max(abs(continuous_trace_view.data))
            template_absolute_max = max(abs(template_trace.data))
            event_magnitude = template_magnitude - log10(template_absolute_max / continuous_absolute_max)
            channel_magnitudes.append(event_magnitude)
    channel_magnitudes = np.array(channel_magnitudes)
    absolute_deviations = np.abs(channel_magnitudes - np.median(channel_magnitudes))
    valid_channel_magnitudes = channel_magnitudes[absolute_deviations < 2 * np.median(absolute_deviations)]
    return valid_channel_magnitudes.mean()
