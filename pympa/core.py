import datetime
import logging
from functools import lru_cache
from math import log10

import numpy as np
from obspy import read, Stream, Trace, UTCDateTime, read_events
from obspy.signal.cross_correlation import correlate_template
from scipy.signal import find_peaks


def range_days(start, stop):
    date = start
    while date < stop:
        yield date
        date += datetime.timedelta(days=1)


def read_templates(templates_dir_path, travel_times_dir_path, catalog_path, settings):
    catalog = read_events(catalog_path)
    start_template, end_template = settings['template_range']
    templates = []
    for template_number in range(start_template, end_template):
        try:
            travel_times = read_travel_times(travel_times_dir_path / f"{template_number}.ttimes")
            if len(travel_times) < settings['nch_min']:
                logging.info(f"Not enough travel times for template {template_number}")
                continue
            elif len(travel_times) > settings['chan_max']:
                channels_to_remove = [name
                                      for n, name in enumerate(sorted(travel_times, key=lambda x: travel_times[x]))
                                      if n >= settings['chan_max']]
                for channel in channels_to_remove:
                    del travel_times[channel]
            template_stream = read_template_stream(templates_dir_path,
                                                   template_number,
                                                   travel_times.keys())
            if len(template_stream) < settings['nch_min']:
                logging.info(f"Not enough channels for template {template_number}")
                continue
            template_magnitude = catalog[template_number].magnitudes[0].mag
            templates.append((template_number, template_stream, travel_times, template_magnitude))
        except Exception as err:
            logging.warning(f"{err} occurred while processing template {template_number}")
            continue
    return templates


def read_travel_times(travel_times_path):
    travel_times = {}
    with open(travel_times_path, "r") as file:
        while line := file.readline():
            key, value = line.split(' ')
            key = tuple(key.split('.'))
            value = float(value)
            travel_times[key] = value
    return travel_times


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
    correlation_dmad = np.mean(np.abs(mean_correlation - np.median(mean_correlation)))
    threshold = settings['factor_thre'] * correlation_dmad
    peaks, properties = find_peaks(mean_correlation, height=threshold)
    starttime = min(trace.stats.starttime for trace in stacked_stream)
    delta = stacked_stream[0].stats.delta
    reference_time = min(travel_times[(trace.stats.network,
                                       trace.stats.station,
                                       trace.stats.channel)]
                         for trace in correlation_stream)
    for peak, peak_height in zip(peaks, properties['peak_heights']):
        channels = fix_correlations(stacked_stream,
                                    peak,
                                    sample_tolerance=settings['sample_tol'])
        trigger_time = starttime + peak * delta
        event_magnitude = magnitude(continuous_stream, template_stream, trigger_time, template_magnitude)
        event_date = trigger_time + reference_time
        event_correlation = sum(corr for _, corr, _ in channels) / len(channels)
        record = (event_date, event_magnitude, event_correlation,
                  peak_height, correlation_dmad, channels)
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

    stds = np.fromiter((np.std(np.abs(trace.data)) for trace in correlation_stream), dtype=float)
    relative_stds = stds / stds.mean()
    for std, tr in zip(relative_stds, correlation_stream):
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
        name = stats.network + "." + stats.station + ".." + stats.channel
        sample_shift = np.argmax(trace.data[lower:upper]) - sample_tolerance
        correlation = trace.data[trigger_sample + sample_shift]
        channels.append((name, correlation, sample_shift))
    return channels


def magnitude(continuous_stream, template_stream, trigger_time, template_magnitude, mad_threshold=2):
    reference_time = min(tr.stats.starttime for tr in template_stream)
    channel_magnitudes = []
    for continuous_trace in continuous_stream:
        if stream := template_stream.select(station=continuous_trace.stats.station,
                                            channel=continuous_trace.stats.channel):
            template_trace, = stream
            timestart = trigger_time + (template_trace.stats.starttime - reference_time)
            timend = timestart + (template_trace.stats.endtime - template_trace.stats.starttime)
            continuous_trace_view = continuous_trace.slice(starttime=timestart, endtime=timend)
            continuous_absolute_max = np.max(np.abs(continuous_trace_view.data))
            template_absolute_max = np.max(np.abs(template_trace.data))
            event_magnitude = template_magnitude - log10(template_absolute_max / continuous_absolute_max)
            channel_magnitudes.append(event_magnitude)
    channel_magnitudes = np.array(channel_magnitudes)
    absolute_deviations = np.abs(channel_magnitudes - np.median(channel_magnitudes))
    valid_channel_magnitudes = channel_magnitudes[absolute_deviations < mad_threshold * np.median(absolute_deviations)]
    return valid_channel_magnitudes.mean()
