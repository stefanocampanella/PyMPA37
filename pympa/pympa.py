import datetime
import logging
import re
from concurrent.futures import as_completed
from math import log10

import bottleneck as bn
import numpy as np
from obspy import read, Stream, Trace, UTCDateTime, read_events
from scipy.signal import find_peaks


def read_continuous_stream(dir_path, day, freqmin=3.0, freqmax=8.0):
    continuous_stream = Stream()
    logging.info(f"Reading continuous data from {dir_path}")
    for filepath in dir_path.glob(f"{day.strftime('%y%m%d')}*"):
        try:
            logging.debug(f"Reading {filepath}")
            with filepath.open('rb') as file:
                if stream := read(file, dtype="float32"):
                    trace, = stream
                    trace.filter("bandpass",
                                 freqmin=freqmin,
                                 freqmax=freqmax,
                                 zerophase=True)
                    begin = UTCDateTime(day)
                    trace.trim(starttime=begin,
                               endtime=begin + datetime.timedelta(days=1),
                               pad=True,
                               fill_value=0)
                    continuous_stream += trace
                else:
                    logging.warning(f"Empty stream found while reading {filepath}")
        except OSError as err:
            logging.warning(f"{err} occurred while reading {filepath}")
    return continuous_stream


def read_templates(templates_dirpath, travel_times_dirpath, catalog_filepath, num_channels_max):
    logging.info(f"Reading catalog from {catalog_filepath}")
    catalog = read_events(catalog_filepath)
    logging.info(f"Reading travel times from {travel_times_dirpath}")
    logging.info(f"Reading templates from {templates_dirpath}")
    for template_number, travel_times_filepath in range_templates(travel_times_dirpath):
        try:
            travel_times = read_travel_times(travel_times_filepath, num_channels_max)
            template_stream = read_template_stream(templates_dirpath,
                                                   template_number,
                                                   travel_times.keys())
            template_magnitude = catalog[template_number].magnitudes[0].mag
            yield template_number, template_stream, travel_times, template_magnitude
        except OSError as err:
            logging.warning(f"{err} occurred while reading template {template_number}")
            continue


def range_templates(travel_times_dirpath):
    file_regex = re.compile(r'(?P<template_number>\d+).ttimes')
    for travel_times_filepath in travel_times_dirpath.glob('*.ttimes'):
        match = re.match(file_regex, travel_times_filepath.name)
        template_number = int(match['template_number'])
        yield template_number, travel_times_filepath


def read_travel_times(filepath, num_channels_max):
    travel_times = {}
    logging.debug(f"Reading {filepath}")
    with open(filepath, "r") as file:
        while line := file.readline():
            key, value = line.split(' ')
            key = tuple(key.split('.'))
            value = float(value)
            travel_times[key] = value
        if len(travel_times) > num_channels_max:
            channels_to_remove = [name
                                  for n, name in enumerate(sorted(travel_times, key=lambda x: travel_times[x]))
                                  if n >= num_channels_max]
            for channel in channels_to_remove:
                del travel_times[channel]
    return travel_times


def read_template_stream(dir_path, template_number, channel_list):
    template_stream = Stream()
    for network, station, channel in channel_list:
        try:
            filepath = dir_path / f"{template_number}.{network}.{station}..{channel}.mseed"
            logging.debug(f"Reading {filepath}")
            with filepath.open('rb') as file:
                template_stream += read(file, dtype="float32")
        except OSError as err:
            logging.warning(f"{err} occurred while reading template {network}.{station}..{channel}")
    return template_stream


def correlation_detector(template_stream, continuous_stream, travel_times, template_magnitude,
                         threshold_factor, tolerance, executor, correlations_std_bounds=(0.25, 1.5)):
    events_list = []
    correlation_stream = correlate_streams(template_stream, continuous_stream, executor, std_bounds=correlations_std_bounds)
    stacked_stream = stack(correlation_stream, travel_times)
    mean_correlation = bn.nanmean([trace.data for trace in stacked_stream], axis=0)
    correlation_dmad = bn.nanmean(np.abs(mean_correlation - bn.median(mean_correlation)))
    threshold = threshold_factor * correlation_dmad
    peaks, properties = find_peaks(mean_correlation, height=threshold)
    starttime = min(trace.stats.starttime for trace in stacked_stream)
    delta = stacked_stream[0].stats.delta
    reference_time = min(travel_times[(trace.stats.network,
                                       trace.stats.station,
                                       trace.stats.channel)]
                         for trace in correlation_stream)
    for peak, peak_height in zip(peaks, properties['peak_heights']):
        channels = fix_correlation(stacked_stream,
                                   peak,
                                   tolerance,
                                   executor)
        trigger_time = starttime + peak * delta
        event_magnitude = magnitude(continuous_stream, template_stream, trigger_time, template_magnitude, executor)
        event_date = trigger_time + reference_time
        event_correlation = sum(corr for _, corr, _ in channels) / len(channels)
        record = (event_date, event_magnitude, event_correlation,
                  peak_height, correlation_dmad, channels)
        events_list.append(record)
    return events_list


def correlate_streams(template_stream, continuous_stream, executor, std_bounds=(0.25, 1.5)):
    correlation_stream = Stream()
    futures = []
    for template_trace in template_stream:
        network = template_trace.stats.network
        station = template_trace.stats.station
        channel = template_trace.stats.channel
        if selection_stream := continuous_stream.select(network=network, station=station, channel=channel):
            continuous_trace, = selection_stream
            header = {"network": network, "station": station, "channel": channel,
                      "starttime": continuous_trace.stats.starttime,
                      "sampling_rate": continuous_trace.stats.sampling_rate}
            correlation_future = executor.submit(lambda x, y, h: Trace(data=correlate_template(x, y), header=h),
                                                 continuous_trace.data, template_trace.data, header)
            futures.append(correlation_future)
        else:
            logging.debug(f"Trace {network}.{station}..{channel} not found in continuous data")
    for future in as_completed(futures):
        correlation_stream += future.result()

    stds = np.fromiter(executor.map(lambda trace: bn.nanstd(np.abs(trace.data)), correlation_stream), dtype=float)
    relative_stds = stds / bn.nanmean(stds)
    for std, tr in zip(relative_stds, correlation_stream):
        if std < std_bounds[0] or std > std_bounds[1]:
            correlation_stream.remove(tr)
            logging.debug(f"Removed trace {tr} with std {std} from correlation stream")
    return correlation_stream


def correlate_template(data, template):
    template = template - bn.nanmean(template)
    template_length = len(template)
    cross_correlation = np.correlate(data, template, mode='valid')
    pad = len(cross_correlation) - (len(data) - template_length)
    pad1, pad2 = (pad + 1) // 2, pad // 2
    data = np.hstack([np.zeros(pad1), data, np.zeros(pad2)])
    norm = np.sqrt(template_length * bn.move_var(data, template_length)[template_length:] * bn.ss(template))
    mask = norm > np.finfo(np.float32).eps
    np.divide(cross_correlation, norm, where=mask, out=cross_correlation)
    cross_correlation[~mask] = 0
    return cross_correlation


def stack(stream, travel_times):
    stacked_stream = stream.copy()
    for trace in stacked_stream:
        key = trace.stats.network, trace.stats.station, trace.stats.channel
        starttime = trace.stats.starttime + travel_times[key]
        endtime = starttime + datetime.timedelta(days=1)
        trace.trim(starttime=starttime, endtime=endtime, nearest_sample=True, pad=True, fill_value=0)
    return stacked_stream


def fix_correlation(stacked_stream, trigger_sample, tolerance, executor):
    futures = []
    for trace in stacked_stream:
        future = executor.submit(fix_channel, trace, trigger_sample, tolerance)
        futures.append(future)
    return [future.result() for future in as_completed(futures)]

def fix_channel(correlation_trace, trigger_sample, tolerance):
    lower = max(trigger_sample - tolerance, 0)
    upper = min(trigger_sample + tolerance + 1, len(correlation_trace.data))
    stats = correlation_trace.stats
    name = stats.network + "." + stats.station + ".." + stats.channel
    sample_shift = bn.nanargmax(correlation_trace.data[lower:upper]) - tolerance
    correlation_trace = correlation_trace.data[trigger_sample + sample_shift]
    return name, correlation_trace, sample_shift


def magnitude(continuous_stream, template_stream, trigger_time, template_magnitude, executor, mad_threshold=2):
    reference_time = min(tr.stats.starttime for tr in template_stream)
    futures = []
    for continuous_trace in continuous_stream:
        if stream := template_stream.select(station=continuous_trace.stats.station,
                                            channel=continuous_trace.stats.channel):
            template_trace, = stream
            future = executor.submit(channel_magnitude, template_trace, continuous_trace, template_magnitude,
                                     trigger_time, reference_time)
            futures.append(future)
    channel_magnitudes = np.fromiter((future.result() for future in as_completed(futures)), dtype=float)
    absolute_deviations = np.abs(channel_magnitudes - bn.median(channel_magnitudes))
    valid_channel_magnitudes = channel_magnitudes[absolute_deviations < mad_threshold * bn.median(absolute_deviations)]
    return bn.nanmean(valid_channel_magnitudes)


def channel_magnitude(template, data, template_magnitude, trigger_time, reference_time):
    timestart = trigger_time + (template.stats.starttime - reference_time)
    timend = timestart + (template.stats.endtime - template.stats.starttime)
    continuous_trace_view = data.slice(starttime=timestart, endtime=timend)
    continuous_absolute_max = bn.nanmax(np.abs(continuous_trace_view.data))
    template_absolute_max = bn.nanmax(np.abs(template.data))
    event_magnitude = template_magnitude - log10(template_absolute_max / continuous_absolute_max)
    return event_magnitude
