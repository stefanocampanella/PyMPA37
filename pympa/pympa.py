import datetime
import logging
import re
from collections import namedtuple
from concurrent.futures import Executor
from functools import reduce
from math import log10
from pathlib import Path
from typing import Tuple, Dict, List, Generator

import bottleneck as bn
import numpy as np
import pandas as pd
from obspy import read, Stream, Trace, UTCDateTime
from obspy.core import Stats
from scipy.signal import find_peaks

TemplateReadTuple = Tuple[int, Stream, Dict[str, int], float]
CorrelationFix = Tuple[str, float, int]
Event = Tuple[UTCDateTime, float, float, float, float, List[CorrelationFix]]

TracePair = namedtuple('TracePair', 'template continuous')


def read_continuous_stream(directory: Path, day: datetime.datetime, executor: Executor, freqmin: float = 3.0,
                           freqmax: float = 8.0) -> Stream:
    logging.info(f"Reading continuous data from {directory}")

    def reader(filepath: Path):
        begin = UTCDateTime(day)
        try:
            logging.debug(f"Reading {filepath}")
            with filepath.open('rb') as file:
                if stream := read(file, dtype=np.float32):
                    trace, = stream
                    trace.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
                    trace.trim(starttime=begin, endtime=begin + datetime.timedelta(days=1), pad=True, fill_value=0)
                    return trace
                else:
                    logging.warning(f"Empty stream found while reading {filepath}")
                    return None
        except OSError as err:
            logging.warning(f"{err} occurred while reading {filepath}")
            return None

    return reduce(lambda a, b: a + b if b else a,
                  executor.map(reader, directory.glob(f"{day.strftime('%y%m%d')}*")), Stream())


def read_templates(templates_directory: Path, travel_times_directory: Path, catalog_filepath: Path,
                   num_channels_max: int, executor: Executor) -> Generator[TemplateReadTuple, None, None]:
    logging.info(f"Reading catalog from {catalog_filepath}")
    template_magnitudes = pd.read_csv(catalog_filepath, sep=r'\s+', usecols=(5,), squeeze=True, dtype=float)
    logging.info(f"Reading travel times from {travel_times_directory}")
    logging.info(f"Reading templates from {templates_directory}")
    for template_number, travel_times_filepath in range_templates(travel_times_directory):
        try:
            travel_times = read_travel_times(travel_times_filepath, num_channels_max)
            template_stream = read_template_stream(templates_directory,
                                                   template_number,
                                                   travel_times.keys(),
                                                   executor)
            yield template_number, template_stream, travel_times, template_magnitudes.iloc[template_number - 1]
        except OSError as err:
            logging.warning(f"{err} occurred while reading template {template_number}")
            continue


def range_templates(travel_times_directory: Path) -> Generator[Tuple[int, Path], None, None]:
    file_regex = re.compile(r'(?P<template_number>\d+).ttimes')
    for filepath in travel_times_directory.glob('*.ttimes'):
        match = file_regex.match(filepath.name)
        if match:
            template_number = int(match.group('template_number'))
            yield template_number, filepath


def read_travel_times(filepath: Path, max_channels: int) -> Dict[str, float]:
    travel_times = {}
    logging.debug(f"Reading {filepath}")
    with open(filepath, "r") as file:
        while line := file.readline():
            key, value_string = line.split(' ')
            network, station, channel = key.split('.')
            trace_id = f"{network}.{station}..{channel}"
            value = float(value_string)
            travel_times[trace_id] = value
        if len(travel_times) > max_channels:
            sorted_ids = list(sorted(travel_times, key=lambda trace: travel_times[trace]))
            for n, trace_id in enumerate(sorted_ids):
                if n >= max_channels:
                    del travel_times[trace_id]
    return travel_times


def read_template_stream(directory: Path, template_number: int, channel_list, executor: Executor) -> Stream:
    def reader(trace_id):
        try:
            filepath = directory / f"{template_number}.{trace_id}.mseed"
            logging.debug(f"Reading {filepath}")
            with filepath.open('rb') as file:
                return read(file, dtype=np.float32)
        except OSError as err:
            logging.warning(f"{err} occurred while reading template {trace_id}")
            return None

    return reduce(lambda a, b: a + b if b else a, executor.map(reader, channel_list), Stream())


def correlation_detector(template_stream: Stream, continuous_stream: Stream, travel_times: Dict[str, float],
                         template_magnitude: float, threshold_factor: float, tolerance: int, executor: Executor,
                         correlations_std_bounds: Tuple[float, float] = (0.25, 1.5)) -> List[Event]:
    events_list = []
    correlation_stream = correlate_streams(template_stream, continuous_stream, executor,
                                           std_bounds=correlations_std_bounds)
    stacked_stream = stack(correlation_stream, travel_times, executor)
    mean_correlation = bn.nanmean([trace.data for trace in stacked_stream], axis=0)
    correlation_dmad = bn.nanmean(np.abs(mean_correlation - bn.median(mean_correlation)))
    threshold = threshold_factor * correlation_dmad
    peaks, properties = find_peaks(mean_correlation, height=threshold)
    starttime = min(trace.stats.starttime for trace in stacked_stream)
    delta = stacked_stream[0].stats.delta
    reference_time = min(travel_times[trace.id] for trace in correlation_stream)
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


def correlate_streams(template_stream: Stream, continuous_stream: Stream, executor: Executor,
                      std_bounds: Tuple[float, float] = (0.25, 1.5)) -> Stream:
    correlation_stream = Stream(traces=executor.map(lambda pair: correlate_template(pair.continuous.data,
                                                                                    pair.template.data,
                                                                                    pair.continuous.stats),
                                                    zip_streams(template_stream, continuous_stream)))
    stds = np.fromiter(executor.map(lambda trace: bn.nanstd(np.abs(trace.data)), correlation_stream), dtype=float)
    relative_stds = stds / bn.nanmean(stds)
    for std, tr in zip(relative_stds, correlation_stream):
        if std < std_bounds[0] or std > std_bounds[1]:
            correlation_stream.remove(tr)
            logging.debug(f"Removed trace {tr} with std {std} from correlation stream")
    return correlation_stream


def zip_streams(template: Stream, continuous: Stream) -> Generator[Tuple[Trace, Trace], None, None]:
    for master_trace in template:
        if selection := continuous.select(id=master_trace.id):
            slave_trace, = selection
            yield TracePair(master_trace, slave_trace)
        else:
            logging.debug(f"Trace {master_trace.id} not found in continuous data")


def correlate_template(data: np.ndarray, template: np.ndarray, stats: Stats) -> Trace:
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
    header = {"network": stats.network, "station": stats.station, "channel": stats.channel,
              "starttime": stats.starttime, "sampling_rate": stats.sampling_rate}
    return Trace(data=cross_correlation, header=header)


def stack(stream: Stream, travel_times: Dict[str, float], executor: Executor) -> Stream:
    def align(trace: Trace):
        trace_copy = trace.copy()
        starttime = trace_copy.stats.starttime + travel_times[trace_copy.id]
        endtime = starttime + datetime.timedelta(days=1)
        trace_copy.trim(starttime=starttime, endtime=endtime, nearest_sample=True, pad=True, fill_value=0)
        return trace_copy

    return Stream(traces=executor.map(align, stream))


def fix_correlation(stacked_stream: Stream, trigger_sample: int, tolerance: int,
                    executor: Executor) -> List[CorrelationFix]:
    def fixer(trace: Trace):
        lower = max(trigger_sample - tolerance, 0)
        upper = min(trigger_sample + tolerance + 1, len(trace.data))
        sample_shift = bn.nanargmax(trace.data[lower:upper]) - tolerance
        correlation = trace.data[trigger_sample + sample_shift]
        return trace.id, correlation, sample_shift

    return list(executor.map(fixer, stacked_stream))


def magnitude(continuous_stream: Stream, template_stream: Stream, trigger_time: UTCDateTime, template_magnitude: float,
              executor: Executor, mad_threshold: float = 2.0) -> float:
    reference_time = min(tr.stats.starttime for tr in template_stream)

    def channel_magnitude(continuous_trace: Trace, template_trace: Trace) -> float:
        starttime = trigger_time + (template_trace.stats.starttime - reference_time)
        endtime = starttime + (template_trace.stats.endtime - template_trace.stats.starttime)
        continuous_trace_view = continuous_trace.slice(starttime=starttime, endtime=endtime)
        continuous_absolute_max = bn.nanmax(np.abs(continuous_trace_view.data))
        template_absolute_max = bn.nanmax(np.abs(template_trace.data))
        event_magnitude = template_magnitude - log10(template_absolute_max / continuous_absolute_max)
        return event_magnitude

    channel_magnitudes = np.fromiter(executor.map(lambda pair: channel_magnitude(pair.continuous, pair.template),
                                                  zip_streams(template_stream, continuous_stream)), dtype=float)
    absolute_deviations = np.abs(channel_magnitudes - bn.median(channel_magnitudes))
    valid_channel_magnitudes = channel_magnitudes[absolute_deviations < mad_threshold * bn.median(absolute_deviations)]
    return bn.nanmean(valid_channel_magnitudes)
