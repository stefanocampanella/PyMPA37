import datetime
import logging
import re
from collections import OrderedDict
from itertools import repeat
from math import log10
from pathlib import Path
from typing import Tuple, Dict, List, Generator, Iterable, Callable

import bottleneck as bn
import numpy as np
import pandas as pd
from obspy import read, Stream, Trace
from scipy.signal import find_peaks

TemplateReadTuple = Tuple[int, Stream, Dict[str, int], float]
CorrelationFix = Tuple[str, float, int]
Event = Tuple[int, datetime.datetime, float, float, float, float, int]


def read_data(path: Path, freqmin: float = 3.0, freqmax: float = 8.0) -> Stream:
    logging.info(f"Reading continuous data from {path}")
    with path.open('rb') as file:
        data = read(file)
        data.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
        starttime = min(trace.stats.starttime for trace in data)
        endtime = max(trace.stats.endtime for trace in data)
        data.trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)
        return data


def read_templates(templates_directory: Path, travel_times_directory: Path,
                   catalog_filepath: Path) -> Generator[TemplateReadTuple, None, None]:
    logging.info(f"Reading catalog from {catalog_filepath}")
    template_magnitudes = pd.read_csv(catalog_filepath, sep=r'\s+', usecols=(5,), squeeze=True, dtype=float)
    logging.info(f"Reading travel times from {travel_times_directory}")
    logging.info(f"Reading templates from {templates_directory}")
    for template_number, travel_times_filepath in range_templates(travel_times_directory):
        try:
            travel_times = read_travel_times(travel_times_filepath)
            filepath = templates_directory / f"{template_number}.mseed"
            logging.debug(f"Reading {filepath}")
            with filepath.open('rb') as file:
                template_stream = read(file)
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


def read_travel_times(filepath: Path) -> Dict[str, float]:
    travel_times = OrderedDict()
    logging.debug(f"Reading {filepath}")
    with open(filepath, "r") as file:
        while line := file.readline():
            key, value_string = line.split(' ')
            network, station, channel = key.split('.')
            trace_id = f"{network}.{station}..{channel}"
            value = float(value_string)
            travel_times[trace_id] = value
    return travel_times


def correlation_detector(template: Stream, data: Stream, travel_times: Dict[str, float], template_magnitude: float,
                         max_channels: int = 16, threshold_factor: float = 8.0, distance_factor: float = 2.0,
                         tolerance: int = 6, cc_threshold: float = 0.35, min_channels: int = 6,
                         magnitude_threshold: float = 2.0, cc_min_std_factor: float = 0.25,
                         cc_max_std_factor: float = 1.5, mapf: Callable = map) -> Generator[Event, None, None]:
    continuous, template, travel_times = sieve_and_sort(data, template, travel_times, max_channels)
    correlations = Stream(traces=mapf(correlate_traces, continuous, template))
    quality_check(correlations, continuous, template, travel_times, cc_min_std_factor=cc_min_std_factor,
                  cc_max_std_factor=cc_max_std_factor, mapf=mapf)
    stacked_stream = Stream(traces=mapf(align, correlations, travel_times.values()))
    mean_correlation = bn.nanmean([trace.data for trace in stacked_stream], axis=0)
    correlation_dmad = bn.nanmean(np.abs(mean_correlation - bn.median(mean_correlation)))
    threshold = threshold_factor * correlation_dmad
    distance = distance_factor * (template[0].stats.endtime - template[0].stats.starttime) / template[0].stats.delta
    peaks, properties = find_peaks(mean_correlation, height=threshold, distance=distance)
    stack_zero = min(trace.stats.starttime for trace in stacked_stream)
    stack_delta = stacked_stream[0].stats.delta
    travel_zero = min(travel_times.values())
    template_zero = min(trace.stats.starttime for trace in template)
    for peak, peak_height in zip(peaks, properties['peak_heights']):
        trigger_time = stack_zero + peak * stack_delta
        event_date = trigger_time + travel_zero
        channels = list(mapf(fix_correlation, stacked_stream, repeat(peak), repeat(tolerance)))
        num_channels = sum(1 for _, corr, _ in channels if corr > cc_threshold)
        if num_channels >= min_channels:
            channel_magnitudes = np.fromiter(mapf(magnitude, template, repeat(template_magnitude), continuous,
                                                  repeat(trigger_time - template_zero)), dtype=float)
            event_magnitude = estimate_magnitude(channel_magnitudes, magnitude_threshold)
            event_correlation = sum(corr for _, corr, _ in channels) / len(channels)
            yield event_date.datetime, event_magnitude, event_correlation, peak_height, correlation_dmad, num_channels
        else:
            logging.debug(f"Skipping detection at {event_date} with peak {peak_height}: "
                          f"only {num_channels} channel(s) above threshold")


def sieve_and_sort(data: Stream, template: Stream, travel_times: Dict[str, float],
                   max_channels: int) -> Tuple[Stream, Stream, Dict[str, float]]:
    channels = sorted(set.intersection({trace.id for trace in data},
                                       {trace.id for trace in template},
                                       {trace_id for trace_id in travel_times}),
                      key=lambda trace_id: travel_times[trace_id])[:max_channels]
    logging.debug(f"Traces used: {', '.join(channels)}")
    sieved_data = select_traces(data, channels)
    sieved_template = select_traces(template, channels)
    sieved_ttimes = OrderedDict([(trace_id, travel_times[trace_id]) for trace_id in channels])
    return sieved_data, sieved_template, sieved_ttimes


def select_traces(stream: Stream, channels: Iterable[str]) -> Stream:
    new_stream = Stream()
    for trace_id in channels:
        trace, = stream.select(id=trace_id)
        new_stream.append(trace)
    return new_stream


def correlate_traces(continuous: Trace, template: Trace):
    header = {"network": continuous.stats.network,
              "station": continuous.stats.station,
              "channel": continuous.stats.channel,
              "starttime": continuous.stats.starttime,
              "sampling_rate": continuous.stats.sampling_rate}
    return Trace(data=correlate_data(continuous.data, template.data), header=header)


def correlate_data(data: np.ndarray, template: np.ndarray) -> np.ndarray:
    template = template - bn.nanmean(template)
    template_length = len(template)
    cross_correlation = np.correlate(data, template, mode='valid')
    pad = len(cross_correlation) - (len(data) - template_length)
    pad1, pad2 = (pad + 1) // 2, pad // 2
    data = np.hstack([np.zeros(pad1), data, np.zeros(pad2)])
    norm = np.sqrt(template_length * bn.move_var(data, template_length)[template_length:] * bn.ss(template))
    mask = norm > np.finfo(cross_correlation.dtype).eps
    np.divide(cross_correlation, norm, where=mask, out=cross_correlation)
    cross_correlation[~mask] = 0
    return cross_correlation


def quality_check(correlations: Stream, continuous: Stream, template: Stream, travel_times: Dict[str, float],
                  cc_min_std_factor: float = 0.25, cc_max_std_factor: float = 1.5, mapf: Callable = map):
    correlation_stds = np.fromiter(mapf(lambda trace: bn.nanstd(np.abs(trace.data)), correlations), dtype=float)
    mean_std = bn.nanmean(correlation_stds)
    traces = zip(correlations, continuous, template, list(travel_times))
    for std, (xcor_trace, cont_trace, temp_trace, trace_id) in zip(correlation_stds, traces):
        if not cc_min_std_factor < std / mean_std < cc_max_std_factor:
            logging.debug(f"Ignored trace {xcor_trace} with std {std} (mean: {mean_std})")
            correlations.remove(xcor_trace)
            continuous.remove(cont_trace)
            template.remove(temp_trace)
            del travel_times[trace_id]


def align(trace: Trace, delay: float):
    trace_copy = trace.copy()
    starttime = trace_copy.stats.starttime + delay
    endtime = starttime + datetime.timedelta(days=1)
    trace_copy.trim(starttime=starttime, endtime=endtime, nearest_sample=True, pad=True, fill_value=0)
    return trace_copy


def fix_correlation(trace: Trace, trigger_sample: int, tolerance: int) -> CorrelationFix:
    lower = max(trigger_sample - tolerance, 0)
    upper = min(trigger_sample + tolerance + 1, len(trace.data))
    sample_shift = bn.nanargmax(trace.data[lower:upper]) - tolerance
    correlation = trace.data[trigger_sample + sample_shift]
    return trace.id, correlation, sample_shift


def magnitude(template_trace: Trace, template_magnitude: float, continuous_trace: Trace,
              delta: datetime.timedelta) -> float:
    starttime = template_trace.stats.starttime + delta
    endtime = starttime + (template_trace.stats.endtime - template_trace.stats.starttime)
    continuous_trace_view = continuous_trace.slice(starttime=starttime, endtime=endtime)
    continuous_max_amp = bn.nanmax(np.abs(continuous_trace_view.data))
    template_max_amp = bn.nanmax(np.abs(template_trace.data))
    event_magnitude = template_magnitude - log10(template_max_amp / continuous_max_amp)
    return event_magnitude


def estimate_magnitude(channel_magnitudes: np.ndarray, threshold_factor: float) -> float:
    magnitude_deviations = np.abs(channel_magnitudes - bn.median(channel_magnitudes))
    magnitude_mad = bn.median(magnitude_deviations)
    threshold = threshold_factor * magnitude_mad + np.finfo(magnitude_mad).eps
    return bn.nanmean(channel_magnitudes[magnitude_deviations < threshold])


def save_records(events: List[Event], output: Path) -> None:
    events_dataframe = pd.DataFrame.from_records(events, columns=['template', 'date', 'magnitude', 'correlation',
                                                                  'stack_height', 'stack_dmad', 'num_channels'])
    events_dataframe.sort_values(by=['template', 'date'], inplace=True)
    events_dataframe['crt_pre'] = events_dataframe['stack_height'] / events_dataframe['stack_dmad']
    events_dataframe['crt_post'] = events_dataframe['correlation'] / events_dataframe['stack_dmad']
    logging.info(f"Writing outputs to {output}")
    events_dataframe.to_csv(output, index=False, header=False, na_rep='NA', sep=' ',
                            date_format='%Y-%m-%dT%H:%M:%S.%fZ',
                            float_format='%.3f', columns=['template', 'date', 'magnitude', 'correlation', 'crt_post',
                                                          'stack_height', 'crt_pre', 'num_channels'])
