import argparse
import logging
import sys
from pathlib import Path
from time import perf_counter as timer
import importlib.resources as resources

from obspy import UTCDateTime
from obspy.core.event import read_events
from yaml import load, FullLoader
from tqdm.auto import trange

from pympa.core import read_travel_times, read_template_stream, range_days, read_continuous_stream, find_events

parser = argparse.ArgumentParser(prog="PyMPA37",
                                 description="A software package for phase match filtering")
parser.add_argument("templates_dir_path", help="Path to templates directory")
parser.add_argument("travel_times_dir_path", help="Path to travel times directory")
parser.add_argument("continuous_dir_path", help="Path to continuous traces directory")
parser.add_argument("catalog_path", help="Path to event catalog")
parser.add_argument("settings_path", help="Path to settings file")
parser.add_argument("output_path", help="Output file path")
cli_args = parser.parse_args()

with resources.open_text('pympa', 'defaults.yaml') as default_settings_file:
    default_settings = load(default_settings_file, FullLoader)

logging.basicConfig(stream=sys.stderr,
                    format='%(levelname)s: %(message)s',
                    level=logging.WARNING)

if __name__ == '__main__':
    tic = timer()

    templates_dir_path = Path(cli_args.templates_dir_path).resolve()
    if not templates_dir_path.is_dir():
        raise FileNotFoundError(f"{templates_dir_path} is not a directory")
    logging.info(f"Reading templates from {templates_dir_path}")

    travel_times_dir_path = Path(cli_args.travel_times_dir_path).resolve()
    if not travel_times_dir_path.is_dir():
        raise FileNotFoundError(f"{travel_times_dir_path} is not a directory")
    logging.info(f"Reading travel times from {travel_times_dir_path}")

    continuous_dir_path = Path(cli_args.continuous_dir_path).resolve()
    if not continuous_dir_path.is_dir():
        raise FileNotFoundError(f"{continuous_dir_path} is not a directory")
    logging.info(f"Reading continuous traces from {continuous_dir_path}")

    catalog_path = Path(cli_args.catalog_path).resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"{catalog_path} is not a file")
    logging.info(f"Catalog file located at {catalog_path}")

    settings_path = Path(cli_args.settings_path).resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"{settings_path} is not a file")
    logging.info(f"Settings file located at {settings_path}")

    with open(settings_path) as settings_file:
        runtime_settings = load(settings_file, FullLoader)
    settings = default_settings.copy()
    settings.update(runtime_settings)

    catalog = read_events(catalog_path)

    UTCDateTime.DEFAULT_PRECISION = settings['utc_prec']

    events_found = {}
    for template_number in trange(*settings['template_range']):
        travel_times = read_travel_times(travel_times_dir_path,
                                         template_number,
                                         chan_max=settings['chan_max'])
        template_stream = read_template_stream(templates_dir_path,
                                               template_number,
                                               travel_times.keys())
        if len(template_stream) >= settings['nch_min']:
            mt = catalog[template_number].magnitudes[0].mag
            for day in range_days(*settings['date_range']):
                day_stream = read_continuous_stream(continuous_dir_path,
                                                    day,
                                                    travel_times.keys(),
                                                    freqmin=settings['lowpassf'],
                                                    freqmax=settings['highpassf'])
                if len(day_stream) >= settings['nch_min']:
                    new_events = find_events(template_stream,
                                             day_stream,
                                             travel_times,
                                             mt,
                                             settings)
                    events_found[template_number] = new_events
                else:
                    logging.info(f"{day}, not enough channels for "
                                 f"template {template_number} (nch_min={settings['nch_min']})")

    output_path = Path(cli_args.output_path)
    logging.info(f"Writing outputs at {output_path}")
    with open(output_path, "w") as output_file:
        for template_number, events_list in events_found.items():
            output_file.write(f"==== event list begin for template {template_number} ====\n")
            for event in events_list:
                output_file.write(" ".join(f"{x}" for x in event[:-1]) + "\n")
                for channel in event[-1]:
                    output_file.write(" ".join(f"{x}" for x in channel) + "\n")
            output_file.write(f"==== event list end ====\n\n")

    toc = timer()
    print(f"Elapsed time: {toc - tic:.2f} seconds.")
