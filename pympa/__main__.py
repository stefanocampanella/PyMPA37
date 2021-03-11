import logging
from pathlib import Path
from time import perf_counter as timer

from obspy import UTCDateTime
from obspy.core.event import read_events
from yaml import load, FullLoader
import importlib.resources as resources

from pympa.core import get_travel_times, get_template_stream, range_days, get_continuous_stream, find_events
import argparse

parser = argparse.ArgumentParser(prog="PyMPA37",
                                 description="A software package for phase match filtering")
parser.add_argument("settings_path",
                    help="Path to settings")
cli_args = parser.parse_args()

with resources.open_text('pympa', 'defaults.yaml') as default_settings_file:
    default_settings = load(default_settings_file, FullLoader)

if __name__ == '__main__':
    tic = timer()

    settings_path = Path(cli_args.settings_path).resolve()
    with open(settings_path) as settings_file:
        runtime_settings = load(settings_file, FullLoader)
    settings = default_settings.copy()
    settings.update(runtime_settings)

    root_path = settings_path.parent

    log_path = root_path / settings['log']
    logging.basicConfig(filename=log_path,
                        filemode='w',
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)

    catalog_path = root_path / settings['ev_catalog']
    catalog = read_events(catalog_path)

    travel_times_dir_path = root_path / settings['travel_dir']
    templates_dir_path = root_path / settings['temp_dir']
    continuous_dir_path = root_path / settings['cont_dir']

    UTCDateTime.DEFAULT_PRECISION = settings['utc_prec']

    events_list = []
    for template_number in range(*settings['template_range']):
        travel_times = get_travel_times(travel_times_dir_path,
                                        template_number,
                                        chan_max=settings['chan_max'])
        template_stream = get_template_stream(templates_dir_path,
                                              template_number,
                                              travel_times.keys())
        if len(template_stream) >= settings['nch_min']:
            mt = catalog[template_number].magnitudes[0].mag
            for day in range_days(*settings['date_range']):
                day_stream = get_continuous_stream(continuous_dir_path,
                                                   day,
                                                   travel_times.keys(),
                                                   freqmin=settings['lowpassf'],
                                                   freqmax=settings['highpassf'])
                if len(day_stream) >= settings['nch_min']:
                    new_events = find_events(template_number,
                                             template_stream,
                                             day_stream,
                                             travel_times,
                                             mt,
                                             settings)
                    events_list.extend(new_events)
                else:
                    logging.info(f"{day}, not enough channels for "
                                 f"template {template_number} (nch_min: {settings['nch_min']}")

    with open(settings['output'], "w") as file:
        for event in events_list:
            file.write(" ".join(f"{x}" for x in event[:-1]) + "\n")
            for channel in event[-1]:
                str22 = "%s %s %s %s \n" % channel
                file.write(str22)
    toc = timer()
    print(f"Elapsed time: {toc - tic:.2f} seconds.")
