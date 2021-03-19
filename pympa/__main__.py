import argparse
import logging
import sys
from pathlib import Path
import importlib.resources as resources

from yaml import load, FullLoader
from tqdm import tqdm
import pandas as pd

from pympa.core import read_templates, range_days, read_continuous_stream, correlation_detector

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

    templates = read_templates(templates_dir_path, travel_times_dir_path, catalog_path, settings)

    start_date, end_date = settings['date_range']
    num_days = (end_date - start_date).days
    events_found = []
    for day in tqdm(range_days(start_date, end_date), total=num_days):
        for template_number, template_stream, travel_times, template_magnitude in templates:
            day_stream = read_continuous_stream(continuous_dir_path,
                                                day,
                                                tuple(travel_times.keys()),
                                                freqmin=settings['lowpassf'],
                                                freqmax=settings['highpassf'])
            if len(day_stream) < settings['nch_min']:
                logging.info(f"{day}, not enough channels in continuous stream")
                continue
            try:
                new_events = correlation_detector(template_stream,
                                                  day_stream,
                                                  travel_times,
                                                  template_magnitude,
                                                  settings)
                events_found.extend((template_number,) + event for event in new_events)
            except Exception as err:
                logging.warning(f"{err}")
    events_dataframe = pd.DataFrame.from_records(events_found,
                                                 columns=['template', 'date',
                                                          'magnitude', 'correlation', 'channels'])
    output_path = Path(cli_args.output_path)
    logging.info(f"Writing outputs to {output_path}")
    events_dataframe.to_csv(output_path)
