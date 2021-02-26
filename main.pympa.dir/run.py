#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2016/08/23 Version 34 - parameters24 input file needed
# 2017/10/27 Version 39 - Reformatted PEP8 Code
# 2017/11/05 Version 40 - Corrections to tdifmin, tstda calculations
# 2019/10/15 Version pympa - xcorr substitued with correlate_template from obspy

# First Version August 2014 - Last October 2017 (author: Alessandro Vuan)

# Code for the detection of microseismicity based on cross correlation
# of template events. The code exploits multiple cores to speed up time
#
# Method's references:
# The code is developed and maintained at
# Istituto Nazionale di Oceanografia e Geofisica di Trieste (OGS)
# and was inspired by collaborating with Aitaro Kato and collegues at ERI.
# Kato A, Obara K, Igarashi T, Tsuruoka H, Nakagawa S, Hirata N (2012)
# Propagation of slow slip leading up to the 2011 Mw 9.0 Tohoku-Oki
# earthquake. Science doi:10.1126/science.1215141
#
# For questions comments and suggestions please send an email to avuan@inogs.it
# The kernel function xcorr used from Austin Holland is modified in pympa
# Recommended the use of Obspy v. 1.2.0 with the substitution of xcorr function with
# correlate_template

# Software Requirements: the following dependencies are needed (check import
# and from statements below)
# Python "obspy" package installed via Anaconda with all numpy and scipy
# packages
# Python "math" libraries
# Python "bottleneck" utilities to speed up numpy array operations
#
from time import perf_counter as timer

from obspy.core import UTCDateTime
from obspy.core.event import read_events
from obspy.signal.trigger import coincidence_trigger

from pympa import *

if __name__ == '__main__':
    logging.basicConfig(filename='run.log',
                        filemode='w',
                        format='%(levelname)s: %(message)s')
    start_time = timer()
    settings = read_settings("settings.yaml")
    catalog = read_events(settings['ev_catalog'])

    # set time precision for UTCDATETIME
    UTCDateTime.DEFAULT_PRECISION = settings['utc_prec']

    events_list = []
    for day in listdays(*settings['date_range']):
        logging.debug(f"Processing {day}")
        for itemp in range(*settings['template_range']):
            logging.debug(f"itemp = {itemp}")
            travel_dir = Path(settings['travel_dir'])
            with open(travel_dir / f"{itemp}.ttimes", "r") as ttim:
                travel_times = dict(x.rstrip().split(None, 1) for x in ttim)
            template_stream = get_template_stream(itemp, travel_times, settings)
            nch_min = settings['nch_min']
            if len(template_stream) >= nch_min:
                mt = catalog[itemp].magnitudes[0].mag
                events_list.extend(find_events(day, itemp, template_stream, travel_times, mt, settings))
            else:
                logging.info(f"{day} {itemp} num.  templates lower than nch_min\n")

    with open("output.stats", "w") as file:
        for event in events_list:
            channels_list = event[-1]
            for channel in channels_list:
                str22 = "%s %s %s %s \n" % channel
                file.write(str22)
            file.write(" ".join(f"{x}" for x in event[:-1]) + "\n")

    print(" elapsed time ", timer() - start_time, " seconds")
