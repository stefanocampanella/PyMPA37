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
import logging
from time import perf_counter as timer

from obspy import UTCDateTime
from obspy.core.event import read_events
from yaml import load, FullLoader

from pympa import get_travel_times, get_template_stream, listdays, get_continuous_stream, find_events

if __name__ == '__main__':
    logging.basicConfig(filename='run.log', filemode='w', format='%(levelname)s: %(message)s')
    start_time = timer()
    with open('settings.yaml') as file:
        settings = load(file, FullLoader)
    catalog = read_events(settings['ev_catalog'])
    UTCDateTime.DEFAULT_PRECISION = settings['utc_prec']

    events_list = []
    for itemp in range(*settings['template_range']):
        travel_times = get_travel_times(itemp, settings)
        template_stream = get_template_stream(itemp, travel_times, settings)
        if len(template_stream) >= settings['nch_min']:
            mt = catalog[itemp].magnitudes[0].mag
            for day in listdays(*settings['date_range']):
                day_stream = get_continuous_stream(template_stream, day, settings)
                if len(day_stream) >= settings['nch_min']:
                    new_events = find_events(itemp, template_stream, day_stream, travel_times, mt, settings)
                    events_list.extend(new_events)
                else:
                    logging.info(f"{day}, not enough channels for template {itemp} (nch_min: {settings['nch_min']}")

    with open("output.stats", "w") as file:
        for event in events_list:
            file.write(" ".join(f"{x}" for x in event[:-1]) + "\n")
            for channel in event[-1]:
                str22 = "%s %s %s %s \n" % channel
                file.write(str22)

    print(" elapsed time ", timer() - start_time, " seconds")
