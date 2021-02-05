import os
import os.path
import datetime
from math import log10

import numpy as np
import logging
from obspy import read, Stream, Trace
from obspy.signal.cross_correlation import correlate_template


def listdays(year, month, day, period):
    """Create a list of days for scanning by templates"""
    init = datetime.datetime(year=year, month=month, day=day)
    days = []
    for n in range(period):
        date = init + datetime.timedelta(days=n)
        days.append(date.strftime("%y%m%d"))
    return days


def read_parameters(par):
    # read 'parameters24' file to setup useful variables

    with open(par) as fp:
        data = fp.read().splitlines()

        stations = data[23].split(" ")
        logging.debug(f"Stations: {stations}")
        channels = data[24].split(" ")
        logging.debug(f"Channels: {channels}")
        networks = data[25].split(" ")
        logging.debug(f"Networks: {networks}")
        lowpassf = float(data[26])
        highpassf = float(data[27])
        sample_tol = int(data[28])
        cc_threshold = float(data[29])
        nch_min = int(data[30])
        temp_length = float(data[31])
        utc_prec = int(data[32])
        cont_dir = "./" + data[33] + "/"
        temp_dir = "./" + data[34] + "/"
        travel_dir = "./" + data[35] + "/"
        dateperiod = data[36].split(" ")
        ev_catalog = str(data[37])
        start_itemp = int(data[38])
        logging.debug(f"Starting template = {start_itemp}")
        stop_itemp = int(data[39])
        logging.debug(f"Ending template = {stop_itemp}")
        factor_thre = int(data[40])
        stdup = float(data[41])
        stddown = float(data[42])
        chan_max = int(data[43])
        nchunk = int(data[44])
        return (
            stations,
            channels,
            networks,
            lowpassf,
            highpassf,
            sample_tol,
            cc_threshold,
            nch_min,
            temp_length,
            utc_prec,
            cont_dir,
            temp_dir,
            travel_dir,
            dateperiod,
            ev_catalog,
            start_itemp,
            stop_itemp,
            factor_thre,
            stdup,
            stddown,
            chan_max,
            nchunk,
        )


def trim_fill(tc, t1, t2):
    tc.trim(starttime=t1, endtime=t2, pad=True, fill_value=0)
    return tc


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# itemp = template number, nn =  network code, ss = station code,
# ich = channel code, stream_df = Stream() object as defined in obspy
# library
def process_input(temp_dir, itemp, nn, ss, ich, stream_df):
    st_cft = Stream()
    temp_file = f"{itemp}.{nn}.{ss}..{ich}.mseed"
    finpt = f"{temp_dir}{temp_file}"
    if os.path.isfile(finpt):
        try:
            tsize = os.path.getsize(finpt)
            if tsize > 0:
                # print "ok template exist and not empty"
                st_temp = read(finpt, dtype="float32")
                tt = st_temp[0]
                # continuous data are stored in stream_df
                sc = stream_df.select(station=ss, channel=ich)
                if sc.__nonzero__():
                    tc = sc[0]
                    logging.debug("Warning issue: using dirty data with spikes and gaps 'fft' "
                          "method could not work properly. "
                          "Try 'direct' to ensure more robustness "
                          "The correlate_template function is set now "
                          "to 'auto' and different environments "
                          "as Windows or Mac could not have consistent results")

                    fct = correlate_template(tc.data, tt.data, normalize="full", method="auto")
                    fct = np.nan_to_num(fct)
                    stats = {"network": tc.stats.network,
                             "station": tc.stats.station,
                             "location": "",
                             "channel": tc.stats.channel,
                             "starttime": tc.stats.starttime,
                             "npts": len(fct),
                             "sampling_rate": tc.stats.sampling_rate,
                             "mseed": {"dataquality": "D"}}
                    trnew = Trace(data=fct, header=stats)
                    tc = trnew.copy()
                    st_cft = Stream(traces=[tc])
                else:
                    logging.debug("No stream is found")
            else:
                logging.debug(f"Template event is empty {finpt}")
        except OSError:
            pass
    return st_cft


def quality_cft(trac):
    return np.nanstd(abs(trac.data))


def stack(stall, df, tstart, d, npts, stdup, stddown, nch_min):
    """
    Function to stack traces in a stream with different trace.id and
    different starttime but the same number of datapoints.
    Returns a trace having as starttime
    the earliest startime within the stream
    """
    std_trac = np.empty(len(stall))
    td = np.empty(len(stall))
    for itr, tr in enumerate(stall):
        std_trac[itr] = quality_cft(tr)
    avestd = np.nanmean(std_trac[0:])
    avestdup = avestd * stdup
    avestddw = avestd * stddown

    for jtr, tr in enumerate(stall):
        if std_trac[jtr] >= avestdup or std_trac[jtr] <= avestddw:
            stall.remove(tr)
            logging.debug(f"Removed Trace n Stream = {tr} {std_trac[jtr]} {avestd}")
            td[jtr] = 99.99
        else:
            sta = tr.stats.station
            chan = tr.stats.channel
            net = tr.stats.network
            s = f"{net}.{sta}.{chan}"
            td[jtr] = float(d[s])

    itr = len(stall)
    logging.debug(f"itr = {itr}")
    header = {"network": "STACK",
              "station": "BH",
              "channel": "XX",
              "starttime": tstart,
              "sampling_rate": df,
              "npts": npts}
    if itr >= nch_min:
        tdifmin = min(td)
        data = np.nansum([tr.data for tr in stall], axis=0) / itr

    else:
        tdifmin = None
        data = np.zeros(npts)
    tt = Trace(data=data, header=header)
    return tt, tdifmin


def csc(stall, stcc, trg, tstda, sample_tol, cc_threshold, nch_min, f1):
    """
    The function check_singlechannelcft compute the maximum CFT's
    values at each trigger time and counts the number of channels
    having higher cross-correlation
    nch, cft_ave, crt are re-evaluated on the basis of
    +/- 2 sample approximation. Statistics are written in stat files
    """
    # important parameters: a sample_tolerance less than 2 results often
    # in wrong magnitudes
    sample_tolerance = sample_tol
    single_channelcft = cc_threshold
    trigger_time = trg["time"]
    tcft = stcc[0]
    t0_tcft = tcft.stats.starttime
    trigger_shift = trigger_time.timestamp - t0_tcft.timestamp
    trigger_sample = int(round(trigger_shift / tcft.stats.delta))
    max_sct = np.empty(len(stall))
    max_trg = np.empty(len(stall))
    max_ind = np.empty(len(stall))
    chan_sct = np.chararray(len(stall), 12)

    for icft, tsc in enumerate(stall):
        # get cft amplitude value at corresponding trigger and store it in
        # check for possible 2 sample shift and eventually change
        # trg['cft_peaks']
        chan_sct[icft] = (
                tsc.stats.network + "." + tsc.stats.station + " " + tsc.stats.channel
        )
        tmp0 = trigger_sample - sample_tolerance

        if tmp0 < 0:
            tmp0 = 0
        tmp1 = trigger_sample + sample_tolerance + 1
        max_sct[icft] = max(tsc.data[tmp0:tmp1])
        max_ind[icft] = np.nanargmax(tsc.data[tmp0:tmp1])
        max_ind[icft] = sample_tolerance - max_ind[icft]
        max_trg[icft] = tsc.data[trigger_sample: trigger_sample + 1]
    nch = (max_sct > single_channelcft).sum()

    if nch >= nch_min:
        nch09 = (max_sct > 0.9).sum()
        nch07 = (max_sct > 0.7).sum()
        nch05 = (max_sct > 0.5).sum()
        nch03 = (max_sct > 0.3).sum()
        # print("nch ==", nch03, nch05, nch07, nch09)
        cft_ave = np.nanmean(max_sct[:])
        crt = cft_ave / tstda
        cft_ave_trg = np.nanmean(max_trg[:])
        crt_trg = cft_ave_trg / tstda
        max_sct = max_sct.T
        max_trg = max_trg.T
        chan_sct = chan_sct.T
        for idchan in range(0, len(max_sct)):
            str22 = "%s %s %s %s \n" % (
                chan_sct[idchan].decode(),
                max_trg[idchan],
                max_sct[idchan],
                max_ind[idchan],
            )
            f1.write(str22)

    else:
        nch = 1
        cft_ave = 1
        crt = 1
        cft_ave_trg = 1
        crt_trg = 1
        nch03 = 1
        nch05 = 1
        nch07 = 1
        nch09 = 1

    return nch, cft_ave, crt, cft_ave_trg, crt_trg, nch03, nch05, nch07, nch09


def mag_detect(magt, amaxt, amaxd):
    """
    mag_detect(mag_temp,amax_temp,amax_detect)
    Returns the magnitude of the new detection by using the template/detection
    amplitude trace ratio
    and the magnitude of the template event
    """
    amaxr = amaxt / amaxd
    magd = magt - log10(amaxr)
    return magd


def reject_moutliers(data, m=1.0):
    nonzeroind = np.nonzero(data)[0]
    nzlen = len(nonzeroind)
    data = data[nonzeroind]
    datamed = np.nanmedian(data)
    d = np.abs(data - datamed)
    mdev = 2 * np.median(d)
    if mdev == 0:
        inds = np.arange(nzlen)
        data[inds] = datamed
    else:
        s = d / mdev
        inds = np.where(s <= m)
    return data[inds]


def mad(dmad):
    # calculate daily median absolute deviation
    ccm = dmad[dmad != 0]
    med_val = np.nanmedian(ccm)
    return np.nansum(abs(ccm - med_val) / len(ccm))
