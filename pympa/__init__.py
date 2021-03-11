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
