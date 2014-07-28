__author__ = 'casey'

"""
@package coverage_model.util.time_utils
@file coverage_model/util/time_utils.py
@author Casey Bryant
@brief Time conversion utilities
"""

import datetime
import time

SYSTEM_EPOCH = datetime.date(*time.gmtime(0)[0:3])
NTP_EPOCH = datetime.date(1900, 1, 1)
NTP_DELTA = (SYSTEM_EPOCH - NTP_EPOCH).days * 24 * 3600


def ntp_to_system_time(date):
    """convert a NTP time to system time"""
    return date - NTP_DELTA


def system_to_ntp_time(date):
    """convert a system time to a NTP time"""
    return date + NTP_DELTA


def get_current_ntp_time(utc=True):
    """convert a system time to a NTP time"""
    offset = 0
    if utc:
        offset = time.altzone
    return system_to_ntp_time(time.time()+offset)

