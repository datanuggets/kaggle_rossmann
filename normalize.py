#!/usr/bin/env python


import csv
import gzip
import sys


from datetime import datetime, time
from isoweek import Week
from progressbar import ProgressBar


# global variables
PROMO2_INTERVAL = {
    'Jan,Apr,Jul,Oct': {1, 4, 7, 10},
    'Feb,May,Aug,Nov': {2, 5, 8, 11},
    'Mar,Jun,Sept,Dec': {3, 6, 9, 12},
}
START_DATE = datetime(2013, 1, 1, 0, 0)


def csv_dict_reader(path):
    with open(path) as f:
        csvf = csv.DictReader(f)
        for record in csvf:
            yield record


def line_count(path):
    with open(path) as f:
        return sum(1 for l in f)


def one_hot_encode(value, options):
    return tuple(int(o == value) for o in options)


def normalize_record(r):
    date = datetime.strptime(r['Date'], "%Y-%M-%d")
    weekday = date.weekday()
    r['NowDayFromStart'] = (date - START_DATE).days
    r['NowDayOfWeek'] = int(weekday)
    r['NowDay'] = int(date.day)
    r['NowMonth'] = int(date.month)
    r['NowYear'] = int(date.year)
    r['NowIsWeekend'] = int(weekday >= 5)
    r['NowWeek'] = int(date.date().isocalendar()[1])
    del r['Date'], r['DayOfWeek']

    r['Store'] = int(r['Store'])
    r['Promo'] = int(r['Promo'])
    r['SchoolHoliday'] = int(r['SchoolHoliday'])
    if 'Customers' in r:
        r['Customers'] = int(r['Customers'])
    if 'Sales' in r:
        r['Sales'] = int(r['Sales'])
    if 'Id' in r:
        r['Id'] = int(r['Id'])

    (
        r['IsOpenYes'],
        r['IsOpenNo'],
        r['IsOpenUnknown'],
    ) = one_hot_encode(r['Open'], ('1', '0', ''))
    del r['Open']

    (
        r['StateHolidayNone'],
        r['StateHolidayPublic'],
        r['StateHolidayEaster'],
        r['StateHolidayChristmas'],
    ) = one_hot_encode(r['StateHoliday'], ('0', 'a', 'b', 'c'))
    del r['StateHoliday']

    r['Promo2'] = int(r['Promo2'])
    r['CompetitionDistance'] = int(r['CompetitionDistance']) if r['CompetitionDistance'] else sys.maxint

    if r['CompetitionOpenSinceMonth'] and r['CompetitionOpenSinceYear']:
        competition_open_since = datetime(int(r['CompetitionOpenSinceYear']), int(r['CompetitionOpenSinceMonth']), 1)
        r['CompetitionOpenAvailable'] = 1
    else:
        competition_open_since = datetime(2010, 1, 1)
        r['CompetitionOpenAvailable'] = 0
    r['CompetitionOpenDays'] = int((date - competition_open_since).days)
    r['CompetitionOpenSinceMonth'] = int(competition_open_since.month)
    r['CompetitionOpenSinceYear'] = int(competition_open_since.year)

    (
        r['AssortmentBasic'],
        r['AssortmentExtra'],
        r['AssortmentExtended'],
    ) = one_hot_encode(r['Assortment'], ('a', 'b', 'c'))
    del r['Assortment']

    (
        r['StoreTypeA'],
        r['StoreTypeB'],
        r['StoreTypeC'],
        r['StoreTypeD'],
    ) = one_hot_encode(r['StoreType'], ('a', 'b', 'c', 'd'))
    del r['StoreType']

    if r['Promo2SinceYear'] and r['Promo2SinceWeek']:
        promo2_since = Week(int(r['Promo2SinceYear']), int(r['Promo2SinceWeek']))
        r['Promo2SinceAvailable'] = 1
    else:
        promo2_since = Week(2000, 1)
        r['Promo2SinceAvailable'] = 0
    promo2_since = datetime.combine(promo2_since.monday(), time(0, 0))
    r['Promo2RunDays'] = int((date - promo2_since).days)
    r['Promo2SinceWeek'] = int(promo2_since.date().isocalendar()[1])
    r['Promo2SinceMonth'] = int(promo2_since.month)
    r['Promo2SinceYear'] = int(promo2_since.year)
    del r['Promo2']

    (
        r['Promo2IntervalJanAprJulOct'],
        r['Promo2IntervalFebMayAugNov'],
        r['Promo2IntervalMarJunSepDec'],
    ) = one_hot_encode(r['PromoInterval'], tuple(PROMO2_INTERVAL))
    r['Promo2Now'] = int(r['NowMonth'] in PROMO2_INTERVAL.get(r['PromoInterval'], set()))
    del r['PromoInterval']


if __name__ == '__main__':
    """
    Combines a train or test set with the stores set while normalizing existing
    features and calculating new features.

    Usage: python normalize.py data/train.csv data/store.csv data/normalized.csv.gz
    """

    data_path = sys.argv[1]
    stores_path = sys.argv[2]
    output_path = sys.argv[3]

    # Load stores in a lookup dictionairy
    stores_map = {r['Store']: r for r in csv_dict_reader(stores_path)}

    with gzip.open(output_path, 'wb') as f:
        # First fetch a single row to setup the writer with the required columns
        reader = csv_dict_reader(data_path)
        record = reader.next()
        record.update(stores_map[record['Store']])
        normalize_record(record)
        writer = csv.DictWriter(f, list(record.iterkeys()))
        writer.writeheader()
        writer.writerow(record)

        # Fancy progressbar indicator
        pb = ProgressBar(maxval=line_count(data_path))
        pb.start()

        # Normalize and write all rows
        for i, record in enumerate(csv_dict_reader(data_path)):
            record.update(stores_map[record['Store']])
            normalize_record(record)
            writer.writerow(record)
            pb.update(i)

    pb.finish()
