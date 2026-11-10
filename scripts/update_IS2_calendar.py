#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update the ICESat-2 cycle calendar (altimetryFit/package_data/IS2_calendar.json)

The calendar maps each cycle number to the [year, month, day] of the cycle's
start, and is used by altimetryFit.read_ICESat2.select_gIs_by_cycle to select
geoIndex files by cycle.  The cycle dates are read from the ICESat-2 science
specs page, which lists each cycle as e.g.
    Cycle 2: December 28 (2018) - March 29, 2019
    Cycle 3: March 29 - June 28, 2019
    Cycle 6: December 26, 2019 - March 26, 2020
including, at the end, provisional future cycles.  The start dates are taken
as listed, so a cycle that started late (e.g. after a safe-hold) has its
listed start, not the end of the previous cycle.
"""

import argparse
import datetime
import html
import json
import os
import re
import urllib.request
from importlib import resources

SPECS_URL = 'https://icesat-2.gsfc.nasa.gov/science/specs'

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
          'september', 'october', 'november', 'december']

# Cycle N: Month D[, YYYY| (YYYY)] - Month D, YYYY
CYCLE_RE = re.compile(r'Cycle\s+(\d+)\s*:\s*'
                      r'([A-Za-z]+)\.?\s+(\d{1,2})(?:\s*,\s*(\d{4})|\s*\(\s*(\d{4})\s*\))?'
                      r'\s*[-–—]+\s*'
                      r'([A-Za-z]+)\.?\s+(\d{1,2})\s*,\s*(\d{4})')


def default_calendar_file():
    return os.path.join(str(resources.files('altimetryFit')), 'package_data', 'IS2_calendar.json')


def read_page(url=SPECS_URL, html_file=None, timeout=60):
    '''
    Read the specs page, from url or from a saved copy (html_file)
    '''
    if html_file is not None:
        with open(html_file, 'r', encoding='utf-8', errors='replace') as fh:
            return fh.read()
    req = urllib.request.Request(url, headers={'User-Agent': 'altimetryFit update_IS2_calendar.py'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(resp.headers.get_content_charset() or 'utf-8', errors='replace')


def page_text(page):
    '''
    The text of an html page: tags removed, entities (&nbsp;, &ndash;) decoded
    '''
    text = re.sub(r'<[^>]*>', ' ', page)
    return re.sub(r'\s+', ' ', html.unescape(text))


def _date(month, day, year):
    # full month names, or abbreviations (Sep, Sept.)
    matches = [num for num, name in enumerate(MONTHS, start=1)
               if len(month) >= 3 and name.startswith(month.lower())]
    if len(matches) != 1:
        raise ValueError(f'unknown month: {month}')
    return datetime.date(int(year), matches[0], int(day))


def parse_cycles(page):
    '''
    Cycle start and end dates from the text of the specs page

    Returns
    -------
    dict : {cycle number : (start date, end date)}, sorted by cycle number
    '''
    cycles = {}
    for match in CYCLE_RE.finditer(page_text(page)):
        cycle = int(match.group(1))
        end = _date(match.group(6), match.group(7), match.group(8))
        start_year = match.group(4) or match.group(5)
        if start_year is None:
            # the start year is that of the end, unless the cycle crosses a new year
            start = _date(match.group(2), match.group(3), end.year)
            if start > end:
                start = _date(match.group(2), match.group(3), end.year-1)
        else:
            start = _date(match.group(2), match.group(3), start_year)
        if cycle in cycles and cycles[cycle] != (start, end):
            raise ValueError(f'cycle {cycle} is listed twice, with different dates: '
                             f'{cycles[cycle]} and {(start, end)}')
        cycles[cycle] = (start, end)
    return dict(sorted(cycles.items()))


def check_cycles(cycles):
    '''
    Check that the cycles are numbered 1 to N, and that their dates increase
    '''
    if len(cycles) == 0:
        raise ValueError('no cycles found: the page format may have changed')
    numbers = list(cycles)
    if numbers != list(range(1, len(numbers)+1)):
        missing = sorted(set(range(1, max(numbers)+1)) - set(numbers))
        raise ValueError(f'the cycles are not numbered 1 to {max(numbers)}, missing: {missing}')
    for cycle, (start, end) in cycles.items():
        if end <= start:
            raise ValueError(f'cycle {cycle} ends ({end}) before it starts ({start})')
        if cycle > 1 and start < cycles[cycle-1][0]:
            raise ValueError(f'cycle {cycle} starts ({start}) before cycle {cycle-1} ({cycles[cycle-1][0]})')


def calendar_from_cycles(cycles):
    '''
    The calendar: {str(cycle) : [year, month, day] of the start}
    '''
    return {str(cycle): [start.year, start.month, start.day] for cycle, (start, end) in cycles.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Update the ICESat-2 cycle calendar (IS2_calendar.json) from the cycle dates on '
                    'the ICESat-2 science specs page.')
    parser.add_argument('--url', default=SPECS_URL, help=f'specs page, default {SPECS_URL}')
    parser.add_argument('--html_file', help='read a saved copy of the specs page instead of the url')
    parser.add_argument('--out_file', help='calendar file to write, default: the IS2_calendar.json '
                        'in the package data of the installed altimetryFit')
    parser.add_argument('--force', action='store_true',
                        help='write the calendar even if it has fewer cycles than the existing one, '
                             'or changes their dates')
    parser.add_argument('--dry_run', '-n', action='store_true', help='print the changes, but write nothing')
    args = parser.parse_args(argv)
    out_file = args.out_file if args.out_file is not None else default_calendar_file()

    cycles = parse_cycles(read_page(args.url, html_file=args.html_file))
    check_cycles(cycles)
    calendar = calendar_from_cycles(cycles)
    print(f'found cycles 1-{len(calendar)}: cycle {len(calendar)} starts {cycles[len(calendar)][0]}')

    old = {}
    if os.path.isfile(out_file):
        with open(out_file, 'r') as fh:
            old = json.load(fh)
    added = [key for key in calendar if key not in old]
    changed = [key for key in calendar if key in old and old[key] != calendar[key]]
    dropped = [key for key in old if key not in calendar]
    for key in changed:
        print(f'cycle {key}: start changes from {old[key]} to {calendar[key]}')
    if added:
        print(f'new cycles: {", ".join(added)}')
    if dropped:
        print(f'cycles in {out_file} but not on the page: {", ".join(dropped)}')
    if not (added or changed or dropped):
        print(f'{out_file} is up to date')
        return 0
    if (dropped or changed) and not args.force and old:
        print(f'not writing {out_file}: the page drops or changes cycles (use --force to write it anyway)')
        return 1
    if args.dry_run:
        print(f'dry run: {out_file} not written')
        return 0
    with open(out_file, 'w') as fh:
        json.dump(calendar, fh, indent=2)
        fh.write('\n')
    print(f'wrote {out_file}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
