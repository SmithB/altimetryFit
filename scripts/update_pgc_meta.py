#!/usr/bin/env python3
"""
Download the PGC metadata (json) for the strip DEMs in a directory

The metadata for each SETSM*.tif in DEM_dir are saved in DEM_dir/../meta,
where altimetryFit.get_pgc looks for them.
"""
import argparse
import glob
import os
import sys
from altimetryFit.get_pgc import get_pgc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Download the PGC metadata (json) for the SETSM*.tif strip DEMs in a directory, '
                    'into the meta directory next to it')
    parser.add_argument('DEM_dir', help='directory containing SETSM*.tif files')
    parser.add_argument('url_file', help='file listing the PGC url for each DEM')
    args = parser.parse_args(argv)

    # absolute paths: get_pgc saves the metadata for DEM_dir/file.tif in
    # DEM_dir/../meta, found from the file's path
    DEM_dir = os.path.abspath(args.DEM_dir)
    os.makedirs(os.path.join(os.path.dirname(DEM_dir), 'meta'), exist_ok=True)

    n_failed = 0
    for file in sorted(glob.glob(os.path.join(DEM_dir, 'SETSM*.tif'))):
        try:
            get_pgc(file, args.url_file, targets=['meta'])
        except IndexError as e:
            # no url for this file in url_file
            print(f'{os.path.basename(file)}: {e}', file=sys.stderr)
            n_failed += 1
    if n_failed > 0:
        print(f'no metadata for {n_failed} files', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
