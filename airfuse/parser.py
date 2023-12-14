__all__ = ['get_parser', 'parse_args']


def str2bbox(bboxstr):
    import numpy as np
    return np.array(bboxstr.split(','), dtype='f')


def get_parser():
    """
    Create a parser object that can be used by pm.py or ozone.py
    """
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Add -v to make more verbose. Add -v -v for more verbose. etc'
    )
    parser.add_argument(
      '-O', '--overwrite', default=False, action='store_true',
      help='Overwrite existing files'
    )
    helpstr = 'Bounding box (lllon,lllat,urlon,urlat) in decimal degrees'
    parser.add_argument(
        '-b', '--bbox', default=str2bbox('-135,15,-55,60'), type=str2bbox,
        help=helpstr
    )
    parser.add_argument(
        '--outdir', default=None,
        help='Path for outputs; defaults to %%Y/%%m/%%d'
    )
    parser.add_argument(
        '-m', '--model', choices={'naqfc', 'geoscf', 'goes'}, default='naqfc',
        help='Model to use as mediating layer (default naqfc)'
    )
    parser.add_argument(
        '-s', '--species', choices={'o3', 'pm25'}, default='o3',
        help='Species to produce a fusion product for'
    )
    opt = {'airnow', 'aqs', 'purpleair'}
    hst = 'Observations can come from airnow, aqs, or purpleiar (pm only)'
    parser.add_argument('--obssource', choices=opt, default='airnow', help=hst)
    parser.add_argument(
      '-c', '--cv-only', default=False, action='store_true',
      help='Only run cross validation'
    )
    parser.add_argument('-a', '--api-key', help='PurpleAir API Key')
    parser.add_argument(
        '-f', '--format', choices={'csv', 'nc'}, default='csv',
        help='Output format csv or NetCDF (nc)'
    )
    hstr = 'Start Date YYYY-MM-DDTHHZ of obs and model'
    parser.add_argument('startdate', type=pd.to_datetime, help=hstr)
    return parser


def parse_args():
    """Parser user supplied arguments and supply the result."""
    return get_parser().parse_args()
