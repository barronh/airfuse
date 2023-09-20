from .parser import parse_args
from .mod import get_model
from .obs import pair_airnow, pair_purpleair
from .models import applyfusion, get_fusions
from .ensemble import distweight
import time
import pyproj
import os
import logging

args = parse_args()
date = args.startdate


model = args.model.upper()
outdir = f'{date:%Y/%m/%d}'
os.makedirs(outdir, exist_ok=True)
stem = f'{outdir}/Fusion_PM25_{model}_{date:%Y-%m-%dT%H}Z'
logpath = f'{stem}.log'
pacvpath = f'{stem}_PurpleAir_CV.csv'
ancvpath = f'{stem}_AirNow_CV.csv'
fusepath = f'{stem}.csv'

found = set()
for path in [pacvpath, ancvpath, fusepath]:
    if os.path.exists(path):
        found.add(path)

if len(found) > 0 and not args.overwrite:
    raise IOError(f'Outputs exist; delete or use -O to continue:\n{found}')

# Divert all logging during this script to the associated
# log file at the INFO level.
logging.basicConfig(filename=logpath, level=logging.INFO)

bbox = args.bbox

pm = get_model(date, key='pm25', bbox=bbox, model=model)

proj = pyproj.Proj(pm.attrs['crs_proj4'], preserve_units=True)
logging.info(proj.srs)

andf = pair_airnow(date, bbox, proj, pm, 'pm25')
padf = pair_purpleair(date, bbox, proj, pm, 'pm25')

models = get_fusions()

if args.cv_only:
    tgtdf = None
else:
    outdf = pm.to_dataframe().reset_index()
    tgtdf = outdf.query(f'{pm.name} == {pm.name}').copy()

# Apply all models to AirNow observations
for mkey, mod in models.items():
    logging.info(f'AN {mkey} begin')
    t0 = time.time()
    applyfusion(
        mod, f'{mkey}_AN', andf, tgtdf=tgtdf, obskey='pm25', modkey=pm.name,
        verbose=9
    )
    t1 = time.time()
    logging.info(f'AN {mkey} {t1 - t0:.0f}s')

# Apply all models to PurpleAir observations
for mkey, mod in models.items():
    logging.info(f'PA {mkey} begin')
    t0 = time.time()
    applyfusion(
        mod, f'{mkey}_PA', padf, tgtdf=tgtdf, loodf=andf,
        obskey='pm25', modkey=pm.name, verbose=9
    )
    t1 = time.time()
    logging.info(f'PA {mkey} finish: {t1 - t0:.0f}s')

# Perform fusions on LOO data for aVNA
distkeys = ['LOO_VNA_AN_DIST', 'LOO_VNA_PA_DIST']
valkeys = ['LOO_aVNA_AN', 'LOO_aVNA_PA']
wgtdf = distweight(
    andf, distkeys, valkeys, modkey=model, ykey='FUSED_aVNA', power=-2,
    add=True, LOO_aVNA_PA=0.25
)
# Perform fusions on LOO data for eVNA
valkeys = ['LOO_eVNA_AN', 'LOO_eVNA_PA']
wgtdf = distweight(
    andf, distkeys, valkeys, modkey=model, ykey='FUSED_eVNA', power=-2,
    add=True, LOO_eVNA_PA=0.25
)
# Save results to disk as CSV files
andf.to_csv(ancvpath, index=False)
padf.to_csv(pacvpath, index=False)

if not args.cv_only:
    # Perform fusions on Target Dataset for aVNA
    distkeys = ['VNA_AN_DIST', 'VNA_PA_DIST']
    valkeys = ['aVNA_AN', 'aVNA_PA']
    wgtdf = distweight(
        tgtdf, distkeys, valkeys, modkey=model, ykey='FUSED_aVNA', power=-2,
        add=True, aVNA_PA=0.25
    )
    # Save final results to disk
    tgtdf.to_csv(fusepath, index=False)

logging.info('Successful Completion')
