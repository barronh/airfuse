__all__ = [
    'epa_aqi_cmap', 'epa_aqi2_cmap',
    'epa_pmaqi_norm', 'epa_pmaqi2_norm',
    'epa_o3aqi_norm', 'epa_o3aqi2_norm'
]

import matplotlib.colors as mc
import matplotlib.cm as cm
import numpy as np
# From AQI guidance on airnow
aqicolors = [mc.to_hex(c) for c in np.array([
    [0, 228, 0, ], [255, 255, 0], [255, 126, 0], [255, 0, 0],
    [143, 63, 151], [126, 0, 35]
]) / 256]

ant1hpmcolors = [
    [0, 150, 0], [153, 204, 0], [255, 255, 153], [255, 255, 0],
    [255, 204, 0], [247, 153, 0], [255, 0, 0], [214, 0, 147],
]
ant1hpmcolors = [mc.to_hex(c) for c in np.array(ant1hpmcolors) / 256]
ant1hpmedges = [-5, 10, 20, 30, 50, 70, 90, 120, 1000]

ant1ho3colors = [
    [0, 255, 0],
    [255, 255, 128],
    [255, 255, 0],
    [252, 191, 67],
    [255, 128, 0],
    [255, 0, 0],
]
ant1ho3colors = [mc.to_hex(c) for c in np.array(ant1ho3colors) / 256]
ant1ho3edges = [0, 60, 80, 100, 112, 125, 1000]

aqiedges = np.array([0, 50, 100, 150, 200, 300, 500])
aqiedges2 = np.array(
    [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]
)

pmedges = np.array([0, 12, 35.5, 55.5, 150.5, 250.5, 255])
pmedges2 = np.array([
    0, 6, 12, 23.75, 35.5, 45.5, 55.5, 103., 150.5, 200.5, 250.5, 252.75, 255
])
o3edges = np.array([0, 54, 70, 85, 105, 200, 255])
o3edges2 = np.array([
    0, 27, 54, 62, 70, 77.5, 85, 95, 105, 152.5, 200, 225, 250
])

from_list = mc.LinearSegmentedColormap.from_list
epa_aqi_cmap = from_list('epa_aqi', aqicolors, len(pmedges) - 1)
epa_aqi_cmap.set_under(aqicolors[0])
epa_aqi_cmap.set_over(aqicolors[-1])
epa_aqi2_cmap = from_list('epa_aqi2', aqicolors, len(pmedges2)-1)
epa_aqi2_cmap.set_under(aqicolors[0])
epa_aqi2_cmap.set_over(aqicolors[-1])
epa_aqi_norm = mc.BoundaryNorm(aqiedges, len(aqiedges) - 1)
epa_aqi2_norm = mc.BoundaryNorm(aqiedges2, len(aqiedges2) - 1)
epa_pmaqi2_norm = mc.BoundaryNorm(pmedges2, len(pmedges2) - 1)
epa_pmaqi_norm = mc.BoundaryNorm(pmedges, len(pmedges) - 1)
epa_o3aqi2_norm = mc.BoundaryNorm(o3edges2, len(o3edges2) - 1)
epa_o3aqi_norm = mc.BoundaryNorm(o3edges, len(o3edges) - 1)
ant_1ho3_cmap = from_list('ant_1ho3', ant1ho3colors, len(ant1ho3edges) - 1)
ant_1ho3_cmap.set_under(ant1ho3colors[0])
ant_1ho3_cmap.set_over(ant1ho3colors[-1])
ant_1ho3_norm = mc.BoundaryNorm(ant1ho3edges, len(ant1ho3edges) - 1)
ant_1hpm_cmap = from_list('ant_1hpm', ant1hpmcolors, len(ant1hpmedges) - 1)
ant_1hpm_cmap.set_under(ant1hpmcolors[0])
ant_1hpm_cmap.set_over(ant1hpmcolors[-1])
ant_1hpm_norm = mc.BoundaryNorm(ant1hpmedges, len(ant1hpmedges) - 1)

cm.register('epa_aqi', epa_aqi_cmap)
cm.register('epa_aqi2', epa_aqi2_cmap)
cm.register('ant_1hpm', ant_1hpm_cmap)
cm.register('ant_1ho3', ant_1ho3_cmap)
