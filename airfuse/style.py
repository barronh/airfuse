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
    [0, 228, 0, ],
    [255, 255, 0],
    [255, 126, 0],
    [255, 0, 0],
    [143, 63, 151],
    [126, 0, 35]
]) / 256]

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
cm.register_cmap('epa_aqi', epa_aqi_cmap)
cm.register_cmap('epa_aqi2', epa_aqi2_cmap)
