# plot_utils/my_cmaps.py
from typing import Sequence
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --- 1) minimal: make + register a continuous map from stops ---
def register_cmap_from_list(name: str, stops: Sequence[str], N: int = 256):
    """
    Build a smooth colormap from `stops` and register it under `name`.
    Returns the cmap object too (so you can pass it directly if you prefer).
    """
    cmap = LinearSegmentedColormap.from_list(name, list(stops), N=N)
    # Matplotlib API changed across versions; keep it simple:
    try:
        mpl.colormaps.register(cmap, name=name)  # modern
    except TypeError:
        mpl.cm.register_cmap(name=name, cmap=cmap)  # older fallback
    return cmap

# --- 2) optional: get a discrete N-color version sampled from a continuous map ---
def make_discrete(name: str, stops: Sequence[str], N: int) -> ListedColormap:
    base = LinearSegmentedColormap.from_list(f"{name}__base", list(stops), N=256)
    cols = base(np.linspace(0, 1, N))
    return ListedColormap(cols, name=name)

# --- 3) your preset(s) ready to call once per process ---
# --- Colormap dictionary ---
lto_all = ['#0b2c4d', '#1b6b5a', '#2a9d2f']
lto_unex = ['#0b2c4d',"#146B94","#1cacb6", ]
lto_15m = ['#1b6b5a',"#162c1f","#8CB682",]
lto_1w = ['#2a9d2f', "#263b0a", "#a39e01",]

lfp_all = ['#a11616', '#e03e8f', '#7e1e9c']
lfp_unex = ['#a11616',"#cf6502","#d3c501",]
lfp_unex2 = ["#e45858",'#a11616',"#290a0a",]
lfp_15m = ["#e03e8f","#40378d","#01A1FD",]
lfp_15m2 = ["#c2759b","#e03e8f","#1f0813",]
lfp_1w = ['#7e1e9c',"#f030d6","#d10d2e",]


def register_ltodefaults():
    # registers 'lto_unex' so plt.get_cmap('lto_unex') works
    register_cmap_from_list('lto_unex', lto_unex, N=512)
    register_cmap_from_list('lto_all', lto_all, N=512)
    register_cmap_from_list('lto_15m', lto_15m, N=512)
    register_cmap_from_list('lto_1w', lto_1w, N=512)

def register_lfpdefaults():
    # registers 'lfp_unex' so plt.get_cmap('lfp_unex') works
    register_cmap_from_list('lfp_unex', lfp_unex, N=512)
    register_cmap_from_list('lfp_unex2', lfp_unex2, N=512)
    register_cmap_from_list('lfp_all', lfp_all, N=512)
    register_cmap_from_list('lfp_15m', lfp_15m, N=512)
    register_cmap_from_list('lfp_15m2', lfp_15m2, N=512)
    register_cmap_from_list('lfp_1w', lfp_1w, N=512)


# how to call them: (when you put into .py file)
""" import numpy as np
import matplotlib.pyplot as plt
from my_colormaps import lto_all, make_continuous_cmap, register_cmap, get_hex_colors

ltoall_cont = make_continuous_cmap("ltoall_cont", lto_all, N=256)
ltoall_9    = make_discrete_cmap("ltoall_9", lto_all, N=9)
# You can register them at import if you like:
register_cmap(ltoall_cont)
register_cmap(ltoall_9) """
