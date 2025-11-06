# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 18:11:48 2025

@author: nancy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_ce_trials_and_aggregate(file_paths):
    """
    Loads individual trial CSVs and computes Coulombic Efficiency (CE).
    Returns a DataFrame with CE mean and SEM per cycle.
    """
    trial_dfs = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        # Ensure correct column names
        charge_col = [col for col in df.columns if 'charge' in col.lower()][0]
        discharge_col = [col for col in df.columns if 'discharge' in col.lower()][0]

        # Calculate CE as %: (Discharge / Charge) * 100
        #df['CE'] = (df[charge_col] / df[discharge_col]) * 100 #LTO
        df['CE'] = (df[discharge_col]/df[charge_col]) * 100 #LFP
        df = df[['Cycle', 'CE']].rename(columns={'CE': file_path})
        trial_dfs.append(df)

    # Outer merge on Cycle to preserve all available data
    merged_df = trial_dfs[0]
    for trial_df in trial_dfs[1:]:
        merged_df = pd.merge(merged_df, trial_df, on='Cycle', how='outer')

    merged_df = merged_df.sort_values('Cycle').reset_index(drop=True)

    # Calculate mean
    mean_series = merged_df.drop(columns='Cycle').mean(axis=1, skipna=True)

    # Calculate standard deviation
    std_series = merged_df.drop(columns='Cycle').std(axis=1, skipna=True)

    # Count number of valid trials per cycle
    count_series = merged_df.drop(columns='Cycle').count(axis=1)

    # Calculate SEM
    sem_series = std_series / count_series.pow(0.5)

    result_df = pd.DataFrame({
        'Cycle': merged_df['Cycle'],
        'MeanCE': mean_series,
        'StdCE': sem_series  # Now actually SEM instead of stdev
    })

    return result_df

def plot_ce_chemistries(file_groups, labels, colors, markers, xlim=50, title_size=16, label_size=18, tick_size=16, legend_size=18):
    fig, ax = plt.subplots(figsize=(6, 5))

    for file_paths, label, color, marker in zip(file_groups, labels, colors, markers):
        data = load_ce_trials_and_aggregate(file_paths)

        if xlim is not None:
            data = data[data['Cycle'] <= xlim]

        ax.errorbar(data['Cycle'], data['MeanCE'], yerr=data['StdCE'],
                    label=label, color=color, fmt=marker, capsize=3, markersize=9)
        print(data['MeanCE'].mean())
        print(data['StdCE'].mean())

    plt.rcParams['font.family'] = 'Arial'
    ax.set_xlabel('Cycle Number', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('Coulombic Efficiency (%)', fontsize=label_size, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.set_xticks(np.arange(0, 50+2, 10))
    ax.legend(fontsize=legend_size, loc='lower left', shadow=True,)
    ax.grid(False, )#which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.9)
    ax.minorticks_on()
    ax.grid(False, )#which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.8)
    #ax.set_ylim(90, 100.05)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    if xlim is not None:
        ax.set_xlim(0, 25)

    plt.tight_layout()
    plt.show()

file_group_1 = [
   # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc254 - unexposed.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc262 - unexposed.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc273 - unexposed.csv", 
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc274 - unexposed.csv",
    #r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc306 - unexposed.csv",
   #r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc307 - unexposed.csv"
   # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP10.csv", 
   # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP11.csv",
   r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP73.csv",
   r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP74.csv",
   r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP91.csv",
   # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP110.csv",
   # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP111.csv",
   # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP112.csv",
   # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP113.csv",
   
]#unexposed

file_group_2 = [
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc283 - 15mins.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc284 - 15mins.csv", 
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc285 - 15mins.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc288 - 15mins.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc289 - 15mins.csv",
    #r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc296 - 15mins.csv",
    #r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc297 - 15mins.csv",
    #r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc292 - 15mins.csv",
    #r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc293 - 15mins.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP32.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP33.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP34.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP37.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP38.csv",
    r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP77.csv",
    r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP78.csv",
    r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP82.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP102.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP103.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP104.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP105.csv",
]#15 mins

file_group_3 = [
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc277 - 1week.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc278 - 1week.csv", 
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc282 - 1week.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc298 - 1week.csv",
    # r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\CE\LTOc300 - 1week.csv"
    #r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP30.csv",
    r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP65.csv",
    r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP66.csv",
     #r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP118.csv",
     r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP119.csv",
     r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP120.csv",
     r"C:\Users\nancy\OneDrive\Desktop\HF Exposure\Cycling Data\Manuscript\Reviews\LFP\CE and CapCYC\LFP121.csv",
]#1w

file_groups = [file_group_1, file_group_2, file_group_3]
labels = ['Unexposed', '15-mins Exposed', '1 week Exposed']
colors =  ['#0b2c4d', '#1b6b5a', '#2a9d2f']
markers = ['D-',  'o-', '^-', ]

plot_ce_chemistries(file_groups, labels, colors, markers)
