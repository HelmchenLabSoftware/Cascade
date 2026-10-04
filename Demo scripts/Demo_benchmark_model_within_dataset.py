#!/usr/bin/env python3
"""
Leave-one-neuron-out cross-validation for an existing model trained on a single dataset.
Usage: python Demo_benchmark_model_leave_one_neuron_out.py <model_name>
"""

import os, sys, glob, shutil, csv
import numpy as np
from scipy.ndimage import gaussian_filter

if 'Demo scripts' in os.getcwd():
    sys.path.append(os.path.abspath('..'))
    os.chdir('..')

from cascade2p import cascade, config, utils

model_name = sys.argv[1]
noise_levels = [2, 4, 6, 8]

cfg = config.read_config(os.path.join('Pretrained_models', model_name, 'config.yaml'))
dataset = cfg['training_datasets'][0]
neuron_files = sorted(glob.glob(os.path.join('Ground_truth', dataset, '*_mini.mat')))
assert neuron_files, 'No ground truth files found for ' + dataset

sigma = cfg['smoothing'] * cfg['sampling_rate']
center = int(np.round(cfg['before_frac'] * cfg['windowsize']))
temp_model, temp_gt = model_name + '_LOO_temp', 'Ground_truth_LOO_temp'
cfg['model_name'], cfg['noise_levels'] = temp_model, noise_levels

csv_path = f'LOO_benchmark_{model_name}.csv'
csv_file = open(csv_path, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['neuron', 'noise_level', 'correlation', 'error', 'bias'])

for i, test_file in enumerate(neuron_files):

    shutil.rmtree(temp_gt, ignore_errors=True)
    shutil.rmtree(os.path.join('Pretrained_models', temp_model), ignore_errors=True)
    os.makedirs(os.path.join(temp_gt, dataset))
    for f in neuron_files:
        if f != test_file:
            shutil.copy(f, os.path.join(temp_gt, dataset))

    cascade.create_model_folder(cfg)
    cascade.train_model(temp_model, ground_truth_folder=temp_gt)

    for noise_level in noise_levels:
        try:
            calcium, ground_truth = utils.preprocess_groundtruth_artificial_noise_balanced(
                ground_truth_folders=[os.path.join('Ground_truth', dataset)],
                before_frac=cfg['before_frac'], windowsize=cfg['windowsize'],
                after_frac=1 - cfg['before_frac'], noise_level=noise_level,
                sampling_rate=cfg['sampling_rate'], smoothing=sigma,
                omission_list=[j for j in range(len(neuron_files)) if j != i],
                permute=0, verbose=cfg['verbose'], replicas=0, causal_kernel=cfg['causal_kernel'])

            spike_rates = np.squeeze(cascade.predict(temp_model, calcium[:, center].T, verbosity=0))
            ground_truth = np.squeeze(ground_truth)
            ok = ~np.isnan(spike_rates)
            ground_truth, spike_rates = ground_truth[ok], spike_rates[ok]

            gt_smooth = gaussian_filter(ground_truth.astype(float), sigma=sigma)
            sr_smooth = gaussian_filter(spike_rates.astype(float), sigma=sigma)
            diff = sr_smooth - gt_smooth
            signal = np.sum(gt_smooth)
            corr = np.corrcoef(ground_truth, spike_rates)[0, 1]
            err, bias = np.sum(np.abs(diff)) / signal, np.sum(diff) / signal
        except Exception as e:
            print(f'Neuron {os.path.basename(test_file)}, noise level {noise_level}: {e}')
            corr = err = bias = np.nan

        writer.writerow([os.path.basename(test_file), noise_level, corr, err, bias])
        csv_file.flush()

shutil.rmtree(temp_gt, ignore_errors=True)
shutil.rmtree(os.path.join('Pretrained_models', temp_model), ignore_errors=True)
csv_file.close()


## Summarize benchmark results

import pandas as pd

df = pd.read_csv(csv_path)
metrics = ['correlation', 'error', 'bias']
 
def stats(x):
    x = x.dropna()
    return pd.Series({'n': len(x), 'mean': x.mean(), 'sd': x.std(), 'sem': x.sem(),
                      'median': x.median(), 'q25': x.quantile(0.25), 'q75': x.quantile(0.75),
                      'min': x.min(), 'max': x.max()})
 
per_noise = df.groupby('noise_level')[metrics].apply(lambda g: g.apply(stats)).unstack()
overall = df[metrics].apply(stats).unstack().to_frame('all').T
summary = pd.concat([per_noise, overall])
summary.index.name = 'noise_level'
 
pd.set_option('display.width', 250, 'display.precision', 3)
for m in metrics:
    print(f'\n{m}\n', summary[m])
print('\nMissing values:', df[metrics].isna().any(axis=1).sum(), 'rows:',
      df.loc[df[metrics].isna().any(axis=1), ['neuron', 'noise_level']].values.tolist())
 
summary.to_csv(csv_path.replace('.csv', '_summary.csv'))
