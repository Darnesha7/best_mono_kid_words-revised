import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def rescore(weight_on_freq, aoa, freq):
    x = freq.multiply(weight_on_freq)
    return aoa + x

def flip_values(x):
    flipped = x.max() - x
    return flipped

def scale_between(x, minval, maxval):
    y = x - x.min()
    z = y.div(x.max())
    m = z.multiply(x.max() - x.min())
    scaled = m + x.min()
    return scaled

def log_transform(x):
    log_trans = np.log10(x + 1)
    return log_trans

def log_transform_legacy(x):
    log_trans = np.log(x)
    return log_trans

mono_partial = pd.read_csv('mono_partial.csv', header=0, keep_default_na=False, na_values="#N/A")
sl = pd.read_excel("word-level_data/SUBTLEXusfrequencyabove1.xls")
ms= mono_partial.merge(sl.loc[:,("Word","SUBTLWF")], left_on = 'word', right_on='Word', how = 'left')
ms['freq_lcnl_extended'] = ms['freq_lcnl']
z = ms['freq_lcnl'].isnull()
ms.loc[z, 'freq_lcnl_extended'] = ms.loc[z, 'SUBTLWF'].divide(ms.loc[z,'SUBTLWF'].max())
print(ms)

ms['freq_lcnl_log'] = log_transform(ms['freq_lcnl'])
ms['freq_lcnl_extended_log'] = log_transform(ms['freq_lcnl_extended'])

ms['freq_lcnl_log_flipscale'] = scale_between(flip_values(ms['freq_lcnl_log']), 0, 10)
ms['freq_lcnl_extended_log_flipscale'] = scale_between(flip_values(ms['freq_lcnl_extended_log']),0 , 10)

weight_on_freq = 4.0
aoa = ms['aoa_mean']

freq = ms['freq_lcnl_log_flipscale']
ms['score'] = rescore(weight_on_freq, aoa, freq)
ms['freq_lcnl_log_flipscale'].corr(ms['score'])
ms['aoa_mean_scale'].corr(ms['score'])

freq_ext = ms['freq_lcnl_extended_log_flipscale']
ms['score_extended'] = rescore(weight_on_freq, aoa, freq_ext)
ms['freq_lcnl_extended_log_flipscale'].corr(ms['score_extended'])
ms['aoa_mean_scale'].corr(ms['score_extended'])

ms = ms[ms.score_extended.notnull()].sort_values(by ='score_extended', ascending = True)
ms.to_csv('mono_scored_extended.csv', na_rep = '#N/A') 
mono_score_words = ms.loc[ms['source'] == '3k']
low_3k_score_words = mono_score_words.head(300)
low_3k_score_words.to_csv('low_3k_score_words.csv', na_rep = '#N/A')