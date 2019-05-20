import pandas as pd
aoa = pd.read_excel ("word-level_data/aoa.xlsx")
tasa = pd.read_excel ("word-level_data/tasa_1995.xlsx")
mono = pd.read_csv ("words/monoSort.csv", index_col = 1, names = ["source","word"], header = 1)
#print (mono)

freq = pd.read_csv("word-level_data/lcnl_kid_words.csv")
mono_freq = mono.merge(freq, on = 'word', how = 'left')
mono_aoa = mono_freq.merge(aoa, on = 'word', how = 'left')
mono_tasa = mono_aoa.merge (tasa, on = 'word', how = 'left')
#print (mono_tasa)

x =pd.isnull(mono_tasa.iloc[:, 1:7])
NoData = x.all(axis=1)
y = ~x
FullData = y.all(axis=1)
mono_wData = pd.DataFrame()
mono_wData = mono_tasa.loc[~NoData,:]
mono_wData.to_csv('words/mono_wData.csv')

mono_3k = pd.read_csv ('words/3k.csv', header = None, names = ["word"])
mono_3k = mono_3k.merge(aoa, on = 'word', how = 'inner')
mono_3k.shape 

mono_partial = pd.DataFrame()
mono_partial = mono_wData[mono_wData.aoa_mean.notnull()]
mono_partial.to_csv('mono_partial.csv')

monofull= pd.DataFrame()
monofull= mono_wData.dropna()
monofull.to_csv('monofull.csv')

monofull_use = monofull.sort_values(by=['aoa_mean'])
mono_partial_use = mono_partial.sort_values(by=['aoa_mean'])
monofull300 = pd.DataFrame()
monofull300 = monofull_use.head(300)
monofull300.to_csv('monofull300.csv')

monofull500 = pd.DataFrame()
monofull500 = monofull_use.head(500)
monofull500.to_csv('monofull500.csv')

monofull1000 = pd.DataFrame()
monofull1000 = monofull_use.head(1000)
monofull1000.to_csv('monofull1000.csv')

mono_partial300 = pd.DataFrame()
mono_partial300 = mono_partial_use.head(300)
mono_partial300.to_csv('mono_partial300.csv')

mono_partial500 = pd.DataFrame()
mono_partial500 = mono_partial_use.head(500)
mono_partial500.to_csv('mono_partial500.csv')

mono_partial1000 = pd.DataFrame()
mono_partial1000 = mono_partial_use.head(1000)
mono_partial1000.to_csv('mono_partial1000.csv')

