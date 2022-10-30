# TODO popuniti kodom za problem 2a
#zadatak 2a
import pandas as pd
import numpy as np
from datetime import timedelta

print('zadatak 2a')
#https://stackoverflow.com/questions/31645466/give-column-name-when-read-csv-file-pandas
#https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-pandas-dataframe

column_names= ['datum pocetka','vreme pocetka','datum kraja','vreme kraja','pm2amb']
colnames=['datum', 'vreme', 'temp', 'vlaznost', 'pm2konc']

podaci_skupi_senzori = pd.read_csv("/content/skupi_senzori.XLS",sep ='\t',usecols=[0, 1, 2, 3, 6],names = column_names,header=None,skiprows=1)
podaci_jeftini_senzori = pd.read_csv("/content/jeftini_senzori.TXT", sep=';', usecols=[1, 2, 4, 5, 8],names=colnames, header=None)

podaci_jeftini_senzori['datum'] = podaci_jeftini_senzori['datum'] + ' ' + podaci_jeftini_senzori['vreme']
podaci_jeftini_senzori['datum'] = pd.to_datetime(podaci_jeftini_senzori['datum']).dt.strftime('%m-%d-%y %H:%M:%S')
podaci_jeftini_senzori['datum'] = pd.to_datetime(podaci_jeftini_senzori['datum']) + pd.DateOffset(hours=0)
podaci_jeftini_senzori.drop('vreme', axis='columns', inplace=True)

podaci_skupi_senzori['datum pocetka'] = podaci_skupi_senzori['datum pocetka'] + ' ' + podaci_skupi_senzori['vreme pocetka']
podaci_skupi_senzori['datum pocetka'] = pd.to_datetime(podaci_skupi_senzori['datum pocetka']).dt.strftime('%d-%m-%y %H:%M:%S')
podaci_skupi_senzori['datum pocetka'] = pd.to_datetime(podaci_skupi_senzori['datum pocetka']) + pd.DateOffset(hours=-2)
podaci_skupi_senzori.drop('vreme pocetka', axis='columns', inplace=True)

podaci_skupi_senzori['datum kraja'] = podaci_skupi_senzori['datum kraja'] + ' ' + podaci_skupi_senzori['vreme kraja']
podaci_skupi_senzori['datum kraja'] = pd.to_datetime(podaci_skupi_senzori['datum kraja']).dt.strftime('%d-%m-%y %H:%M:%S')
podaci_skupi_senzori['datum kraja'] = pd.to_datetime(podaci_skupi_senzori['datum kraja']) + pd.DateOffset(hours=-2)
podaci_skupi_senzori.drop('vreme kraja', axis='columns', inplace=True)

dataframe = pd.DataFrame(columns = ['datum','temp','vlaznost','pm2konc','datum pocetka','datum kraja','pm2amb'])

i = 0

for index, row in podaci_skupi_senzori.iterrows():
    result = podaci_jeftini_senzori[np.logical_and(row['datum pocetka'] <= podaci_jeftini_senzori['datum'], podaci_jeftini_senzori['datum'] < row['datum kraja'])]
    if result.empty:
        continue

    for j, r in result.iterrows():
        dataframe.loc[i] = [r['datum'], r['temp'], r['vlaznost'], r['pm2konc'], row['datum pocetka'], row['datum kraja'], row['pm2amb']]
        i += 1


dataframe['pm2amb'] = dataframe['pm2amb'].str.replace(',', '.')
dataframe.to_csv('./2a.xls', index=False, header=True, encoding="utf-8", sep="\t")

# print(podaci_skupi_senzori)
# print("####")
# print(podaci_jeftini_senzori)