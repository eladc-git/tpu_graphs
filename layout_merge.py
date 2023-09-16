import os
import pandas as pd
import glob
import tensorflow as tf


folder = "/data/projects/swat/users/eladco/models/TpuGraphs/v0.4.0_merge_xla_random"
#N = 17  # nlp_default
N = 8  # xla_random

K = 1000
merge_list = [("", {}) for i in range(N)]
first_file = True
# Scan all results
for file in list(glob.glob(os.path.join(folder,"*"))):
    if file.endswith(".csv"):
        df = pd.read_csv(file)
        for i in range(len(df)):
            id = df.loc[i]['ID']
            topConfigs = df.loc[i]['TopConfigs']
            if first_file:
                merge_list[i] = (id, {})
            merge_dict = merge_list[i][1]
            for k,config in enumerate(topConfigs.split(';')):
                if config in merge_dict:
                    merge_dict[config] += K-k
                else:
                    merge_dict.update({config: K-k})
        first_file = False

# Create merge results
results_csv = os.path.join(folder,"results_merged_layout.csv")

with tf.io.gfile.GFile(results_csv, 'w') as fout:
    fout.write('ID,TopConfigs\n')
    for (id, merge_dict) in merge_list:
        new_dict = {k: v for k, v in sorted(merge_dict.items(), reverse=True, key=lambda item: item[1])}
        fout.write(f'{id},')
        for i in range(K-1):
            a = list(new_dict.keys())[i]
            fout.write(f'{a};')
        a = list(new_dict.keys())[K-1]
        fout.write(f'{a}')
        fout.write(f'\n')
print('\n\n   ***  Wrote', results_csv, '\n\n')


