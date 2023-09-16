import pandas as pd
import glob
import tensorflow as tf
import os

folder = "/data/projects/swat/users/eladco/models/TpuGraphs/v0.4.1"
N = 844

merge_list = [("", {}) for i in range(N)]
first_file = True
# Scan all results
for file in list(glob.glob(folder+"/*")):
    if file.endswith(".csv"):
        df = pd.read_csv(file)
        for i in range(len(df)):
            id = df.loc[i]['ID']
            topConfigs = df.loc[i]['TopConfigs']
            if first_file:
                merge_list[i] = (id, {})
            merge_dict = merge_list[i][1]
            for config in topConfigs.split(';'):
                if config in merge_dict:
                    merge_dict[config] += 1
                else:
                    merge_dict.update({config: 1})
        first_file = False

# Create merge results
results_csv = os.path.join(folder, "results_merged_tile.csv")

with tf.io.gfile.GFile(results_csv, 'w') as fout:
    fout.write('ID,TopConfigs\n')
    for (id, merge_dict) in merge_list:
        new_dict = {k: v for k, v in sorted(merge_dict.items(), reverse=True, key=lambda item: item[1])}
        a = list(new_dict.keys())[0]
        b = list(new_dict.keys())[1] if len(list(new_dict.keys()))>1 else list(new_dict.keys())[-1]
        c = list(new_dict.keys())[2] if len(list(new_dict.keys()))>2 else list(new_dict.keys())[-1]
        d = list(new_dict.keys())[3] if len(list(new_dict.keys()))>3 else list(new_dict.keys())[-1]
        e = list(new_dict.keys())[4] if len(list(new_dict.keys()))>4 else list(new_dict.keys())[-1]
        fout.write(f'{id},{a};{b};{c};{d};{e}\n')
print('\n\n   ***  Wrote', results_csv, '\n\n')


