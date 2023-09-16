import numpy as np
import gzip
import json
import glob

folder = "/data/projects/swat/users/eladco/models/TpuGraphs/12_best/*"

acc_metric_total = 0
for file in list(glob.glob(folder)):
    if file.endswith("jsonz"):
        with gzip.open(open(file, 'rb'), 'rb') as fin:
            json_data = json.loads(fin.read().decode())
            json_data.pop('test_predictions',None)
            print("----------------------------------------------------------------------------------------------------------------------------")
            print(file)
            print("----------------------------------------------------------------------------------------------------------------------------")
            acc_metric = np.max(json_data['train_curve']['acc_metric'])
            acc_metric_total += acc_metric
            print(acc_metric)
            print(json_data)

print("ACC: ", acc_metric_total/5)