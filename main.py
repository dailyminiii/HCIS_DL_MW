import numpy as np
import torch
import argparse
from copy import deepcopy
import csv

import model
import exp
import dataset
import preprocess

# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Data Loading ====== #
#args.batch_size = 64
args.x_frames = 1
args.y_frames = 3 # the number of classes


# ====== Model Capacity ===== #
# args.input_dim = [5,62,6,63]
#args.input_dim = 5
args.hid_dim = 10
#args.n_layers = 1
args.n_filters = 64
args.filter_size = 1
args.str_len = 1

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.5
args.use_bn = True

# ====== Optimizer & Training ====== #
#args.optim = 'Adam'
#args.model = 'Conv1D'
#args.lr = 0.001
#args.epoch = 10

# ====== Experiment Variable ====== #
name_var1 = 'lr'
name_var2 = 'n_layers'
name_var3 = 'data'
name_var4 = 'input_dim'
name_var5 = 'model'
name_var6 = 'optim'
name_var7 = 'batch_size'
name_var8 = 'epoch'
list_var1 = [0.001, 0.01, 0.1]
list_var2 = [1,2,3]
list_var3 = [["data/ProcessedData_1.csv", 5], ["data/ProcessedData_2.csv", 62], ["data/ProcessedData_3.csv", 6], ["data/ProcessedData_4.csv", 63]]
list_var4 = ['LSTM', 'Conv1D', 'ConvLSTM']
# list_var5 = ['SGD', 'RMSprop', 'Adam']
list_var5 = ['Adam']
list_var6 = [64]
list_var7 = [10]

md = int(input("Enter the number of mode(1: preprocess, 2: exp): "))

if md == 1:
    preprocess.preprocess()

elif md == 2:

    for var1 in list_var1:
        for var2 in list_var2:
            for var3 in list_var3:
                for var4 in list_var4:
                    for var5 in list_var5:
                        for var6 in list_var6:
                            for var7 in list_var7:
                                setattr(args, name_var1, var1)
                                setattr(args, name_var2, var2)
                                setattr(args, name_var3, var3[0])
                                setattr(args, name_var4, var3[1])
                                setattr(args, name_var5, var4)
                                setattr(args, name_var6, var5)
                                setattr(args, name_var7, var6)
                                setattr(args, name_var8, var7)
                                print(args)

                                # trainset, valset, testset 8 : 1 :1로 나누기
                                trainset = dataset.NumDataset(args.data, args.x_frames, args.y_frames, args.str_len)
                                train_num = round(len(trainset) * 0.8)
                                print('train_num :', train_num)
                                train_num = train_num // args.batch_size
                                train_num = train_num * args.batch_size
                                val_num = round(len(trainset) * 0.1)
                                print('val_num :', val_num)
                                test_num = len(trainset) - train_num - val_num
                                trainset, valset, testset = torch.utils.data.random_split(trainset, [train_num, val_num, test_num])

                                partition = {'train': trainset, 'val': valset, 'test': testset}
                                setting, result = model.experiment(partition, deepcopy(args))
                                exp.save_exp_result(setting, result)

                                #print('result :', result)
                                print('Settings:', setting)
                                print('train_accuracy :', result['train_acc'])
                                print('val_accuracy :', result['val_acc'])

                                with open('results.csv', 'w', newline='') as f:
                                    writer = csv.writer(f)
                                    for k, v in setting.items():
                                        writer.writerow([k, v])
                                    for k, v in result.items():
                                        writer.writerow([k, v])
                                    f.close()

                                list = []
                                list2 = []

                                with open('results.csv', 'r') as f:
                                    reader = csv.reader(f)
                                    for row in reader:
                                        list.append(row)
                                    f.close()

                                list2.append(list[11])
                                list2.append(list[12])
                                list2.append(list[13])
                                list2.append(list[15])
                                list2.append(list[16])
                                list2.append(list[17])
                                list2.append(list[22])
                                list2.append(list[23])
                                list2.append(list[24])


                                with open('210927_3_results.csv', 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(list2)
                                    f.close()


    var_1 = 'lr'
    var_2 = 'n_layers'
    var_3 = 'data'
    var_4 = 'input_dim'
    var_5 = 'model'
    var_6 = 'optim'
    var_7 = 'batch_size'
    var_8 = 'epoch'
    df = exp.load_exp_result('exp1')

    # exp.plot_acc(var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, df)
    # exp.plot_loss_variation(var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, df, sharey=False)
    # exp.plot_acc_variation(var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, df, margin_titles=True, sharey=True)
else:
    raise ValueError('In-valid mode choice')
