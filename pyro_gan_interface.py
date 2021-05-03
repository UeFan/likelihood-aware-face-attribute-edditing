import argparse
import os
import pyro
import pyro.params.param_store as param_store
import pandas as pd
import numpy as np
from pyro_models import *
import random
import json
import torch


att_name_list = ["sex", "young", "mustache", "makeup", "ear", "bag"]

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--attribute', nargs='+', type=str, help='Attributes supposed to be modified (0-5)')
parser.add_argument('-v', '--att_value', nargs='+', type=str,  help='Attribute values for the attributes we want to modify (0/1)')
# parser.add_argument('-d', '--dataroot', type=str, default='../Dataset')
parser.add_argument('-i', '--img', type=str, nargs='+', default=None, help='e.g., --img 182638')
parser.add_argument('-g', '--graph', type=str, default='./pyro_params', help='parameters from pyro')
parser.add_argument('-e', '--experiment_name', help='experiment_name')
parser.add_argument('-t', '--times', type=int, default=1)


arg = parser.parse_args()

target_att = arg.attribute
target_att_value_ = arg.att_value
target_att_value = {}
for i in range(len(target_att)):
    target_att_value[int(target_att[i])] = float(target_att_value_[i])

# dataroot = arg.dataroot
img = arg.img
imgs =[int(i) for i in img]


def get_random_imgs(num):
    imgs_list = range(1, 202600)
    res_list = random.sample(imgs_list, num)
    return res_list

if imgs is None:
    imgs = get_random_imgs(1000)
experiment = arg.experiment_name
times = arg.times

pyro.clear_param_store()
graph_params = param_store.ParamStoreDict()
graph_params.load(arg.graph)

model = sex_mus_model(graph_params)

dataset = pd.read_csv("list_attr_celeba.txt", sep = ' ', header=1, skiprows = 0)
data = np.vstack((dataset['Male'], dataset["Young"],
                      np.minimum(dataset["Mustache"] + (1 - dataset["No_Beard"]) + dataset["Goatee"], 1),
                      dataset["Heavy_Makeup"], dataset["Bags_Under_Eyes"], dataset["Wearing_Earrings"]))
data = np.maximum(data, 0)[:,imgs]
# print("0:Male, 1:Young, 2:Mustache, 3:Heavy makeup, 4:Bags_under_eyes, 5:Wear earrings")

def get_diff_att(model, idx_img, times):
    '''
    Take supposed modified attributes and return the corresponding label list
    '''

    def decimalToBinary(n):
        return bin(n).replace("0b", "")



    scale_log_joint = model.make_log_joint()

    res_list = []
    intensity_list = []

    # dictionary for attributes which will be modified in the original picture
    target_att_needed_mod = {}

    # Check the difference for the target attributes
    att_ori = {}
    att_modified = {}
    for i in range(len(att_name_list)):
        att_ori[att_name_list[i]] = torch.tensor(float(data[i][idx_img]))
        att_modified[att_name_list[i]] = torch.tensor(float(data[i][idx_img]))

        if i in target_att_value.keys():
            att_modified[att_name_list[i]] = torch.tensor(target_att_value[i])

    if scale_log_joint(att_modified).exp().item() / scale_log_joint(att_ori).exp().item() < 0.5:
        num_of_rand_sample_var = len(att_name_list) - len(target_att_value.keys())
        binery_list = [[c for c in decimalToBinary(i)] for i in range(2 ** num_of_rand_sample_var)]
        for time in range(times):

            prob_list = []
            att_modified_sample_result = []
            for b in binery_list: # Each case
                b_ = [c for c in ('0' * num_of_rand_sample_var)]
                b_[-len(b):] = b
                b = b_
                att_modified_ = {}
                ii = 0
                for i in range(len(att_name_list)):
                    if(i in target_att_value.keys()):
                        att_modified_[att_name_list[i]] = att_modified[att_name_list[i]]
                    else:
                        att_modified_[att_name_list[i]] = torch.tensor(float(b[ii]))
                        ii+=1
                att_modified_sample_result.append(att_modified_)
                prob_list.append(scale_log_joint(att_modified_).exp().item())
            prob_list = (prob_list / np.sum(prob_list)).tolist()
            sample_idx = np.random.choice(a=np.arange(len(prob_list)).tolist(), size=1, p=prob_list)[0]
            res_list.append(att_modified_sample_result[int(sample_idx)])


    else:
        for time in range(times):
            res_list.append(att_modified)

    print(att_ori)
    print(res_list)
    print()
    for time in range(times):
        intensity_list.append(np.random.rand(len(att_name_list)).tolist())



    return res_list, intensity_list


if __name__ == '__main__':

    modified_att_list = []
    intensity_list = []
    res_dict = {}

    for i in range(len(imgs)):
        each_att_list, each_intensity_list = get_diff_att(model, i, times=times)
        modified_att_list.append(each_att_list)
        intensity_list += each_intensity_list



    res_dict['modified_att_list'] = modified_att_list
    res_dict['times'] = times
    res_dict['imgs'] = imgs
    res_dict['experiment'] = experiment
    res_dict['intensity_list'] = intensity_list

    print(res_dict)
    # with open('interface_param.json', 'w') as f:
    #     json.dump(res_dict, f)
    #
    # os.system('python stgan.py')