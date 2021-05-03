import argparse
import os
import pyro
import pyro.params.param_store as param_store
import pandas as pd
import numpy as np
from pyro_models import *
import random
import json

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--attribute', nargs='+',  type=str, help='Attributes supposed to be modified')
parser.add_argument('-v', '--att_value', nargs='+', type=int, help='Attribute values for the attributes we want to modify')
parser.add_argument('-d', '--dataroot', type=str, default='../Dataset')
parser.add_argument('-i', '--img', type=str, nargs='+', default=None, help='e.g., --img 182638')
parser.add_argument('-g', '--graph', type=str, help='parameters from pyro')
parser.add_argument('-e', '--experiment_name', help='experiment_name')
parser.add_argument('-t', '--times', type=int)

arg = parser.parse_args()

target_att = arg.attribute
target_att_list = arg.att_value
dataroot = arg.dataroot
imgs = arg.img

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

dataset = pd.read_csv("list_attr_celeba.txt", sep=' ', header=1, skiprows=0)


def get_diff_att(model, img, times):
    '''
    Take supposed modified attributes and return the corresponding label list
    '''

    # dictionary for attributes which will be modified in the original picture
    target_modified_att = {}

    # Check the difference for the target attributes
    for idx, each_att in enumerate(target_att):
        if dataset[each_att][img + 1] != target_att_list[idx]:
            target_modified_att[each_att] = target_att_list[idx]

    res_list = []
    intensity_list = []

    for time in range(times):
        cur_list = [i for i in target_modified_att.copy().keys()]
        for each_target_att, value in target_modified_att.items():
            if each_target_att == 'Mustache':
                parent_dict_prob = model.get_conditional_prob(Mustache=value)
            for key, prob in parent_dict_prob.items():
                sample_res = np.random.binomial(n=1, p=prob)
                sample_res = sample_res*2 - 1

                joint_prob = model.get_joint_prob(Mustache=value, Male=dataset[key][img + 1])
                if joint_prob > 0.7:
                    continue
                if dataset[key][img + 1] != sample_res:
                    cur_list.append(key)

        res_list.append(cur_list)
        intensity_list.append(np.random.rand(len(cur_list)).tolist())

    return res_list, intensity_list


if __name__ == '__main__':

    target_modified_att_list = []
    intensity_list = []
    res_dict = {}

    for each_img in imgs:
        each_att_list, each_intensity_list = get_diff_att(model, each_img, times=times)
        target_modified_att_list += each_att_list
        intensity_list += each_intensity_list

    res_dict['target_modified_att_list'] = target_modified_att_list
    res_dict['times'] = times
    res_dict['imgs'] = imgs
    res_dict['experiment'] = experiment
    res_dict['intensity_list'] = intensity_list

    with open('interface_param.json', 'w') as f:
        json.dump(res_dict, f)

    os.system('python stgan.py')