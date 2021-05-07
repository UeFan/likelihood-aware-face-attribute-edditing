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
from itertools import product

def get_random_imgs(num):
    imgs_list = range(1, 202600)
    res_list = random.sample(imgs_list, num)
    return res_list

att_name_list = ["sex", "young", "mustache", "beard", "bald"]
output_att_name_list =  {"sex":"Male", "young":"Young", "mustache":"Mustache", "beard":"No_Beard", "bald":"Bald"}
parser = argparse.ArgumentParser()

parser.add_argument('-a', '--attribute', nargs='+', type=str, help='Attributes supposed to be modified (0-4)')
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
# imgs = get_random_imgs(100)

#
# if imgs is None:
#     imgs = get_random_imgs(100)

experiment = arg.experiment_name
times = arg.times

pyro.clear_param_store()
graph_params = param_store.ParamStoreDict()
graph_params.load(arg.graph)

model = sex_mus_model(graph_params)

dataset = pd.read_csv("list_attr_celeba.txt", sep = ' ', header=1, skiprows = 0)
data = np.vstack((dataset['Male'], dataset["Young"], dataset["Mustache"], (-dataset["No_Beard"]), dataset["Bald"]))

data = np.maximum(data, 0)[:,np.array(imgs) - 1]
# print("0:Male, 1:Young, 2:Mustache, 3:Heavy makeup, 4:Bags_under_eyes, 5:Wear earrings")




def get_diff_att(model, idx_img, times, pyro_effect_time):
    '''
    Take supposed modified attributes and return the corresponding label list
    '''
    def attribute_needed_to_flip(ori,new):
        att_list = []
        for k in ori:
            if ori[k] != new[k]:
                att_list.append(output_att_name_list[k])
        # if ori['sex'] == 1 and "Male" not in att_list:
        #     print("!!!!!!")
        #     att_list.append(output_att_name_list['sex'])
        return att_list

    def decimalToBinary(n):
        return bin(n).replace("0b", "")



    scale_log_joint = model.make_log_joint()

    res_list = []
    intensity_list = []
    ori_dic_list = []
    modified_dic_list = []
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
        pyro_effect_time += 1
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
            # print(prob_list)
            sample_idx = np.random.choice(a=np.arange(len(prob_list)).tolist(), size=1, p=prob_list)[0]

            ori_dic_list.append(att_modified)
            modified_dic_list.append(att_modified_sample_result[int(sample_idx)])
            res_list.append( attribute_needed_to_flip(att_ori, att_modified_sample_result[int(sample_idx)]) )
            intensity_list.append(np.random.rand(len(res_list[-1])).tolist())

    else:
        for time in range(times):
            ori_dic_list.append(att_modified)
            modified_dic_list.append(att_modified)
            res_list.append(attribute_needed_to_flip(att_ori, att_modified))
            intensity_list.append(np.random.rand(len(res_list[-1])).tolist())
    # print(att_ori)
    # print(res_list)
    # print()




    return res_list, intensity_list, ori_dic_list, modified_dic_list, pyro_effect_time


def get_RMSE_of_condition_prob(data_new):
    ori_joint_prob_array = np.array([7.32037177e-02, 1.97434341e-05, 2.02370199e-04, 0.00000000e+00,
                                     4.93585852e-06, 0.00000000e+00, 4.93585852e-06, 0.00000000e+00,
                                     5.09262139e-01, 6.41661607e-05, 4.78778276e-04, 0.00000000e+00,
                                     4.93585852e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                     8.62985503e-02, 1.04344049e-02, 3.10317425e-02, 4.33861964e-03,
                                     5.57752013e-04, 6.91020193e-05, 1.78332568e-02, 2.38401966e-03,
                                     1.52745078e-01, 1.75222977e-03, 8.60764367e-02, 2.54690300e-03,
                                     4.83714135e-04, 3.94868681e-05, 1.93683088e-02, 7.94673221e-04])

    perm_array = np.array(list(product([0, 1], repeat=5)))
    generated_joint_prob_array = []

    for arr in perm_array:
        p = ((data_new.T == arr).sum(axis=1) == 5).mean()
        generated_joint_prob_array.append(p)

    generated_joint_prob_array = np.array(generated_joint_prob_array)

    def prob_edited_attr(prob_a):
        if (len(target_att) == 3):
            return prob_a[np.where(np.logical_and(np.logical_and(perm_array[:, int(target_att[0])] == float(target_att_value_[0]),
                                                  perm_array[:, int(target_att[1])] == float(target_att_value_[1])),
                                                  perm_array[:, int(target_att[2])] == float(target_att_value_[2])))[0]]
        if (len(target_att) == 2):
            return prob_a[np.where(np.logical_and(perm_array[:, int(target_att[0])] == float(target_att_value_[0]),
                                                  perm_array[:, int(target_att[1])] == float(target_att_value_[1])))[0]]
        if (len(target_att) == 1):
            return prob_a[np.where(perm_array[:, int(target_att[0])] == float(target_att_value_[0]))[0]]
        # return prob_a[np.where(np.logical_and(perm_array[:, 2] == 1, perm_array[:, 4] == 1))[0]]

    #
    print()
    print("No.1")
    print(prob_edited_attr(ori_joint_prob_array))

    print(prob_edited_attr(generated_joint_prob_array))
    # print(prob_edited_attr(generated_joint_prob_array).sum())

    generated_conditoned_prob_array = prob_edited_attr(generated_joint_prob_array) / prob_edited_attr(
        generated_joint_prob_array).sum()
    ori_conditoned_prob_array = prob_edited_attr(ori_joint_prob_array) / prob_edited_attr(ori_joint_prob_array).sum()

    return np.mean((generated_conditoned_prob_array - ori_conditoned_prob_array) ** 2) ** 0.5
    # return generated_conditoned_prob_array

if __name__ == '__main__':

    ori_dic_list_list = []
    modified_dic_list_list = []
    modified_att_list = []
    intensity_list = []
    res_dict = {}
    pyro_effect_time = 0
    for i in range(len(imgs)):
        each_att_list, each_intensity_list, ori_dic_list, modified_dic_list, pyro_effect_time = get_diff_att(model, i, times, pyro_effect_time)
        modified_att_list += each_att_list
        intensity_list += each_intensity_list

        ori_dic_list_list += ori_dic_list
        modified_dic_list_list += modified_dic_list

    # a = modified_att_list
    # count = 0
    # for i in a:
    #     for j in i:
    #         if j == 'Male':
    #             count += 1
    # print(count / len(a))

    res_dict['modified_att_list'] = modified_att_list
    res_dict['times'] = times
    res_dict['imgs'] = imgs
    res_dict['experiment'] = experiment
    res_dict['intensity_list'] = intensity_list

    # print(ori_dic_list_list)
    # print(modified_dic_list_list)

    ori_list = []
    modified_list = []

    for i in range(len(ori_dic_list_list)):
        ori_list_ = []
        modified_list_ = []
        for name in att_name_list:
            ori_list_.append(ori_dic_list_list[i][name].item())
            modified_list_.append(modified_dic_list_list[i][name].item())
        ori_list.append(ori_list_)
        modified_list.append(modified_list_)

    np.save("./ori_list.npy", ori_list)
    np.save("./modified_list.npy", modified_list)

    ori_list = np.array(ori_list).T
    modified_list = np.array(modified_list).T

    print(get_RMSE_of_condition_prob(ori_list))
    print(get_RMSE_of_condition_prob(modified_list))
    print(pyro_effect_time)

    # print(res_dict)
    # with open('interface_param.json', 'w') as f:
    #     json.dump(res_dict, f)
    #
    # os.system('python stgan.py')
