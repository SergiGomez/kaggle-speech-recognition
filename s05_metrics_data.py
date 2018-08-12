#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:55:27 2018

@author: sergigomezpalleja
"""

import pandas as pd
import numpy as np

def get_err_perc(conf_matrix):
    labels_names = ['silence','unknown','yes','no','up','down','left','right','on','off','stop','go']
    df_errors = pd.DataFrame(columns = ['label','perc_missed'])
    labels = []
    missed_values = []
    sum_missed = 0
    sum_row = 0
    for i in range(0,len(labels_names)):
        labels.append(labels_names[i])
        for j in range(0,len(labels_names)):
            sum_row += conf_matrix[i,j]
            if i != j:
                sum_missed += conf_matrix[i,j]     
        missed_values.append(sum_missed*100/sum_row)
        sum_row = 0
        sum_missed = 0
    df_errors['label'] = labels
    df_errors['perc_missed'] = missed_values
    
    sum_col = 0
    sum_err = 0
    labels = []
    err_pred_values =[]
    for j in range(0,len(labels_names)):
        labels.append(labels_names[j])
        for i in range(0,len(labels_names)):
            sum_col += conf_matrix[i,j]
            if j != i:
                sum_err += conf_matrix[i,j] 
        err_pred_values.append(sum_err*100/sum_col)
        sum_col = 0
        sum_err = 0
    df_errors['perc_error'] = err_pred_values
    return(df_errors)
    
#More unknowns
print('change: unknowns from 10% to 15% and silence from 10% to 5%')
conf_matrix_more_unknowns =  np.array([[129,0,0,0,0,0,   0,   0,   0,   0,   0,   0],
                                [  2, 315,   5,   2,   4,  11,  5,  10,   8,   3,   4,  17],
                                [  0,   7, 238,   2,   1,   0,   7,   1,   0,   0,   0,   0],
                                [  1,   6,   1, 214,   0,   6,   1,  3,   1,   0,   2,  17],
                                [  0,   4,   0,   0, 254,   1,   4,   0,   0,   3,   5,   1],
                                [  2,  10,   0,  11,   1, 215,   1,   0,   1,   0,   2,  10],
                                [  0,   3,  20,   1,   5,   0, 236,   2,   0,   0,   0,   0],
                                [  1,  17,   0,   1,   3,   0,   1, 233,   0,   1,   0,   2],
                                [  0,   8,   0,   0,   0,   1,   1,   2, 232,   1,   1,   0],
                                [  1,   6,   0,   0,  14,   0,   2,   2,   5, 228,   3,   1],
                                [  0,   1,   1,   0,   8,   1,   2,   0,   0,   2, 234,   0],
                                [  0,  13,   0,  23,   4,   5,   4,   1,   0,   0,   2, 199]])   

err_more_unknowns = get_err_perc(conf_matrix_more_unknowns)
    
#more noise
print('change: volume of noise from 0.1 to 0.2')
conf_matrix_more_noise = np.array([[257,   0,   0,   0,   0,   0,   0 ,  0,    0,   0,   0,   0],
                                    [  1, 189,   3,   5,   6,   8,   6,  12,   8,   1,   3,  15],
                                    [  1,   8, 230,   1,   1,   4,   8,   2,   0,   0,   1,   0],
                                    [  1,   5,   0, 214,   3,   9,   3,   2,   1,   0,   1,  13],
                                    [  0,   3,   0,   0, 256,   2,   3,   0,   0,   1,   5,   2],
                                    [  3,   4,   0,  14,   2, 219,   2,   0,   0,   0,   0,   9],
                                    [  0,   4,  12,   0,   5,   1, 243,   2,   0,   0,   0,   0],
                                    [  1,   9,   0,   0,   2,   1,   2, 241,   0,   2,   0,   1],
                                    [  0,   8,   0,   0,   1,   1,   1,   2, 231,   1,   0,   1],
                                    [  1,   4,   0,   0,  17,   1,   3,   0,   8, 224,   3,   1],
                                    [  0,   2,   0,   0,   7,   4,   1,   0,   0,   3, 231,   1],
                                    [  0,   9,   1,  25,   5,   8,   5,   2,   1,   1,   3, 191]])   

err_more_noise = get_err_perc(conf_matrix_more_noise)

#lower learning rate
conf_matrix_lower_learning_rate = np.array([[257,   0,   0,   0,  0,   0,   0,   0,   0,   0,   0,   0],
                                             [  1, 193,   5,   3,   8,   7,   4,  11,  10,   1,   4,  10],
                                             [  1,   5, 236,   0,   1,   1,  11,   1,   0,   0, 0,   0],
                                             [  1,   3,   1, 212,   3,   9,   5,   2,   0,   0,  2,  14],
                                             [  0,   1,   0,   0, 257,   0,   5,   0,   0,   3,   5,   1],
                                             [  2,   8,   0,  13,   2, 213,   2,   0,   2,   0,   0,  11],
                                             [  0,   1,  15,   1,   3,   0, 245,   1,   0,   0,   1,   0],
                                             [  1,  10,   1,   1,   1,   1,   4, 239,   0,   1,   0,   0],
                                             [  0,   9,   0,   0,   1,   2,   2,   1, 230,   1,   0,   0],
                                             [  1,   2,   0,   0,  14,  1,  2,   3,   7, 229,   3,   0],
                                             [  0,   1,   0,   0,   8,   1,   2,   0,   0,   2, 233,   2],
                                             [  0,  11,   0,  29,   4,   5,   7,   2,   0,   0,   1, 192]])
err_lower_lr = get_err_perc(conf_matrix_lower_learning_rate)

# energy coef
conf_matrix_energy_coef = np.array([[  0,   0,   0,   0, 257,   0,   0,   0,   0,   0,   0,   0],
                                    [  3, 171,   6,   9,   4,   6,   8,  17,   7,   1,  11,  14],
                                    [  2,   6, 228,   5,   1,   1,   7,   2,   0,   0,   4,   0],
                                    [  1,   5,   1, 199,   2,  14,   3,   2,   0,   0,   3,  22],
                                    [  2,   1,   0,   0, 241,   0,   7,   4,   0,   4,  10,   3],
                                    [  3,   1,   1,  20,   2, 211,   2,   0,   0,   0,   4,   9],
                                    [  2,   2,  21,   2,   5,  0, 230,   3,  0,   0,   2,   0],
                                    [  2,  12,   0,   2,   0,  0,   5, 238,   0,   0,   0,   0],
                                    [  0,   6,   0,   0,   2,   1,   1,   3, 220,  12,   0,   1],
                                    [  1,   3,   0,   0,  18,   2,   2,   2,   9, 215,  10,   0],
                                    [  0,   1,   0,   1,  12,   3,   1,   0,   0,   2, 226,   3],
                                    [  1,  7,   1, 34,   2,  12,   8,   4,   0,   0, 16, 166]])
err_energy_coef = get_err_perc(conf_matrix_energy_coef)

#new model --> cnn_tpool2
conf_matrix_cnn_tpool2 = np.array([[289,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                    [  1, 196,   7,   6,  10,   7,  12,  16,  10,   2,   6,  16],
                                    [  0,   1, 269,   4,   1,   2,  12,   1,   1,   0,   1,   0],
                                    [  0,   2,   2, 237,   3,   9,   5,   2,   1,   0,   3, 21],
                                    [  2,   5,   0,   0, 276,   0,  5,  2,   5,      3,   4,   1],
                                    [  1,   4,   2,  26,   1, 253,  2,   0,   0,   0,   1,  12],
                                    [  1,   2,  19,   2,   3,   0, 254,   3,   0,   0,   1,   0],
                                    [  0,  13,   0,   1,   2,   2,   4, 262,   1,   1,   0,   0],
                                    [  1,   4,   0,   0,   3,   2,   1,   4, 271,   2,   0,   1],
                                    [  0,   3,   0,   0,  27,   0,   3,   2,  11, 224,   0,   1],
                                    [  0,   2,   0,   1,  16,   3,   4,   0,   0,   1, 249,   0],
                                    [  5,   6,   0,  35,   5,   6,   2,   6,   3,   0,   0, 226]])
 
print('More unknowns')
print(err_more_unknowns)
print('more noise')
print(err_more_noise)
print('lower learning rate')
print(err_lower_lr)
print('energy coef')
print(err_energy_coef)

print('More unknowns missed')
print(np.sum(err_more_unknowns['perc_missed']))
print('more noise missed')
print(np.sum(err_more_noise['perc_missed']))
print('lower learning rate missed')
print(np.sum(err_lower_lr['perc_missed']))
print('energy coef missed')
print(np.sum(err_energy_coef['perc_missed']))

print('More unknowns errors')
print(np.sum(err_more_unknowns['perc_error']))
print('more noise errors')
print(np.sum(err_more_noise['perc_error']))
print('lower learning rate errors')
print(np.sum(err_lower_lr['perc_error']))
print('energy coef rate errors')
print(np.sum(err_energy_coef['perc_error']))


#len_train = len(data_index['training'])
#len_dev = len(data_index['development'])
#len_test = len(data_index['testing'])
#len_total = len_train + len_dev + len_test
#len_train / len_total
#len_dev / len_total 
#len_test / len_total
#
##Percentage of each label
#  #Training set
#data_train = data_index['training']
#  #Dev set
#data_dev = data_index['development']  
#  #Test set
#data_test = data_index['testing']  
#
#labels_train  = pd.DataFrame(columns=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','unknown','silence'])
#labels_dev  = pd.DataFrame(columns=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','unknown','silence'])
#labels_test  = pd.DataFrame(columns=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','unknown','silence'])

