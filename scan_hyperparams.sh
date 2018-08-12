#!/bin/bash

#training_steps="2,1"
training_steps="8000,2000"
training_steps2="15000,5000"
name1="cnn_tpool2_lr9_0.00018_0.000018"
param1="--name_model=cnn_tpool2_lr9 --learning_rate=0.00018,0.000018 --how_many_training_steps=${training_steps}"
name2="cnn_tpool2_lr10_0.0001_0.00001"
param2="--name_model=cnn_tpool2_lr10 --learning_rate=0.0001,0.00001 --how_many_training_steps=${training_steps}"
name3="cnn_tpool2_mel_coef1"
param3="--name_model=cnn_tpool2_mel_coef1 --dct_coefficient_count=26 --how_many_training_steps=${training_steps}"
name4="cnn_tpool2_batch_size1"
param4="--name_model=cnn_tpool2_batch_size1 --batch_size=32 --how_many_training_steps=${training_steps}"
name5="cnn_tpool2_beta1b"
param5="--name_model=cnn_tpool2_beta1b --beta1=-1.5 --how_many_training_steps=${training_steps}"
name6="cnn_tpool2_beta1c"
param6="--name_model=cnn_tpool2_beta1c --beta1=-2.0 --how_many_training_steps=${training_steps}"
name7="cnn_tpool2_beta2a"
param7="--name_model=cnn_tpool2_beta2a --beta2=-2.75 --how_many_training_steps=${training_steps}"
name8="cnn_tpool2_beta2b"
param8="--name_model=cnn_tpool2_beta2b --beta2=-3.25 --how_many_training_steps=${training_steps}"
name9="cnn_tpool2_beta2c"
param9="--name_model=cnn_tpool2_beta2c --beta2=-3.75 --how_many_training_steps=${training_steps}"
name10="cnn_tpool2_moreregul_moresteps"
param10="--name_model=cnn_tpool2_moreregul_moresteps --regul_coef=0.05 --how_many_training_steps=${training_steps2}"

#paramslist=("$param1" "$param2" "$param3" "$param4" "$param5" "$param6" "$param7" "$param8" "$param9")
#nameslist=("$name1" "$name2" "$name3" "$name4" "$name5" "$name6" "$name7" "$name8" "$name9")
paramslist=("$param3" "$param10" "$param4")
nameslist=("$name3" "$name10" "$param4")

for ((i = 0; i < ${#paramslist[@]}; i++)); do
  logfile="${nameslist[$i]}.log"
  echo "$(date): python s_train.py ${paramslist[$i]} > $logfile 2>&1"
  python s_train.py ${paramslist[$i]} > $logfile 2>&1
done

# sudo shutdown

