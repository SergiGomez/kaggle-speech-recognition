#!/bin/bash

#training_steps="2,1"
training_steps="8000,2000"
training_steps2="12000,4000"
training_steps3="4000,3000,3000"

# name1="cnn_tpool2_lr6_0.00045_0.000045"
# param1="--name_model=cnn_tpool2_lr6 --learning_rate=0.00045,0.000045 --how_many_training_steps=${training_steps}"
# name2="cnn_tpool2_lr7_0.00056_0.000056"
# param2="--name_model=cnn_tpool2_lr7 --learning_rate=0.00056,0.000056 --how_many_training_steps=${training_steps}"
# name3="cnn_tpool2_lr8_0.00071_0.000071"
# param3="--name_model=cnn_tpool2_lr8 --learning_rate=0.00071,0.000071 --how_many_training_steps=${training_steps}"
# name4="cnn_tpool2_beta1a"
# param4="--name_model=cnn_tpool2_beta1a --beta1=-0.5 --how_many_training_steps=${training_steps}"
# name5="cnn_tpool2_beta1b"
# param5="--name_model=cnn_tpool2_beta1b --beta1=-1.5 --how_many_training_steps=${training_steps}"
# name6="cnn_tpool2_beta1c"
# param6="--name_model=cnn_tpool2_beta1c --beta1=-2.0 --how_many_training_steps=${training_steps}"
# name7="cnn_tpool2_beta2a"
# param7="--name_model=cnn_tpool2_beta2a --beta2=-2.75 --how_many_training_steps=${training_steps}"
# name8="cnn_tpool2_beta2b"
# param8="--name_model=cnn_tpool2_beta2b --beta2=-3.25 --how_many_training_steps=${training_steps}"
# name9="cnn_tpool2_beta2c"
# param9="--name_model=cnn_tpool2_beta2c --beta2=-3.75 --how_many_training_steps=${training_steps}"

name1="cnn_tpool2_beta2b"
param1="--name_model=cnn_tpool2_beta2b --beta2=-3.25 --how_many_training_steps=${training_steps}"
name2="cnn_tpool2_beta2c"
param2="--name_model=cnn_tpool2_beta2c --beta2=-3.75 --how_many_training_steps=${training_steps}"
name3="cnn_tpool2_coefregul1"
param3="--name_model=cnn_tpool2_coefregul1 --regul_coef=0.01 --how_many_training_steps=${training_steps}"
name4="cnn_tpool2_coefregul2"
param4="--name_model=cnn_tpool2_coefregul2 --regul_coef=0.02 --how_many_training_steps=${training_steps}"
name5="cnn_tpool2_coefregul3_moresteps"
param5="--name_model=cnn_tpool2_coefregul3_moresteps --regul_coef=0.01 --how_many_training_steps=${training_steps2}"
name6="cnn_tpool2_coefregul4_newlr"
param6="--name_model=cnn_tpool2_coefregul4_newlr --regul_coef=0.02 --learning_rate=0.00032,0.00010,0.000032 --how_many_training_steps=${training_steps3}"

paramslist=("$param1" "$param2" "$param3" "$param4" "$param5" "$param6" )
nameslist=("$name1" "$name2" "$name3" "$name4" "$name5" "$name6" )
#paramslist=("$param1c")
#nameslist=("$name1c")

for ((i = 0; i < ${#paramslist[@]}; i++)); do
  logfile="${nameslist[$i]}.log"
  echo "$(date): python s_train.py ${paramslist[$i]} > $logfile 2>&1"
  echo "python s_train.py ${paramslist[$i]} > $logfile 2>&1" > $logfile
  python s_train.py ${paramslist[$i]} > $logfile 2>&1
done

# sudo shutdown

