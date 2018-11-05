#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:13:25 2018

@author: sergigomezpalleja
"""
import os
from os.path import join
from glob import glob
import tensorflow as tf
import importlib
import numpy as np
import pandas as pd 
import re

df = pd.read_csv('submission.csv')
df.loc[df['fname'] == 'clip_0a0a99fbe.wav']

df.loc[:,'label']= df.loc[:,'label'].replace([0],'silence')
df.loc[:,'label']= df.loc[:,'label'].replace([1],'unknown')
df.loc[:,'label']= df.loc[:,'label'].replace([2],'yes')
df.loc[:,'label']= df.loc[:,'label'].replace([3],'no')
df.loc[:,'label']= df.loc[:,'label'].replace([4],'up')
df.loc[:,'label']= df.loc[:,'label'].replace([5],'down')
df.loc[:,'label']= df.loc[:,'label'].replace([6],'left')
df.loc[:,'label']= df.loc[:,'label'].replace([7],'right')
df.loc[:,'label']= df.loc[:,'label'].replace([8],'on')
df.loc[:,'label']= df.loc[:,'label'].replace([9],'off')
df.loc[:,'label']= df.loc[:,'label'].replace([10],'stop')
df.loc[:,'label']= df.loc[:,'label'].replace([11],'go')    
 
df.label.unique()

df.to_csv("submission.csv", index=False)