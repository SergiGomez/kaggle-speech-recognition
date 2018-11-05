# Kaggle Speach Reconition 

# Link to the Kaggle challenge
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

# Project Overview
We might be on the verge of too many screens. It seems like everyday, new versions of common objects are “re-invented” with built-in wifi and bright touchscreens. A promising antidote to our screen addiction are voice interfaces.

But, for independent makers and entrepreneurs, it’s hard to build a simple speech detector using free, open data and code. Many voice recognition datasets require preprocessing before a neural network model can be built on them. To help with this, TensorFlow recently released the Speech Commands Datasets. It includes 65,000 one-second long utterances of 30 short words, by thousands of different people.

In this competition, you're challenged to use the Speech Commands Dataset to build an algorithm that understands simple spoken commands. By improving the recognition accuracy of open-sourced voice interface tools, we can improve product effectiveness and their accessibility.

# Code structure

Important note: The main skeleton of the code, as well as some specific functions, were taken from the great TensorFlow tutorial (www.tensorflow.org/tutorials/sequences/audio_recognition). It would not have been possible to get this project started without the great help of such a great tutorial prepared by the incredible team working at TensorFlow

- scan_hyperparameters.sh: Bash program that schedules training executions with different combinations of hyperparameters given
- s00_input_params: Parameters that user should input before executing the main script s_train
- s_train.py: Main Exec
- s01_input_data.py: Reads the data in Audio format 
- s02_get_data.py: Class that transforms it to the format that will be fed to the NN
- s03_models.py: NN models that are used in this project. CNN were used in this project with few variations on the architecture.
- s04_make_predictions.py: Script that does the predictions taking the last model trained. 
- s05_metrics_data.py: Script that gets the performance metrics (Accuracy) of the predictions made.


Script where NN model is defined: s02

# Datasets
* data/train: dataset to develop our algorithm. We divide these data into:
  * ``train``
  * ``dev``
  * ``validation``

* data/test: dataset to submit and rank our results to Kaggle

## RAW data import
To load RAW data into a pandas dataframe, split in `train`, `dev` and `test`, just run the following code:
```python
import load_data as ld

data_path = 'data'
prepared_data_df = ld.prepare_data(data_path, random_state=42)
```

It will return a dataframe as follows:

|label |path |uid |wav |set|
|------|-----|----|----|---|
|yes|data/train/audio/yes/004ae714_nohash_0.wav|004ae714|\[-91, -176, -111, -95, -120, -151, -133, -133,..|train|
|...|...|...|...|...|

