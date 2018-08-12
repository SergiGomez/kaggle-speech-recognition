# Kaggle Speach Reconition - Project: parrot_recognition
## Datasets
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
