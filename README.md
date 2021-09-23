# Estimated Time of Arrival Prediction via Modeling the Spatial-Temporal Interactions between Links and Crosses (GIS Cup)
**Competition Page**: [DiDi-ETA](https://www.biendata.xyz/competition/didi-eta/)
## Feature Generation

### Path

```
python feature.py
```

* Original train data should be stored at 'GISCUP-submit/data/raw_data/train/'. Original test data should be stored at 'GISCUP-submit/data/raw_data/'
* In 'feature.py'. The variable ```ws``` on line 184 represents your workspace, which should be modified to ```your workspace path```
* This script will generate two feature files for each day's data, ```df``` and ```id_ time```. They will be located in ```'ws/data/feature/df/'``` and ```'ws/data/feature/id_time/'```, respectively. You don't need to create these directories in advance. If these two directories don't exist, the script will generate them for you
* An additional ```.pkl``` file will be created under /id_time directory

### Function

* The ```df``` feature files contain information of each order and will be used in the ```W``` and ```D``` module of our model
* There is sequence information in the ```id_time``` file. For each sequence, we elaborately insert each cross on a appropriate position into the link sequence
* For the crosses inserted into the link sequence, we assigned new cross_id for them to make their ids and link_id has the same form. ```cross_dict.npy``` file records the mapping relationship between old and new IDS
* This script will take a long time to run

## Creating the validation set

### Path

```
python val_random_day.py
```

* In 'val_random_day.py'. The variable ```ws``` on line 62 represents your workspace, which should be modified to ```your workspace path```.
* This script will generate two files in  ```'ws/data/feature/df/'``` and ```'ws/data/feature/id_time/'```, respectively, which are the new validation set.

### Function

* The script will traverse the generated feature files and randomly sample one thirtieth of the data from the feature file of each day as the validation data set.
* The validation set produced will have a similar size and distribution to a single-day training set

### Remarks

This script should be run after the ```feature.py```.

## Get Results

### **Path**

```
python train_val_test.py
```

* In 'train_val_test.py'. The variable ```ws``` on line 409 represents your workspace, which should be modified to ```your workspace path```
* After running this file, you can get the result of a single model stored as "model_epoch2_day31.npy" in '~/GISCUP-submit/results

### Function

* After running this file, you can get the result of a single model stored as "model_epoch2_day31.npy" in '~/GISCUP-submit/results

* We adopt ensemble averaging to get the final prediction.

  









