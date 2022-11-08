```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/blood-transfusion-dataset/transfusion.csv
    


```python
import pandas as pd
import numpy as np
data=pd.read_csv('../input/blood-transfusion-dataset/transfusion.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recency (months)</th>
      <th>Frequency (times)</th>
      <th>Monetary (c.c. blood)</th>
      <th>Time (months)</th>
      <th>whether he/she donated blood in March 2007</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>50</td>
      <td>12500</td>
      <td>98</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>13</td>
      <td>3250</td>
      <td>28</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>16</td>
      <td>4000</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>20</td>
      <td>5000</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>24</td>
      <td>6000</td>
      <td>77</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.isnull().sum()
```




    Recency (months)                              0
    Frequency (times)                             0
    Monetary (c.c. blood)                         0
    Time (months)                                 0
    whether he/she donated blood in March 2007    0
    dtype: int64




```python
data=data.drop_duplicates()
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 533 entries, 0 to 747
    Data columns (total 5 columns):
     #   Column                                      Non-Null Count  Dtype
    ---  ------                                      --------------  -----
     0   Recency (months)                            533 non-null    int64
     1   Frequency (times)                           533 non-null    int64
     2   Monetary (c.c. blood)                       533 non-null    int64
     3   Time (months)                               533 non-null    int64
     4   whether he/she donated blood in March 2007  533 non-null    int64
    dtypes: int64(5)
    memory usage: 25.0 KB
    


```python
print(data['whether he/she donated blood in March 2007'].value_counts())
cls_0=data[data['whether he/she donated blood in March 2007']==0]
cls_1=data[data['whether he/she donated blood in March 2007']==1]
```

    0    384
    1    149
    Name: whether he/she donated blood in March 2007, dtype: int64
    


```python
cls_0=cls_0.sample(500,replace=True)
cls_1=cls_1.sample(500,replace=True)
data=pd.concat([cls_0,cls_1],axis=0)
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1000 entries, 642 to 508
    Data columns (total 5 columns):
     #   Column                                      Non-Null Count  Dtype
    ---  ------                                      --------------  -----
     0   Recency (months)                            1000 non-null   int64
     1   Frequency (times)                           1000 non-null   int64
     2   Monetary (c.c. blood)                       1000 non-null   int64
     3   Time (months)                               1000 non-null   int64
     4   whether he/she donated blood in March 2007  1000 non-null   int64
    dtypes: int64(5)
    memory usage: 46.9 KB
    


```python
y=data['whether he/she donated blood in March 2007']
x=data.drop(['whether he/she donated blood in March 2007'],axis=1)
for column in x.columns:
    x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min()) 
x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recency (months)</th>
      <th>Frequency (times)</th>
      <th>Monetary (c.c. blood)</th>
      <th>Time (months)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>642</th>
      <td>0.152778</td>
      <td>0.102041</td>
      <td>0.102041</td>
      <td>0.406250</td>
    </tr>
    <tr>
      <th>134</th>
      <td>0.027778</td>
      <td>0.204082</td>
      <td>0.204082</td>
      <td>0.895833</td>
    </tr>
    <tr>
      <th>401</th>
      <td>0.319444</td>
      <td>0.081633</td>
      <td>0.081633</td>
      <td>0.322917</td>
    </tr>
    <tr>
      <th>685</th>
      <td>0.291667</td>
      <td>0.122449</td>
      <td>0.122449</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>525</th>
      <td>0.027778</td>
      <td>0.061224</td>
      <td>0.061224</td>
      <td>0.093750</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.2,stratify=y)
```


```python
from sklearn.svm import SVC
svc_model=SVC(kernel='rbf',gamma=8)
svc_model.fit(X_train,y_train)
```




    SVC(gamma=8)




```python
from sklearn.metrics import accuracy_score, confusion_matrix
predictions= svc_model .predict(X_train)
percentage=svc_model.score(X_train,y_train)
res=confusion_matrix(y_train,predictions)
print("Training confusion matrix")
print(res)
predictions= svc_model .predict(X_test)
train_percentage=svc_model.score(X_train,y_train)
test_percentage=svc_model.score(X_test,y_test)
res=confusion_matrix(y_test,predictions)
print("Testing confusion matrix")
print(res)
# check the accuracy on the training set
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))
print(f"Train set:{len(X_train)}")
print(f"Train Accuracy={train_percentage*100}%")
print(f"Test set:{len(X_test)}")
print(f"Test Accuracy={test_percentage*100}%")
```

    Training confusion matrix
    [[262 138]
     [115 285]]
    Testing confusion matrix
    [[62 38]
     [28 72]]
    0.68375
    0.67
    Train set:800
    Train Accuracy=68.375%
    Test set:200
    Test Accuracy=67.0%
    


```python
from xgboost import XGBClassifier
svc_model1=XGBClassifier()
svc_model1.fit(X_train,y_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                  max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                  missing=nan, monotone_constraints='()', n_estimators=100,
                  n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
                  reg_alpha=0, reg_lambda=1, ...)




```python
from sklearn.metrics import accuracy_score, confusion_matrix
predictions= svc_model1 .predict(X_train)
percentage=svc_model1.score(X_train,y_train)
res=confusion_matrix(y_train,predictions)
print("Training confusion matrix")
print(res)
predictions= svc_model1 .predict(X_test)
train_percentage=svc_model1.score(X_train,y_train)
test_percentage=svc_model1.score(X_test,y_test)
res=confusion_matrix(y_test,predictions)
print("Testing confusion matrix")
print(res)
# check the accuracy on the training set
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))
print(f"Train set:{len(X_train)}")
print(f"Train Accuracy={train_percentage*100}%")
print(f"Test set:{len(X_test)}")
print(f"Test Accuracy={test_percentage*100}%")
```

    Training confusion matrix
    [[367  33]
     [ 12 388]]
    Testing confusion matrix
    [[87 13]
     [ 3 97]]
    0.68375
    0.67
    Train set:800
    Train Accuracy=94.375%
    Test set:200
    Test Accuracy=92.0%
    
