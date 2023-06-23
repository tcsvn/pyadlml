# Activities of Daily Living - Machine Learning
> Contains data preprocessing and visualization methods for ADL datasets.

![PyPI version](https://img.shields.io/pypi/v/pyadlml?style=flat-square)
![Download Stats](https://img.shields.io/pypi/dd/pyadlml?style=flat-square)
![Read the Docs (version)](https://img.shields.io/readthedocs/pyadlml/latest?style=flat-square)
![License](https://img.shields.io/pypi/l/pyadlml?style=flat-square)
<p align="center"><img width=95% src="https://github.com/tcsvn/pyadlml/blob/master/media/pyadlml_banner.png"></p>
Activities of Daily living (ADLs) such as eating, working, sleeping and Smart Home device readings are recorded by inhabitants. Predicting resident activities based on the device event-stream allows for a range of applications, including automation, action recommendation and abnormal activity detection in the context of assisted living for elderly inhabitants. Pyadlml offers an easy way to fetch, visualize and preprocess common datasets.


## !! Disclaimer !!
Package is still an alpha-version and under active development. 
As of now do not expect anything to work! APIs are going to change, 
stuff breaks and the documentation may lack behind. Nevertheless, feel 
free to take a look. The safest point to start is probably the API reference.

## Last (stable) Release
```sh 
$ pip install pyadlml
```
## Latest Development Changes
```sh
$ git clone https://github.com/tcsvn/pyadlml
$ cd pyadlml
$ pip install .
```

## Usage example


### Simple

```python
# Fetch dataset
from pyadlml.dataset import fetch_amsterdam
data = fetch_amsterdam()
df_devs, df_acts = data['devices'], data['activities']


# Plot the residents activity density over one day
from pyadlml.plot import plot_activity_density
fig = plot_activity_density(df_acts)
fig.show()


# Create a vector representing the state of all Smart Home devices
# at a certain time and discretize the events into 20 second bins
from pyadlml.preprocessing import Event2Vec, LabelMatcher
e2v = Event2Vec(encode='state', dt='20s')
states = e2v.fit_transform(df_devs)

# Label each datum with the corresponding activity.
# When an event matches no activity set the activity to "other"
lbls = LabelMatcher(other=True).fit_transform(df_acts, states)

# Extract numpy arrays without timestamps (1st column)
X, y = states.values[:,1:], lbls.values[:,1:]

# Proceed with machine learning stuff 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X, y)
clf.score(X, y)
...
```

### Less simple


```python
from pyadlml.dataset import fetch_amsterdam
from pyadlml.constants import VALUE
from pyadlml.pipeline import Pipeline
from pyadlml.preprocessing import IndexEncoder, LabelMatcher, DropTimeIndex, \
                                  EventWindows, DropColumn
from pyadlml.model_selection import train_test_split
from pyadlml.model import DilatedModel
from pyadlml.dataset.torch import TorchDataset
from torch.utils.data import DataLoader
from torch.optim import Adam 
from torch.nn import functional as F

# Featch data and split into train/val/test based on time rather than #events
data = fetch_amsterdam()
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
    data['devices'], data['activities'], \
    split=(0.6, 0.2, 0.2), temporal=True,
)

# Formulate all preprocessing steps using a sklearn styled pipeline 
pipe = Pipeline([
    ('enc', IndexEncoder()),            # Encode devices strings with indices
    ('drop_obs', DropColumn(VALUE)),    # Disregard device observations
    ('lbl', LabelMatcher(other=True)),  # Generate labels y  
    ('drop_time', DropTimeIndex()),     # Remove timestamps for x and y
    ('windows', EventWindows(           # Create sequences S with a sliding window
                  rep='many-to-one',    # Only one label y_i per sequence S_i
                  window_size=16,       # Each sequence contains 16 events 
                  stride=2)),           # A sequence is created every 2 events
    ('passthrough', 'passthrough')      # Do not use a classifier in the pipeline
])

# Create a dataset to sample from
dataset = TorchDataset(X_train, y_train, pipe) 
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = DilatedModel(
    n_features=14,       # Number of devices
    n_classes=8          # Number of activities
)
optimizer = Adam(model.parameters(), lr=3e-4)

# Minimal loop to overfit the data
for s in range(10000):
    for (Xb, yb) in train_loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()

...
```



_For more examples and how to use, please refer to the [documentation](https://pyadlml.readthedocs.io/en/latest/)._

## Features
  - Access to 14 Datasets 
  - Importing data from [Home Assistant](https://www.home-assistant.io/) or [Activity Assistant](https://github.com/tcsvn/activity-assistant)
  - Tools for data cleaning
    - Relabel activities and devices
    - Merge overlapping activities
    - Find and replace specific patterns in device signals
    - Interactive dashboard for data exploration
  - Various statistics and visualizations for devices, activities and their interaction
  - Preprocessing methods
    - Device encoder (index, state, changepoint, last_fired, ...)
    - Feature extraction (inter-event-times, intensity, time2vec, ...)
    - Sliding windows (event, temporal, explicit or (fuzzytime))
    - Many more ... 
  - Cross validation iterators and pipeline adapted for ADLs
    - LeaveKDayOutSplit, TimeSeriesSplit
    - Conditional transformer: YTransformer, XorYTransformer, ...
  - Online metrics to compare models regardless of resample frequency
    - Accuracy, TPR, PPV, ConfMat, Calibration
  - Ready to use models (TODO)
    - RNNs
    - WaveNet
    - Transformer
  - Translate datasets to sktime formats
 
### Supported Datasets
  - [x] Amsterdam [1]
  - [x] Aras [2]
  - [x] Casas Aruba (2011) [3]
  - [X] Casas Cairo [4]
  - [X] Casas Milan (2009) [4]
  - [X] Casas Tulum [4]
  - [X] Casas Kyoto (2010) [4]
  - [x] Kasteren House A,B,C [5]
  - [x] MITLab [6]
  - [x] UCI Adl Binary [8]
  - [ ] Chinokeeh [9]
  - [ ] Orange [TODO]

## Examples, benchmarks and replications
The project includes (TODO) a ranked model leaderboard evaluated on the cleaned dataset versions.
Additionaly, here is a useful list of awesome references (todo include link) to papers
and repos related to ADLs and machine learning.


## Contributing 
1. Fork it (<https://github.com/tcsvn/pyadlml/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Related projects
  - [Activity Assistant](https://github.com/tcsvn/activity-assistant) - Recording, predicting ADLs within Home assistant.
  - [Sci-kit learn](https://github.com/sklearn) - The main inspiration and some borrowed source code.
  
## Support 
[![Buy me a coffee][buy-me-a-coffee-shield]][buy-me-a-coffee]
  
## How to cite
If you are using pyadlml for publications consider citing the package
```
@software{pyadlml,
  author = {Christian Meier},
  title = {PyADLMl - Machine Learning library for Activities of Daily Living},    
  url = {https://github.com/tcsvn/pyadlml},
  version = {0.0.22-alpha},
  date = {2023-01-03}
}
```

## Sources

#### Dataset


[1]: T.L.M. van Kasteren; A. K. Noulas; G. Englebienne and B.J.A. Kroese, Tenth International Conference on Ubiquitous Computing 2008  
[2]: H. Alemdar, H. Ertan, O.D. Incel, C. Ersoy, ARAS Human Activity Datasets in Multiple Homes with Multiple Residents, Pervasive Health, Venice, May 2013.  
[3,4]: WSU CASAS smart home project: D. Cook. Learning setting-generalized activity models for smart spaces. IEEE Intelligent Systems, 2011.  
[5]: Transferring Knowledge of Activity Recognition across Sensor networks. Eighth International Conference on Pervasive Computing. Helsinki, Finland, 2010.  
[6]: E. Munguia Tapia. Activity Recognition in the Home Setting Using Simple and Ubiquitous sensors. S.M Thesis  
[7]: Activity Recognition in Smart Home Environments using Hidden Markov Models. Bachelor Thesis. Uni Tuebingen.   
[8]: Ordonez, F.J.; de Toledo, P.; Sanchis, A. Activity Recognition Using Hybrid Generative/Discriminative Models on Home Environments Using Binary Sensors. Sensors 2013, 13, 5460-5477.  
[9]: D. Cook and M. Schmitter-Edgecombe, Assessing the quality of activities in a smart environment. Methods of information in Medicine, 2009

#### Software

TODO add software used in TPPs 

## License
MIT  © [tcsvn](http://deadlink)


[buy-me-a-coffee-shield]: https://img.shields.io/static/v1.svg?label=%20&message=Buy%20me%20a%20coffee&color=6f4e37&logo=buy%20me%20a%20coffee&logoColor=white
[buy-me-a-coffee]: https://www.buymeacoffee.com/tscvn
