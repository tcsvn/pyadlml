# Activity of Daily Living - Machine Learning
> Contains data preprocessing and visualization methods for ADL datasets.

[![NPM Version][npm-image]][npm-url]  
[![Build Status][travis-image]][travis-url]  
[![Downloads Stats][npm-downloads]][npm-url]  

Activity of Daily livings (ADLs) e.g cooking, sleeping, and devices readings are recorded by smart homes inhabitants. The goal is to predict the activities of an inhabitant using the device readings. Pyadlml offers a way to fetch, visualize and preprocess common datasets. A further Goal is to replicate prominent works in this domain.
![](header.png)

## Last Stable Release
```sh 
$ pip install pyadlml
```
## Latest Development Changes
```sh
$ git clone https://github.com/tcsvn/pyadlml
$ cd pyadlml
```

## Usage example
```python
from pyadlml.dataset import fetch_amsterdam

# Fetch dataset
data = fetch_amsterdam(cache=True)

# plot the activity density distribution for the person over one day
from pyadlml.dataset.plot.activities import ridge_line
ridge_line(data.df_activities)

# plot the signal cross correlation between the devices
from pyadlml.dataset.plot.devices import heatmap_cross_correlation
heatmap_cross_correlation(data.df_devices)

# create a raw representation with timeslices of 20 seconds
from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
enc_dat = DiscreteEncoder(rep='raw', t_res='20s')
X = enc_dat.fit_transform(data.df_devices).values

# label the datapoints with the corresponding activity
y = LabelEncoder(X).fit_transform(data.df_activities)

# do all the other fancy machine learning stuff
from sklearn import SVM 
SVM().fit(X).score(X,y)
...
```

_For more examples and usage, please refer to the Notebooks _

## Features
  - 8 Datasets to fetch
  - A bunch of plots visualizing devices, activities and their interaction
  - Different data representations
    - Discrete timeseries
      - raw
      - changepoint
      - lastfired
    - Timeseries as images 
 
### Supported Datasets
  - [x] Casas Aruba (2011)
  - [x] Casas Milan (2009)
  - [x] Aras
  - [x] Amsterdam 
  - [x] MitLab
  - [x] Tuebingen 2019
  - [ ] Kasteren House A,B,C
  
### Models
#### iid data
  - [x] SVM
  - [ ] winnow algorithm
  - [ ] Naive bayes
  - [x] Decision Trees
#### sequential discretized 
  - [ ] RNNs
  - [ ] LSTMs
  - [ ] HMMs
  - [ ] TCNs
#### images  
  - [ ] CNN
  - [ ] Transformer
#### temporal points
  - [ ] THP
  
### Replication list  
Here are papers I plan to replicate


## Contributing 
1. Fork it (<https://github.com/tcsvn/pyadlml/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Related projects
  - [activity-assistant](https://github.com/tcsvn/activity-assistant) - Recording, Prediciting ADLs in Home assistant.
  
## Support 
  - Todo buy me o coffee batch
  
## Sources
  - TODO cite every dataset
  - TODO cite every algorithm package
  - https://github.com/anfederico/Clairvoyant#readme
  
## License
MIT  © [tcsvn](http://deadlink)
