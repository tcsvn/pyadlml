# Activities of Daily Living - Machine Learning
> Contains data preprocessing and visualization methods for ADL datasets.

![PyPI version](https://img.shields.io/pypi/v/pyadlml?style=flat-square)
![Download Stats](https://img.shields.io/pypi/dd/pyadlml?style=flat-square)
![License](https://img.shields.io/pypi/l/pyadlml?style=flat-square)

Activities of Daily living (ADLs) e.g cooking, sleeping, and devices readings are recorded by smart homes inhabitants. The goal is to predict inhabitants activities using device readings. Pyadlml offers an easy way to fetch, visualize and preprocess common datasets. A further goal is to replicate prominent work in this domain.
<p align="center"><img width=95% src="https://github.com/tcsvn/pyadlml/blob/master/media/pyadlml_banner.png"></p>


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

# plot the persons activity density distribution over one day
from pyadlml.dataset.plot.activities import ridge_line
ridge_line(data.df_activities)

# plot the signal cross correlation between devices
from pyadlml.dataset.plot.devices import heatmap_cross_correlation
heatmap_cross_correlation(data.df_devices)

# create a raw representation with timeslices of 20 seconds
from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
enc_dat = DiscreteEncoder(rep='raw', t_res='20s')
X = enc_dat.fit_transform(data.df_devices).values

# label the datapoints with the corresponding activity
y = LabelEncoder(X).fit_transform(data.df_activities)

# do all the other fancy machine learning stuff you already know
from sklearn import SVM 
SVM().fit(X).score(X,y)
...
```

_For more examples and usage, please refer to the Notebooks _

## Features
  - 8 Datasets
  - A bunch of plots visualizing devices, activities and their interaction
  - Different data representations
    - Discrete timeseries
      - raw
      - changepoint
      - lastfired
    - Timeseries as images 
 - methods for importing data from Home Assistant/Activity Assistant
 
### Supported Datasets
  - [x] Amsterdam 
  - [x] Aras
  - [x] Casas Aruba (2011)
  - [ ] Casas Milan (2009)
  - [ ] Kasteren House A,B,C
  - [x] MitLab
  - [x] Tuebingen 2019
  - [x] UCI Adl Binary 
  
  
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
  
## Replication list  
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
