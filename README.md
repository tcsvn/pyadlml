# Activity of Daily Living - Machine Learning
> Contains data processing and visualization methods for ADL datasets.

[![NPM Version][npm-image]][npm-url] 
[![Build Status][travis-image]][travis-url] 
[![Downloads Stats][npm-downloads]][npm-url] 

With pyadlml you can fetch common datasets and visualze them. 
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

# plot the activity density distribution over one day
from pyadlml.dataset.plot.activities import ridge_line
ridge_line(data.df_activities)

# plot sth. device related
from pyadlml.dataset.plot.devices import heatmap_cross_correlation
heatmap_cross_correlation(data.df_devices)

# create a raw representation with timeslices of 20 seconds
from pyadlml.preprocessing import DiscreteEncoder
enc_dat = DiscreteEncoder(rep='raw', t_res='20s')
raw = enc_dat.fit_transform(data.df_devices)

# no do all the other machine learning related stuff

```

_For more examples and usage, please refer to the Notebooks _

## Features
  - A bunch of plots visualizing devices and activities 
  - Different data representations
    - discrete timeseries
    - timeseries as images 
 
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
Papers that could be replicated

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
