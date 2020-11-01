# Activities of Daily Living - Machine Learning
> Contains data preprocessing and visualization methods for ADL datasets.

![PyPI version](https://img.shields.io/pypi/v/pyadlml?style=flat-square)
![Download Stats](https://img.shields.io/pypi/dd/pyadlml?style=flat-square)
![License](https://img.shields.io/pypi/l/pyadlml?style=flat-square)
<p align="center"><img width=95% src="https://github.com/tcsvn/pyadlml/blob/master/media/pyadlml_banner.png"></p>
Activities of Daily living (ADLs) e.g cooking, working, sleeping and devices readings are recorded by smart home inhabitants. The goal is to predict inhabitants activities using device readings. Pyadlml offers an easy way to fetch, visualize and preprocess common datasets. My further goal is to replicate prominent work in this domain.



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
From a jupyter notebook run
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

# create a raw representation with 20 second timeslices
from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
enc_dat = DiscreteEncoder(rep='raw', t_res='20s')
raw = enc_dat.fit_transform(data.df_devices)

# label the datapoints with the corresponding activity
lbls = LabelEncoder(raw).fit_transform(data.df_activities)

X = raw.values
y = lbls.values

# from here on do all the other fancy machine learning stuff you already know
from sklearn import svm
clf = svm.SVC()
clf.fit(X, y)
...
```

_For more examples and and how to use, please refer to the Documentation (to come) or the Notebooks_

## Features
  - 8 Datasets
  - A bunch of plots visualizing devices, activities and their interaction
  - Different data representations
    - Discrete timeseries
      - raw
      - changepoint
      - lastfired
    - Timeseries as images 
 - Methods for importing data from Home Assistant/Activity Assistant
 
### Supported Datasets
  - [x] Amsterdam [1]
  - [x] Aras [2]
  - [x] Casas Aruba (2011) [3]
  - [ ] Casas Milan (2009) [4]
  - [ ] Kasteren House A,B,C [5]
  - [x] MitLab [6]
  - [x] Tuebingen 2019 [7]
  - [x] UCI Adl Binary [8]
  
  
### Models
#### Iid data
  - [x] SVM
  - [ ] Winnow algorithm
  - [ ] Naive bayes
  - [x] Decision Trees
#### Sequential discretized 
  - [ ] RNNs
  - [ ] LSTMs
  - [ ] HMMs
  - [ ] HSMMs
  - [ ] TCNs
#### Images  
  - [ ] CNN
  - [ ] Transformer
#### Temporal points
  - [ ] TPPs
  
## Replication list  
Here are papers I plan to replicate


## Contributing 
1. Fork it (<https://github.com/tcsvn/pyadlml/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Related projects
  - [activity-assistant](https://github.com/tcsvn/activity-assistant) - Recording, predicting ADLs within Home assistant.
  
## Support 
  - Todo buy me a coffee batch
  
## Sources
  - Datasets (TODO get all correct citations)  
    [1]: https://sites.google.com/site/tim0306/   
    [2]: H. Alemdar, H. Ertan, O.D. Incel, C. Ersoy, ARAS Human Activity Datasets in Multiple Homes with Multiple Residents, Pervasive Health, Venice, May 2013.  
    [3]: WSU CASAS smart home project: D. Cook. Learning setting-generalized activity models for smart spaces. IEEE Intelligent Systems, 2011.  
    [4]: WSU CASAS smart home project: D. Cook. Learning setting-generalized activity models for smart spaces. IEEE Intelligent Systems, 2011.  
    [5]:  
    [6]:  
    [7]: Me :)  
    [8]: Ordonez, F.J.; de Toledo, P.; Sanchis, A. Activity Recognition Using Hybrid Generative/Discriminative Models on Home Environments Using Binary Sensors. Sensors 2013, 13, 5460-5477.
  - TODO cite every algorithm package
  
## License
MIT  © [tcsvn](http://deadlink)
