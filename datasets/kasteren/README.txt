Thank you for downloading the kasterenDataset. This document explains how the dataset is structured and how to use the tools that were provided to ease its use. If you have any questions or suggestions please email me at: T.L.M.vanKasteren@uva.nl 

This dataset and experiments performed on it, is carefully described in: 
	Accurate Activity Recognition in a Home Setting
	T.L.M. van Kasteren, A. K. Noulas, G. Englebienne and B.J.A. Kröse
	Tenth International Conference on Ubiquitous Computing 2008 (Ubicomp'08). 
Please refer to this paper when you include experiments on this dataset in your publications.

Future datasets and publications can be found at: http://www.science.uva.nl/~tlmkaste

Contents of dataset package: 
- Actual data in matlab form (kasterenDataset.mat)
- Same data in text form (kasterenActData.txt and kasterenSenseData.txt)
- sensorGUI, a tool for visualizing the data and annotation in matlab
- convert scripts, scripts for converting the data structure to a discrete format in matlab
- actstruct and sensorstruct are structures used for storing the data in matlab


Using the dataset in matlab:
----------------------------
Run the following code (or use the runScript.m) in matlab to load, visualize and convert the dataset. Make sure both the kasterenDataset and its subdirectory sensorGUI are included in your path.

>> load kasterenDataset
>> sensorGUI(ss,ss.getIDs, sensor_labels, as, as.getIDs, activity_labels)
>> [FeatMatRaw, Labels, Dates] = convert2RawFeatMat(ss, as, 60);
>> [FeatMatChange, Labels, Dates] = convert2ChangeFeatMat(ss, as, 60);
>> [FeatMatLast, Labels, Dates] = convert2LastFiredFeatMat(ss, as, 60);


Acknowledgement: 
----------------
In creating the sensorGUI visualization tool some segments of support code from Emmanuel Munguia Tapia's dataset were used. His dataset and the original code can be found at: http://courses.media.mit.edu/2004fall/mas622j/04.projects/home/