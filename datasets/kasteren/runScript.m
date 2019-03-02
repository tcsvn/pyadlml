% Loads dataset from file
load kasterenDataset

% Visualizes the sensor and annotation data
sensorGUI(ss,ss.getIDs, sensor_labels, as, as.getIDs, activity_labels)

% Converts the sensordata to a feature matrix
[FeatMatRaw, Labels, Dates] = convert2RawFeatMat(ss, as, 60);
[FeatMatChange, Labels, Dates] = convert2ChangeFeatMat(ss, as, 60);
[FeatMatLast, Labels, Dates] = convert2LastFiredFeatMat(ss, as, 60);
