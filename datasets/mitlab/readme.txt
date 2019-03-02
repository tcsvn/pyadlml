First of all, thanks for any references or acknowledgements to this work 
:P


This directory contains the data collected for the thesis:

"Activity Recognition in the Home Setting Using Simple and Ubiquitous sensors"
by Emmanuel Munguia Tapia

for any referece to this work or dataset please use:

-E. Munguia Tapia. Activity Recognition in the Home Using Simple and Ubiquitous
 Sensors. Pervasive 2004,Vienna, Austria.

-E. Munguia Tapia. Activity Recognition in the Home Setting Using Simple and 
Ubiquitous Sensors. S.M. Thesis, Massachusetts Institute of Technology, 2003.


In this thesis, between 80-100 reed switch sensors where installed in two single-person
apartments collecting data about human activity for two weeks. The sensors
were installed in everyday objects such as drawers, refrigerators, containers,
etc to record opening-closing events (activation deactivation events) as
the subject carried out everyday activities.


each directory contains the data of a different subject and each subdirectory
contains the following files:

The data is stored in text format and coma separated values (.csv). To visualize
the files, you can use Excel or Wordpad.


Each directory contains the following files:


1) sensors.csv

file containing the sensor information in the following forma:

SENSOR_ID,LOCATION,OBJECT

example:

100,Bathroom,Toilet Flush 
101,Bathroom,Light switch
104,Foyer,Light switch
...
..
.


2)activities.csv

is the file containing all the activities analyzed in the following format

Heading,Category,Subcategory,Code

example:

Employment related,Employment work at home,Work at home,1
Employment related,Travel employment,Going out to work,5
Personal needs,Eating,Eating,10
Personal needs,Personal hygiene,Toileting,15



3)activities_data.csv

is the data of the activities in the following format:

ACTIVITY_LABEL,DATE,START_TIME,END_TIME
SENSOR1_ID, SENSOR2_ID, ......
SENSOR1_OBJECT,SENSOR2_OBJECT, .....
SENSOR1_ACTIVATION_TIME,SENSOR2_ACTIVATION_TIME, .....
SENSOR1_DEACTIVATION_TIME,SENSOR2_DEACTIVATION_TIME, .....

where date is in the mm/dd/yyyy format
where time is in the hh:mm:ss format


NOTE: ACTIVITY_LABEL = Subcategory

example:

Toileting,17:30:36,17:46:41
100,68
Toilet Flush,Sink faucet - hot
17:39:37,17:39:46
18:10:57,17:39:52


Send any questions or comments to


-Emmanuel Munguia Tapia [emunguia@media.mit.edu]




