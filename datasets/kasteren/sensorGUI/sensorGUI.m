function h = sensorGUI(ss, sensor_list, sensor_labels, as, activity_list, activity_labels, infered_as, infered_activity_list, infered_activity_labels, date);

% Create window and set resize callback
h = figure;%('ResizeFcn','sensorGUIHandler(3)');

% Set figure to ADD mode
hold on;

% Check variables
if (exist('sensor_list', 'var')==0 |isempty(sensor_list)),
    sensor_list = ss.getIDs;
end

if (exist('sensor_labels', 'var')==0 |isempty(sensor_labels))
    sensor_labels = [];
end

if (exist('as', 'var')==0 |isempty(as)),
    as = [];
end
if (exist('infered_as', 'var')==0 |isempty(as)),
    infered_as = [];
end

if (exist('activity_list', 'var')==0 |isempty(activity_list)),
    if (isempty(as))
        activity_list = [];
    else
        activity_list = as.getIDs;
    end
end

if (exist('activity_labels', 'var')==0 |isempty(activity_labels))
    activity_labels = [];
end

if (exist('infered_activity_list', 'var')==0 |isempty(activity_list)),
    if (isempty(infered_as))
        infered_activity_list = [];
    else
        infered_activity_list = as.getIDs;
    end
end

if (exist('infered_activity_labels', 'var')==0 |isempty(activity_labels))
    infered_activity_labels = [];
end

if (exist('date', 'var')==0 |isempty(date))
    date = floor(ss(1,1));
end

% Store variables in info structure
info.ss = ss;
info.sensor_list = sensor_list;
info.sensor_labels = sensor_labels;

info.as = as;
info.activity_list = activity_list;
info.activity_labels = activity_labels;
info.infered_as = infered_as;
info.infered_activity_list = infered_activity_list;
info.infered_activity_labels = infered_activity_labels;
info.drawInfAct = 0;
info.drawAct = 0;
info.drawText= 1;
info.date = date;

info.xmin = 0;
info.xmax = 86400;
info.ymin = 0;
info.ymax = size(sensor_list,1)*2;
% Add area for text
info.ymax = info.ymax * 1.15;

% Store info structure inside figure
set(h,'UserData',info)

% Call initializer
sensorGUIHandler(0);
