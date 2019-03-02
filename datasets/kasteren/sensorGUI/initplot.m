function initplot(ss, sensor_list, sensor_labels, date, xmin, xmax, ymin, ymax)

% Check variables
if (exist('sensor_list', 'var')==0 ||isempty(sensor_list)),
    sensor_list = [72 139 81  101 68 57 58 88 67 71]'; %ss.getIDs' TODO!!!!!!! FIX getIDs;
end
if (exist('sensor_labels', 'var')==0 ||isempty(sensor_labels))
    sensor_labels = [];
end
if (exist('date', 'var')==0 ||isempty(date))
    date = floor(ss(1,1));
end
if (exist('xmin', 'var')==0 ||isempty(xmin)),
    xmin = 0;
end
if (exist('xmax', 'var')==0 ||isempty(xmax)),
    xmax = 86400;
end
if (exist('ymin', 'var')==0 ||isempty(ymin)),
    ymin = 0;
end
if (exist('ymax', 'var')==0 ||isempty(ymax)),
    ymax = size(sensor_list,1)*2;
end

nticks = 13; % Number of ticks on x-axis, advised to put at 25 or 13 

% set axis 86400 seconds in one day
axis([xmin xmax ymin ymax]);

hold on;

%changing axis properties
set(gca,'FontSize',7);
set(gca,'TickDir','out');
%zoom off;

%print date
szTitle = sprintf('%s - %s', datestr(date, 'dddd'), datestr(date));
title(szTitle);

%putting ticks at every half hour on x axis
[xlabels,interval] = timeTicks(xmin,xmax,nticks,0,0,1); 
set(gca,'XTick',xmin:interval:xmax,'XTickLabel',xlabels);

% putting ticks on y axis
fticks = 1:2:size(sensor_list,1)*2;
set(gca,['y','tick'],fticks);
