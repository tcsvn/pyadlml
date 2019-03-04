% author: Jennifer, May, 2003

% EXAMPLE OF USE:
% startSec = 28800; % 8:00am
% endSec = 36000; % 10:00am
% get labels for 10 ticks
% [xlabels,interval] = timeTicks(startSec,endSec,10,0,0,0)
% set(gca,'XTick',startSec:interval:endSec,'XTickLabel',xlabels);


function [timeLabels,interval] = timeTicks(startSeconds, endSeconds, nTicks, interval,includeSec, includeAMPM)

totalSeconds = endSeconds - startSeconds;
 
if (nTicks == 0) %if nTicks is 0, use interval to determine number of ticks
 nTicks = floor(totalSeconds/interval) + 1;
else %if nTicks is specified, determine interval
 interval = totalSeconds/(nTicks-1);
end
 
timeLabels = cell(1,nTicks);
for index=1:nTicks
 [h,m,s,a] = sec2Time(startSeconds + (index-1)*interval);
 h = num2str(h);
 if (m < 10) m = strcat('0',num2str(m)); %pad minutes with leading 0
 else m = num2str(m);
 end
 
 timeLabels{1,index} = strcat(h,':',m); %#:##
 if (includeSec == 1) %include seconds
    timeLabels{1,index} = strcat(timeLabels{1,index},':',s); %#:##:##
 end
 if (includeAMPM == 1) %include am or pm
     if (s < 10) s = strcat('0',num2str(s)); %pad secons with leading 0
     else s = num2str(s);
     end
     timeLabels{1,index} = strcat(timeLabels{1,index} ,' ',a); %#:##am
 end
end

