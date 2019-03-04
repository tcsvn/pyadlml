function plotas(as, activity_list, activity_labels, date, drawText, color);

%Assuming figure is already prepared by plotss
hold on;

if (exist('activity_list', 'var')==0 |isempty(activity_list)),
    activity_list = [4 13 5 7 6 2 14 17 18]';%as.getIDs' TODO!!!!!!! FIX getIDs;
end

if (exist('activity_labels', 'var')==0 |isempty(activity_labels))
    activity_labels = [];
end

if (exist('date', 'var')==0 |isempty(date))
    date = floor(as(1,1));
end

v = axis;

xmin = v(1);
xmax = v(2);
ymin = v(3);
ymax = v(4);

nticks = 13; % Advised to put at 25 or 13 

% Filter out the requested day
%idxday = find(floor(as.start)==date | floor(as.end)==date);
idxday = find(date >= floor(as.start) & date <= floor(as.end));

asday = as(idxday);

offset = 2;

labels = {};
row = 1;

for i=size(activity_list,1):-1:1,
    % Filter out the requested id on the requested day
    idxid = find(asday.id==activity_list(i));
    asid  = asday(idxid);
    
    if (asid.len ==0)
        continue;
    end
    
    for j=1:asid.len,
        % |   |
        % |   | 
        % |___|   <-- store points to create that figure
        % activity

        % label for this activity
        label = activity_labels(activity_list(i));
        if (asid(j).startdate < date)
            actStartTime = 0;
        else            
            actStartTime = asid(j).starttime;
        end
        
        if (asid(j).enddate > date)
            actEndTime = xmax;
        else            
            actEndTime = asid(j).endtime;
        end

        h=rectangle('Position', [actStartTime 0 (actEndTime-actStartTime) ymax], 'FaceColor', color, 'EdgeColor', [0 0 0]);
        set(h, 'ButtonDownFcn', @lineClickCallback, 'tag', label{:});
        h=plot([actStartTime, actEndTime], [ymax, ymax], 'k-');
        set(h, 'ButtonDownFcn', @lineClickCallback, 'tag', label{:});
        h=plot(actStartTime, ymax, 'k<');
        set(h, 'ButtonDownFcn', @lineClickCallback, 'tag', label{:});
        h=plot(actEndTime, ymax, 'k>');
        set(h, 'ButtonDownFcn', @lineClickCallback, 'tag', label{:});
        
        if (drawText)
            text(actStartTime + (actEndTime-actStartTime)/2, ymax-1, label, 'Rotation', -90);
        end

        if (offset+4 > ymax)
            offset = 2;
        else
            offset = offset + 2;
        end
    end
end
