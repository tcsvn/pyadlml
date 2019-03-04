function plotss(ss, sensor_list, sensor_labels, date, xmin, xmax, ymin, ymax)

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

% Filter out the requested day
idxday = find(floor(ss.start)==date & floor(ss.end)==date);
ssday = ss(idxday);

offset = 2;

labels = {};
row = 1;

max_val = max(ss.val);

hWait = waitbar(0, 'Please Wait...');

xlist={};
ylist={};

for i=1:size(sensor_list,1),
    % Update progressbar
    waitbar(i/size(sensor_list,1), hWait);

    % Filter out the requested id on the requested day
    idxid = find(ssday.id==sensor_list(i));
    ssid  = ssday(idxid);
    
    xpoints = [0];    
    ypoints = [offset-1];
    
    for j=1:ssid.len,
        %       _________
        % _____|         |_______   <-- store points to create that figure
        xpoints = [xpoints, ssid(j).starttime, ssid(j).starttime, ssid(j).endtime, ssid(j).endtime ];
        
        rel_val = ssid(j).val/max_val;
        ypoints = [ypoints, offset-1, offset-(1-rel_val),   offset-(1-rel_val),    offset-1];
    end

    xpoints = [xpoints, xmax];
    ypoints = [ypoints, offset-1];

    offset = offset + 2;
    
    if (isempty(sensor_labels))
        if (ss.idishex==1)
            sinfo = sprintf('%s',num2hex(sensor_list(i)));
        else
            sinfo = sprintf('%d',sensor_list(i));
        end
    else
       if (ss.idishex==1)
           sinfo = sprintf('%s %s',num2hex(sensor_list(i)), getLabel(sensor_labels, sensor_list(i)));
       else
           sinfo = sprintf('%d %s',sensor_list(i), getLabel(sensor_labels, sensor_list(i)));
       end
    end 
    labels{row,1} = sinfo;
    row = row + 1;      
   
    xlist{i} = xpoints;
    ylist{i} = ypoints;
end

close(hWait);

for i=1:size(sensor_list,1),
    plot(xlist{i}, ylist{i});
end

if (ssday.len==0 & (date > ss(ss.len).enddate | date < ss(1).startdate))
    h = text(xmax/2, ymax/2,'No data!');
    set(h, 'FontSize', 32);
    e = get(h, 'Extent');
    delete(h);
    
    h = text((xmax/2)-e(3)/2, ymax/2,'No data!');
    set(h, 'FontSize', 32);
    set(h,'Color',[1.0 0 0]);
end
set(gca,['y','ticklabel'],labels); %setting the tick labels

