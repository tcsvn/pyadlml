function sensorGUIHandler(selector)

% Retrieve info structure from figure
fig = gcf;
info = get(fig,'UserData');

% Handle events
switch selector
case 0 % Initialize GUI application

case 3 % resizing window   
	
case 21 % Prev day
	info.date = info.date - 1;
case 22 % Next day
    info.date = info.date + 1;
case 23 % Prev week
   	info.date = info.date - 7;
case 24 % Next week
	info.date = info.date + 7;
case 25 % Prev month
    info.date = info.date - 30;
case 26 % Next month
    info.date = info.date + 30;
case 27 % Toggle Zoom
    if (info.xmin==0 & info.xmax == 86400)
        idxday = find(floor(info.ss.start)==info.date);
        ssday = info.ss(idxday);

        info.xmin = ssday(1).starttime;
        info.xmax = ssday(ssday.len).endtime;
    else
        info.xmin = 0;
        info.xmax = 86400;
    end
case 28 % Toggle Activities
    info.drawAct = ~info.drawAct;
case 29 % Toggle Activities
    info.drawText = ~info.drawText;
case 30
    info.drawInfAct = ~info.drawInfAct;
end

% Clear screen from last drawings
clf;

% Set figure to ADD mode
hold on;

%Initialize plot
initplot(info.ss, info.sensor_list, info.sensor_labels, info.date, info.xmin, info.xmax, info.ymin, info.ymax);

% Check whether activities need to be drawn
if (info.drawAct)
    plotas(info.as, info.activity_list, info.activity_labels, info.date, info.drawText,[0.75 0.75 1.0]);

%     % Create static text
%     hUI = uicontrol('Style','text',...
%                 'Position',[20 25 250 20],...
%                 'String','Activity selected: (none)',...
%                 'tag', 'unique_static_output');
end

if (info.drawInfAct)
    plotas(info.infered_as, info.infered_activity_list, info.infered_activity_labels, info.date, info.drawText,[1.0 0.75 0.75]);
end
% Draw sensor readings
plotss(info.ss, info.sensor_list, info.sensor_labels, info.date, info.xmin, info.xmax, info.ymin, info.ymax);


% Create buttons
uicontrol('Style','pushbutton',...
			'Position',[20 1 60 20],...
			'String','Prev Day',...
			'CallBack','sensorGUIHandler(21)');
        
uicontrol('Style','pushbutton',...
			'Position',[90 1 60 20],...
			'String','Next Day',...
			'CallBack','sensorGUIHandler(22)');        

uicontrol('Style','pushbutton',...
			'Position',[160 1 60 20],...
			'String','Prev Week',...
			'CallBack','sensorGUIHandler(23)'); 

uicontrol('Style','pushbutton',...
			'Position',[230 1 60 20],...
			'String','Next Week',...
			'CallBack','sensorGUIHandler(24)'); 
        
uicontrol('Style','pushbutton',...
			'Position',[300 1 60 20],...
			'String','Prev Month',...
			'CallBack','sensorGUIHandler(25)'); 

uicontrol('Style','pushbutton',...
			'Position',[370 1 60 20],...
			'String','Next Month',...
			'CallBack','sensorGUIHandler(26)'); 

uicontrol('Style','pushbutton',...
        'Position',[440 1 80 20],...
        'String','Zoom toggle',...
        'CallBack','sensorGUIHandler(27)'); 

if (~isempty(info.as))
    uicontrol('Style','pushbutton',...
			'Position',[530 1 60 20],...
			'String','Activities',...
			'CallBack','sensorGUIHandler(28)'); 

    if (info.drawAct)
    uicontrol('Style','pushbutton',...
            'Position',[600 1 80 20],...
            'String','Text toggle',...
            'CallBack','sensorGUIHandler(29)'); 
    end

end

if (~isempty(info.infered_as))
    uicontrol('Style','pushbutton',...
			'Position',[690 1 60 20],...
			'String','Infered',...
			'CallBack','sensorGUIHandler(30)'); 
end

        
% Restore info structure
set(fig,'UserData',info)
