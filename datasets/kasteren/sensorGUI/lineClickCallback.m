function lineClickCallback(src, evt)

% Read label from clicked item 
label = get(src, 'tag');

szOut = ['Activity selected: ' label];
hUI = findobj('tag', 'unique_static_output');
set(hUI, 'String', label);