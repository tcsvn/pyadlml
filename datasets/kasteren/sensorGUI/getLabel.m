function label = getLabel(sensor_labels, id);

label = [];

for i=1:size(sensor_labels,1),
    if (sensor_labels{i,1}==id),
        if (size(sensor_labels,2)==3)
            label = [sensor_labels{i,2} ' ' sensor_labels{i,3}];
        else
            label = sensor_labels{i,2};
        end
        return;
    end
end