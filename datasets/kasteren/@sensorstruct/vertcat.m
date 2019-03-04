function [ds2] = vertcat(ds, varargin)

if (isempty(ds))
    d = [];
else
    d = ds.d;
end    

for i=1:nargin-1,
  d2 = varargin{i}.d;
  d = [d; d2];
end;

[dummy idx] = sort(d(:,1));
ds2 = sensorstruct(d(idx,1),d(idx,2),d(idx,3),d(idx,4));
