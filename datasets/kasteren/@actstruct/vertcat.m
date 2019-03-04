function [as2] = vertcat(as, varargin)

if (isempty(as))
    d = [];
else
    d = as.d;
end    

for i=1:nargin-1,
  d2 = varargin{i}.d;
  d = [d; d2];
end;

[dummy idx] = sort(d(:,1));
if(size(d,2)==3)
    as2 = actstruct(d(idx,1),d(idx,2),d(idx,3));
elseif(size(d,2)==4)
    as2 = actstruct(d(idx,1),d(idx,2),d(idx,3), d(idx,4));
else
    error('Incorrect dataformat for actstruct')
end
    
