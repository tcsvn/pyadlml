function [as] = actstruct(varargin)
% ACTSTRUCT/ACTSTRUCT		A class for storing data from datasets in a
%                             standardized way
%
%  as = ACTSTRUCT(START, END, ID)
%
% Inputs :
%	START : Starttime of activity (in datenum format)
%   END : Endtime of activity   (in datenum format)
%   ID : activity ID (int)
% LOCATION: location (optional)
%
% Outputs :
%	None
%
% Usage Example : as = actstruct(732877.4520, 732877.4521, 20)
%   activity 20 took place on 19-Jul-2006 from 10:50:52 till 10:51:01
%
% Note	:
% See also datenum

% Uses :

% Change History :
% Date           Time	Prog	Note
% 26-July-2006	 11:44  TvK     Created under MATLAB 7.1.0.246

% Tvk = Tim van Kasteren
% University of Amsterdam (UvA) - Intelligent Autonomous Systems (IAS) group
% e-mail : tlmkaste@science.uva.nl
% website: http://www.science.uva.nl/~tlmkaste/

if nargin == 0
   as.d = [];
   as = class(as,'actstruct')
   return;
end

as = struct('d',[]);

x = varargin{1};

% Handle various sorts of inputs as described in Matlab guidelines
switch class(x),
  case 'actstruct',
    as = x;
    return;
  case 'double',
    switch nargin,
        case 1,
            if (size(varargin{1},2)==3 | size(varargin{1},2)==4)
                d = varargin{1};
            else
                error('Invalid number of input variables, please use format: actstruct([start, end, id])');
            end
        case 3,
           d = [varargin{1} varargin{2} varargin{3}];
        case 4,
           d = [varargin{1} varargin{2} varargin{3} varargin{4}];
        otherwise,
           error('Invalid number of input variables, please use format: actstruct(start, end, id)');
    end,
    [dummy idx] = sort(d(:,1));
    as.d = d(idx,:);
  otherwise,
    error('Please check arguments to the ACTSTRUCT constructor...');
end;

as = class(as,'actstruct');
