function [ss] = sensorstruct(varargin)
% SENSORSTRUCT/SENSORSTRUCT		A class for storing data from datasets in a
%                             standardized way
%
%  ss = SENSORSTRUCT(START, END, ID, VAL)
%
% Inputs :
%	START : Starttime of sensor measurement (in datenum format)
%   END : Endtime of sensor measurement   (in datenum format)
%   ID : Sensor ID (int)
%   VAL: Value of sensor reading (double)
%
% Outputs :
%	None
%
% Usage Example : ss = datestruct(732877.4520, 732877.4521, 20, 1)
%   sensor 20 gave reading 1 on 19-Jul-2006 from 10:50:52 till 10:51:01
%
% Note	:
% See also datenum

% Uses :

% Change History :
% Date           Time	Prog	Note
% 19-July-2006	 10:25  TvK     Created under MATLAB 7.1.0.246

% Tvk = Tim van Kasteren
% University of Amsterdam (UvA) - Intelligent Autonomous Systems (IAS) group
% e-mail : tlmkaste@science.uva.nl
% website: http://www.science.uva.nl/~tlmkaste/

ss = struct('d',[], 'idishex', 0);

if nargin == 0
   ss.d = [];
   ss = class(ss,'sensorstruct');
   return;
end

x = varargin{1};

% Handle various sorts of inputs as described in Matlab guidelines
switch class(x),
  case 'sensorstruct',
    ss = x;
    return;
  case 'double',
    switch nargin,
        case 1,
            if (size(varargin{1},2)== 4)
                d = varargin{1};
            else
                error('Invalid number of input variables, please use format: sensorstruct([start, end, id, val])');
            end
        case 4,
           d = [varargin{1} varargin{2} varargin{3} varargin{4}];
        otherwise,
           error('Invalid number of input variables, please use format: sensorstruct(start, end, id, val)');
    end,
    [dummy idx] = sort(d(:,1));
	ss.d = d(idx,:);
  otherwise,
    error('Please check arguments to the SENSORSTRUCT constructor...');
end;

ss = class(ss,'sensorstruct');
