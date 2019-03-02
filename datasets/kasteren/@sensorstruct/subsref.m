function ret = subsref(ss, S)
% SENSORSTRUCT/SENSORSTRUCT		A class for storing data from datasets in a
%                             standardized way
%
% Change History :
% Date           Time	Prog	Note
% 19-July-2006	 13:04  TvK     Created under MATLAB 7.1.0.246

% Tvk = Tim van Kasteren
% University of Amsterdam (UvA) - Intelligent Autonomous Systems (IAS) group
% e-mail : tlmkaste@science.uva.nl
% website: http://www.science.uva.nl/~tlmkaste/

switch S(1).type
case '()'
    N = length(S(1).subs);
    switch N
        case 1
            idx = S(1).subs{1};
            ret = sensorstruct(ss.d(idx,:));
            if (size(S,2)>1), ret = subsref(ret, S(2:end)); end;
        case 2
            idx = S(1).subs{1};
            idx2= S(1).subs{2};
            ret = ss.d(idx,idx2);
        otherwise
            error('see : help sensorstruct/subsref;'); 
    end
case '.'
  switch S(1).subs
    case 'len'
        ret = size(ss.d,1);
    case 'd'
        ret = ss.d;
    case 'start'
        ret = ss.d(:,1);
    case 'startdate'
        ret = floor(ss.d(:,1));
    case 'starttime'
        ret = round((ss.d(:,1)-floor(ss.d(:,1)))*86400);
    case 'startsecs' %start time in seconds
        ret = round(ss.d(:,1)*86400);
    case 'startstr'
        ret = datestr(ss.d(:,1));
    case 'end'
        ret = ss.d(:,2);
    case 'enddate'
        ret = floor(ss.d(:,2));
    case 'endtime'
        ret = round((ss.d(:,2)-floor(ss.d(:,2)))*86400);
    case 'endsecs' %end time in seconds
        ret = round(ss.d(:,2)*86400);
    case 'endstr'
        ret = datestr(ss.d(:,2));
    case 'dur'
        % USE DATEVEC TO GET TIME NOT DATESTR!!
        ret = as.d(:,2) - as.d(:,1);
    case 'durvec'
        % USE DATEVEC TO GET TIME NOT DATESTR!!
        ret = datevec(as.d(:,2) - as.d(:,1));
    case 'id'
        ret = ss.d(:,3);
    case 'getIDs'
        ret = unique(ss.d(:,3));        
    case 'val'
        ret = ss.d(:,4);
    case 'idishex'
        ret = ss.idishex;          
    otherwise
        error('Invalid field name')
  end
  
case '{}'
    error('Cell array indexing not supported by sensorstruct objects')
end