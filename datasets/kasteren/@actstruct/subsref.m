function ret = subsref(as, S)
% ACTSTRUCT/ACTSTRUCT		A class for storing data from datasets in a
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
            ret = actstruct(as.d(idx,:));
            if (size(S,2)>1), ret = subsref(ret, S(2:end)); end;
        case 2
            idx = S(1).subs{1};
            idx2= S(1).subs{2};
            ret = as.d(idx,idx2);
        otherwise
            error('see : help actstruct/subsref;'); 
    end
case '.'
  switch S(1).subs
    case 'len'
        ret = size(as.d,1);
    case 'd'
        ret = as.d;
    case 'start'
        ret = as.d(:,1);
    case 'startstr'
        ret = datestr(as.d(:,1));
    case 'startdate'
        ret = floor(as.d(:,1));
    case 'starttime'
        ret = round((as.d(:,1)-floor(as.d(:,1)))*86400);
    case 'startsecs'
        ret = as.d(:,1)*86400;
    case 'end'
        ret = as.d(:,2);
    case 'endstr'
        ret = datestr(as.d(:,2));
    case 'enddate'
        ret = floor(as.d(:,2));
    case 'endtime'
        ret = round((as.d(:,2)-floor(as.d(:,2)))*86400);
    case 'endsecs'
        ret = as.d(:,2)*86400;    
    case 'dur'
        % USE DATEVEC TO GET TIME NOT DATESTR!!
        ret = as.d(:,2) - as.d(:,1);
    case 'durvec'
        % USE DATEVEC TO GET TIME NOT DATESTR!!
        ret = datevec(as.d(:,2) - as.d(:,1));
    case 'id'
        ret = as.d(:,3);
    case 'getIDs'
        ret = unique(as.d(:,3));        
    case 'loc'
        ret = as.d(:,4);
    case 'getLocs'
        ret = dedup(as.d(:,4));        
    case 'haveLocs'
        ret = (size(as.d,2)==4);
    otherwise
        error('Invalid field name')
  end
  
case '{}'
    error('Cell array indexing not supported by actstruct objects')
end