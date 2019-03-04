function  display(as)

% ACTSTRUCT/DISPLAY		Display Function
%
%
% Note	:

% Uses :

% Change History :
% Date           Time	Prog	Note
% 26-July-2006	 11:45  TvK     Created under MATLAB 7.1.0.246

% Tvk = Tim van Kasteren
% University of Amsterdam (UvA) - Intelligent Autonomous Systems (IAS) group
% e-mail : tlmkaste@science.uva.nl
% website: http://www.science.uva.nl/~tlmkaste/

if (size(as.d,2)==3)
    fprintf(1,'\nStart time          \tEnd time            \tID\n'); 
    fprintf(1,  '--------------------\t--------------------\t--\n'); 
elseif(size(as.d,2)==4)
    fprintf(1,'\nStart time          \tEnd time            \tID\tLocation\n'); 
    fprintf(1,  '--------------------\t--------------------\t--\t--------\n');    
end

if (size(as.d,2)==3)
    for i=1:size(as.d,1)
        fprintf(1, '%s\t%s\t%d\n', datestr(as.d(i,1)),datestr(as.d(i,2)),as.d(i,3));
    end
elseif(size(as.d,2)==4)
    for i=1:size(as.d,1)
        fprintf(1, '%s\t%s\t%d\t%d\n', datestr(as.d(i,1)),datestr(as.d(i,2)),as.d(i,3), as.d(i,4));
    end
end
if (size(as.d,1)>1)
    fprintf(1, 'Length: %d\n', size(as.d,1));
end

    

