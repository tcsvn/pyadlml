function  display(ss)

% SENSORSTRUCT/DISPLAY		Display Function
%
%
% Note	:

% Uses :

% Change History :
% Date           Time	Prog	Note
% 19-July-2006	 11:28  TvK     Created under MATLAB 7.1.0.246

% Tvk = Tim van Kasteren
% University of Amsterdam (UvA) - Intelligent Autonomous Systems (IAS) group
% e-mail : tlmkaste@science.uva.nl
% website: http://www.science.uva.nl/~tlmkaste/

if (ss.idishex==0)
    fprintf(1,'\nStart time          \tEnd time            \tID\tVal\n'); 
    fprintf(1,  '--------------------\t--------------------\t--\t---\n'); 
else
    fprintf(1,'\nStart time          \tEnd time            \tID\t\t\t\t\tVal\n'); 
    fprintf(1,  '--------------------\t--------------------\t----------------\t---\n'); 
end

for i=1:size(ss.d,1)
    if (ss.idishex==0)
        fprintf(1, '%s\t%s\t%d\t%d\n', datestr(ss.d(i,1)),datestr(ss.d(i,2)),ss.d(i,3),ss.d(i,4));
    else
        fprintf(1, '%s\t%s\t%s\t%d\n', datestr(ss.d(i,1)),datestr(ss.d(i,2)),num2hex(ss.d(i,3)),ss.d(i,4));
    end
end
    
if (size(ss.d,1)>1)
    fprintf(1, 'Length: %d\n', size(ss.d,1));
end
