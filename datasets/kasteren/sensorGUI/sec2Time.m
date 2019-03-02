function [hours, minutes, seconds, ampm] = sec2Time(totalseconds)
 
hours = floor(totalseconds/3600);
 
if (hours < 12) ampm = 'a';
else ampm = 'p';
end

if (hours == 0) hours = 12;
elseif (hours > 12) hours = mod(hours, 13) + 1;
end
    
minutes = floor(mod(totalseconds,3600)/60);
 
seconds = mod(mod(totalseconds,3600),60);
