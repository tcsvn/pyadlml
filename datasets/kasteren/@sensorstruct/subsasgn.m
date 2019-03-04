function [ss] = subsasgn(ss, S, x)
% NOTETRACK/SUBSASGN		Subscripted assignment
%
%  NT.FIELD = X
%  NT(IDX).FIELD = X
%  NT{T_START, <T_END>}.FIELD = X
% 
% Inputs :
%	NT : Notetrack Object
%       FIELD : A Fields name from {'t', 'dur', 'pitch', 'vol', 'chan', 'len'};   
%       X : Values
%
% Usage Example : nt(idx).vol = nt(idx).vol + 10
%                 nt.len = 2
%                 nt(nt.pitch < 60) = [];
%
%
% Note	:
% See also

% Uses :

% Change History :
% Date		Time		Prog	Note
% 03-Feb-2000	 4:30 PM	ATC	Created under MATLAB 5.2.0.3084

% ATC = Ali Taylan Cemgil,
% SNN - University of Nijmegen, Department of Medical Physics and Biophysics
% e-mail : cemgil@mbfys.kun.nl 

switch S(1).type,
  case '.',
      switch(S(1).subs)
          case 'idishex'
              ss.idishex = x;
          otherwise
              error('Cannot assign objects this way, use constructor and concatenation')
      end
  case '()',
    error('Cannot assign objects this way, use constructor and concatenation')
  case '{}',
    error('Cell array indexing not supported by sensorstruct objects')
  otherwise,
    error('see : help sensorstruct;');
end;  
