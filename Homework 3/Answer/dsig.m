% BENG420/520 Homework #3
% function ds = dsig(a)
%     a - activation potential
% return value of the first order derivative of a (scaled) sigmoid function

function ds = dsig(a)

 ds = 2 * exp(-a) ./ ((1+exp(-a)).^2);
 
end