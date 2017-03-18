function [filename] = dateFormat()
string = char(datetime);
string = regexprep(string, ' ', '_');
filename = regexprep(string, '[-:]', '');
end