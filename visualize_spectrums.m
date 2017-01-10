clear;
close all;

prefix = 'featvec_ey';

list = randperm(1000, 100);
for i = 1:length(list)
    ind = list(i);
    filename = strcat(num2str(ind), '.txt');
    show_log_spec(fullfile(prefix, filename));
    pause(2);
end