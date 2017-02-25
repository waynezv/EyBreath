clear;
addpath(genpath('../export_fig'));

% Linda_Wertheimer_Female_Native_Planned_High_Clean_Off_
% ey_list = {
%     'j960607a_78601.wav',
%     'j960607a_78347.wav',
%     'j960531d_77887.wav',
%     'j960531c_77271.wav',
%     'j960607a_78513.wav'
%     };
% saveto = 'linda_logspec';

% Craig_Wintom_Male_Native_Planned_High_Clean_Off_
% ey_list = {
%     'j960617_86927.wav',
%     'j960617_85701.wav',
%     'j960618b_88083.wav',
%     'j960617_85727.wav',
%     'j960618b_88123.wav'
%     };
% saveto = 'craig_logspec';

% j960522a_F_NUS_002_Female_Nonnative_Spontaneous_High_Clean_Off_
ey_list = {
    'j960522a_72149.wav',
    'j960522a_72155.wav',
    'j960522a_72171.wav',
    'j960522a_72211.wav',
    'j960522a_72201.wav'
    };
saveto = 'j960522a_logspec';

figure;
for i = 1:5
    wf = ey_list{i};
    wf = regexp(wf, '.wav', 'split');
    wf = wf(1);
    fn = strcat(wf,'.80-7200_40filts.lspec.ascii');
    fn = fn{1};
    
    fid = fopen(fn);
    raw = textscan(fid,'%s','Delimiter','\n');
    fclose(fid);
    
    spec = [];
    raw = raw{1};
    nr = size(raw, 1);
    for j = 3:nr
        ln = raw{j};
        ln = regexp(ln, '\S+', 'match');
        ln = cellfun(@(x) str2num(x), ln);
        ln = ln(2:end);
        spec = [spec; ln];
    end
    subplot(2,3,i);
    imagesc(spec');
    title(wf);
end

export_fig(saveto, '-jpg');