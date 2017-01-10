% Convert wav files to constant-q transformed files.
%
% wzhao1 cs cmu edu
% 11/20/2016

clear;
close all;
addpath(genpath('./CQT_toolbox_2013'));

% Paras
fs = 44100;
fmin = 27.5;
num_bin_per_octave = 48;
gamma = 20;
fmax = fs/2;

% Data config
% wav_path = './wavs/EY_full';
wav_path = './wavs/BREATH_full';
% interest_list = './ey.interested';
interest_list = './breath.interested';
% save_path = './feat_constq/ey';
save_path = './feat_constq/breath';

fid = fopen(interest_list);
raw = textscan(fid, '%s%s%s', 'Delimiter', ' ');
file_list = raw{1};
speaker_ids = raw{2};
num_files = length(file_list);

parfor f = 1:num_files
    fn = fullfile(wav_path, file_list{f});
    x = audioread(fn);
    x = x(:);
    xlen = length(x);
    % full rasterized transform
    Xcq = cqt(x, num_bin_per_octave, fs, fmin, fmax, 'rasterize', 'full','gamma', gamma);
    % piecewise rasterized transform
%     Xcq = cqt(x, num_bin_per_octave, fs, fmin, fmax,  'rasterize', 'piecewise', 'format', 'sparse', 'gamma', gamma);
    % no rasterization
%     Xcq = cqt(x, num_bin_per_octave, fs, fmin, fmax, 'rasterize', 'none', 'gamma', gamma);
    
    c = Xcq.c;
    
    % Plot
    if 0
        switch(Xcq.rast)
            case 'full'
                figure; imagesc(20*log10(abs(flipud(c))+eps));
                hop = xlen/size(c,2);
                xtickVec = 0:round(fs/hop)/100:size(c,2)-1;
                set(gca,'XTick',xtickVec);
                ytickVec = 0:num_bin_per_octave:size(c,1)-1;
                set(gca,'YTick',ytickVec);
                ytickLabel = round(fmin * 2.^( (size(c,1)-ytickVec)/num_bin_per_octave));
                set(gca,'YTickLabel',ytickLabel);
                xtickLabel = 0 : length(xtickVec) ;
                set(gca,'XTickLabel',xtickLabel);
                xlabel('time [0.01s]', 'FontSize', 12, 'Interpreter','latex');
                ylabel('frequency [Hz]', 'FontSize', 12, 'Interpreter','latex');
                set(gca, 'FontSize', 10);
                pause(0.8);
            case 'piecewise'
                if strcmp(Xcq.format, 'sparse')
                    cFill = cqtFillSparse(c,Xcq.M,Xcq.B);
                    figure; imagesc(20*log10(abs(flipud(cFill))));
                    
                    hop = xlen/size(c,2);
                    xtickVec = 0:round(fs/hop):size(c,2)-1;
                    set(gca,'XTick',xtickVec);
                    ytickVec = 0:num_bin_per_octave:size(c,1)-1;
                    set(gca,'YTick',ytickVec);
                    ytickLabel = round(fmin * 2.^( (size(c,1)-ytickVec)/num_bin_per_octave));
                    set(gca,'YTickLabel',ytickLabel);
                    xtickLabel = 0 : length(xtickVec) ;
                    set(gca,'XTickLabel',xtickLabel);
                    xlabel('time [s]', 'FontSize', 12, 'Interpreter','latex');
                    ylabel('frequency [Hz]', 'FontSize', 12, 'Interpreter','latex');
                    set(gca, 'FontSize', 10);
                else
                    figure; plotnsgtf(c.',Xcq.shift,fs,fmin,fmax,num_bin_per_octave,2,60);
                end
                
            otherwise
                figure; plotnsgtf({Xcq.cDC Xcq.c{1:end} Xcq.cNyq}.',Xcq.shift,fs,fmin,fmax,num_bin_per_octave,2,60);
        end
    end
    
    % Save
    sfn = regexp(file_list{f}, '(.+)([^.wav])', 'match');
    sfn = strcat(sfn{1}, '.txt');
    dlmwrite(fullfile(save_path, sfn), abs(c));
    disp(strcat('pool ', num2str(f)));
    disp(strcat('Saved ', sfn));
end

delete(gcp);
