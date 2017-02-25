
%% PARAMETERS
clear;

fs = 44100;
fmin = 27.5;
B = 48;
gamma = 20; 
fmax = fs/2;

addpath(genpath('../export_fig'));

%% INPUT SIGNAL
% ey interested
% Andrea_Arsenault_Female_Native_Planned_High_Clean_Off_
ey_int = 'wavs/ey_interest';
% ey_list = {'d960515_23947.wav', 'd960515_23955.wav',...
% 'd960515_24767.wav', 'd960515_24941.wav' 'd960515_25017.wav'};

% Linda_Wertheimer_Female_Native_Planned_High_Clean_Off_
ey_list = {
    'j960607a_78601.wav',
    'j960607a_78347.wav',
    'j960531d_77887.wav',
    'j960531c_77271.wav',
    'j960607a_78513.wav'
    };

% Craig_Wintom_Male_Native_Planned_High_Clean_Off_
% ey_list = {
%     'j960617_86927.wav',
%     'j960617_85701.wav',
%     'j960618b_88083.wav',
%     'j960617_85727.wav',
%     'j960618b_88123.wav'
%     };

% Alan_Cheuse_Male_Native_Planned_High_Clean_Off_
% with less samples
% ey_list = {
%     'j960613d_82295.wav',
%     'j960613d_82297.wav',
%     'j960613d_82299.wav',
%     'j960613d_82301.wav',
%     'j960613d_82303.wav'
%     };

% Amir_Noiman_Male_Nonnative_Spontaneous_High_Clean_Off_
% ey_list = {
%     'b960530_10369.wav',
%     'b960530_10371.wav',
%     'b960530_10377.wav',
%     'b960530_10379.wav',
%     'b960530_10381.wav'
%     };

% j960522a_F_NUS_002_Female_Nonnative_Spontaneous_High_Clean_Off_
% ey_list = {
%     'j960522a_72149.wav',
%     'j960522a_72155.wav',
%     'j960522a_72171.wav',
%     'j960522a_72211.wav',
%     'j960522a_72201.wav'
%     };

% breath interested
% David_Brancaccio_Male_Native_Planned_High_Clean_Off_
br_int = 'wavs/br_interest';
% br_list = {'k960529_61885.wav', 'k960529_61925.wav',... 
% 'k960529_61997.wav', 'k960529_62177.wav', 'k960529_62395.wav'};

% ATC_M_US_A7_Male_Native_Planned_High_Clean_Off_
% with less samples
% br_list = {
%     'j960617_58571.wav',
%     'j960617_58579.wav',
%     'j960617_58593.wav',
%     'j960617_58595.wav',
%     'j960617_58609.wav'
%     };

% Amy_Bernstein_Female_Native_Planned_High_Clean_Off_
br_list = {
    'j960522b_46989.wav',
    'j960522b_46995.wav',
    'j960522b_47019.wav',
    'j960522b_47053.wav',
    'j960522b_47063.wav'
    };

% Linda_Wertheimer_Female_Native_Planned_High_Clean_Off_
% br_list = {
%     'j960521d_46169.wav',
%     'j960522a_46419.wav',
%     'j960522a_46589.wav',
%     'j960522a_46591.wav',
%     'j960522a_46595.wav'
%     };

% figure;
cc = cell(5);
for i = 1:5
    wav = fullfile(ey_int, ey_list{i});
    save_to = 'ey_andrea';
%     wav = fullfile(br_int, br_list{i});
%     save_to = 'br_david';

x = audioread(wav);
x = x(:); xlen = length(x);
%% COMPUTE COEFFIENTS
% full rasterized transform
Xcq = cqt(x, B, fs, fmin, fmax, 'rasterize', 'full','gamma', gamma);

% piecewise rasterized transform
% Xcq = cqt(x, B, fs, fmin, fmax,  'rasterize', 'piecewise', 'format', 'sparse', 'gamma', gamma);

% no rasterization
% Xcq = cqt(x, B, fs, fmin, fmax, 'rasterize', 'none', 'gamma', gamma);

c = Xcq.c;
cc{i} = c;
%% ICQT
[y gd] = icqt(Xcq);

%% RECONSTRUCTION ERROR [dB]
SNR = 20*log10(norm(x-y)/norm(x));
disp(['reconstruction error = ' num2str(SNR) ' dB']);

%% REDUNDANCY
if iscell(c)
   disp(['redundancy = ' num2str( (2*sum(cellfun(@numel,c)) + ...
       length(Xcq.cDC) + length(Xcq.cNyq)) / length(x))]); 
elseif issparse(c)
   disp(['redundancy = ' num2str( (2*nnz(c) + length(Xcq.cDC) + ...
       length(Xcq.cNyq)) / length(x))]);  
else
   disp(['redundancy = ' num2str( (2*size(c,1)*size(c,2) + ...
       length(Xcq.cDC) + length(Xcq.cNyq)) / length(x))]); 
end

% TEST
cb = abs(c);
c_dct = dct(cb);
dct_h = c_dct;
dct_l = c_dct;
dct_m = c_dct;
dct_h(50:end, :) = 0; % preserve high
dct_l(1:50, :) = 0; % preserve low
uc = envelope(cb);
subplot(221), imagesc(idct(c_dct));
subplot(222), imagesc(idct(dct_h));
subplot(223), imagesc(idct(dct_l));
% subplot(224), imagesc(20*log10(cb));

pause;

%% PLOT
if 0
switch(Xcq.rast)
    case 'full'
        subplot(2, 3, i);
        imagesc(20*log10(abs(flipud(c))+eps));
        hop = xlen/size(c,2);
        xtickVec = 0:round(fs/hop):size(c,2)-1;
        set(gca,'XTick',xtickVec);
        ytickVec = 0:B:size(c,1)-1;
        set(gca,'YTick',ytickVec);
        ytickLabel = round(fmin * 2.^( (size(c,1)-ytickVec)/B));
        set(gca,'YTickLabel',ytickLabel);
        xtickLabel = 0 : length(xtickVec) ;
        set(gca,'XTickLabel',xtickLabel);
        xlabel('time [s]', 'FontSize', 12, 'Interpreter','latex'); 
        ylabel('frequency [Hz]', 'FontSize', 12, 'Interpreter','latex');
        set(gca, 'FontSize', 10);
        title(ey_list{i});

    case 'piecewise'
        if strcmp(Xcq.format, 'sparse')
            cFill = cqtFillSparse(c,Xcq.M,Xcq.B);
            figure; imagesc(20*log10(abs(flipud(cFill))));

            hop = xlen/size(c,2);
            xtickVec = 0:round(fs/hop):size(c,2)-1;
            set(gca,'XTick',xtickVec);
            ytickVec = 0:B:size(c,1)-1;
            set(gca,'YTick',ytickVec);
            ytickLabel = round(fmin * 2.^( (size(c,1)-ytickVec)/B));
            set(gca,'YTickLabel',ytickLabel);
            xtickLabel = 0 : length(xtickVec) ;
            set(gca,'XTickLabel',xtickLabel);
            xlabel('time [s]', 'FontSize', 12, 'Interpreter','latex'); 
            ylabel('frequency [Hz]', 'FontSize', 12, 'Interpreter','latex');
            set(gca, 'FontSize', 10);
        else
            figure; plotnsgtf(c.',Xcq.shift,fs,fmin,fmax,B,2,60);
        end
        
    otherwise
        figure; plotnsgtf({Xcq.cDC Xcq.c{1:end} Xcq.cNyq}.',Xcq.shift,fs,fmin,fmax,B,2,60); 
end
end

end

% export_fig(save_to, '-jpg');

