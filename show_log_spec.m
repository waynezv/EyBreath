% Visualize log-spectrum from file.

function show_log_spec(filename)
% close all;
fid = fopen(filename);
raw = textscan(fid,'%s','Delimiter','\n');
fclose(fid);
raw = raw{1};
id = raw{1};
num_frm = str2num(raw{2});
frm_cnt = 0;
feat = [];
expr1 = '\[\s+';
expr2 = '\s*\]';
for i = 3:length(raw)
    tmp = raw{i};
    if ~isempty(regexp(tmp, expr1, 'match')) % match begin
        tmp = regexp(tmp, expr1, 'split');
        tmp = regexp(tmp{2}, '\s+', 'split');
        tmp = cellfun(@(x) str2num(x), tmp);
    elseif ~isempty(regexp(tmp, expr2, 'match')) % match end
        tmp = regexp(tmp, expr2, 'split');
        tmp = regexp(tmp{1}, '\s+', 'split');
        tmp = cellfun(@(x) str2num(x), tmp);
        frm_cnt = frm_cnt + 1;
    else
        tmp = str2num(tmp);
    end
    feat = [feat tmp];
    if frm_cnt >= num_frm
        break;
    end
end
feat = reshape(feat, [num_frm, 40]);
% Visualize
% figure;
% subplot(211);
% imagesc(feat');
% colormap jet;
% xlabel('Time');
% ylabel('Freq');
% title(strcat('log mel spectrumgram, speaker ', num2str(id)));
% subplot(212);
% plot(feat);
% xlabel('Time');
% ylabel('Power');
end