% To use this script, add the eeglab toolbox to working directory first
% Correspondance: cyj.sci@gmail.com

addpath(genpath('code_matlab'))
load('global_var', 'p_prepro');

%% batch export task psd
for subject = [201:230, 232:241]
    export_task_psd(subject, 5);
    export_task_psd(subject, 10);
    export_task_psd(subject, 20);
    export_task_psd(subject, 0);
end

%% export ERP
subjects = [201:230 232:241];
%triggersets = {[1,5],[11,15],[21,25],[41,45]};  % standard
%triggersets = {[2],[12],[22],[42]};             % minus_60
%triggersets = {[3],[13],[23],[43]};             % minus_40
%triggersets = {[4],[14],[24],[44]};             % minus_20
%triggersets = {[6],[16],[26],[46]};             % plus_20
%triggersets = {[7],[17],[27],[47]};             % plus_40
triggersets = {[8],[18],[28],[48]};              % plus_60
%triggersets = {[1],[11],[21],[41]};             % standard_de
%triggersets = {[5],[15],[25],[45]};             % standard_in
labels = {'0Hz', '5Hz', '10Hz', '20Hz'};
timewin = [-0.6,0.6];
channels = 'all';
sessions = []; 
offset = 1; 
correctness = 1;
load('global_var','chanlocs')

% loading data
erp_conds = {};
for ti = 1:length(triggersets)
    triggers = triggersets{ti};
    [dat, times,srate, subs_all] = getData_N(subjects,triggers, timewin,channels,offset,correctness,sessions);
    erp = zeros(size(dat,1), size(dat,2), length(subjects));
    fprintf('Computing ERPs...%s...',labels{ti})
    for si = 1:length(subjects)
        subject = subjects(si);
        tp = dat(:,:,subs_all==subject);
        erp(:,:,si) = mean(tp,3);
    end
    erp_conds{ti} = erp;
    fprintf('Done.\n')
end
%f_save = 'erp_gratings_standard';
f_save = fullfile('results_ERP', 'erp_offset_plus_60');
save(f_save, 'erp_conds', 'srate', 'times', 'subs_all', 'triggersets', 'labels', 'timewin', 'channels');

% all gratings
triggers = [triggersets{:}];
[dat, times,srate, subs_all] = getData_N(subjects,triggers, timewin,channels,offset,correctness,sessions);
erp_all = zeros(size(dat,1), size(dat,2), length(subjects));
fprintf('Computing ERPs...');
for si = 1:length(subjects)
    subject = subjects(si);
    tp = dat(:,:,subs_all==subject);
    erp_all(:,:,si) = mean(tp,3);
end
fprintf('Done.\n')
save(f_save, 'erp_all', '-append');




