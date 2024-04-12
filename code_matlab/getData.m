function [dat, times, srate] = getData(sub, triggers, timewin, channels, offsetLockedOn, correctness, session)
% load data mat with the specifed triggers and channels
% channel = 'all' to select all channels
% correctness = [], load all trials; 0/1 load correct/incorrect trials only
if nargin < 5
    offsetLockedOn = 0;
end

if nargin < 6
    correctness = [];
end

if nargin < 7
    session = []; % all sessions
end

if offsetLockedOn
    [dat, times, srate] = getData_offset(sub, triggers, timewin, channels, correctness,session);
    disp('Time locked at the OFFSET of the stimulus.')
    return
end

p_prepro = 'preprocessing';

%load('global_var.mat')

EEG = pop_loadset(fullfile(p_prepro, [num2str(sub),'_epochs_ica_a2.set']));

if ~strcmp(channels, 'all')
    chanid = ismember({EEG.chanlocs.labels}, channels);
else
    chanid = ones(1, size(EEG.data,1));
end
chans  = find(chanid);


if isnumeric(triggers(1))
    triggers = prepareTriggers(triggers);
end
eeg = pop_epoch(EEG, triggers, timewin); 

rows = repelem(logical(1),eeg.trials)';

T = struct2table(eeg.event);
T(~ismember(T.type, triggers),:) = []; % remove ghose events
% load by correctness
if ~isempty(correctness)
    disp('Loading trials by the specified CORRECTNESS')
    try
        rows_logical = cellfun(@(x) isequal(x, ~correctness), T.corr);  % exclude the unwanted trials
    catch
        rows_logical = T.corr == ~correctness;
    end
    rows_logical = ~rows_logical; % flip for selection
    rows = rows & rows_logical;
end

% load by session
if ~isempty(session)
    disp('Loading trials by the specified SESSION')
    try
        rows_logical = cellfun(@(x) isequal(x, session), T.sess);
    catch
        rows_logical = T.sess == session;
    end
    rows = rows & rows_logical;
end

dat = eeg.data(chans, :, rows);

times = eeg.times;
srate = EEG.srate;

disp('Time locked at the ONSET of the stimulus.')
disp(['Load ', num2str(size(dat,3)), ' trials.'])

end

