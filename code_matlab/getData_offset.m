% load data mat with the specifed triggers and channels
% channel = 'all' to select all channels

function [dat_concat, times, srate] = getData_offset(sub, triggers, offset_win, channels, correctness, session)

%load('global_var.mat')
p_prepro = 'preprocessing';

EEG = pop_loadset(fullfile(p_prepro, [num2str(sub),'_epochs_ica_a2.set']));
dat_concat = [];
for ti = 1:length(triggers)
    trigger = triggers(ti);
    duration = getDuration(trigger);

    timewin = offset_win + duration;
    baseline = [-0.2 0] + duration;

    EEG = pop_rmbase(EEG, baseline*1000);

    if ~strcmp(channels, 'all')
        chanid = ismember({EEG.chanlocs.labels}, channels);
    else
        chanid = ones(1, size(EEG.data,1));
    end
    chans = find(chanid);

    trigger = prepareTriggers(trigger);
    eeg = pop_epoch(EEG, trigger, timewin); 
    tp = pop_epoch(EEG, trigger, [-0.2 0.2]); % use a window containing 0 to load the events 
    eeg.event = tp.event;

    if eeg.trials==0
        continue
    end

    rows = repelem(logical(1),eeg.trials)';

    T = struct2table(eeg.event);
    T(~ismember(T.type, trigger),:) = []; % remove ghose events
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
    
    dat = eeg.data(chans,:,rows);

    if isempty(dat_concat)
        dat_concat = dat;
    else
        dat_concat(:,:,end+1:end+size(dat,3)) = dat;
    end

    if 1
        disp(['Trigger is ', trigger])
        disp(['Load ', num2str(size(dat,3)), ' data'])
    end

end

times = eeg.times - duration*1000;
srate = eeg.srate;
disp(['Load ', num2str(size(dat_concat,3)), ' trials.'])
end

