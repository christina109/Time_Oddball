function [dat, times, srate, subject_arr] = getData_N(subjects, triggers, timewin,  channels, offsetLockedOn, correctness, session)
% load and return data matrix from mulitple subjects
% concatenated at the trial dimension

wb = waitbar(0, 'Loading datasets...');
for si = 1:length(subjects)
    subject = subjects(si);
    [tp, times, srate] = getData(subject, triggers, timewin,  channels, offsetLockedOn, correctness, session);
       
    if si == 1
        dat = tp;
        subject_arr = repelem(subject, size(tp,3));
    else
        dat(:,:,end+1:end+size(tp,3)) = tp;
        subject_arr(end+1:end+size(tp,3)) = subject;
    end
    waitbar(si/length(subjects), wb, 'Loading datasets...');
end
close(wb)