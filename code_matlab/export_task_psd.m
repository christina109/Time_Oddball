function export_task_psd(subject, grating_freq)
% export PSD: before vs. after stimulus onset, per motion speed, per

triggers  = [1,5]+grating_freq*2;
timewin  = [-0.8, 1.2];
channels = 'all';
offset   = 0;
correctness = 1;

[dat, times, srate] = getData(subject, triggers, timewin, channels, offset, correctness, []);

win_rs = [-800,-200];
win_st = [0,600];
pnt_rs = dsearchn(times', win_rs');
pnt_st = dsearchn(times', win_st');

dat_rs = dat(:,pnt_rs(1):pnt_rs(2)-1,:);
dat_st = dat(:,pnt_st(1):pnt_st(2)-1,:);

%dat_rs = [zeros(size(dat_rs,1), 4700, size(dat_rs,3)), dat_rs];
%dat_st = [zeros(size(dat_st,1), 4700, size(dat_st,3)), dat_st];

if 0
    figure
    chani = 16;
    trial = 1000;
    plot(real(dat_rs(chani,:,trial)))
    hold on
    plot(real(dat_st(chani,:,trial)))

    figure
    %[psd0, frex0] = calc_welchPSD(dat_rs(chani,4701:end,trial), srate, 0, 1);
    [psd1, frex1] = calc_welchPSD(dat_rs(chani,:,trial), srate, 0, 1);
    %plot(frex0, psd0)
    %hold on
    plot(frex1, psd1)
end

wb = waitbar(0,'Getting the PSDs...');
counti = 0;
for chani = 1:size(dat_rs,1)
    for trial = 1:size(dat_rs,3)
        [psd1, frex] = calc_welchPSD(dat_rs(chani,:,trial), srate, 0, 0);
        [psd2, frex] = calc_welchPSD(dat_st(chani,:,trial), srate, 0, 0);
        if chani == 1 && trial == 1
            psd_rs = zeros(size(dat_rs,1), length(psd1), size(dat_rs,3));
            psd_st = psd_rs;
        end
        psd_rs(chani, :, trial) = psd1;
        psd_st(chani, :, trial) = psd2;
        counti = counti+1;
    	waitbar(counti/size(dat_rs,1)/size(dat_rs,3), wb,'Getting the PSDs...')
    end
end
close(wb)

save(fullfile('data_psd', [num2str(grating_freq),'Hz'], num2str(subject)), 'psd_rs', 'psd_st', 'srate', 'frex');

end