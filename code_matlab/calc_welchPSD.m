function [psd, freq, mat] = calc_welchPSD(dat, srate, plotOn, logOn)
% input: nEpoch x nPnt
% output: PSD averaged across epochs
% multiple input channels will be averaged
% correspondance: cyj.sci@gmail.com

if nargin < 4
    logOn = 1;
end

mat = zeros(size(dat,1), size(dat,2)/2+1);
for ei = 1:size(dat,1)
    x = dat(ei,:);
    if size(x,1)==1
        x = x';
    end
    Nx = length(x);
    nfft = Nx;
    % Window data
    w = hanning(Nx); 
    [Pxx2,~] = pwelch(x,w,0,nfft,srate);
    mat(ei, :) = Pxx2;
end

mean_psdx = mean(mat, 1);
if logOn
    psd = 10*log10(mean_psdx);
else
    psd = mean_psdx;
end

freq = 0:srate/Nx:srate/2;

if plotOn
    plot(freq,psd)
    grid on
    title('Periodogram Using FFT')
    xlabel('Frequency (Hz)')
    if logOn
        ylabel('Power/Frequency (dB/Hz)')
    else
        ylabel('Power/Frequency (V^2/Hz)')
    end
end


end