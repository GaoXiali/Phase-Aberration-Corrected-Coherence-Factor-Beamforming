% File: denoise_sinogram.m
% Purpose: Band-pass filtering helper for photoacoustic sinogram data.
% Authors: Xiali Gao
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function clean_sinogram = denoise_sinogram(raw_sinogram)

    fs = 40e6; % Sampling frequency, Hz.
    fc = 3.6e6; % Transducer center frequency, Hz.
    fwhm_pct = 1.2; % Fractional bandwidth used for the band-pass filter.

    f_low = fc * (1 - fwhm_pct/2);
    f_high = fc * (1 + fwhm_pct/2);

    f_low = max(f_low, 100e3);
    f_nyquist = fs / 2;
    f_high = min(f_high, f_nyquist - 1e5);

    Wn = [f_low, f_high] / (fs / 2);
    order = 4;
    [b, a] = butter(order, Wn, 'bandpass');

    [~, ~, num_frames] = size(raw_sinogram);
    clean_sinogram = zeros(size(raw_sinogram), 'like', raw_sinogram);

    fprintf('Band-pass filtering (%0.2f - %0.2f MHz)...\n', f_low/1e6, f_high/1e6);

    for k = 1:num_frames

        current_frame = raw_sinogram(:, :, k);

        temp = current_frame';

        temp_filtered = filtfilt(b, a, double(temp));

        clean_sinogram(:, :, k) = temp_filtered';
    end

    fprintf('Denoising finished.\n');

end
