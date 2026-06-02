% File: ds_fwhm_metrics.m
% Purpose: FWHM metric helper for normalized reconstructed volumes.
% Authors: Xiali Gao; Hao Huang
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function fwhm = ds_fwhm_metrics(img, grid)

[~, peak_idx] = max(img(:));
[ix, iy, iz] = ind2sub(size(img), peak_idx);

fwhm = struct();
fwhm.x_mm = local_fwhm_1d(grid.x, squeeze(img(:, iy, iz))) * 1e3;
fwhm.y_mm = local_fwhm_1d(grid.y, squeeze(img(ix, :, iz))) * 1e3;
fwhm.z_mm = local_fwhm_1d(grid.z, squeeze(img(ix, iy, :))) * 1e3;
fwhm.peak_index = [ix, iy, iz];
fwhm.peak_m = [grid.x(ix), grid.y(iy), grid.z(iz)];
end

function width = local_fwhm_1d(axis_m, profile)
profile = double(profile(:));
axis_m = axis_m(:);
if isempty(profile) || max(profile) <= 0
    width = NaN;
    return;
end

profile = profile ./ max(profile);
[~, peak] = max(profile);
half = 0.5;

left = find(profile(1:peak) < half, 1, 'last');
right_rel = find(profile(peak:end) < half, 1, 'first');

if isempty(left) || isempty(right_rel)
    width = NaN;
    return;
end

right = peak + right_rel - 1;
x_left = interp1(profile([left, left + 1]), axis_m([left, left + 1]), half, 'linear');
x_right = interp1(profile([right - 1, right]), axis_m([right - 1, right]), half, 'linear');
width = abs(x_right - x_left);
end
