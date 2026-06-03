% File: ds_gcnr.m
% Purpose: Generalized contrast-to-noise ratio helper.
% Authors: Xiali Gao
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function value = ds_gcnr(target_values, background_values, num_bins)

target_values = double(target_values(:));
background_values = double(background_values(:));

lo = min([target_values; background_values]);
hi = max([target_values; background_values]);
if hi <= lo
    value = 0;
    return;
end

edges = linspace(lo, hi, num_bins + 1);
target_hist = histcounts(target_values, edges, 'Normalization', 'probability');
background_hist = histcounts(background_values, edges, 'Normalization', 'probability');
value = 1 - sum(min(target_hist, background_hist));
value = min(max(value, 0), 1);
end
