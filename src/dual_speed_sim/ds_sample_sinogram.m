% File: ds_sample_sinogram.m
% Purpose: Linear interpolation helper for sampling sinogram values at target arrival times.
% Authors: Xiali Gao
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function values = ds_sample_sinogram(sensor_data, t_array, tof)

num_pixels = size(tof, 1);
num_elements = size(tof, 2);
values = zeros(num_pixels, num_elements);

dt = t_array(2) - t_array(1);
t0 = t_array(1);
num_samples = numel(t_array);

sample_pos = (tof - t0) ./ dt + 1;
idx0 = floor(sample_pos);
frac = sample_pos - idx0;
valid = idx0 >= 1 & idx0 < num_samples;

for elem = 1:num_elements
    elem_valid = valid(:, elem);
    idx = idx0(elem_valid, elem);
    f = frac(elem_valid, elem);
    trace = sensor_data(elem, :).';
    values(elem_valid, elem) = trace(idx) .* (1 - f) + trace(idx + 1) .* f;
end
end
