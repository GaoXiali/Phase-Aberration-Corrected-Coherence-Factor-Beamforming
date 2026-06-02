% File: ds_fibonacci_hemisphere.m
% Purpose: Fibonacci sampling helper for a hemispherical detector layout.
% Authors: Xiali Gao; Hao Huang
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function [xyz_m, mask] = ds_fibonacci_hemisphere(num_elements, radius_m)

i = (0:num_elements-1)';
golden_angle = pi * (3 - sqrt(5));
z = (i + 0.5) / num_elements;
r_xy = sqrt(max(0, 1 - z.^2));
theta = i * golden_angle;

x = r_xy .* cos(theta);
y = r_xy .* sin(theta);

xyz_m = radius_m * [x, y, z];
mask = xyz_m';
end
