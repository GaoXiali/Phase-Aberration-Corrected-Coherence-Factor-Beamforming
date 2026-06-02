% File: ds_uniform_tof.m
% Purpose: Uniform-speed time-of-flight calculator.
% Authors: Xiali Gao; Hao Huang
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function tof = ds_uniform_tof(points_m, sensor_xyz_m, sound_speed_mps)

tof = sqrt( ...
    (points_m(:,1) - sensor_xyz_m(:,1)').^2 + ...
    (points_m(:,2) - sensor_xyz_m(:,2)').^2 + ...
    (points_m(:,3) - sensor_xyz_m(:,3)').^2) ./ sound_speed_mps;
end
