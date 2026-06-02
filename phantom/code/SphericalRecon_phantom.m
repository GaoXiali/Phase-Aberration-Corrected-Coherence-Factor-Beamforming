% File: SphericalRecon_phantom.m
% Purpose: Phantom reconstruction demo comparing DAS, CF, and PAC-CF modes.
% Authors: Xiali Gao; Hao Huang
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

clc;

paths = resolvePhantomPaths();
addpath(paths.src_dir);

gpu = gpuDevice(1);
reset(gpu);

raw = load(paths.data_file, 'datax');
data = raw.datax(:,:,1:2:20);
Aline = mean(data,3);
Aline = squeeze(mean(Aline,1));
[~,DL1] = max(Aline(1:100));

detector = load(paths.coordinate_file);
detector(:,1) = detector(:,1)+0.555;
detector(:,2) = detector(:,2)+0.39;
[~, ~, Nframe] = size(data);

reconstruct_mode = [1 2 3]; % 1: DAS; 2: CF; 3: PAC-CF.
frames_to_reconstruct = 1; % Frame index or vector of frame indices to reconstruct.

T = 23.3;
V_M = 1446.5;
VM_out = waterSoundSpeed(T);
VM_in = 1260; % Inner sound speed for the phantom inclusion, m/s.

Is_Denoising = 0; % 1: apply band-pass denoising before reconstruction.

x_size = 20;
y_size = 20;
z_size = 20;
resolution_factor = 30; % Voxel density per millimeter.
center_x = 0;
center_y = -0.7;
center_z = -0.5;

Ellipse.a = 25; % PAC-CF ellipsoid x semi-axis, mm.
Ellipse.b = 25;
Ellipse.c = 7.2;
Ellipse.centerx = 0;
Ellipse.centery = 0;
Ellipse.centerz = 1.9;

if Is_Denoising == 1
    data = denoise_sinogram(data);
end

predelay = -DL1;
pa_data = -data;
fs = 40;
R = 100;

Npixel_x = x_size * resolution_factor+1;
Npixel_y = y_size * resolution_factor+1;
Npixel_z = z_size * resolution_factor+1;
x_range = ((1:Npixel_x)-(Npixel_x+1)/2)*x_size/(Npixel_x-1) + center_x;
y_range = ((1:Npixel_y)-(Npixel_y+1)/2)*y_size/(Npixel_y-1) + center_y;
z_range = ((1:Npixel_z)-(Npixel_z+1)/2)*z_size/(Npixel_z-1) + center_z;

[X_img, Y_img, Z_img] = meshgrid(x_range, y_range, z_range);

theta_x = 4.0;
theta_y = 5;
theta_z = 65;
trans_x = 0;
trans_y = 0;
trans_z = 0;

base_affine_mat = makeAffineMatrix(theta_x, theta_y, theta_z, trans_x, trans_y, trans_z);
detector_base = [detector, detector(:,1)*0+1] * base_affine_mat';

X_img = gpuArray(single(X_img));
Y_img = gpuArray(single(Y_img));
Z_img = gpuArray(single(Z_img));
Points_img = cat(4, X_img, Y_img, Z_img);

params = struct();
params.frames_to_reconstruct = frames_to_reconstruct;
params.T = T;
params.V_M = V_M;
params.VM_out = VM_out;
params.VM_in = VM_in;
params.Is_Denoising = Is_Denoising;
params.predelay = predelay;
params.fs = fs;
params.R = R;
params.x_size = x_size;
params.y_size = y_size;
params.z_size = z_size;
params.resolution_factor = resolution_factor;
params.center_x = center_x;
params.center_y = center_y;
params.center_z = center_z;
params.Ellipse = Ellipse;

validateFrames(frames_to_reconstruct, Nframe);

summary_records = struct([]);
tic;

for mode_idx = 1:numel(reconstruct_mode)
    mode = reconstruct_mode(mode_idx);
    method = phantomMethodInfo(mode);
    fprintf('\n===== %s reconstruction =====\n', method.label);

    pa_total = zeros(size(Points_img(:,:,:,1)), 'single');
    detector_run = detector_base;

    switch mode
        case 1
            for frame = frames_to_reconstruct
                [pa_img1, total_angle_weight, ~] = runDualSpeedMexAsUniform( ...
                    detector_run, Points_img, pa_data(:,:,frame), fs, predelay, V_M, R, Ellipse);
                pa_recon = max(pa_img1, 0) ./ max(total_angle_weight, eps('single'));
                pa_total = pa_total + pa_recon;
                fprintf('frame: %d\n', frame);
            end

        case 2
            delta_angle = -5000*0.800/11000;
            frame_affine_mat = makeAffineMatrix(0, 0, delta_angle, 0, 0, 0);

            for frame = frames_to_reconstruct
                detector_run = detector_run * frame_affine_mat';
                [pa_img1, total_angle_weight, coherent_factor] = runDualSpeedMexAsUniform( ...
                    detector_run, Points_img, pa_data(:,:,frame), fs, predelay, V_M, R, Ellipse);
                pa_recon = pa_img1 .* coherent_factor.^2 ./ max(total_angle_weight, eps('single'));
                pa_total = pa_total + pa_recon;
                fprintf('frame: %d\n', frame);
            end

        case 3
            delta_angle = -5000*0.800/11000;
            frame_affine_mat = makeAffineMatrix(0, 0, delta_angle, 0, 0, 0);

            for frame = frames_to_reconstruct
                detector_run = detector_run * frame_affine_mat';
                [pa_img1, total_angle_weight, coherent_factor] = runDualSpeedMex( ...
                    detector_run, Points_img, pa_data(:,:,frame), fs, predelay, VM_out, VM_in, R, Ellipse);
                pa_recon = pa_img1 .* coherent_factor.^2 ./ max(total_angle_weight, eps('single'));
                pa_total = pa_total + pa_recon;
                fprintf('frame: %d\n', frame);
            end

        otherwise
            error('Undefined reconstruct mode: %d', mode);
    end

    record = collectPhantomMethodResult(method, pa_total, x_range, y_range);
    summary_records = [summary_records; record];
    disp(struct2table(record));
end

if ~isempty(summary_records)
    metrics_table = struct2table(summary_records);
    disp(metrics_table);
end
fprintf('\nReconstruction finished in %.1f s.\n', toc);

function paths = resolvePhantomPaths()
    script_dir = fileparts(mfilename('fullpath'));
    if isempty(script_dir)
        script_dir = pwd;
    end
    project_root = fileparts(fileparts(script_dir));
    paths.project_root = project_root;
    paths.script_dir = script_dir;
    paths.src_dir = fullfile(project_root, 'src');
    paths.data_file = fullfile(project_root, 'phantom', 'data', 'raw', 'data_phantom.mat');
    paths.coordinate_file = fullfile(paths.src_dir, 'coordinate.txt');
    mustExist(paths.src_dir, 'src folder');
    mustExist(paths.data_file, 'data_phantom.mat');
    mustExist(paths.coordinate_file, 'coordinate.txt');
end

function validateFrames(frames_to_reconstruct, Nframe)
    if any(frames_to_reconstruct < 1) || any(frames_to_reconstruct > Nframe)
        error('frames_to_reconstruct exceeds available frame range 1:%d.', Nframe);
    end
end

function method = phantomMethodInfo(mode)
    switch mode
        case 1
            method.id = "DAS";
            method.label = "DAS";
            method.cn_label = "DAS";
        case 2
            method.id = "CF";
            method.label = "CF";
            method.cn_label = "Conventional CF";
        case 3
            method.id = "PAC_CF";
            method.label = "PAC-CF";
            method.cn_label = "PAC-CF";
        otherwise
            error('Undefined reconstruct mode: %d', mode);
    end
end

function affine_mat = makeAffineMatrix(theta_x, theta_y, theta_z, trans_x, trans_y, trans_z)
    rotate_x_mat = [1 0 0 0; 0 cosd(theta_x) -sind(theta_x) 0; 0 sind(theta_x) cosd(theta_x) 0; 0 0 0 1];
    rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0; 0 1 0 0; sind(theta_y) 0 cosd(theta_y) 0; 0 0 0 1];
    rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0; sind(theta_z) cosd(theta_z) 0 0; 0 0 1 0; 0 0 0 1];
    trans_mat = [1 0 0 trans_x; 0 1 0 trans_y; 0 0 1 trans_z; 0 0 0 1];
    affine_mat = trans_mat * rotate_x_mat * rotate_y_mat * rotate_z_mat;
end

function [pa_img1, total_angle_weight, coherent_factor] = runDualSpeedMexAsUniform(detector_run, Points_img, pa_data_frame, fs, predelay, V_M, R, Ellipse)
    [pa_img1, total_angle_weight, coherent_factor] = runDualSpeedMex( ...
        detector_run, Points_img, pa_data_frame, fs, predelay, V_M, V_M, R, Ellipse);
end

function [pa_img1, total_angle_weight, coherent_factor] = runDualSpeedMex(detector_run, Points_img, pa_data_frame, fs, predelay, VM_out, VM_in, R, Ellipse)
    x_sensor = gpuArray(single(detector_run(:,1)));
    y_sensor = gpuArray(single(detector_run(:,2)));
    z_sensor = gpuArray(single(-detector_run(:,3)));
    Points_sensor_all = gpuArray(single([x_sensor, y_sensor, z_sensor]));
    pa_data_gpu = gpuArray(single(pa_data_frame));

    [pa_img, total_angle_weight, coherent_factor, ~] = DualSpeedReconstraction_cof_mex( ...
        [Ellipse.a, Ellipse.b, Ellipse.c, Ellipse.centerx, Ellipse.centery, Ellipse.centerz], ...
        Points_sensor_all, Points_img, pa_data_gpu, ...
        single(fs), single(predelay), single(VM_out), single(VM_in), single(R));

    pa_img1 = gather(pa_img);
    total_angle_weight = gather(total_angle_weight);
    coherent_factor = gather(coherent_factor);
end

function record = collectPhantomMethodResult(method, pa_recon, x_range, y_range)
    projections = makeProjectionImages(pa_recon);
    [metrics, ~] = computeProjectionMetrics(projections.xy, x_range, y_range);
    metrics.method_id = method.id;
    metrics.method_label = method.label;
    record = metrics;
end

function projections = makeProjectionImages(pa_recon)
    projections.zx = squeeze(max(pa_recon(end:-1:1,:,:), [], 1));
    projections.zy = squeeze(max(pa_recon(end:-1:1,:,:), [], 2));
    projections.xy = squeeze(max(pa_recon(end:-1:1,:,:), [], 3));
end

function out = normalizeImage(img)
    img = double(img);
    img = img - min(img(:));
    max_val = max(img(:));
    if max_val > 0
        out = img ./ max_val;
    else
        out = zeros(size(img));
    end
end

function [metrics, analysis] = computeProjectionMetrics(proj_xy, x_range, y_range)
    img = normalizeImage(proj_xy);
    analysis = buildAnalysisRegions(img, x_range, y_range);

    signal_values = img(analysis.signal_mask);
    background_values = img(analysis.background_mask);

    metrics.signal_mean = mean(signal_values);
    metrics.background_mean = mean(background_values);
    metrics.background_std = std(background_values);
    metrics.SNR = metrics.signal_mean / max(metrics.background_std, eps);
    metrics.CR_dB = 20*log10((metrics.signal_mean + eps) / (metrics.background_mean + eps));
    metrics.gCNR = computeGCNR(signal_values, background_values);

    metrics.peak1_x_mm = analysis.vertical_wire_x_mm;
    metrics.peak1_y_mm = analysis.vertical_profile_y_mm;
    metrics.FWHM_wire1_mm = profileFwhm(img(analysis.vertical_profile_row,:), x_range, analysis.vertical_wire_col);
    metrics.peak2_x_mm = analysis.horizontal_profile_x_mm;
    metrics.peak2_y_mm = analysis.horizontal_wire_y_mm;
    metrics.FWHM_wire2_mm = profileFwhm(img(:,analysis.horizontal_profile_col), y_range, analysis.horizontal_wire_row);
    metrics.FWHM_mean_mm = mean([metrics.FWHM_wire1_mm, metrics.FWHM_wire2_mm], 'omitnan');
    metrics.signal_roi_radius_mm = analysis.signal_band_half_width_mm;
    metrics.background_roi_width_mm = analysis.background_roi_width_mm;
end

function gcnr = computeGCNR(signal_values, background_values)
    edges = linspace(0, 1, 101);
    signal_hist = histcounts(signal_values, edges, 'Normalization', 'probability');
    background_hist = histcounts(background_values, edges, 'Normalization', 'probability');
    gcnr = 1 - sum(min(signal_hist, background_hist));
end

function analysis = buildAnalysisRegions(img, x_range, y_range)
    [X, Y] = meshgrid(x_range, y_range);
    dx = abs(x_range(2)-x_range(1));
    dy = abs(y_range(2)-y_range(1));

    signal_band_half_width_mm = 0.25;
    background_width_mm = 2.5;
    exclude_crossing_mm = 0.7;
    central_x_window_mm = 3.0;
    line_search_y_window_mm = 6.0;

    horizontal_score_mask = abs(x_range) > central_x_window_mm;
    horizontal_search_mask = abs(y_range) <= line_search_y_window_mm;
    row_score = sum(img(:, horizontal_score_mask), 2);
    row_score(~horizontal_search_mask(:)) = 0;
    row_score = smoothVector(row_score, max(3, round(0.6 / dy)));
    [~, horizontal_wire_row] = max(row_score);
    horizontal_wire_y_mm = y_range(horizontal_wire_row);

    vertical_score_mask = abs(y_range - horizontal_wire_y_mm) > 1.0;
    vertical_search_mask = abs(x_range) <= central_x_window_mm;
    col_score = sum(img(vertical_score_mask, :), 1);
    col_score(~vertical_search_mask) = 0;
    col_score = smoothVector(col_score, max(3, round(0.6 / dx)));
    [~, vertical_wire_col] = max(col_score);
    vertical_wire_x_mm = x_range(vertical_wire_col);

    vertical_band = abs(X - vertical_wire_x_mm) <= signal_band_half_width_mm;
    horizontal_band = abs(Y - horizontal_wire_y_mm) <= signal_band_half_width_mm;
    signal_mask = (vertical_band & abs(Y - horizontal_wire_y_mm) > exclude_crossing_mm) | ...
        (horizontal_band & abs(X - vertical_wire_x_mm) > exclude_crossing_mm);

    vertical_profile_candidate = vertical_band & abs(Y - horizontal_wire_y_mm) > exclude_crossing_mm;
    vertical_profile_score = sum(img .* vertical_profile_candidate, 2);
    [~, vertical_profile_row] = max(vertical_profile_score);
    vertical_profile_y_mm = y_range(vertical_profile_row);

    horizontal_profile_candidate = horizontal_band & abs(X - vertical_wire_x_mm) > exclude_crossing_mm;
    horizontal_profile_score = sum(img .* horizontal_profile_candidate, 1);
    [~, horizontal_profile_col] = max(horizontal_profile_score);
    horizontal_profile_x_mm = x_range(horizontal_profile_col);

    background_mask = (X <= min(x_range)+background_width_mm & Y <= min(y_range)+background_width_mm) | ...
        (X >= max(x_range)-background_width_mm & Y <= min(y_range)+background_width_mm) | ...
        (X <= min(x_range)+background_width_mm & Y >= max(y_range)-background_width_mm) | ...
        (X >= max(x_range)-background_width_mm & Y >= max(y_range)-background_width_mm);

    background_mask = background_mask & ~imdilateLogical(signal_mask, round(1.0 / min(dx, dy)));

    analysis.normalized_xy = img;
    analysis.signal_mask = signal_mask;
    analysis.background_mask = background_mask;
    analysis.vertical_wire_x_mm = vertical_wire_x_mm;
    analysis.horizontal_wire_y_mm = horizontal_wire_y_mm;
    analysis.vertical_wire_col = vertical_wire_col;
    analysis.horizontal_wire_row = horizontal_wire_row;
    analysis.vertical_profile_y_mm = vertical_profile_y_mm;
    analysis.vertical_profile_row = vertical_profile_row;
    analysis.horizontal_profile_x_mm = horizontal_profile_x_mm;
    analysis.horizontal_profile_col = horizontal_profile_col;
    analysis.signal_band_half_width_mm = signal_band_half_width_mm;
    analysis.background_roi_width_mm = background_width_mm;
    analysis.pixel_size_x_mm = dx;
    analysis.pixel_size_y_mm = dy;
    analysis.signal_pixel_count = nnz(signal_mask);
    analysis.background_pixel_count = nnz(background_mask);
end

function smoothed = smoothVector(values, window_size)
    window_size = max(1, window_size);
    if mod(window_size, 2) == 0
        window_size = window_size + 1;
    end
    smoothed = movmean(values, window_size);
end

function dilated = imdilateLogical(mask, radius_pixels)
    radius_pixels = max(0, radius_pixels);
    if radius_pixels == 0
        dilated = mask;
        return;
    end
    [kx, ky] = meshgrid(-radius_pixels:radius_pixels, -radius_pixels:radius_pixels);
    kernel = (kx.^2 + ky.^2) <= radius_pixels^2;
    dilated = conv2(double(mask), double(kernel), 'same') > 0;
end

function width = profileFwhm(profile, axis_range, peak_index)
    profile = double(profile(:));
    profile = profile - min(profile);
    peak_value = profile(peak_index);
    if peak_value <= 0
        width = NaN;
        return;
    end

    half_value = peak_value / 2;
    left = peak_index;
    while left > 1 && profile(left) >= half_value
        left = left - 1;
    end
    right = peak_index;
    while right < numel(profile) && profile(right) >= half_value
        right = right + 1;
    end
    width = abs(axis_range(right) - axis_range(left));
end

function mustExist(path_name, label)
    if exist(path_name, 'file') ~= 2 && exist(path_name, 'dir') ~= 7
        error('Missing %s: %s', label, path_name);
    end
end
