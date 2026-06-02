% File: run_paccf_boundary_percent_error_study.m
% Purpose: Synthetic simulation for checking PAC-CF robustness to percentage boundary errors.
% Authors: Xiali Gao; Hao Huang
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function result = run_paccf_boundary_percent_error_study()

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
addpath(fullfile(project_root, 'src'));
addpath(fullfile(project_root, 'src', 'dual_speed_sim'));

cfg = local_config();
[sensor_xyz_m, ~] = ds_fibonacci_hemisphere(cfg.num_elements, cfg.sensor_radius_m);
grid = local_recon_grid(cfg);
[sensor_data, t_array] = local_make_sinogram(sensor_xyz_m, cfg);

fprintf('\n=== PAC-CF percentage boundary error study ===\n');

fprintf('Computing uniform-speed DAS / CF references...\n');
ref = local_reconstruct_uniform(sensor_data, t_array, sensor_xyz_m, grid, cfg);
ref_metrics = local_reference_metrics(ref, grid, cfg);
fprintf('Computing correct-boundary PAC-CF normalization baseline...\n');
true_recon = local_reconstruct_paccf(sensor_data, t_array, sensor_xyz_m, grid, cfg, cfg.true_ellipse);
true_basic = local_basic_metrics(true_recon.Image, grid, cfg);
ref_metrics.TrueTargetMeanRaw = true_basic.TargetMeanRaw;

metrics = table();
translation_selected = struct([]);
axis_selected = struct([]);

for i = 1:numel(cfg.error_percent)
    pct = cfg.error_percent(i);

    ell_t = cfg.true_ellipse;
    ell_t.center_m = ell_t.center_m + [0, 0, pct / 100 * cfg.true_ellipse.c_m];
    fprintf('translation error %+g%%...\n', pct);
    recon_t = local_reconstruct_paccf(sensor_data, t_array, sensor_xyz_m, grid, cfg, ell_t);
    row_t = local_paccf_metrics(recon_t, grid, cfg, ref_metrics, "translation error", pct);
    metrics = [metrics; row_t];
    if ismember(pct, cfg.selected_percent)
        k = numel(translation_selected) + 1;
        translation_selected(k).percent = pct;
        translation_selected(k).ellipse = ell_t;
        translation_selected(k).recon = recon_t;
    end

    ell_s = cfg.true_ellipse;
    scale = 1 + pct / 100;
    ell_s.a_m = ell_s.a_m * scale;
    ell_s.b_m = ell_s.b_m * scale;
    ell_s.c_m = ell_s.c_m * scale;
    fprintf('axis-length error %+g%%...\n', pct);
    recon_s = local_reconstruct_paccf(sensor_data, t_array, sensor_xyz_m, grid, cfg, ell_s);
    row_s = local_paccf_metrics(recon_s, grid, cfg, ref_metrics, "axis-length error", pct);
    metrics = [metrics; row_s];
    if ismember(pct, cfg.selected_percent)
        k = numel(axis_selected) + 1;
        axis_selected(k).percent = pct;
        axis_selected(k).ellipse = ell_s;
        axis_selected(k).recon = recon_s;
    end
end

local_plot_error_type(metrics, cfg, "translation error", 'translation_error_effect.png');
local_plot_error_type(metrics, cfg, "axis-length error", 'axis_length_error_effect.png');
local_plot_system(sensor_xyz_m, cfg, translation_selected, "translation error", 'translation_error_system.png');
local_plot_system(sensor_xyz_m, cfg, axis_selected, "axis-length error", 'axis_length_error_system.png');
local_plot_recon_montage(translation_selected, grid, cfg, 'translation_error_recons.png', 'translation error');
local_plot_recon_montage(axis_selected, grid, cfg, 'axis_length_error_recons.png', 'axis-length error');

result = struct();
result.cfg = cfg;
result.metrics = metrics;
result.ref_metrics = ref_metrics;
result.sensor_xyz_m = sensor_xyz_m;
result.t_array = t_array;

disp(metrics);
end

function cfg = local_config()
cfg = struct();
cfg.num_elements = 1024;
cfg.sensor_radius_m = 10e-3;
cfg.c_out = 1490;
cfg.c_in = 1950;
cfg.source_m = [-2.2, 0.8, -6.4] * 1e-3;
cfg.true_ellipse = struct('a_m', 5.2e-3, 'b_m', 4.1e-3, 'c_m', 5.6e-3, ...
    'center_m', cfg.source_m);
cfg.fs = 80e6;
cfg.t_end = 18e-6;
cfg.wavelet_f0 = 5.0e6;
cfg.noise_snr_db = 34;
cfg.recon_fov_m = [5.2e-3, 5.2e-3, 3.2e-3];
cfg.recon_dx_m = 0.10e-3;
cfg.recon_center_m = cfg.source_m;
cfg.cf_power = 2; % Coherence-factor exponent.
cfg.chunk_pixels = 8000;
cfg.roi_radius_m = 0.35e-3;
cfg.background_center_m = cfg.source_m + [1.9e-3, -1.8e-3, 0.45e-3];
cfg.error_percent = [-15, -10, -5, 0, 5, 10, 15]; % Boundary perturbations as percentages.
cfg.selected_percent = [-15, 0, 15]; % Representative cases shown in figures.
end

function [sensor_data, t_array] = local_make_sinogram(sensor_xyz_m, cfg)
t_array = 0:1/cfg.fs:cfg.t_end;
tof_true = ds_dual_speed_tof(cfg.source_m, sensor_xyz_m, cfg.true_ellipse, cfg.c_out, cfg.c_in);
sensor_data = zeros(cfg.num_elements, numel(t_array));
for elem = 1:cfg.num_elements
    t = t_array - tof_true(elem);
    wave = (1 - 2 * (pi * cfg.wavelet_f0 * t).^2) .* exp(-(pi * cfg.wavelet_f0 * t).^2);
    sensor_data(elem, :) = wave ./ norm(sensor_xyz_m(elem,:) - cfg.source_m);
end

rng(20260529);
noise_rms = rms(sensor_data(:)) / 10^(cfg.noise_snr_db / 20);
sensor_data = sensor_data + noise_rms * randn(size(sensor_data));
end

function grid = local_recon_grid(cfg)
x = local_axis(cfg.recon_fov_m(1), cfg.recon_dx_m) + cfg.recon_center_m(1);
y = local_axis(cfg.recon_fov_m(2), cfg.recon_dx_m) + cfg.recon_center_m(2);
z = local_axis(cfg.recon_fov_m(3), cfg.recon_dx_m) + cfg.recon_center_m(3);
[X, Y, Z] = ndgrid(x, y, z);
grid = struct('x', x, 'y', y, 'z', z, 'X', X, 'Y', Y, 'Z', Z, ...
    'points', [X(:), Y(:), Z(:)], 'size', size(X), 'dx_m', cfg.recon_dx_m);
end

function axis_m = local_axis(fov_m, dx_m)
n = floor(fov_m / dx_m);
if mod(n, 2) == 1
    n = n + 1;
end
axis_m = ((0:n) - n / 2) * dx_m;
end

function ref = local_reconstruct_uniform(sensor_data, t_array, sensor_xyz_m, grid, cfg)
num_pixels = size(grid.points, 1);
das = zeros(num_pixels, 1);
cf_img = zeros(num_pixels, 1);
cf_map = zeros(num_pixels, 1);
for start_idx = 1:cfg.chunk_pixels:num_pixels
    stop_idx = min(start_idx + cfg.chunk_pixels - 1, num_pixels);
    pix = grid.points(start_idx:stop_idx, :);
    tof = ds_uniform_tof(pix, sensor_xyz_m, cfg.c_out);
    values = ds_sample_sinogram(sensor_data, t_array, tof);
    [das_chunk, cf_chunk] = local_das_cf(values);
    das(start_idx:stop_idx) = abs(das_chunk);
    cf_img(start_idx:stop_idx) = abs(das_chunk .* cf_chunk.^cfg.cf_power);
    cf_map(start_idx:stop_idx) = cf_chunk;
end
ref = struct();
ref.DAS = reshape(das, grid.size);
ref.CF = reshape(cf_img, grid.size);
ref.CFMap = reshape(cf_map, grid.size);
end

function recon = local_reconstruct_paccf(sensor_data, t_array, sensor_xyz_m, grid, cfg, ellipse)
num_pixels = size(grid.points, 1);
img = zeros(num_pixels, 1);
cf_map = zeros(num_pixels, 1);
for start_idx = 1:cfg.chunk_pixels:num_pixels
    stop_idx = min(start_idx + cfg.chunk_pixels - 1, num_pixels);
    pix = grid.points(start_idx:stop_idx, :);
    tof = ds_dual_speed_tof(pix, sensor_xyz_m, ellipse, cfg.c_out, cfg.c_in);
    values = ds_sample_sinogram(sensor_data, t_array, tof);
    [das_chunk, cf_chunk] = local_das_cf(values);
    img(start_idx:stop_idx) = abs(das_chunk .* cf_chunk.^cfg.cf_power);
    cf_map(start_idx:stop_idx) = cf_chunk;
end
recon = struct();
recon.Image = reshape(img, grid.size);
recon.CFMap = reshape(cf_map, grid.size);
end

function [das, cf] = local_das_cf(values)
sum_values = sum(values, 2);
sum_abs = sum(abs(values), 2);
das = sum_values ./ size(values, 2);
cf = abs(sum_values) ./ max(sum_abs, eps);
end

function ref_metrics = local_reference_metrics(ref, grid, cfg)
ref_metrics = struct();
ref_metrics.DAS = local_basic_metrics(ref.DAS, grid, cfg);
ref_metrics.CF = local_basic_metrics(ref.CF, grid, cfg);
ref_metrics.UniformSourceCF = local_value_at_source(ref.CFMap, grid, cfg);
end

function row = local_paccf_metrics(recon, grid, cfg, ref_metrics, error_type, error_percent)
basic = local_basic_metrics(recon.Image, grid, cfg);
source_cf = local_value_at_source(recon.CFMap, grid, cfg);
target_rel_to_true = basic.TargetMeanRaw / max(ref_metrics.TrueTargetMeanRaw, eps);
gain_vs_cf_db = 20 * log10(max(basic.TargetMeanRaw, eps) / max(ref_metrics.CF.TargetMeanRaw, eps));
gain_vs_das_db = 20 * log10(max(basic.TargetMeanRaw, eps) / max(ref_metrics.DAS.TargetMeanRaw, eps));
row = table(string(error_type), error_percent, ...
    basic.FWHM_lateral_mm, basic.PeakError_mm, basic.TargetMeanRaw, ...
    target_rel_to_true, gain_vs_cf_db, gain_vs_das_db, source_cf, ...
    basic.SNR_dB, basic.gCNR, ...
    'VariableNames', {'ErrorType', 'ErrorPercent', 'FWHM_lateral_mm', ...
    'PeakError_mm', 'TargetMeanRaw', 'TargetMeanRelativeToTruePAC', ...
    'GainVsCF_dB', 'GainVsDAS_dB', 'SourceCF', 'SNR_dB', 'gCNR'});
end

function basic = local_basic_metrics(img, grid, cfg)
img = double(img);
peak = max(img, [], 'all');
img_norm = img ./ max(peak, eps);
width = ds_fwhm_metrics(img_norm, grid);
[~, peak_error_mm] = local_source_values(img, grid, cfg);
[target_raw, background_raw] = local_roi(img, grid, cfg);
basic = struct();
basic.FWHM_lateral_mm = mean([width.x_mm, width.y_mm], 'omitnan');
basic.PeakError_mm = peak_error_mm;
basic.TargetMeanRaw = mean(target_raw);
basic.BackgroundStdRaw = std(background_raw);
basic.SNR_dB = 20 * log10(max(mean(target_raw), eps) / max(std(background_raw), eps));
basic.gCNR = ds_gcnr(target_raw, background_raw, 128);
end

function [source_value, peak_error_mm] = local_source_values(img, grid, cfg)
source_value = local_value_at_source(img, grid, cfg);
[~, peak_idx] = max(img(:));
[px, py, pz] = ind2sub(size(img), peak_idx);
peak_error_mm = norm([grid.x(px), grid.y(py), grid.z(pz)] - cfg.source_m) * 1e3;
end

function value = local_value_at_source(img, grid, cfg)
[~, ix] = min(abs(grid.x - cfg.source_m(1)));
[~, iy] = min(abs(grid.y - cfg.source_m(2)));
[~, iz] = min(abs(grid.z - cfg.source_m(3)));
value = img(ix, iy, iz);
end

function [target, background] = local_roi(img, grid, cfg)
target_mask = (grid.X - cfg.source_m(1)).^2 + (grid.Y - cfg.source_m(2)).^2 + ...
    (grid.Z - cfg.source_m(3)).^2 <= cfg.roi_radius_m^2;
background_mask = (grid.X - cfg.background_center_m(1)).^2 + ...
    (grid.Y - cfg.background_center_m(2)).^2 + ...
    (grid.Z - cfg.background_center_m(3)).^2 <= cfg.roi_radius_m^2;
target = img(target_mask);
background = img(background_mask);
end

function local_plot_error_type(metrics, cfg, error_type, file_name)
rows = metrics(metrics.ErrorType == error_type, :);
rows = sortrows(rows, 'ErrorPercent');

fig = figure('Visible', 'on', 'Color', 'w', 'Position', [90, 90, 1250, 820]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
yyaxis left;
plot(rows.ErrorPercent, rows.TargetMeanRelativeToTruePAC, '-o', 'LineWidth', 2, ...
    'DisplayName', 'target signal intensity');
ylabel('Target intensity / correct-boundary PAC-CF');
ylim([0, max(1.25, max(rows.TargetMeanRelativeToTruePAC) * 1.12)]);
yyaxis right;
plot(rows.ErrorPercent, rows.PeakError_mm, '-s', 'LineWidth', 2, ...
    'DisplayName', 'peak localization error');
ylabel('peak localization error (mm)');
grid on;
xlabel('Boundary error (%)');
title(error_type + " effect on PAC-CF target intensity and localization error");
legend('Location', 'best');

nexttile;
plot(rows.ErrorPercent, rows.GainVsCF_dB, '-o', 'LineWidth', 2, ...
    'DisplayName', 'gain over CF');
hold on;
plot(rows.ErrorPercent, rows.GainVsDAS_dB, '-s', 'LineWidth', 2, ...
    'DisplayName', 'gain over DAS');
grid on;
xlabel('Boundary error (%)');
ylabel('Target-intensity gain (dB)');
title(error_type + ": PAC-CF gain over conventional reconstruction");
legend('Location', 'best');

sgtitle(error_type + " percentage error robustness of PAC-CF");
end

function local_plot_system(sensor_xyz_m, cfg, selected, error_type, file_name)
fig = figure('Visible', 'on', 'Color', 'w', 'Position', [100, 100, 1250, 580]);
theta = linspace(0, 2*pi, 400);

subplot(1, 2, 1);
hold on;
scatter3(sensor_xyz_m(:,1)*1e3, sensor_xyz_m(:,2)*1e3, sensor_xyz_m(:,3)*1e3, ...
    8, [0.1, 0.35, 0.8], 'filled', 'MarkerFaceAlpha', 0.25);
[X,Y,Z] = ellipsoid(cfg.true_ellipse.center_m(1), cfg.true_ellipse.center_m(2), cfg.true_ellipse.center_m(3), ...
    cfg.true_ellipse.a_m, cfg.true_ellipse.b_m, cfg.true_ellipse.c_m, 48);
surf(X*1e3, Y*1e3, Z*1e3, 'FaceColor', [0.9, 0.35, 0.1], 'FaceAlpha', 0.22, 'EdgeColor', 'none');
plot3(cfg.source_m(1)*1e3, cfg.source_m(2)*1e3, cfg.source_m(3)*1e3, 'r.', 'MarkerSize', 26);
axis equal; grid on; box on; view(38, 24);
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
title("True system geometry: " + error_type);

subplot(1, 2, 2);
hold on;
plot(local_ellipse_xz(cfg.true_ellipse, theta, 1), local_ellipse_xz(cfg.true_ellipse, theta, 2), ...
    'Color', [0.85, 0.25, 0.05], 'LineWidth', 2.2, 'DisplayName', 'true boundary');
for i = 1:numel(selected)
    plot(local_ellipse_xz(selected(i).ellipse, theta, 1), local_ellipse_xz(selected(i).ellipse, theta, 2), ...
        '--', 'LineWidth', 1.4, 'DisplayName', sprintf('%+g%%', selected(i).percent));
end
plot(cfg.source_m(1)*1e3, cfg.source_m(3)*1e3, 'r.', 'MarkerSize', 24, 'DisplayName', 'source');
axis equal; grid on; box on;
xlabel('x (mm)'); ylabel('z (mm)');
title("Perturbed boundary used by PAC-CF: " + error_type);
legend('Location', 'bestoutside');
end

function v = local_ellipse_xz(ellipse, theta, dim)
if dim == 1
    v = (ellipse.center_m(1) + ellipse.a_m * cos(theta)) * 1e3;
else
    v = (ellipse.center_m(3) + ellipse.c_m * sin(theta)) * 1e3;
end
end

function local_plot_recon_montage(selected, grid, cfg, file_name, error_type)
if isempty(selected)
    return;
end
fig = figure('Visible', 'on', 'Color', 'w', 'Position', [100, 100, 1250, 420]);
for i = 1:numel(selected)
    img = double(selected(i).recon.Image);
    img = img ./ max(img, [], 'all');
    [~, iz] = min(abs(grid.z - cfg.source_m(3)));
    subplot(1, numel(selected), i);
    imagesc(grid.x*1e3, grid.y*1e3, squeeze(img(:,:,iz))', [0, 1]);
    axis image tight; set(gca, 'YDir', 'normal'); colormap gray; colorbar;
    hold on;
    plot(cfg.source_m(1)*1e3, cfg.source_m(2)*1e3, 'r+', 'MarkerSize', 10, 'LineWidth', 1.2);
    title(sprintf('%s %+g%%', error_type, selected(i).percent));
    xlabel('x (mm)'); ylabel('y (mm)');
end
end
