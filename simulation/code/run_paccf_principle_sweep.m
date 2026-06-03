% File: run_paccf_principle_sweep.m
% Purpose: Synthetic simulation showing how PAC-CF behaves as dual-speed contrast increases.
% Authors: Xiali Gao
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function result = run_paccf_principle_sweep()

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
addpath(fullfile(project_root, 'src'));
addpath(fullfile(project_root, 'src', 'dual_speed_sim'));

cfg = local_config();
[sensor_xyz_m, ~] = ds_fibonacci_hemisphere(cfg.num_elements, cfg.sensor_radius_m);
grid = local_recon_grid(cfg);

metrics = table();
selected = struct([]);
for i = 1:numel(cfg.c_in_values)
    cfg_i = cfg;
    cfg_i.c_in = cfg.c_in_values(i);
    fprintf('\n=== Dual-speed strength sweep %d/%d:c_in = %.0f m/s ===\n', i, numel(cfg.c_in_values), cfg_i.c_in);
    [sensor_data, t_array, diag_info] = local_make_sinogram(sensor_xyz_m, cfg_i);
    recon = local_reconstruct(sensor_data, t_array, sensor_xyz_m, grid, cfg_i);
    row = local_metrics(recon, grid, cfg_i, diag_info);
    metrics = [metrics; row];

    if ismember(cfg_i.c_in, cfg.selected_c_in)
        k = numel(selected) + 1;
        selected(k).c_in = cfg_i.c_in;
        selected(k).recon = recon;
        selected(k).diag = diag_info;
        selected(k).metrics = row;
    end
end

local_plot_summary(metrics, cfg);
local_plot_selected_recons(selected, grid, cfg);
local_plot_geometry(sensor_xyz_m, cfg);

result = struct('cfg', cfg, 'metrics', metrics, 'selected', selected);
disp(metrics);
end

function cfg = local_config()
cfg = struct();
cfg.num_elements = 1024;
cfg.sensor_radius_m = 10e-3;
cfg.c_out = 1490;
cfg.c_in_values = [1490, 1540, 1600, 1660, 1720, 1780, 1850]; % Inner sound speeds to sweep, m/s.
cfg.selected_c_in = [1490, 1660, 1850]; % Cases shown in the montage figures.
cfg.source_m = [-2.4, 1.1, -7.4] * 1e-3;
cfg.ellipse = struct('a_m', 6.0e-3, 'b_m', 3.8e-3, 'c_m', 5.2e-3, ...
    'center_m', [0.8, -0.4, -4.2] * 1e-3);
cfg.fs = 80e6;
cfg.t_end = 18e-6;
cfg.wavelet_f0 = 7.5e6;
cfg.noise_snr_db = 34;
cfg.recon_fov_m = [7.0e-3, 7.0e-3, 4.0e-3];
cfg.recon_dx_m = 0.10e-3;
cfg.recon_center_m = cfg.source_m;
cfg.cf_power = 2; % Coherence-factor exponent.
cfg.chunk_pixels = 10000;
cfg.background_center_m = cfg.source_m + [2.6e-3, -2.4e-3, 0.6e-3];
cfg.roi_radius_m = 0.45e-3;
end

function [sensor_data, t_array, diag_info] = local_make_sinogram(sensor_xyz_m, cfg)
t_array = 0:1/cfg.fs:cfg.t_end;
tof_true = ds_dual_speed_tof(cfg.source_m, sensor_xyz_m, cfg.ellipse, cfg.c_out, cfg.c_in);
tof_wrong = ds_uniform_tof(cfg.source_m, sensor_xyz_m, cfg.c_out);
sensor_data = zeros(cfg.num_elements, numel(t_array));
for elem = 1:cfg.num_elements
    t = t_array - tof_true(elem);
    wave = (1 - 2 * (pi * cfg.wavelet_f0 * t).^2) .* exp(-(pi * cfg.wavelet_f0 * t).^2);
    sensor_data(elem, :) = wave ./ norm(sensor_xyz_m(elem,:) - cfg.source_m);
end

rng(20260528);
noise_rms = rms(sensor_data(:)) / 10^(cfg.noise_snr_db / 20);
sensor_data = sensor_data + noise_rms * randn(size(sensor_data));

values_wrong = ds_sample_sinogram(sensor_data, t_array, tof_wrong);
values_true = ds_sample_sinogram(sensor_data, t_array, tof_true);
diag_info = struct();
diag_info.cf_wrong_at_source = abs(sum(values_wrong)) / sum(abs(values_wrong));
diag_info.cf_true_at_source = abs(sum(values_true)) / sum(abs(values_true));
diag_info.tof_error_std_ns = std((tof_wrong - tof_true) * 1e9);
diag_info.tof_error_range_ns = range((tof_wrong - tof_true) * 1e9);
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

function recon = local_reconstruct(sensor_data, t_array, sensor_xyz_m, grid, cfg)
num_pixels = size(grid.points, 1);
das = zeros(num_pixels, 1);
cf = zeros(num_pixels, 1);
pac = zeros(num_pixels, 1);
for start_idx = 1:cfg.chunk_pixels:num_pixels
    stop_idx = min(start_idx + cfg.chunk_pixels - 1, num_pixels);
    pix = grid.points(start_idx:stop_idx, :);
    tof_uniform = ds_uniform_tof(pix, sensor_xyz_m, cfg.c_out);
    values_uniform = ds_sample_sinogram(sensor_data, t_array, tof_uniform);
    [das_chunk, cf_chunk] = local_das_cf(values_uniform);

    tof_dual = ds_dual_speed_tof(pix, sensor_xyz_m, cfg.ellipse, cfg.c_out, cfg.c_in);
    values_dual = ds_sample_sinogram(sensor_data, t_array, tof_dual);
    [pac_das_chunk, pac_cf_chunk] = local_das_cf(values_dual);

    das(start_idx:stop_idx) = abs(das_chunk);
    cf(start_idx:stop_idx) = abs(das_chunk .* cf_chunk.^cfg.cf_power);
    pac(start_idx:stop_idx) = abs(pac_das_chunk .* pac_cf_chunk.^cfg.cf_power);
end
recon = struct();
recon.DAS = reshape(das, grid.size);
recon.CF = reshape(cf, grid.size);
recon.PAC_CF = reshape(pac, grid.size);
end

function [das, cf] = local_das_cf(values)
sum_values = sum(values, 2);
sum_abs = sum(abs(values), 2);
das = sum_values ./ size(values, 2);
cf = abs(sum_values) ./ max(sum_abs, eps);
end

function row = local_metrics(recon, grid, cfg, diag_info)
methods = ["DAS", "CF", "PAC_CF"];
row = table();
for i = 1:numel(methods)
    method = methods(i);
    img = double(recon.(method));
    peak = max(img, [], 'all');
    img_norm = img ./ max(peak, eps);
    fwhm = ds_fwhm_metrics(img_norm, grid);
    [source_value, peak_error_mm] = local_source_values(img, grid, cfg);
    [target, background] = local_roi(img_norm, grid, cfg);
    one = table(cfg.c_in, cfg.c_in - cfg.c_out, method, ...
        fwhm.x_mm, fwhm.y_mm, fwhm.z_mm, mean([fwhm.x_mm, fwhm.y_mm], 'omitnan'), ...
        source_value / max(peak, eps), peak_error_mm, ...
        mean(target), mean(background), ...
        20 * log10(max(mean(target), eps) / max(std(background), eps)), ...
        ds_gcnr(target, background, 128), ...
        diag_info.cf_wrong_at_source, diag_info.cf_true_at_source, ...
        diag_info.tof_error_std_ns, diag_info.tof_error_range_ns, ...
        'VariableNames', {'CIn_mps', 'DeltaC_mps', 'Method', ...
        'FWHM_x_mm', 'FWHM_y_mm', 'FWHM_z_mm', 'FWHM_lateral_mm', ...
        'SourceToPeakRatio', 'PeakError_mm', 'TargetMean', 'BackgroundMean', ...
        'SNR_dB', 'gCNR', 'CFWrongAtSource', 'PACCFAtSource', ...
        'TOFErrorStd_ns', 'TOFErrorRange_ns'});
    row = [row; one];
end
end

function [source_value, peak_error_mm] = local_source_values(img, grid, cfg)
[~, ix] = min(abs(grid.x - cfg.source_m(1)));
[~, iy] = min(abs(grid.y - cfg.source_m(2)));
[~, iz] = min(abs(grid.z - cfg.source_m(3)));
source_value = img(ix, iy, iz);
[~, peak_idx] = max(img(:));
[px, py, pz] = ind2sub(size(img), peak_idx);
peak_error_mm = norm([grid.x(px), grid.y(py), grid.z(pz)] - cfg.source_m) * 1e3;
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

function local_plot_summary(metrics, cfg)
fig = figure('Visible','on','Color','w','Position',[80,80,1450,820]);
subplot(2,2,1);
local_plot_methods(metrics, 'SourceToPeakRatio', 'True-source intensity / image peak');
set(gca, 'YScale', 'log');
title('True signal suppression');

subplot(2,2,2);
hold on;
cf_rows = metrics(metrics.Method == "CF", :);
plot(cf_rows.DeltaC_mps, cf_rows.CFWrongAtSource, '-o', 'LineWidth', 2, 'DisplayName', 'Conventional CF at source');
plot(cf_rows.DeltaC_mps, cf_rows.PACCFAtSource, '-o', 'LineWidth', 2, 'DisplayName', 'PAC-CF at source');
grid on; xlabel('Inner-outer sound-speed difference (m/s)'); ylabel('CF at source');
legend('Location','best'); title('Channel coherence versus dual-speed contrast');

subplot(2,2,3);
local_plot_methods(metrics, 'FWHM_lateral_mm', 'Lateral FWHM (mm)');
title('Lateral FWHM');

subplot(2,2,4);
local_plot_methods(metrics, 'PeakError_mm', 'peak localization error (mm)');
title('peak localization error');

sgtitle('PAC-CF advantage increases with dual-speed contrast');
end

function local_plot_methods(metrics, field_name, y_label)
methods = ["DAS", "CF", "PAC_CF"];
labels = ["DAS", "CF", "PAC-CF"];
colors = lines(numel(methods));
hold on;
for i = 1:numel(methods)
    rows = metrics(metrics.Method == methods(i), :);
    plot(rows.DeltaC_mps, rows.(field_name), '-o', 'LineWidth', 2, ...
        'Color', colors(i,:), 'DisplayName', labels(i));
end
grid on; xlabel('Inner-outer sound-speed difference (m/s)'); ylabel(y_label);
legend('Location','best');
end

function local_plot_selected_recons(selected, grid, cfg)
fig = figure('Visible','on','Color','w','Position',[80,80,1500,840]);
for k = 1:numel(selected)
    methods = ["CF", "PAC_CF"];
    for j = 1:2
        img = double(selected(k).recon.(methods(j)));
        img = img ./ max(img, [], 'all');
        [~, peak_idx] = max(img(:));
        [~, ~, iz] = ind2sub(size(img), peak_idx);
        subplot(numel(selected), 4, (k-1)*4 + j);
        imagesc(grid.x*1e3, grid.y*1e3, squeeze(img(:,:,iz))');
        axis image tight; set(gca,'YDir','normal'); colormap gray; colorbar;
        hold on; plot(cfg.source_m(1)*1e3, cfg.source_m(2)*1e3, 'r+', 'MarkerSize', 10, 'LineWidth', 1.4);
        title(sprintf('%s, \\Delta c=%g', methods(j), selected(k).c_in - cfg.c_out));
        xlabel('x (mm)'); ylabel('y (mm)');

        subplot(numel(selected), 4, (k-1)*4 + j + 2);
        imagesc(grid.x*1e3, grid.y*1e3, squeeze(20*log10(max(img(:,:,iz),1e-6)))', [-50 0]);
        axis image tight; set(gca,'YDir','normal'); colormap gray; colorbar;
        hold on; plot(cfg.source_m(1)*1e3, cfg.source_m(2)*1e3, 'r+', 'MarkerSize', 10, 'LineWidth', 1.4);
        title(sprintf('%s dB', methods(j)));
        xlabel('x (mm)'); ylabel('y (mm)');
    end
end
sgtitle('As dual-speed contrast increases, conventional CF suppresses the true point while PAC-CF remains stable');
end

function local_plot_geometry(sensor_xyz_m, cfg)
fig = figure('Visible','on','Color','w','Position',[100,100,1250,560]);
subplot(1,2,1); hold on;
scatter3(sensor_xyz_m(:,1)*1e3, sensor_xyz_m(:,2)*1e3, sensor_xyz_m(:,3)*1e3, 8, [0.1,0.35,0.8], 'filled', 'MarkerFaceAlpha', 0.35);
[X,Y,Z] = ellipsoid(cfg.ellipse.center_m(1), cfg.ellipse.center_m(2), cfg.ellipse.center_m(3), cfg.ellipse.a_m, cfg.ellipse.b_m, cfg.ellipse.c_m, 48);
surf(X*1e3,Y*1e3,Z*1e3,'FaceColor',[0.9,0.35,0.1],'FaceAlpha',0.22,'EdgeColor','none');
plot3(cfg.source_m(1)*1e3,cfg.source_m(2)*1e3,cfg.source_m(3)*1e3,'r.','MarkerSize',26);
axis equal; grid on; box on; view(38,24);
xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
title('3D geometry');
subplot(1,2,2); hold on;
theta = linspace(0,2*pi,400);
fill((cfg.ellipse.center_m(1)+cfg.ellipse.a_m*cos(theta))*1e3, ...
    (cfg.ellipse.center_m(3)+cfg.ellipse.c_m*sin(theta))*1e3, [0.9,0.35,0.1], ...
    'FaceAlpha',0.18,'EdgeColor',[0.9,0.35,0.1],'LineWidth',1.5);
plot(cfg.source_m(1)*1e3,cfg.source_m(3)*1e3,'r.','MarkerSize',24);
scatter(sensor_xyz_m(abs(sensor_xyz_m(:,2))<8e-3,1)*1e3, sensor_xyz_m(abs(sensor_xyz_m(:,2))<8e-3,3)*1e3, 12, [0.1,0.35,0.8], 'filled');
axis equal; grid on; box on; xlabel('x (mm)'); ylabel('z (mm)');
title('XZ section');
end
