% File: SphericalRecon_mouse_liver.m
% Purpose: Mouse liver-region PAC-CF reconstruction demo using the dual-side raw mouse dataset.
% Authors: Xiali Gao
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

clc;
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
src_dir = fullfile(project_root, 'src');
addpath(src_dir);
gpuDevice(1).reset()

raw = load(fullfile(project_root, 'mouse', 'data', 'raw', 'datax_dualside.mat'), 'datax');
data = raw.datax;
DL1 = 42;

detector = load(fullfile(src_dir, 'coordinate.txt'));
detector(:,1) = detector(:,1)+0.555;
detector(:,2) = detector(:,2)+0.39;
[Nelemt, Nsample, Nframe] = size(data);

reconstruct_mode = 3; % 1: DAS; 2: CF; 3: PAC-CF.

T = 22.4;

Is_Gating = 1; % 1: use correlation-based static-frame gating; 0: use all frames.
Is_Denoising = 1; % 1: apply band-pass denoising before reconstruction.

V_M = waterSoundSpeed(T);
VM_out = V_M; % Outer sound speed in water, m/s.
VM_out_Range = 1475:0.5:1499;
VM_in = 1540.3; % Inner sound speed used by PAC-CF, m/s.
VM_in_Range = 1530:5:1599;

step_x = 1;
step_y = 1;
step_length_x = 5;
step_length_y = 5;
Nframex_scan = 17;
Nframey_scan = 17;
GaussianMask_FWHM = 40;

x_size = 12;
y_size = 10;
z_size = 40;
resolution_factor = 10; % Voxel density per millimeter.
center_x = 1.5;
center_y = 4.1;
center_z = 0;

Ellipse.a = 7; % PAC-CF ellipsoid x semi-axis, mm.
Ellipse.b = 28;
Ellipse.c = 11.5;
Ellipse.centerx = -1.8;
Ellipse.centery = 2;
Ellipse.centerz = 2.7;

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

theta_x = 1;
theta_y = 5;
theta_z = -12.5;

trans_x = 0;
trans_y = 0;
trans_z = 0;

rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat;

detector_new=[detector,detector(:,1)*0+1]*afine_mat';

x_sensor=detector_new(:,1);
y_sensor=detector_new(:,2);
z_sensor=-detector_new(:,3);

x_sensor = gpuArray(single(x_sensor));
y_sensor = gpuArray(single(y_sensor));
z_sensor = gpuArray(single(z_sensor));

X_img = gpuArray(single(X_img));
Y_img = gpuArray(single(Y_img));
Z_img = gpuArray(single(Z_img));
Points_img = cat(4,X_img,Y_img,Z_img);

tic
switch reconstruct_mode

    case 1
        pa_total = zeros(size(Points_img(:,:,:,1)),'single');

        if Is_Gating == 1
            [T, D, F] = size(pa_data(:,2501:3000,:));
            reshaped_data = reshape(pa_data(:,2501:3000,:), T*D, F);
            corr_mat = corr(reshaped_data);
            corr_line = mean(corr_mat,1);
            corr_line = corr_line/max(corr_line);
            corr_line(1:10) = 0;
            static_frames = 1:Nframe;
            top_vals = maxk(corr_line, 20);
            Similarity_threshold = top_vals(end);
            static_frames = static_frames(corr_line>=Similarity_threshold);

            figure(11),plot(corr_line,'b'),hold on
            for isf = static_frames
                plot(isf,corr_line(isf),'*r'),hold on
            end
            hold off;
        else
            static_frames = 1:Nframe;
        end

        delta_angle = -5000*0.800/11000;
        static_Nframe = size(static_frames,2);

        firstframe_flag = 1;

        for frame = 1:1:static_Nframe

            tic

            theta_x = 0;
            theta_y = 0;
            theta_z = (static_frames(frame)-1)*delta_angle;

            trans_x = 0;
            trans_y = 0;
            trans_z = 0;

            rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
            rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
            rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
            trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];

            afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat;
            detector_corr=detector_new*afine_mat';

            pa_data_frame = gpuArray(single(pa_data(:,:,static_frames(frame))));
            Points_sensor_all = gpuArray(single(detector_corr(:,1:3)));
            tic

            [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));
            toc
            disp(['frame: ', num2str(frame)]);

            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);

            pa_img2 = pa_img1./total_angle_weight;

            if firstframe_flag == 0

            else
                pa_ref = pa_img2;

            end
            pa_total = pa_total+pa_img2;
            firstframe_flag = 0;

            imin=0;
            imax=max(pa_total,[],"all");

            figure(1); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_total(:,:,:),[],1)));
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
            ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[],2)));
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
            ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[],3)));
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out');
            title(frame)
            colormap("hot");colorbar;

        end

    case 2
        pa_total = zeros(size(Points_img(:,:,:,1)));

        delta_angle = -5000*0.800/11000;

        theta_x = 0;
        theta_y = 0;
        theta_z = delta_angle;

        trans_x = 0;
        trans_y = 0;
        trans_z = 0;

        rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
        rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
        rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
        trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
        afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat;

        for frame = 1:Nframe

            detector_new=detector_new*afine_mat';

            x_sensor=detector_new(:,1);
            y_sensor=detector_new(:,2);
            z_sensor=-detector_new(:,3);

            x_sensor = gpuArray(single(x_sensor));
            y_sensor = gpuArray(single(y_sensor));
            z_sensor = gpuArray(single(z_sensor));

            pa_data_frame = gpuArray(single(pa_data(:,:,frame)));
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));

            tic
            [pa_img, total_angle_weight, coherent_factor, ~] = SingleSpeedReconstraction_cof_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));
            toc
            disp(['frame: ', num2str(frame)]);

            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            coherent_factor = gather(coherent_factor);

            pa_img2 = pa_img1.*coherent_factor.^2./total_angle_weight;

            pa_total = pa_total+pa_img2;

            imin=0;
            imax=max(pa_total,[],"all");

            figure(1); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_total(:,:,:),[],1)),[imin,imax]);
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
            ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[],2)),[imin,imax]);
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
            ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[],3)),[imin,imax]);
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out');
            colormap("hot");colorbar;

        end

    case 3
        pa_total = zeros(size(Points_img(:,:,:,1)));

        delta_angle = -5000*0.800/11000;

        theta_x = 0;
        theta_y = 0;
        theta_z = delta_angle;

        trans_x = 0;
        trans_y = 0;
        trans_z = 0;

        rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
        rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
        rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
        trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
        afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat;

        for frame = 1:Nframe

            detector_new=detector_new*afine_mat';

            x_sensor=detector_new(:,1);
            y_sensor=detector_new(:,2);
            z_sensor=-detector_new(:,3);

            x_sensor = gpuArray(single(x_sensor));
            y_sensor = gpuArray(single(y_sensor));
            z_sensor = gpuArray(single(z_sensor));

            pa_data_frame = gpuArray(single(pa_data(:,:,frame)));
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));

            tic
            [pa_img, total_angle_weight, coherent_factor, ~] = DualSpeedReconstraction_cof_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                                                                Points_sensor_all, Points_img, pa_data_frame, ...
                                                                single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
            toc
            disp(['frame: ', num2str(frame)]);

            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            coherent_factor = gather(coherent_factor);

            pa_img2 = pa_img1.*coherent_factor.^2./total_angle_weight;

            if frame>1

            end
            pa_total = pa_total+pa_img2;

            imin=0;
            imax=max(pa_total,[],"all")*1;

            figure(1); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_total(:,:,:),[],1)),[imin,imax]);
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
            ylabel('X'); xlabel('Z'); title('XZ proj');
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[], 2)),[imin,imax]);
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
            ylabel('Y'); xlabel('Z'); title('YZ proj');
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[], 3)),[imin,imax]);
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out');
            colormap("hot");colorbar;

        end
    otherwise

        disp('Error: Undefined reconstruct mode!');

end
