% File: ds_dual_speed_tof.m
% Purpose: Dual-speed time-of-flight calculator for an ellipsoidal inner region.
% Authors: Xiali Gao
% Tested with: MATLAB R2024a, CUDA 12.9, NVIDIA RTX 4090.

function tof = ds_dual_speed_tof(points_m, sensor_xyz_m, ellipse, c_out_mps, c_in_mps)

num_pixels = size(points_m, 1);
num_elements = size(sensor_xyz_m, 1);
tof = zeros(num_pixels, num_elements);

c = ellipse.center_m;
abc = [ellipse.a_m, ellipse.b_m, ellipse.c_m];
if any(abc <= 0)
    tof = ds_uniform_tof(points_m, sensor_xyz_m, c_out_mps);
    return;
end
inv_abc2 = 1 ./ (abc .^ 2);

for elem = 1:num_elements
    s = sensor_xyz_m(elem, :);
    d = points_m - s;

    A = d(:,1).^2 * inv_abc2(1) + d(:,2).^2 * inv_abc2(2) + d(:,3).^2 * inv_abc2(3);
    B = 2 * (d(:,1) .* (s(1) - c(1)) * inv_abc2(1) + ...
             d(:,2) .* (s(2) - c(2)) * inv_abc2(2) + ...
             d(:,3) .* (s(3) - c(3)) * inv_abc2(3));
    C = (s(1) - c(1))^2 * inv_abc2(1) + ...
        (s(2) - c(2))^2 * inv_abc2(2) + ...
        (s(3) - c(3))^2 * inv_abc2(3) - 1;

    disc = B.^2 - 4 .* A .* C;
    sqrt_disc = sqrt(max(disc, 0));
    t1 = (-B - sqrt_disc) ./ (2 .* A);
    t2 = (-B + sqrt_disc) ./ (2 .* A);

    valid1 = disc >= 0 & t1 >= 0 & t1 <= 1;
    valid2 = disc >= 0 & t2 >= 0 & t2 <= 1;
    t_enter = min(t1, t2);
    t_exit = max(t1, t2);

    inside_sensor = local_inside(s, c, inv_abc2);
    inside_point = local_inside(points_m, c, inv_abc2);

    total_dist = sqrt(sum(d.^2, 2));
    inside_fraction = zeros(num_pixels, 1);

    two_hits = valid1 & valid2;
    inside_fraction(two_hits) = max(0, t_exit(two_hits) - t_enter(two_hits));

    one_hit = xor(valid1, valid2);
    hit_t = t1;
    hit_t(~valid1) = t2(~valid1);
    inside_fraction(one_hit & inside_sensor) = hit_t(one_hit & inside_sensor);
    inside_fraction(one_hit & inside_point) = 1 - hit_t(one_hit & inside_point);

    no_hit_inside = ~valid1 & ~valid2 & inside_sensor & inside_point;
    inside_fraction(no_hit_inside) = 1;

    inside_dist = total_dist .* inside_fraction;
    outside_dist = total_dist - inside_dist;
    tof(:, elem) = outside_dist ./ c_out_mps + inside_dist ./ c_in_mps;
end
end

function inside = local_inside(p, c, inv_abc2)
inside = ((p(:,1) - c(1)).^2 * inv_abc2(1) + ...
          (p(:,2) - c(2)).^2 * inv_abc2(2) + ...
          (p(:,3) - c(3)).^2 * inv_abc2(3)) <= 1;
end
