function cfg = avm_config(dataRoot, outputRoot)
% Frozen preparation defaults; runtime consumes exported LUT only.
repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if nargin < 1 || isempty(dataRoot)
    dataRoot = 'D:\Datasets\competition_surround_eval\a2d2_20180810_150607';
end
if nargin < 2 || isempty(outputRoot)
    outputRoot = fullfile(repo,'outputs','avm_fpga_package');
end
cfg.repo = repo; cfg.data_root = dataRoot; cfg.output_root = outputRoot;
cfg.camera_names = {'front_center','front_left','front_right','side_left','side_right','rear_center'};
cfg.camera_dirs = {'front','front_left','front_right','left','right','back'};
cfg.input_width = 640; cfg.input_height = 320; cfg.pixel_format = 'RGB565_LE';
cfg.native_width = 1920; cfg.native_height = 1208;
cfg.bev_width = 480; cfg.bev_height = 640;
cfg.extent_candidates_m = [8 10.6666666667;9.6 12.8;12 16;15 20];
cfg.selected_extent_index = 4;
cfg.x_min = -7.5; cfg.x_max = 7.5; cfg.y_min = -10; cfg.y_max = 10;
cfg.meters_per_pixel = (cfg.x_max-cfg.x_min)/cfg.bev_width;
% A2D2 vehicle axes use x=forward, y=left, z=up.  Keep the dataset vehicle
% origin (no arbitrary translation); the ground plane is z=0.
cfg.a2d2_origin = [0,0,0];
cfg.vehicle_half_width = 1.25; cfg.vehicle_half_length = 2.5;
cfg.feather_pixels = 30; cfg.max_sync_skew_us = 5000;
cfg.golden_indices = [0,24,49,74,99];
cfg.coord_scan = [4,6,8]; cfg.alpha_scan = [6,8,10];
cfg.internal_guard_scan = [0,4];
cfg.quality_max_mae = 0.5; cfg.quality_min_psnr = 45;
cfg.quality_min_ssim = 0.995;
cfg.lut_entry_bytes = 12; cfg.fps_targets = [30,60];
cfg.schema = 'avm-fpga-freeze/v1';
end

