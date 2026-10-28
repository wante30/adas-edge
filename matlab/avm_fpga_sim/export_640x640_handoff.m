function export_640x640_handoff(destination, frame_index)
% Export a fresh 4CH 640x640 AVM handoff for RTL simulation.
% Uses the frozen A2D2 calibration and BEST_4CH camera selection.
if nargin < 1 || isempty(destination)
    destination = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
        'outputs', 'avm_fpga_640x640_20261011');
end
if nargin < 2 || isempty(frame_index), frame_index = 0; end
assert(exist(destination,'dir') ~= 7, 'Refusing to overwrite an existing handoff folder.');

here = fileparts(mfilename('fullpath'));
addpath(here);
cfg = avm_config();
cfg.bev_width = 640;
cfg.bev_height = 640;
cfg.output_root = destination;
repo = fileparts(fileparts(here));
frozen = jsondecode(fileread(fullfile(repo, 'outputs', 'avm_fpga_package', ...
    'calibration', 'frozen_calibration.json')));
assert(isequal(frozen.working_resolution(:)', [cfg.input_width cfg.input_height]));
assert(isequal(frozen.bev_bounds_m(:)', [cfg.x_min cfg.x_max cfg.y_min cfg.y_max]));

cameras = avm_geometry('load', cfg);
bounds = frozen.bev_bounds_m(:)';
maps = avm_geometry('maps', cfg, cameras, cfg.bev_width, cfg.bev_height, bounds, false);
ids = frozen.best_4ch_ids(:)' + 1;
l = avm_lut('select', maps, ids);
q = avm_lut('quantize', l, 4, 6, cfg);
% Convert global A2D2 IDs to dense local FPGA slots 0..3.
for slot = 1:numel(ids)
    q.cam_a(q.cam_a == ids(slot)-1) = slot-1;
    q.cam_b(q.cam_b == ids(slot)-1) = slot-1;
end

mkdir(destination); mkdir(fullfile(destination, 'lut'));
frame_dir = fullfile(destination, 'golden', sprintf('frame_%03d', frame_index));
mkdir(fullfile(destination, 'golden')); mkdir(frame_dir); mkdir(fullfile(frame_dir, 'inputs'));
bytes = avm_lut('pack', q);
write_binary(fullfile(destination, 'lut', 'lut_96bit_le.bin'), bytes.', 'uint8');
write_hex96(fullfile(destination, 'lut', 'lut_readmemh96.hex'), bytes);
write_binary(fullfile(destination, 'lut', 'valid_u8.bin'), uint8(q.valid).', 'uint8');
write_binary(fullfile(destination, 'lut', 'blend_u8.bin'), uint8(q.blend).', 'uint8');
write_binary(fullfile(destination, 'lut', 'alpha_u16_le.bin'), uint16(q.alpha_q).', 'uint16');

images = cell(1, numel(cfg.camera_names));
for k = 1:numel(cfg.camera_names)
    images{k} = imresize(imread(fullfile(cfg.data_root, cfg.camera_dirs{k}, ...
        sprintf('%03d.png', frame_index))), [cfg.input_height cfg.input_width], 'bilinear');
end
for slot = 1:numel(ids)
    words = avm_render('pack565', images{ids(slot)});
    stem = sprintf('slot_%d_%s', slot-1, cfg.camera_names{ids(slot)});
    write_pixels(fullfile(frame_dir, 'inputs', stem), words);
end
fixed = avm_render('fixed', images(ids), q, 'rgb565', 0);
write_pixels(fullfile(frame_dir, 'expected_rgb565'), fixed.words);
imwrite(avm_render('expand565', fixed.words), fullfile(frame_dir, 'expected_rgb565.png'));

params = struct('schema','avm-fpga-handoff/v1-640x640', ...
    'input_width',cfg.input_width,'input_height',cfg.input_height, ...
    'output_width',cfg.bev_width,'output_height',cfg.bev_height, ...
    'channel_count',4,'coordinate_fraction_bits',4,'alpha_fraction_bits',6, ...
    'guard_bits',0,'alpha_unity',64,'lut_entry_bytes',12, ...
    'lut_entries',cfg.bev_width*cfg.bev_height,'pixel_format','RGB565', ...
    'binary_endianness','little','pixel_order','row-major', ...
    'camera_id_encoding','local slot; 7 is invalid', ...
    'source_global_ids',ids-1,'slot_camera_names',{cfg.camera_names(ids)}, ...
    'bev_bounds_m',bounds,'valid_pixels',nnz(q.valid),'blend_pixels',nnz(q.blend), ...
    'source_frame_index',frame_index,'bitstream_included',false,'hardware_verified',false);
write_text(fullfile(destination, 'parameters.json'), jsonencode(params, PrettyPrint=true));
write_text(fullfile(destination, 'README.md'), sprintf([ ...
    '# 640x640 4CH AVM RTL handoff\n\n' ...
    '- Four RGB565 little-endian inputs: 640x320.\n' ...
    '- AVM output: 640x640 RGB565, row-major.\n' ...
    '- LUT entry: 96-bit, 12 bytes, Q0.4 coordinates and Q0.6 alpha.\n' ...
    '- This package is for RTL simulation; it is not a board validation result.\n']));
fprintf('HANDOFF_640=%s\nVALID=%d BLEND=%d LUT_BYTES=%d\n', destination, nnz(q.valid), nnz(q.blend), numel(bytes));
end

function write_binary(path, data, precision)
fid = fopen(path, 'w', 'ieee-le'); assert(fid > 0, 'Cannot write %s', path);
fwrite(fid, data(:), precision); fclose(fid);
end

function write_hex96(path, bytes)
fid = fopen(path, 'w'); assert(fid > 0, 'Cannot write %s', path);
text = reshape(dec2hex(reshape(fliplr(bytes).', [], 1), 2).', 24, []).';
fprintf(fid, '%s', reshape([text repmat(newline, size(text,1), 1)].', 1, []));
fclose(fid);
end

function write_pixels(stem, words)
write_binary([stem '.bin'], words.', 'uint16');
fid = fopen([stem '.hex'], 'w'); assert(fid > 0, 'Cannot write %s', stem);
fprintf(fid, '%04X\n', reshape(words.', [], 1)); fclose(fid);
end

function write_text(path, content)
fid = fopen(path, 'w', 'n', 'UTF-8'); assert(fid > 0, 'Cannot write %s', path);
fwrite(fid, unicode2native(content, 'UTF-8'), 'uint8'); fclose(fid);
end
