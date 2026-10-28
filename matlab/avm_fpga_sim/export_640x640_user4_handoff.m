function export_640x640_user4_handoff(destination)
% Export a 640x640 four-channel LUT using FRONT + LEFT + RIGHT + REAR.
% This variant matches the four supplied A2D2 videos and is for visual
% comparison; it does not replace the frozen competition LUT.
if nargin < 1 || isempty(destination)
    repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    destination = fullfile(repo, 'outputs', 'avm_fpga_640x640_user4_20261011');
end
assert(exist(destination, 'dir') ~= 7, 'Refusing to overwrite an existing handoff folder.');

here = fileparts(mfilename('fullpath'));
addpath(here);
cfg = avm_config();
cfg.bev_width = 640;
cfg.bev_height = 640;
frozen = jsondecode(fileread(fullfile(cfg.output_root, 'calibration', 'frozen_calibration.json')));
assert(isequal(frozen.working_resolution(:)', [cfg.input_width cfg.input_height]));

cameras = avm_geometry('load', cfg);
bounds = frozen.bev_bounds_m(:)';
maps = avm_geometry('maps', cfg, cameras, cfg.bev_width, cfg.bev_height, bounds, false);
% camera_names = front_center, front_left, front_right, side_left, side_right, rear_center
ids = [1 4 5 6];
l = avm_lut('select', maps, ids);
q = avm_lut('quantize', l, 4, 6, cfg);
for slot = 1:numel(ids)
    q.cam_a(q.cam_a == ids(slot)-1) = slot-1;
    q.cam_b(q.cam_b == ids(slot)-1) = slot-1;
end

mkdir(destination); mkdir(fullfile(destination, 'lut'));
bytes = avm_lut('pack', q);
write_binary(fullfile(destination, 'lut', 'lut_96bit_le.bin'), bytes.', 'uint8');
write_hex96(fullfile(destination, 'lut', 'lut_readmemh96.hex'), bytes);
params = struct('schema','avm-fpga-handoff/v1-640x640-user4', ...
    'input_width',cfg.input_width,'input_height',cfg.input_height, ...
    'output_width',cfg.bev_width,'output_height',cfg.bev_height, ...
    'channel_count',4,'coordinate_fraction_bits',4,'alpha_fraction_bits',6, ...
    'alpha_unity',64,'lut_entry_bytes',12,'lut_entries',cfg.bev_width*cfg.bev_height, ...
    'pixel_format','RGB565','binary_endianness','little','pixel_order','row-major', ...
    'camera_id_encoding','local slot; 7 is invalid', ...
    'source_global_ids',ids-1,'slot_camera_names',{{'front_center','side_left','side_right','rear_center'}}, ...
    'bev_bounds_m',bounds,'valid_pixels',nnz(q.valid),'blend_pixels',nnz(q.blend), ...
    'bitstream_included',false,'hardware_verified',false);
write_text(fullfile(destination, 'parameters.json'), jsonencode(params, PrettyPrint=true));
write_text(fullfile(destination, 'README.md'), sprintf([ ...
    '# 640x640 user 4CH AVM LUT\n\n' ...
    '- Camera slots: front_center, side_left, side_right, rear_center.\n' ...
    '- This is a visual comparison LUT for the supplied four videos.\n' ...
    '- It is not the frozen competition LUT and has not been board-verified.\n']));
fprintf('HANDOFF_USER4_640=%s\nVALID=%d BLEND=%d LUT_BYTES=%d\n', destination, nnz(q.valid), nnz(q.blend), numel(bytes));
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

function write_text(path, content)
fid = fopen(path, 'w', 'n', 'UTF-8'); assert(fid > 0, 'Cannot write %s', path);
fwrite(fid, unicode2native(content, 'UTF-8'), 'uint8'); fclose(fid);
end
