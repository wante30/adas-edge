function export_fpga_handoff(destination, includeGolden)
% Export frozen AVM data to a NEW handoff folder; do not rerun run_all.
if nargin<2, includeGolden=true; end
cfg=avm_config(); original=cfg.output_root;
if nargin<1, destination=fullfile(cfg.repo,'outputs',['avm_fpga_handoff_' char(datetime('now','Format','yyyyMMdd_HHmmss'))]); end
assert(exist(destination,'dir')~=7,'Refusing to overwrite an existing handoff folder.');
mkdir(destination);
frozen=jsondecode(fileread(fullfile(original,'calibration','frozen_calibration.json')));
assert(isequal(frozen.working_resolution(:)',[cfg.input_width cfg.input_height]));
assert(isequal(frozen.bev_resolution(:)',[cfg.bev_width cfg.bev_height]));
assert(isequal(frozen.bev_bounds_m(:)',[cfg.x_min cfg.x_max cfg.y_min cfg.y_max]));
saved=load(fullfile(original,'lut','avm_selected.mat'),'q'); q6=saved.q;
report=fileread(fullfile(cfg.repo,'docs','avm_fixed_point_report.md'));
t=regexp(report,'Chosen point: F=(\d+), alpha=(\d+), guard=(\d+)','tokens','once');
assert(~isempty(t)); F=str2double(t{1}); A=str2double(t{2}); G=str2double(t{3});
assert(q6.F==F && q6.A==A && G==0,'Unsupported or inconsistent frozen format.');
assert(isequal(size(q6.valid),[cfg.bev_height cfg.bev_width]));
cameras=avm_geometry('load',cfg);
maps=avm_geometry('maps',cfg,cameras,cfg.bev_width,cfg.bev_height,frozen.bev_bounds_m(:)',false);
ids4=frozen.best_4ch_ids(:)'+1;
q4=avm_lut('quantize',avm_lut('select',maps,ids4),F,A,cfg);
allq={q4,q6}; allids={ids4,1:6}; variants={'4ch','6ch'};
for variant=1:2
    ids=allids{variant}; q=allq{variant}; globalq=q;
    % Four physical inputs must use dense local slots 0..3, not global 2..5.
    for slot=1:numel(ids)
        q.cam_a(globalq.cam_a==ids(slot)-1)=slot-1;
        q.cam_b(globalq.cam_b==ids(slot)-1)=slot-1;
    end
    root=fullfile(destination,variants{variant}); mkdir(root); mkdir(fullfile(root,'lut'));
    bytes=avm_lut('pack',q);
    if variant==2
        fid=fopen(fullfile(original,'lut','avm_selected_96b_le.bin'),'r'); old=fread(fid,Inf,'*uint8'); fclose(fid);
        assert(isequal(old,reshape(bytes.',[],1)),'Six-channel frozen LUT differs.');
    end
    unpacked=avm_lut('unpack',bytes,cfg.bev_height,cfg.bev_width,F,A);
    for field={'addr_a','addr_b','fx_a','fy_a','fx_b','fy_b','alpha_q','cam_a','cam_b','valid','blend'}
        assert(isequal(double(q.(field{1})),double(unpacked.(field{1}))));
    end
    write_data(fullfile(root,'lut','lut_96bit_le.bin'),bytes.','uint8');
    write_hex96(fullfile(root,'lut','lut_readmemh96.hex'),bytes);
    write_data(fullfile(root,'lut','valid_u8.bin'),uint8(q.valid).','uint8');
    write_data(fullfile(root,'lut','blend_u8.bin'),uint8(q.blend).','uint8');
    write_data(fullfile(root,'lut','alpha_u16_le.bin'),uint16(q.alpha_q).','uint16');
    params=struct('schema','avm-fpga-handoff/v1','input_width',cfg.input_width, ...
        'input_height',cfg.input_height,'output_width',cfg.bev_width,'output_height',cfg.bev_height, ...
        'channel_count',numel(ids),'coordinate_fraction_bits',F,'alpha_fraction_bits',A, ...
        'guard_bits',G,'alpha_unity',2^A,'lut_entry_bytes',12,'lut_entries',numel(q.valid), ...
        'pixel_format','RGB565','binary_endianness','little','pixel_order','row-major', ...
        'camera_id_encoding','local slot; 7 is invalid','source_global_ids',ids-1, ...
        'slot_camera_names',{cfg.camera_names(ids)},'bev_bounds_m',frozen.bev_bounds_m(:)', ...
        'valid_pixels',nnz(q.valid),'blend_pixels',nnz(q.blend), ...
        'synchronization_status','UNKNOWN: front-left/right timestamp sidecars absent', ...
        'bitstream_included',false,'hardware_verified',false);
    write_text(fullfile(root,'parameters.json'),jsonencode(params,PrettyPrint=true));
    if includeGolden, export_golden(root,cfg,q,globalq,ids,G); end
end
mkdir(fullfile(destination,'reference'));
mkdir(fullfile(destination,'reference','calibration'));
for file={'cams_lidars.json','frozen_calibration.json','avm_4ch.json','avm_6ch.json'}
    copyfile(fullfile(original,'calibration',file{1}),fullfile(destination,'reference','calibration',file{1}));
end
for file={'timestamp_alignment.csv','download_manifest.csv'}
    copyfile(fullfile(cfg.data_root,file{1}),fullfile(destination,'reference','calibration',file{1}));
end
copyfile(fullfile(original,'reports'),fullfile(destination,'reference','reports'));
copyfile(fullfile(original,'previews'),fullfile(destination,'reference','previews'));
mkdir(fullfile(destination,'reference','docs'));
docs=dir(fullfile(cfg.repo,'docs','avm_*.md'));
for i=1:numel(docs), copyfile(fullfile(docs(i).folder,docs(i).name),fullfile(destination,'reference','docs',docs(i).name)); end
mkdir(fullfile(destination,'matlab'));
sources=dir(fullfile(fileparts(mfilename('fullpath')),'*.m'));
for i=1:numel(sources), copyfile(fullfile(sources(i).folder,sources(i).name),fullfile(destination,'matlab',sources(i).name)); end
copyfile(fullfile(fileparts(mfilename('fullpath')),'fpga_handoff_README.md'),fullfile(destination,'README.md'));
copyfile(fullfile(fileparts(mfilename('fullpath')),'verify_fpga_handoff.py'),fullfile(destination,'verify_fpga_handoff.py'));
fprintf('HANDOFF_DIR=%s\n',destination);
end

function write_data(path,data,precision)
fid=fopen(path,'w','ieee-le'); assert(fid>0); fwrite(fid,data(:),precision); fclose(fid);
end

function write_hex96(path,bytes)
fid=fopen(path,'w'); assert(fid>0);
% Verilog memory words are printed MSB first; binary bytes are LSB first.
text=reshape(dec2hex(reshape(fliplr(bytes).',[],1),2).',24,[]).';
fprintf(fid,'%s',reshape([text repmat(newline,size(text,1),1)].',1,[]));
fclose(fid);
end

function write_text(path,content)
fid=fopen(path,'w','n','UTF-8'); assert(fid>0); fwrite(fid,unicode2native(content,'UTF-8'),'uint8'); fclose(fid);
end

function export_golden(root,cfg,q,globalq,ids,G)
mkdir(fullfile(root,'golden'));
metadata=readtable(fullfile(cfg.data_root,'download_manifest.csv'),'Delimiter',',','TextType','string','VariableNamingRule','preserve');
provenance=cell(numel(cfg.golden_indices)*numel(ids),6); row=0;
for idx=cfg.golden_indices
    dirpath=fullfile(root,'golden',sprintf('frame_%03d',idx)); mkdir(dirpath); mkdir(fullfile(dirpath,'inputs'));
    images=cell(1,numel(cfg.camera_names));
    for k=1:numel(cfg.camera_names)
        image=imread(fullfile(cfg.data_root,cfg.camera_dirs{k},sprintf('%03d.png',idx)));
        images{k}=imresize(image,[cfg.input_height cfg.input_width],'bilinear');
    end
    for slot=1:numel(ids)
        k=ids(slot); words=avm_render('pack565',images{k});
        stem=sprintf('slot_%d_%s',slot-1,cfg.camera_names{k});
        write_pixels(fullfile(dirpath,'inputs',stem),words);
        row=row+1;
        if ismember(k,[2 3])
            src=83+3*idx; filestem=strrep(cfg.camera_names{k},'_','');
            uri=sprintf('https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar/20180810_150607/camera/cam_%s/20180810150607_camera_%s_%09d.png',cfg.camera_names{k},filestem,src);
            evidence='download index sequence; timestamp UNKNOWN';
        else
            selected=metadata.("Logical")==string(cfg.camera_dirs{k}) & metadata.("OutputIndex")==idx;
            assert(nnz(selected)==1);
            src=metadata.("SourceIndex")(selected); uri=char(metadata.("Uri")(selected)); evidence='original download manifest + four-view timestamp alignment';
        end
        provenance(row,:)={idx,slot-1,cfg.camera_names{k},src,uri,evidence};
    end
    localResult=avm_render('fixed',images(ids),q,'rgb565',G);
    write_pixels(fullfile(dirpath,'expected_rgb565'),localResult.words);
    imwrite(avm_render('expand565',localResult.words),fullfile(dirpath,'expected_rgb565.png'));
    imwrite(localResult.rgb,fullfile(dirpath,'internal_rgb888_before_pack.png'));
    floating=avm_render('float',images,globalq);
    imwrite(uint8(min(max(round(floating),0),255)),fullfile(dirpath,'float_reference_rgb888.png'));
end
writetable(cell2table(provenance,VariableNames={'frame_index','slot','camera_name','source_index','source_uri','synchronization_evidence'}),fullfile(root,'golden','source_manifest.csv'));
end

function write_pixels(stem,words)
% MATLAB column-major requires transposition to emit row-major pixels.
write_data([stem '.bin'],words.','uint16');
fid=fopen([stem '.hex'],'w'); assert(fid>0); fprintf(fid,'%04X\n',reshape(words.',[],1)); fclose(fid);
end
