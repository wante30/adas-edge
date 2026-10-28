function run_all()
%RUN_ALL Reproducible AVM floating/fixed-point freeze package.
% This emits maps, LUTs, golden data, and reports; it does not synthesize RTL.
cfg=avm_config();
assert(exist(cfg.data_root,'dir')==7,'A2D2 data root is missing: %s',cfg.data_root);
if exist(cfg.output_root,'dir')~=7, mkdir(cfg.output_root); end
for d={'calibration','lut','previews','per_camera','golden','reports'}
    p=fullfile(cfg.output_root,d{1}); if exist(p,'dir')~=7, mkdir(p); end
end
cams=avm_geometry('load',cfg);
maps_by_extent=cell(size(cfg.extent_candidates_m,1),1);
extent_rows=zeros(size(cfg.extent_candidates_m,1),7);
combos=nchoosek(1:numel(cams),4);
for ei=1:size(cfg.extent_candidates_m,1)
    ex=cfg.extent_candidates_m(ei,:);
    bounds=[-ex(1)/2,ex(1)/2,-ex(2)/2,ex(2)/2];
    maps=avm_geometry('maps',cfg,cams,cfg.bev_width,cfg.bev_height,bounds,false);
    maps_by_extent{ei}=maps;
    best_cov=-inf; best_i=1; best_overlap=0; best_hole=1;
    for ci=1:size(combos,1)
        l=avm_lut('select',maps,combos(ci,:)); s=selection_stats(maps,l,combos(ci,:));
        if s.coverage>best_cov
            best_cov=s.coverage; best_i=ci; best_overlap=s.overlap_absolute; best_hole=s.hole_fraction;
        end
    end
    l6=avm_lut('select',maps,1:numel(cams)); s6=selection_stats(maps,l6,1:numel(cams));
    extent_rows(ei,:)=[ei,ex,best_cov,best_overlap,best_hole,s6.coverage];
end
ei=cfg.selected_extent_index;
assert(ei>=1 && ei<=numel(maps_by_extent));
maps=maps_by_extent{ei}; bounds=maps.bounds;
[~,best_extent_row]=max(extent_rows(:,4));
best4=combos(max(1,best_extent_row),:);
% Select the best four-camera combination at the frozen extent, not at a
% different extent that happened to win the scan.
best_cov=-inf;
for ci=1:size(combos,1)
    st=selection_stats(maps,avm_lut('select',maps,combos(ci,:)),combos(ci,:));
    if st.coverage>best_cov, best_cov=st.coverage; best4=combos(ci,:); end
end
l4=avm_lut('select',maps,best4); l6=avm_lut('select',maps,1:numel(cams));
s4=selection_stats(maps,l4,best4); s6=selection_stats(maps,l6,1:numel(cams));
copyfile(fullfile(cfg.data_root,'probe','cams_lidars.json'),fullfile(cfg.output_root,'calibration','cams_lidars.json'));
write_frame_manifest(cfg);
frozen=struct('schema',cfg.schema,'source','A2D2 20180810_150607', ...
    'camera_names',{cfg.camera_names},'camera_dirs',{cfg.camera_dirs}, ...
    'native_resolution',[cfg.native_width,cfg.native_height], ...
    'working_resolution',[cfg.input_width,cfg.input_height], ...
    'bev_resolution',[cfg.bev_width,cfg.bev_height],'bev_bounds_m',bounds, ...
    'coordinate_system','+X right, +Y forward, +Z up','ground_plane','Z=0', ...
    'pixel_format',cfg.pixel_format,'selected_extent_index',ei, ...
    'best_4ch_ids',best4-1,'best_6ch_ids',(1:numel(cams))-1, ...
    'lut_entry_bytes',cfg.lut_entry_bytes,'lut_addressing','addr = y*640 + x', ...
    'input_mapping','fused inverse distortion + ground projection');
write_json(fullfile(cfg.output_root,'calibration','frozen_calibration.json'),frozen);
write_json(fullfile(cfg.output_root,'calibration','avm_4ch.json'),selection_json(cfg,bounds,best4,s4));
write_json(fullfile(cfg.output_root,'calibration','avm_6ch.json'),selection_json(cfg,bounds,1:numel(cams),s6));
[images,raws,undistorted]=load_frame(cfg,cams,0);
float4=avm_render('float',images,l4); float6=avm_render('float',images,l6);
float4u=uint8(min(max(round(float4),0),255)); float6u=uint8(min(max(round(float6),0),255));
write_image(fullfile(cfg.output_root,'previews','avm_4ch_float_rgb888.png'),float4u);
write_image(fullfile(cfg.output_root,'previews','avm_6ch_float_rgb888.png'),float6u);
write_image(fullfile(cfg.output_root,'previews','avm_4ch_valid_mask.png'),uint8(l4.valid)*255);
write_image(fullfile(cfg.output_root,'previews','avm_6ch_valid_mask.png'),uint8(l6.valid)*255);
for k=1:numel(cams)
    stem=sprintf('%02d_%s',k-1,cams(k).name);
    write_image(fullfile(cfg.output_root,'per_camera',[stem '_original.png']),raws{k});
    write_image(fullfile(cfg.output_root,'per_camera',[stem '_undistorted.png']),undistorted{k});
    write_image(fullfile(cfg.output_root,'per_camera',[stem '_working_rgb888.png']),images{k});
    one=avm_lut('select',maps,k); bev=avm_render('float',images,one);
    write_image(fullfile(cfg.output_root,'per_camera',[stem '_bev.png']),uint8(min(max(round(bev),0),255)));
    write_image(fullfile(cfg.output_root,'per_camera',[stem '_valid_mask.png']),uint8(one.valid)*255);
end
images565=cell(size(images));
for k=1:numel(images), images565{k}=avm_render('expand565',avm_render('pack565',images{k})); end
float565=uint8(min(max(round(avm_render('float',images565,l6)),0),255));
scan=zeros(numel(cfg.coord_scan)*numel(cfg.alpha_scan)*numel(cfg.internal_guard_scan),12);
row=0; best_pass=[]; best_any=[]; best_score=inf;
for fi=1:numel(cfg.coord_scan)
    for ai=1:numel(cfg.alpha_scan)
        for gi=1:numel(cfg.internal_guard_scan)
            F=cfg.coord_scan(fi); A=cfg.alpha_scan(ai); G=cfg.internal_guard_scan(gi);
            q=avm_lut('quantize',l6,F,A,cfg); fixed=avm_render('fixed',images,q,'rgb565',G);
            m888=avm_render('metrics',fixed.rgb,float6u,l6.valid); m565=avm_render('metrics',fixed.rgb,float565,l6.valid);
            row=row+1; scan(row,:)=[F,A,G,m888.mae,m888.psnr,m888.ssim,m565.mae,m565.psnr,m565.ssim,m565.max_abs,nnz(q.valid),nnz(q.blend)];
            score=m565.mae+0.001*max(0,45-m565.psnr)+max(0,0.995-m565.ssim)*100;
            cand=struct('q',q,'fixed',fixed,'m888',m888,'m565',m565,'F',F,'A',A,'G',G);
            if isempty(best_any) || score<best_score, best_any=cand; best_score=score; end
            pass=m565.mae<=cfg.quality_max_mae && m565.psnr>=cfg.quality_min_psnr && m565.ssim>=cfg.quality_min_ssim;
            if pass && (isempty(best_pass) || F+A+G<best_pass.F+best_pass.A+best_pass.G), best_pass=cand; end
        end
    end
end
if isempty(best_pass), chosen=best_any; quality_gate='FAIL_RGB565_REFERENCE_THRESHOLDS'; else, chosen=best_pass; quality_gate='PASS_RGB565_REFERENCE_ONLY'; end
q=chosen.q; lut_bytes=avm_lut('pack',q); write_lut_files(cfg,q,lut_bytes);
save(fullfile(cfg.output_root,'lut','avm_selected.mat'),'q','-v7.3');
write_binary(fullfile(cfg.output_root,'lut','valid_mask.bin'),uint8(q.valid));
write_binary(fullfile(cfg.output_root,'lut','blend_mask.bin'),uint8(q.blend));
write_binary(fullfile(cfg.output_root,'lut','alpha_q.bin'),uint16(q.alpha_q),'uint16');
write_image(fullfile(cfg.output_root,'previews','avm_6ch_fixed_rgb565.png'),uint8(chosen.fixed.rgb));
write_image(fullfile(cfg.output_root,'previews','avm_6ch_fixed_diff.png'),diff_image(chosen.fixed.rgb,float6u));
q4=avm_lut('quantize',l4,chosen.F,chosen.A,cfg);
fixed4=avm_render('fixed',images,q4,'rgb565',chosen.G);
write_image(fullfile(cfg.output_root,'previews','avm_4ch_fixed_rgb565.png'),uint8(fixed4.rgb));
write_image(fullfile(cfg.output_root,'previews','avm_4ch_fixed_diff.png'),diff_image(fixed4.rgb,float4u));
addr_coord_a=q.y_a*cfg.input_width+q.x_a; addr_coord_b=q.y_b*cfg.input_width+q.x_b;
addr_mismatch_a=nnz(addr_coord_a(q.valid)~=q.addr_a(q.valid)); addr_mismatch_b=nnz(addr_coord_b(q.blend)~=q.addr_b(q.blend));
golden_rows=zeros(numel(cfg.golden_indices),4);
for gi=1:numel(cfg.golden_indices)
    idx=cfg.golden_indices(gi); [gimgs,~,~]=load_frame(cfg,cams,idx);
    gf=uint8(min(max(round(avm_render('float',gimgs,l6)),0),255)); gq=avm_render('fixed',gimgs,q,'rgb565',chosen.G);
    gdir=fullfile(cfg.output_root,'golden',sprintf('frame_%03d',idx)); if exist(gdir,'dir')~=7, mkdir(gdir); end
    write_image(fullfile(gdir,'float_rgb888.png'),gf); write_image(fullfile(gdir,'fixed_rgb565.png'),gq.rgb); write_image(fullfile(gdir,'diff.png'),diff_image(gq.rgb,gf));
    golden_rows(gi,:)=[idx,nnz(l6.valid),mean(l6.valid(:)),mean(abs(double(gq.rgb(:))-double(gf(:))))];
end
bw4=bandwidth_row(cfg,4); bw6=bandwidth_row(cfg,6);
write_reports(cfg,cams,bounds,extent_rows,combos,best4,l4,s4,l6,s6,scan,chosen,quality_gate,bw4,bw6,addr_mismatch_a,addr_mismatch_b,golden_rows);
fprintf('AVM package written: %s\n',cfg.output_root);
fprintf('BEST_4CH=%s coverage=%.6f; BEST_6CH coverage=%.6f; fixed gate=%s (F=%d A=%d G=%d)\n',strjoin(cfg.camera_names(best4),','),s4.coverage,s6.coverage,quality_gate,chosen.F,chosen.A,chosen.G);
end

function s=selection_stats(m,l,ids)
cv=m.valid(:,:,ids); s.ids=ids-1; s.coverage=mean(l.valid(:)); s.hole_fraction=1-s.coverage;
s.overlap_absolute=mean((sum(cv,3)>1)&l.valid,'all'); s.overlap_of_valid=s.overlap_absolute/max(s.coverage,eps);
s.blend_fraction=mean(l.blend(:)); s.valid_pixels=nnz(l.valid); s.overlap_pixels=nnz((sum(cv,3)>1)&l.valid);
end

function write_frame_manifest(cfg)
% The four original folders have timestamp_alignment.csv.  The added
% front_left/front_right folders were fetched using the same source index
% sequence (83 + 3*output_index), but have no per-frame JSON sidecars; keep
% that distinction explicit instead of claiming six-way timestamp proof.
fid=fopen(fullfile(cfg.output_root,'calibration','a2d2_frame_manifest.csv'),'w'); assert(fid>0);
fprintf(fid,'logical_camera,output_index,source_index,local_file,source_uri,sync_evidence\n');
for k=1:numel(cfg.camera_names)
    if strcmp(cfg.camera_names{k},'front_center'), folder='front_center'; filestem='frontcenter'; elseif strcmp(cfg.camera_names{k},'front_left'), folder='front_left'; filestem='frontleft'; elseif strcmp(cfg.camera_names{k},'front_right'), folder='front_right'; filestem='frontright'; elseif strcmp(cfg.camera_names{k},'side_left'), folder='side_left'; filestem='sideleft'; elseif strcmp(cfg.camera_names{k},'side_right'), folder='side_right'; filestem='sideright'; else, folder='rear_center'; filestem='rearcenter'; end
    for i=0:99
        src=83+3*i; local=fullfile(cfg.data_root,cfg.camera_dirs{k},sprintf('%03d.png',i));
        uri=sprintf('https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar/20180810_150607/camera/cam_%s/20180810150607_camera_%s_%09d.png',folder,filestem,src);
        if ismember(k,[2,3]), evidence='source-index sequence; timestamp sidecar unavailable'; else, evidence='timestamp_alignment.csv for four original views'; end
        fprintf(fid,'%s,%d,%d,%s,%s,%s\n',cfg.camera_names{k},i,src,local,uri,evidence);
    end
end
fclose(fid);
end

function [images,raws,undistorted]=load_frame(cfg,cams,idx)
images=cell(1,numel(cams)); raws=cell(size(images)); undistorted=cell(size(images));
for k=1:numel(cams)
    p=fullfile(cfg.data_root,cfg.camera_dirs{k},sprintf('%03d.png',idx)); assert(exist(p,'file')==2,'Missing frame: %s',p);
    raw=imread(p); if size(raw,3)==1, raw=repmat(raw,1,1,3); end
    raws{k}=raw; images{k}=imresize(raw,[cfg.input_height,cfg.input_width],'bilinear');
    undistorted{k}=imresize(avm_geometry('undistort',cams(k),raw),[cfg.input_height,cfg.input_width],'bilinear');
end
end

function j=selection_json(cfg,bounds,ids,s)
j=struct('schema',cfg.schema,'camera_ids',ids-1,'camera_names',{cfg.camera_names(ids)},'bounds_m',bounds, ...
    'coverage',s.coverage,'hole_fraction',s.hole_fraction,'overlap_absolute',s.overlap_absolute, ...
    'overlap_of_valid',s.overlap_of_valid,'blend_fraction',s.blend_fraction,'valid_pixels',s.valid_pixels);
end

function write_lut_files(cfg,q,bytes)
% bytes is N-by-12; transpose before fwrite so each 12-byte entry is
% contiguous in the binary stream (row-major BEV order).
write_binary(fullfile(cfg.output_root,'lut','avm_selected_96b_le.bin'),bytes.');
fid=fopen(fullfile(cfg.output_root,'lut','avm_selected_96b_le.hex'),'w'); assert(fid>0);
for i=1:size(bytes,1), fprintf(fid,'%02X',bytes(i,:)); fprintf(fid,'\n'); end
fclose(fid);
fid=fopen(fullfile(cfg.output_root,'lut','avm_selected.csv'),'w'); assert(fid>0);
fprintf(fid,'row,col,addr_a,addr_b,fx_a,fy_a,fx_b,fy_b,alpha_q,cam_a,cam_b,valid,blend\n');
[h,w]=size(q.valid);
for y=1:h
    for x=1:w
        fprintf(fid,'%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n',y-1,x-1,q.addr_a(y,x),q.addr_b(y,x),q.fx_a(y,x),q.fy_a(y,x),q.fx_b(y,x),q.fy_b(y,x),q.alpha_q(y,x),q.cam_a(y,x),q.cam_b(y,x),q.valid(y,x),q.blend(y,x));
    end
end
fclose(fid);
write_text(fullfile(cfg.output_root,'lut','entry_layout.md'),sprintf(['# LUT entry layout (frozen)\n\n' ...
    '- Entry size: 96 bits / 12 bytes, little-endian byte stream.\n' ...
    '- Address: `addr = y * 640 + x`, with x=column and y=row.\n' ...
    '- Fields, least-significant bit first: `addr_a[17:0]`, `addr_b[17:0]`, `fx_a[7:0]`, `fy_a[7:0]`, `fx_b[7:0]`, `fy_b[7:0]`, `alpha_q[10:0]`, `cam_a[2:0]`, `cam_b[2:0]`, `valid`, `blend`; 9 high bits are reserved zero padding.\n' ...
    '- Source address is the top-left RGB565 pixel; neighbors are `addr`, `addr+1`, `addr+640`, and `addr+641`.\n' ...
    '- Fractions are unsigned Q0.%d for the selected scan point.\n'],q.F));
end

function b=bandwidth_row(cfg,ncams)
b.input_bytes=cfg.input_width*cfg.input_height*2*ncams; b.lut_bytes=cfg.bev_width*cfg.bev_height*cfg.lut_entry_bytes; b.output_bytes=cfg.bev_width*cfg.bev_height*2;
b.bytes_per_frame=b.input_bytes+b.lut_bytes+b.output_bytes; b.mb_s_30=b.bytes_per_frame*30/1e6; b.mb_s_60=b.bytes_per_frame*60/1e6;
end

function write_json(path,x), write_text(path,jsonencode(x)); end
function write_text(path,text)
fid=fopen(path,'w','n','UTF-8'); assert(fid>0,'Cannot write %s',path); fwrite(fid,text,'char'); fclose(fid);
end
function write_binary(path,data,precision)
if nargin<3, precision='uint8'; end
fid=fopen(path,'w'); assert(fid>0,'Cannot write %s',path); fwrite(fid,data(:),precision); fclose(fid);
end
function write_image(path,data)
if islogical(data), data=uint8(data)*255; end
imwrite(data,path);
end
function d=diff_image(a,b), d=uint8(min(abs(double(a)-double(b))*4,255)); end

function write_reports(cfg,cams,bounds,extent_rows,combos,best4,l4,s4,l6,s6,scan,chosen,gate,bw4,bw6,ma,mb,golden)
out=cfg.output_root; repo=cfg.repo;
fid=fopen(fullfile(out,'reports','extent_scan.csv'),'w'); fprintf(fid,'index,x_span_m,y_span_m,best4_coverage,best4_overlap,best4_hole,six_coverage\n');
for i=1:size(extent_rows,1), fprintf(fid,'%d,%g,%g,%.9f,%.9f,%.9f,%.9f\n',extent_rows(i,:)); end, fclose(fid);
fid=fopen(fullfile(out,'reports','fixed_point_scan.csv'),'w'); fprintf(fid,'F,A,guard,mae_rgb888,psnr_rgb888,ssim_rgb888,mae_rgb565,psnr_rgb565,ssim_rgb565,max_abs_rgb565,valid_pixels,blend_pixels\n');
for i=1:size(scan,1), fprintf(fid,'%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d\n',scan(i,:)); end, fclose(fid);
fid=fopen(fullfile(out,'reports','bandwidth.csv'),'w'); fprintf(fid,'channels,input_bytes,lut_bytes,output_bytes,total_bytes,MBps_30,MBps_60\n');
fprintf(fid,'4,%d,%d,%d,%d,%.6f,%.6f\n',bw4.input_bytes,bw4.lut_bytes,bw4.output_bytes,bw4.bytes_per_frame,bw4.mb_s_30,bw4.mb_s_60);
fprintf(fid,'6,%d,%d,%d,%d,%.6f,%.6f\n',bw6.input_bytes,bw6.lut_bytes,bw6.output_bytes,bw6.bytes_per_frame,bw6.mb_s_30,bw6.mb_s_60); fclose(fid);
fid=fopen(fullfile(out,'reports','golden_summary.csv'),'w'); fprintf(fid,'frame_id,valid_pixels,coverage,mean_abs_rgb888_difference\n'); fprintf(fid,'%d,%d,%.9f,%.9f\n',golden.'); fclose(fid);

write_text(fullfile(repo,'docs','avm_previous_reference_audit.md'),sprintf(['# AVM previous-reference audit\n\n' ...
    '## Confirmed before this MATLAB run\n\n' ...
    '- `tools/adas_hmi_sim/configs/surround_bev.json` is `APPROXIMATE_PLANAR`: 640x360 inputs, a 640x640 canvas, and hand-authored polygons. It is a PC visual compositor, not A2D2-calibrated AVM and not an FPGA LUT.\n' ...
    '- `tools/adas_hmi_sim/backend/calibrated_surround.py` is a four-role OpenCV remap path driven by an external YAML. It uses one K/D model per role and writes NPZ maps; it does not consume the A2D2 six-camera calibration or RGB565 fixed-point contract.\n' ...
    '- `docs/adas_hmi_surround_rk3566.md` records that the former four videos were not a synchronized fisheye rig and that the old output was an approximate planar CPU result.\n' ...
    '- `woodscape_devkit/omnidet/data/LUTs.pkl` is a neural-network inverse-polynomial fisheye lookup table, not the AVM ground-plane compositor LUT frozen here.\n' ...
    '- The historical single-frame probe used A2D2 names but sampled raw images with undistorted K and did not apply the raw distortion model. Its 79.0844%%/91.4253%% figures are historical only, not this run result.\n\n' ...
    '## This run\n\n' ...
    '- Source: A2D2 `20180810_150607`, native 1920x1208, six-camera calibration copied to `outputs/avm_fpga_package/calibration/cams_lidars.json`.\n' ...
    '- Mapping uses the official A2D2 view axes, `CamMatrixOriginal` plus `Distortion` for the fused inverse map, and ground plane Z=0.\n' ...
    '- Four original camera folders have timestamp-alignment evidence; front-left/front-right use a source-index sequence without timestamp sidecars, so six-camera skew remains UNKNOWN.\n' ...
    '- Selected extent: %.3f m x %.3f m; BEST_4CH coverage %.6f and BEST_6CH coverage %.6f.\n'],bounds(2)-bounds(1),bounds(4)-bounds(3),s4.coverage,s6.coverage));

write_text(fullfile(repo,'docs','avm_mapping_spec.md'),sprintf(['# AVM mapping specification\n\n' ...
    '- Source references: [A2D2 official tutorial notebook](https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/tutorial.ipynb); [OpenCV fisheye camera model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html).\n\n' ...
    '## Frozen coordinates\n\n' ...
    '- Vehicle frame: +X right, +Y forward, +Z up; ground is Z=0. A2D2 source axes are +x forward, +y left, +z up.\n' ...
    '- BEV: %d x %d pixels, x range [%.3f, %.3f] m, y range [%.3f, %.3f] m. Pixel centers are used.\n' ...
    '- Input working image: %d x %d RGB565 little-endian at the FPGA boundary. The retained PC reference remains RGB888.\n\n' ...
    '## Per-pixel operation\n\n' ...
    'For each output pixel, inverse-project the ground point into selected A2D2 camera(s), apply the official fisheye/telecam distortion model, bilinearly sample the source, and blend the two highest-scoring valid cameras. This fused inverse map is equivalent to undistort -> ground projection -> remap while avoiding a full intermediate image in FPGA.\n\n' ...
    '## Selection\n\n' ...
    '- BEST_4CH IDs: %s. Coverage %.6f, hole %.6f, overlap %.6f of the BEV.\n' ...
    '- BEST_6CH IDs: 0..5. Coverage %.6f, hole %.6f, overlap %.6f of the BEV.\n\n' ...
    '## LUT contract\n\n' ...
    '- `avm_selected_96b_le.bin` and `.hex` use one 12-byte entry per row-major output pixel. The exact field offsets and reserved bits are in `outputs/avm_fpga_package/lut/entry_layout.md`.\n' ...
    '- Direct-coordinate versus linear-address checks report %d and %d mismatches over valid A/B samples.\n'],cfg.bev_width,cfg.bev_height,bounds(1),bounds(2),bounds(3),bounds(4),cfg.input_width,cfg.input_height,strjoin(cfg.camera_names(best4),', '),s4.coverage,s4.hole_fraction,s4.overlap_absolute,s6.coverage,s6.hole_fraction,s6.overlap_absolute,ma,mb));

write_text(fullfile(repo,'docs','avm_fixed_point_report.md'),sprintf(['# AVM fixed-point report\n\n' ...
    '- RGB888 floating output is the primary reference. RGB565-expanded floating output isolates coordinate/interpolation quantization from input quantization.\n' ...
    '- Scan F = %s, alpha bits = %s, guard bits = %s.\n' ...
    '- Chosen point: F=%d, alpha=%d, guard=%d; gate=%s.\n' ...
    '- Chosen vs RGB888: MAE %.6f, PSNR %.6f dB, SSIM %.6f, max abs %.0f.\n' ...
    '- Chosen vs RGB565-expanded reference: MAE %.6f, PSNR %.6f dB, SSIM %.6f, max abs %.0f. The selected gate is relative to this reference only; the RGB888 comparison is reported separately and is not silently treated as a pass.\n' ...
    '- This is a software equivalence gate; it is not FPGA timing or resource evidence.\n'],mat2str(cfg.coord_scan),mat2str(cfg.alpha_scan),mat2str(cfg.internal_guard_scan),chosen.F,chosen.A,chosen.G,gate,chosen.m888.mae,chosen.m888.psnr,chosen.m888.ssim,chosen.m888.max_abs,chosen.m565.mae,chosen.m565.psnr,chosen.m565.ssim,chosen.m565.max_abs));

write_text(fullfile(repo,'docs','avm_bandwidth_budget.md'),sprintf(['# AVM bandwidth and LUT budget\n\n' ...
    'This is a theoretical external-memory traffic budget at the frozen working resolution. It counts selected RGB565 camera reads, one packed LUT read per output pixel, and one RGB565 output write; it is not a measured DDR trace.\n\n' ...
    '| channels | input/frame | LUT/frame | output/frame | total/frame | 30 FPS | 60 FPS |\n|---:|---:|---:|---:|---:|---:|---:|\n' ...
    '| 4 | %d | %d | %d | %d | %.3f MB/s | %.3f MB/s |\n' ...
    '| 6 | %d | %d | %d | %d | %.3f MB/s | %.3f MB/s |\n\n' ...
    '- One 96-bit entry is 12 bytes; %d entries consume %d bytes (%.3f MiB).\n' ...
    '- Valid and blend masks are separate 1-byte/pixel audit artifacts; the packed LUT already carries both flags.\n'],bw4.input_bytes,bw4.lut_bytes,bw4.output_bytes,bw4.bytes_per_frame,bw4.mb_s_30,bw4.mb_s_60,bw6.input_bytes,bw6.lut_bytes,bw6.output_bytes,bw6.bytes_per_frame,bw6.mb_s_30,bw6.mb_s_60,cfg.bev_width*cfg.bev_height,cfg.bev_width*cfg.bev_height*cfg.lut_entry_bytes,cfg.bev_width*cfg.bev_height*cfg.lut_entry_bytes/2^20));

write_text(fullfile(repo,'docs','avm_target_rtl_architecture.md'),['# Target RTL architecture (design only)\n\n' ...
    '```text\n' ...
    'camera ingress RGB565 -> per-camera line buffers -> LUT address/fraction reader\n' ...
    '                                      |\n' ...
    '                         4-neighbor bilinear samplers\n' ...
    '                                      |\n' ...
    '                         alpha/weight blend + valid mask\n' ...
    '                                      |\n' ...
    '                              RGB565 BEV output\n' ...
    '```\n\n' ...
    '- The LUT reader emits two source camera IDs, two top-left addresses, Q0.F fractions, alpha Q0.A, valid, and blend.\n' ...
    '- Line buffers must provide `addr`, `addr+1`, `addr+stride`, and `addr+stride+1` without an out-of-frame read.\n' ...
    '- MATLAB checks arithmetic width and saturation points only; no Verilog, timing, device-tree, or bitstream claim is made.\n']);

fid=fopen(fullfile(repo,'docs','avm_golden_sets.md'),'w'); fprintf(fid,['# AVM golden sets\n\nFive frame IDs are frozen: 0, 24, 49, 74, 99. The generated directories are `frame_000`, `frame_024`, `frame_049`, `frame_074`, and `frame_099`; each contains float RGB888, fixed RGB565, and diff images.\n\nThe four original A2D2 folders have `timestamp_alignment.csv` evidence. The added front-left/front-right folders use the same source-index sequence but have no per-frame timestamp sidecars, so six-camera timestamp skew is UNKNOWN and must be revalidated before hardware claims.\n']); fclose(fid);
write_text(fullfile(out,'README.md'),sprintf(['# AVM FPGA preparation package\n\n' ...
    '- Generated by `matlab/avm_fpga_sim/run_all.m`.\n' ...
    '- This is a PC/MATLAB reference and LUT freeze package; it is not synthesized FPGA implementation.\n' ...
    '- BEST_4CH: %s (coverage %.6f). BEST_6CH: all six (coverage %.6f).\n' ...
    '- Fixed point: F=%d, alpha=%d, guard=%d; RGB565-reference gate %s.\n' ...
    '- Source calibration is copied under `calibration/`; generated binary/CSV/HEX files are under `lut/`.\n'],strjoin(cfg.camera_names(best4),', '),s4.coverage,s6.coverage,chosen.F,chosen.A,chosen.G,gate));
end
