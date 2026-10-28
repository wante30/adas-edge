function test_avm_fpga_sim()
%TEST_AVM_FPGA_SIM Fast self-checks for the frozen MATLAB contract.
cfg=avm_config(); cams=avm_geometry('load',cfg);
maps=avm_geometry('maps',cfg,cams,cfg.bev_width,cfg.bev_height,[cfg.x_min,cfg.x_max,cfg.y_min,cfg.y_max],false);
l=avm_lut('select',maps,1:numel(cams)); q=avm_lut('quantize',l,4,6,cfg);
bytes=avm_lut('pack',q); u=avm_lut('unpack',bytes,cfg.bev_height,cfg.bev_width,4,6);
assert(numel(bytes)==cfg.bev_width*cfg.bev_height*cfg.lut_entry_bytes);
assert(nnz(q.addr_a~=u.addr_a)==0 && nnz(q.addr_b~=u.addr_b)==0);
assert(nnz(q.fx_a~=u.fx_a)==0 && nnz(q.fy_a~=u.fy_a)==0);
assert(nnz(q.alpha_q~=u.alpha_q)==0 && nnz(q.valid~=u.valid)==0 && nnz(q.blend~=u.blend)==0);
assert(all(q.addr_a(q.valid)>=0) && all(q.addr_a(q.valid)<cfg.input_width*cfg.input_height));
assert(all(q.addr_b(q.blend)>=0) && all(q.addr_b(q.blend)<cfg.input_width*cfg.input_height));
addr_a=q.y_a*cfg.input_width+q.x_a; addr_b=q.y_b*cfg.input_width+q.x_b;
assert(nnz(addr_a(q.valid)~=q.addr_a(q.valid))==0 && nnz(addr_b(q.blend)~=q.addr_b(q.blend))==0);
rgb=uint8(randi([0,255],[cfg.input_height,cfg.input_width,3]));
roundtrip=avm_render('expand565',avm_render('pack565',rgb));
assert(isequal(size(roundtrip),size(rgb)) && isa(roundtrip,'uint8'));
fprintf('AVM self-check PASS: %d entries, coordinate/address and pack/unpack consistent.\n',numel(q.valid));
end
