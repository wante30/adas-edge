function out = avm_lut(action,varargin)
switch action
    case 'select', out=select_sources(varargin{:});
    case 'quantize', out=quantize(varargin{:});
    case 'pack', out=pack(varargin{:});
    case 'unpack', out=unpack(varargin{:});
    otherwise, error('avm:action','Unknown LUT action');
end
end

function l = select_sources(maps,ids)
score=maps.score(:,:,ids); [h,w,~]=size(score); pix=reshape(1:h*w,h,w);
if numel(ids)==1
    sorted=cat(3,score,zeros(h,w));
    a=ids*ones(h,w); b=a;
else
    [sorted,order]=sort(score,3,'descend');
    a=ids(order(:,:,1)); b=ids(order(:,:,2));
end
ia=pix+(a-1)*h*w; ib=pix+(b-1)*h*w;
l.valid=sorted(:,:,1)>0; l.blend=sorted(:,:,2)>0;
l.cam_a=uint8(a-1); l.cam_b=uint8(b-1);
l.u_a=maps.u(ia); l.v_a=maps.v(ia); l.u_b=maps.u(ib); l.v_b=maps.v(ib);
l.alpha=zeros(h,w); l.alpha(l.valid)=1;
top=sorted(:,:,1); second=sorted(:,:,2);
l.alpha(l.blend)=top(l.blend)./max(top(l.blend)+second(l.blend),eps);
l.alpha(~l.blend & l.valid)=1;
l.cam_a(~l.valid)=7; l.cam_b(~l.blend)=7;
l.u_a(~l.valid)=0; l.v_a(~l.valid)=0;
l.u_b(~l.blend)=0; l.v_b(~l.blend)=0;
l.selected_camera_ids=ids-1;
end

function q = quantize(l,F,A,cfg)
q=l; q.F=F; q.A=A;
ua=floor(l.u_a*2^F+.5); va=floor(l.v_a*2^F+.5);
ub=floor(l.u_b*2^F+.5); vb=floor(l.v_b*2^F+.5);
q.x_a=floor(ua/2^F); q.y_a=floor(va/2^F);
q.x_b=floor(ub/2^F); q.y_b=floor(vb/2^F);
q.fx_a=mod(ua,2^F); q.fy_a=mod(va,2^F);
q.fx_b=mod(ub,2^F); q.fy_b=mod(vb,2^F);
q.addr_a=q.y_a*cfg.input_width+q.x_a;
q.addr_b=q.y_b*cfg.input_width+q.x_b;
q.alpha_q=floor(l.alpha*2^A+.5);
assert(all(q.alpha_q(:)>=0 & q.alpha_q(:)<=2^A));
assert(all(q.x_a(l.valid)<cfg.input_width-1 & q.y_a(l.valid)<cfg.input_height-1));
assert(all(q.x_b(l.blend)<cfg.input_width-1 & q.y_b(l.blend)<cfg.input_height-1));
assert(all(isfinite(q.addr_a(:))) && all(isfinite(q.addr_b(:))));
end

function bytes = pack(q)
% 96-bit entry, 12 bytes LE, row-major. Bit offsets are frozen in spec.
fields={q.addr_a,q.addr_b,q.fx_a,q.fy_a,q.fx_b,q.fy_b,q.alpha_q, ...
    q.cam_a,q.cam_b,q.valid,q.blend};
offset=[0,18,36,44,52,60,68,79,82,85,86];
width=[18,18,8,8,8,8,11,3,3,1,1];
n=numel(q.valid); words=zeros(n,3,'uint32');
for f=1:numel(fields)
    a=uint32(reshape(fields{f}',[],1));
    assert(all(double(a)<2^width(f)));
    for b=0:width(f)-1
        target=offset(f)+b; wi=floor(target/32)+1; bi=mod(target,32);
        words(:,wi)=bitor(words(:,wi),bitshift(bitand(bitshift(a,-b),uint32(1)),bi));
    end
end
bytes=zeros(n,12,'uint8');
for k=1:12
    wi=floor((k-1)/4)+1; bi=8*mod(k-1,4);
    bytes(:,k)=uint8(bitand(bitshift(words(:,wi),-bi),uint32(255)));
end
end

function q = unpack(bytes,h,w,F,A)
names={'addr_a','addr_b','fx_a','fy_a','fx_b','fy_b','alpha_q','cam_a','cam_b','valid','blend'};
offset=[0,18,36,44,52,60,68,79,82,85,86]; width=[18,18,8,8,8,8,11,3,3,1,1];
for k=1:numel(names)
    value=zeros(size(bytes,1),1,'uint32');
    for b=0:width(k)-1
        p=offset(k)+b; by=floor(p/8)+1; bi=mod(p,8);
        value=bitor(value,bitshift(uint32(bitand(bitshift(bytes(:,by),-bi),uint8(1))),b));
    end
    q.(names{k})=reshape(double(value),w,h)';
end
q.valid=logical(q.valid); q.blend=logical(q.blend);
q.F=F; q.A=A;
end
