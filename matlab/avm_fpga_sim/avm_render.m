function out = avm_render(action,varargin)
switch action
    case 'float', out=float_render(varargin{:});
    case 'fixed', out=fixed_render(varargin{:});
    case 'pack565', out=pack565(varargin{:});
    case 'expand565', out=expand565(varargin{:});
    case 'metrics', out=metrics(varargin{:});
    otherwise, error('avm:action','Unknown render action');
end
end

function image=float_render(images,l)
image=zeros([size(l.valid),3]);
for side=1:2
    if side==1
        cams=l.cam_a; u=l.u_a; v=l.v_a; valid=l.valid; weight=l.alpha;
    else
        cams=l.cam_b; u=l.u_b; v=l.v_b; valid=l.blend; weight=1-l.alpha;
    end
    for k=1:numel(images)
        mask=valid & cams==k-1;
        for ch=1:3
            value=interp2(double(images{k}(:,:,ch)),u+1,v+1,'linear',0);
            plane=image(:,:,ch); plane(mask)=plane(mask)+value(mask).*weight(mask);
            image(:,:,ch)=plane;
        end
    end
end
end

function out=fixed_render(images,q,mode,guard)
% All arithmetic operands represent unsigned integers exactly. Double host
% arrays preserve integers below 2^53; asserted accumulators model RTL widths.
F=q.F; A=q.A; outputs=cell(1,2); B=8;
if strcmp(mode,'direct565'), channel_bits=[5,6,5]; else, channel_bits=[8,8,8]; end
for side=1:2
    if side==1
        cam=q.cam_a; addr=q.addr_a; fx=q.fx_a; fy=q.fy_a; valid=q.valid;
    else
        cam=q.cam_b; addr=q.addr_b; fx=q.fx_b; fy=q.fy_b; valid=q.blend;
    end
    values=zeros([size(valid),3]);
    weights=cat(3,(2^F-fx).*(2^F-fy),fx.*(2^F-fy), ...
        (2^F-fx).*fy,fx.*fy);
    for k=1:numel(images)
        mask=valid & cam==k-1;
        if ~any(mask(:)), continue; end
        input=images{k};
        if ~strcmp(mode,'rgb888')
            words=pack565(input);
            if strcmp(mode,'direct565')
                % Keep the native 5/6/5 code values for a direct packed
                % implementation.  The RGB565 path below deliberately
                % expands to 8 bits before interpolation.
                input=cat(3,double(bitshift(words,-11)),double(bitand(bitshift(words,-5),63)),double(bitand(words,31)));
            else
                input=expand565(words);
            end
        end
        stride=size(input,2); offsets=[0,1,stride,stride+1];
        for ch=1:3
            row=reshape(double(input(:,:,ch))',[],1);
            accumulator=zeros(nnz(mask),1);
            for n=1:4
                wp=weights(:,:,n);
                accumulator=accumulator+row(addr(mask)+offsets(n)+1).*wp(mask);
            end
            % Need B+2F bits for an unsigned weighted sum. No saturation in
            % the middle; reject rather than hide an overflow.
            B=channel_bits(ch);
            assert(all(accumulator>=0 & accumulator<2^(B+2*F)));
            shift=2*F-guard; assert(shift>=0);
            rounded=floor((accumulator+2^(shift-1))/2^shift);
            rounded=min(rounded,(2^B-1)*2^guard);
            plane=values(:,:,ch); plane(mask)=rounded; values(:,:,ch)=plane;
        end
    end
    outputs{side}=values;
end
channels=zeros([size(q.valid),3]);
for ch=1:3
    B=channel_bits(ch);
    acc=q.alpha_q.*outputs{1}(:,:,ch)+(2^A-q.alpha_q).*outputs{2}(:,:,ch);
    assert(all(acc(:)>=0 & acc(:)<2^(B+guard+A)));
    shift=A+guard;
    channels(:,:,ch)=min(2^B-1,floor((acc+2^(shift-1))/2^shift));
end
if strcmp(mode,'direct565')
    words=bitor(bitshift(uint16(channels(:,:,1)),11),bitor(bitshift(uint16(channels(:,:,2)),5),uint16(channels(:,:,3))));
    out.words=words; out.rgb=expand565(words);
else
    out.rgb=uint8(channels); out.words=pack565(out.rgb);
end
out.rgb(~repmat(q.valid,1,1,3))=0;
out.words(~q.valid)=0;
out.accumulator_widths=channel_bits+2*F;
end

function words=pack565(image)
image=uint16(image);
words=bitor(bitshift(bitshift(image(:,:,1),-3),11), ...
    bitor(bitshift(bitshift(image(:,:,2),-2),5),bitshift(image(:,:,3),-3)));
end

function rgb=expand565(words)
r=bitshift(words,-11); g=bitand(bitshift(words,-5),63); b=bitand(words,31);
rgb=uint8(cat(3,bitor(bitshift(r,3),bitshift(r,-2)), ...
    bitor(bitshift(g,2),bitshift(g,-4)),bitor(bitshift(b,3),bitshift(b,-2))));
end

function m=metrics(candidate,reference,mask)
d=double(candidate)-double(reference); selected=d(repmat(mask,1,1,3));
m.mae=mean(abs(selected)); mse=mean(selected.^2);
if mse==0, m.psnr=Inf; else, m.psnr=10*log10(255^2/mse); end
% SSIM is over the complete output; MAE/PSNR use valid non-body pixels.
m.ssim=ssim(uint8(candidate),uint8(reference));
m.max_abs=max(abs(selected));
end
