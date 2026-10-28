function out = avm_geometry(action, varargin)
% A2D2 official tutorial uses CamMatrixOriginal for distorted input.
% https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/tutorial.ipynb
% Fisheye model: https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html
switch action
    case 'load', out = load_cameras(varargin{:});
    case 'maps', out = build_maps(varargin{:});
    case 'undistort', out = undistort_preview(varargin{:});
    case 'project', out = project_optical(varargin{:});
    otherwise, error('avm:action','Unknown geometry action %s',action);
end
end

function cams = load_cameras(cfg)
raw = jsondecode(fileread(fullfile(cfg.data_root,'probe','cams_lidars.json')));
% Unified right/forward/up -> dataset forward/left/up.
S = [0,1,0;-1,0,0;0,0,1];
for k = 1:numel(cfg.camera_names)
    n = cfg.camera_names{k}; c = raw.cameras.(n);
    ex = c.view.x_axis(:); ex = ex/norm(ex);
    ey = c.view.y_axis(:); ey = ey-ex*(ex'*ey); ey=ey/norm(ey);
    ez = cross(ex,ey); ez=ez/norm(ez);
    % View coordinates are forward/left/up; optical = right/down/forward.
    R = [-ey';-ez';ex'];
    cams(k).id = k-1; cams(k).name = n; cams(k).lens = c.Lens;
    cams(k).K_raw = c.CamMatrixOriginal; cams(k).K_undistorted = c.CamMatrix;
    cams(k).distortion = c.Distortion(:)';
    cams(k).origin_a2d2 = c.view.origin(:)';
    cams(k).origin = (S'*(c.view.origin(:)-cfg.a2d2_origin(:)))';
    cams(k).R_vehicle_to_optical = R*S;
    cams(k).R_a2d2_to_optical = R;
    cams(k).tstamp_delay_us = c.tstamp_delay;
    cams(k).resolution = c.Resolution(:)';
    assert(isequal(cams(k).resolution,[cfg.native_width,cfg.native_height]));
    cams(k).yaw_deg = atan2d(ex(2),ex(1)); % dataset forward/left/up ZYX
    cams(k).pitch_deg = atan2d(-ex(3),hypot(ex(1),ex(2)));
    cams(k).roll_deg = atan2d(ey(3),ez(3));
    sx=cfg.input_width/cfg.native_width; sy=cfg.input_height/cfg.native_height;
    A=[sx,0,(sx-1)/2;0,sy,(sy-1)/2;0,0,1];
    cams(k).K_work = A*cams(k).K_raw;
    assert(norm(cams(k).R_vehicle_to_optical*cams(k).R_vehicle_to_optical'-eye(3),'fro')<1e-10);
end
end

function maps = build_maps(cfg,cams,width,height,bounds,native)
if nargin<7, native=false; end
[u,v]=meshgrid(0:width-1,0:height-1);
X=bounds(1)+(u+.5)*(bounds(2)-bounds(1))/width;
Y=bounds(4)-(v+.5)*(bounds(4)-bounds(3))/height;
P=[X(:)';Y(:)';zeros(1,numel(X))];
body=abs(X)<=cfg.vehicle_half_width & abs(Y)<=cfg.vehicle_half_length;
maps.body=body; maps.X=X; maps.Y=Y; maps.bounds=bounds;
maps.u=zeros(height,width,numel(cams)); maps.v=maps.u;
maps.valid=false(height,width,numel(cams)); maps.score=maps.u;
if native, sw=cfg.native_width; sh=cfg.native_height; else, sw=cfg.input_width; sh=cfg.input_height; end
for k=1:numel(cams)
    Q=cams(k).R_vehicle_to_optical*(P-cams(k).origin(:));
    if native, K=cams(k).K_raw; else, K=cams(k).K_work; end
    uv=project_optical(Q,K,cams(k).distortion,cams(k).lens);
    mx=reshape(uv(1,:),height,width); my=reshape(uv(2,:),height,width);
    % Margin keeps nearest-coordinate rounding safe even at F=4.
    valid=reshape(Q(3,:)>0.05,height,width) & isfinite(mx) & isfinite(my) & ...
        mx>=0 & my>=0 & mx<sw-1-1/16 & my<sh-1-1/16 & ~body;
    mx(~valid)=0; my(~valid)=0;
    angular=reshape(max(0,Q(3,:)./max(sqrt(sum(Q.^2,1)),eps)),height,width);
    feather=min(bwdist(~valid)/cfg.feather_pixels,1);
    maps.u(:,:,k)=mx; maps.v(:,:,k)=my; maps.valid(:,:,k)=valid;
    maps.score(:,:,k)=feather.*angular.^2.*valid;
end
maps.roi=~body;
end

function uv = project_optical(Q,K,D,lens)
z=Q(3,:); z(abs(z)<1e-12)=1e-12;
x=Q(1,:)./z; y=Q(2,:)./z; r2=x.^2+y.^2;
if strcmp(lens,'Fisheye')
    r=sqrt(r2); theta=atan(r); t2=theta.^2;
    td=theta.*(1+D(1)*t2+D(2)*t2.^2+D(3)*t2.^3+D(4)*t2.^4);
    scale=ones(size(r)); nz=r>1e-12; scale(nz)=td(nz)./r(nz);
    xd=x.*scale; yd=y.*scale;
else
    k3=0; if numel(D)>=5, k3=D(5); end
    radial=1+D(1)*r2+D(2)*r2.^2+k3*r2.^3;
    xd=x.*radial+2*D(3)*x.*y+D(4)*(r2+2*x.^2);
    yd=y.*radial+D(3)*(r2+2*y.^2)+2*D(4)*x.*y;
end
uv=[K(1,1)*xd+K(1,2)*yd+K(1,3);K(2,2)*yd+K(2,3)];
end

function image = undistort_preview(c,raw)
[u,v]=meshgrid(0:size(raw,2)-1,0:size(raw,1)-1);
K=c.K_undistorted;
Q=[(u(:)'-K(1,3))/K(1,1);(v(:)'-K(2,3))/K(2,2);ones(1,numel(u))];
uv=project_optical(Q,c.K_raw,c.distortion,c.lens);
image=zeros(size(raw),'uint8');
for ch=1:3
    image(:,:,ch)=uint8(interp2(double(raw(:,:,ch)),reshape(uv(1,:),size(u))+1, ...
        reshape(uv(2,:),size(v))+1,'linear',0));
end
end
