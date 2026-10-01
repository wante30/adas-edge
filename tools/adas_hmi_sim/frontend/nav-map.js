(function () {
  const COLORS = {
    sky: '#10232b',
    water: '#102c36',
    block: '#162b31',
    blockEdge: '#24434a',
    minor: '#547077',
    major: '#9bb1ad',
    majorEdge: '#233a40',
    route: '#47c9ff',
    routeGlow: '#47c9ff55',
    text: '#dcece8',
    muted: '#7d9b9f',
    amber: '#ffc875',
  };

  // A small local vector tile substitute for the PC demo.  The renderer is
  // deliberately data-shaped: the same draw calls can consume MapLibre/OSM
  // line features later without changing the HMI layout.
  const ROADS = [
    { kind: 'major', name: '中央大道', points: [[0.05, 0.88], [0.2, 0.75], [0.38, 0.66], [0.56, 0.53], [0.73, 0.38], [0.94, 0.24]] },
    { kind: 'major', name: '环城路', points: [[0.1, 0.14], [0.26, 0.2], [0.41, 0.27], [0.58, 0.3], [0.79, 0.24], [0.94, 0.1]] },
    { kind: 'major', name: '滨河路', points: [[0.08, 0.96], [0.19, 0.83], [0.3, 0.68], [0.38, 0.5], [0.42, 0.31], [0.45, 0.05]] },
    { kind: 'minor', name: '青年路', points: [[0.08, 0.42], [0.25, 0.43], [0.43, 0.44], [0.62, 0.43], [0.88, 0.46]] },
    { kind: 'minor', name: '和平街', points: [[0.16, 0.1], [0.2, 0.29], [0.22, 0.5], [0.25, 0.73], [0.29, 0.97]] },
    { kind: 'minor', name: '学府路', points: [[0.57, 0.06], [0.58, 0.23], [0.59, 0.42], [0.6, 0.64], [0.62, 0.94]] },
    { kind: 'minor', name: '长江街', points: [[0.74, 0.04], [0.7, 0.2], [0.69, 0.41], [0.73, 0.62], [0.82, 0.9]] },
    { kind: 'minor', name: '松花江路', points: [[0.04, 0.59], [0.23, 0.59], [0.43, 0.57], [0.66, 0.59], [0.96, 0.66]] },
  ];
  const ROUTE = [[0.5, 0.98], [0.5, 0.84], [0.49, 0.71], [0.52, 0.59], [0.62, 0.49], [0.73, 0.38], [0.78, 0.25], [0.78, 0.09]];

  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function smooth(value) { return value * value * (3 - 2 * value); }

  class NavigationMapRenderer {
    constructor(host) {
      this.host = host;
      this.canvas = host.tagName === 'CANVAS' ? host : document.createElement('canvas');
      if (this.canvas !== host) { this.canvas.className = 'nav-map-fallback'; host.appendChild(this.canvas); }
      this.context = this.canvas.getContext('2d');
      this.frame = null;
      this.resize = this.resize.bind(this);
      window.addEventListener('resize', this.resize);
      this.resize();
    }

    resize() {
      const box = this.host.getBoundingClientRect();
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      this.canvas.width = Math.max(1, Math.floor(box.width * ratio));
      this.canvas.height = Math.max(1, Math.floor(box.height * ratio));
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.width = box.width;
      this.height = box.height;
      this.draw();
    }

    update(frame) {
      this.frame = frame;
      this.draw();
    }

    point(point) { return [point[0] * this.width, point[1] * this.height]; }

    path(points) {
      const ctx = this.context;
      points.forEach((point, index) => {
        const [x, y] = this.point(point);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
    }

    drawRoad(points, kind) {
      const ctx = this.context;
      ctx.save();
      ctx.beginPath(); this.path(points); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.strokeStyle = kind === 'major' ? COLORS.majorEdge : '#10252b';
      ctx.lineWidth = kind === 'major' ? 28 : 16; ctx.stroke();
      ctx.beginPath(); this.path(points);
      ctx.strokeStyle = kind === 'major' ? COLORS.major : COLORS.minor;
      ctx.lineWidth = kind === 'major' ? 20 : 9; ctx.stroke();
      if (kind === 'major') { ctx.beginPath(); this.path(points); ctx.strokeStyle = '#d7e2de55'; ctx.lineWidth = 1; ctx.setLineDash([8, 12]); ctx.stroke(); }
      ctx.restore();
    }

    drawRoute(progress) {
      const ctx = this.context;
      const route = ROUTE.map((point) => this.point(point));
      ctx.save(); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath(); route.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.strokeStyle = COLORS.routeGlow; ctx.lineWidth = 18; ctx.stroke();
      ctx.beginPath(); route.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.strokeStyle = COLORS.route; ctx.lineWidth = 6; ctx.stroke();
      ctx.beginPath(); route.slice(0, Math.max(2, Math.ceil(progress * route.length))).forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.strokeStyle = '#b9efff'; ctx.lineWidth = 2; ctx.stroke();
      ctx.restore();
    }

    marker(progress) {
      const scaled = clamp(progress, 0, 0.999) * (ROUTE.length - 1);
      const index = Math.floor(scaled); const amount = smooth(scaled - index);
      const a = ROUTE[index]; const b = ROUTE[Math.min(ROUTE.length - 1, index + 1)];
      return [a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount];
    }

    drawCar(progress) {
      const ctx = this.context; const [nx, ny] = this.marker(progress); const [x, y] = this.point([nx, ny]);
      const next = this.point(this.marker(Math.min(0.999, progress + 0.015))); const angle = Math.atan2(next[1] - y, next[0] - x);
      ctx.save(); ctx.translate(x, y); ctx.rotate(angle + Math.PI / 2);
      ctx.shadowColor = '#3ad7ff99'; ctx.shadowBlur = 20; ctx.fillStyle = '#55d7f2';
      ctx.beginPath(); ctx.roundRect(-13, -25, 26, 50, 8); ctx.fill();
      ctx.shadowBlur = 0; ctx.fillStyle = '#12303a'; ctx.beginPath(); ctx.roundRect(-8, -12, 16, 20, 4); ctx.fill();
      ctx.fillStyle = '#e5fbff'; ctx.fillRect(-8, -22, 5, 3); ctx.fillRect(3, -22, 5, 3);
      ctx.restore();
    }

    label(text, x, y, color = COLORS.text, size = 11) {
      const ctx = this.context; ctx.save(); ctx.font = `600 ${size}px "Segoe UI", sans-serif`; ctx.fillStyle = color; ctx.shadowColor = '#081217'; ctx.shadowBlur = 5; ctx.fillText(text, x, y); ctx.restore();
    }

    draw() {
      if (!this.width || !this.height) return;
      const ctx = this.context; const frame = this.frame || {}; const nav = frame.navigation || {};
      ctx.clearRect(0, 0, this.width, this.height);
      const gradient = ctx.createLinearGradient(0, 0, 0, this.height); gradient.addColorStop(0, '#10242b'); gradient.addColorStop(1, '#0b1a21'); ctx.fillStyle = gradient; ctx.fillRect(0, 0, this.width, this.height);
      ctx.fillStyle = COLORS.water; ctx.beginPath(); ctx.moveTo(0, this.height * .76); ctx.bezierCurveTo(this.width * .16, this.height * .68, this.width * .2, this.height * .88, this.width * .42, this.height); ctx.lineTo(0, this.height); ctx.fill();
      for (let x = 0.03; x < 1; x += 0.14) for (let y = 0.04; y < 1; y += 0.16) { const [px, py] = this.point([x, y]); ctx.fillStyle = COLORS.block; ctx.fillRect(px, py, this.width * .095, this.height * .095); ctx.strokeStyle = COLORS.blockEdge; ctx.lineWidth = 1; ctx.strokeRect(px, py, this.width * .095, this.height * .095); }
      ROADS.forEach((road) => this.drawRoad(road.points, road.kind));
      const progress = ((Number(frame.frame_id) || 0) % 600) / 600;
      this.drawRoute(progress); this.drawCar(progress);
      const action = String(nav.action || 'STRAIGHT').replaceAll('_', ' ');
      this.label('OFFLINE VECTOR MAP  /  DEMO ROUTE', 22, 28, COLORS.muted, 10);
      this.label('中央城区 · 导航演示', 22, 52, COLORS.text, 14);
      this.label('中央大道', this.width * .63, this.height * .34, COLORS.text, 11);
      this.label('青年路', this.width * .15, this.height * .41, COLORS.muted, 10);
      this.label('松花江路', this.width * .68, this.height * .63, COLORS.muted, 10);
      ctx.save(); ctx.fillStyle = '#071218dd'; ctx.strokeStyle = '#5ebdca66'; ctx.lineWidth = 1; ctx.beginPath(); ctx.roundRect(18, this.height - 70, 210, 46, 8); ctx.fill(); ctx.stroke();
      this.label('下一动作', 31, this.height - 49, COLORS.muted, 9); this.label(`${action}  ·  ${nav.distance_m == null ? '—' : `${Math.round(nav.distance_m)} m`}`, 31, this.height - 31, COLORS.amber, 16); ctx.restore();
      ctx.save(); ctx.fillStyle = '#071218dd'; ctx.strokeStyle = '#5ebdca66'; ctx.lineWidth = 1; ctx.beginPath(); ctx.roundRect(this.width - 115, 18, 93, 38, 8); ctx.fill(); ctx.stroke(); this.label('跟随车辆', this.width - 98, 42, COLORS.route, 10); ctx.restore();
    }
  }

  // A real offline OSM raster pack for the PC/RK demo. This is the robust
  // fallback for devices where WebGL workers or MapLibre's raster pipeline
  // are unavailable. It keeps the map imagery real while leaving route and
  // vehicle overlays in the HMI's control layer.
  class OfflineRasterNavigationMap {
    constructor(host) {
      this.host = host;
      this.canvas = document.createElement('canvas');
      this.canvas.className = 'nav-map-fallback';
      this.host.innerHTML = '';
      this.host.appendChild(this.canvas);
      this.context = this.canvas.getContext('2d');
      this.frame = null;
      this.zoom = 14;
      this.center = [126.642, 45.752];
      this.tileX = Math.floor((this.center[0] + 180) / 360 * (2 ** this.zoom));
      this.tileY = Math.floor((1 - Math.asinh(Math.tan(this.center[1] * Math.PI / 180)) / Math.PI) / 2 * (2 ** this.zoom));
      this.tiles = new Map();
      this.resize = this.resize.bind(this);
      window.addEventListener('resize', this.resize);
      this.resize();
      this.loadTiles();
    }

    resize() {
      const box = this.host.getBoundingClientRect();
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      this.canvas.width = Math.max(1, Math.floor(box.width * ratio));
      this.canvas.height = Math.max(1, Math.floor(box.height * ratio));
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.width = box.width;
      this.height = box.height;
      this.draw();
    }

    loadTiles() {
      for (let dx = -2; dx <= 2; dx += 1) for (let dy = -2; dy <= 2; dy += 1) {
        const x = this.tileX + dx;
        const y = this.tileY + dy;
        const image = document.createElement('img');
        image.decoding = 'async';
        image.onload = () => { this.tiles.set(`${x}/${y}`, image); this.host.dataset.mapMode = 'offline-osm-raster'; this.draw(); };
        image.onerror = () => this.draw();
        image.src = `./tiles/${this.zoom}/${x}/${y}.png`;
      }
    }

    update(frame) { this.frame = frame; this.draw(); }

    project(lng, lat) {
      const scale = 2 ** this.zoom;
      const x = (lng + 180) / 360 * scale * 256;
      const sin = Math.sin(lat * Math.PI / 180);
      const y = (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * scale * 256;
      const cx = (this.tileX + 0.5) * 256;
      const cy = (this.tileY + 0.5) * 256;
      return [x - cx + this.width / 2, y - cy + this.height / 2];
    }

    drawRoute(progress) {
      const ctx = this.context;
      const points = ROUTE_GEO.map(([lng, lat]) => this.project(lng, lat));
      ctx.save(); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath(); points.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.strokeStyle = '#28b9ff88'; ctx.lineWidth = 22; ctx.stroke();
      ctx.beginPath(); points.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.strokeStyle = '#1d9df0'; ctx.lineWidth = 8; ctx.stroke();
      const end = Math.max(2, Math.ceil(progress * points.length));
      ctx.beginPath(); points.slice(0, end).forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.strokeStyle = '#e5fbff'; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();
    }

    drawCar(progress) {
      const scaled = clamp(progress, 0, 0.999) * (ROUTE_GEO.length - 1);
      const index = Math.floor(scaled); const amount = smooth(scaled - index);
      const a = ROUTE_GEO[index]; const b = ROUTE_GEO[Math.min(ROUTE_GEO.length - 1, index + 1)];
      const [x, y] = this.project(a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount);
      const [nx, ny] = this.project(b[0], b[1]); const angle = Math.atan2(ny - y, nx - x);
      const ctx = this.context; ctx.save(); ctx.translate(x, y); ctx.rotate(angle + Math.PI / 2);
      ctx.shadowColor = '#19bfff'; ctx.shadowBlur = 24; ctx.fillStyle = '#45d4ff';
      ctx.beginPath(); ctx.roundRect(-14, -27, 28, 54, 8); ctx.fill();
      ctx.shadowBlur = 0; ctx.fillStyle = '#15333c'; ctx.beginPath(); ctx.roundRect(-9, -13, 18, 21, 4); ctx.fill();
      ctx.fillStyle = '#e5fbff'; ctx.fillRect(-9, -23, 6, 3); ctx.fillRect(3, -23, 6, 3); ctx.restore();
    }

    draw() {
      if (!this.width || !this.height) return;
      const ctx = this.context; const frame = this.frame || {}; const nav = frame.navigation || {};
      ctx.clearRect(0, 0, this.width, this.height); ctx.fillStyle = '#d8e0d8'; ctx.fillRect(0, 0, this.width, this.height);
      const centerWorldX = (this.tileX + 0.5) * 256; const centerWorldY = (this.tileY + 0.5) * 256;
      for (const [key, image] of this.tiles) { const [x, y] = key.split('/').map(Number); ctx.drawImage(image, x * 256 - centerWorldX + this.width / 2, y * 256 - centerWorldY + this.height / 2, 256, 256); }
      ctx.fillStyle = '#07151bb8'; ctx.fillRect(0, 0, this.width, this.height);
      const gradient = ctx.createLinearGradient(0, 0, 0, this.height); gradient.addColorStop(0, '#07151b26'); gradient.addColorStop(1, '#07151b88'); ctx.fillStyle = gradient; ctx.fillRect(0, 0, this.width, this.height);
      const progress = ((Number(frame.frame_id) || 0) % 600) / 600;
      this.drawRoute(progress); this.drawCar(progress);
      ctx.save(); ctx.font = '600 10px "Segoe UI", sans-serif'; ctx.fillStyle = '#e6faf6'; ctx.shadowColor = '#061015'; ctx.shadowBlur = 5; ctx.fillText('OFFLINE OSM RASTER  /  DEMO ROUTE', 22, 28); ctx.font = '600 14px "Segoe UI", sans-serif'; ctx.fillText('哈尔滨 · 导航演示', 22, 52); ctx.restore();
      ctx.save(); ctx.fillStyle = '#071218dd'; ctx.strokeStyle = '#5ebdca99'; ctx.lineWidth = 1; ctx.beginPath(); ctx.roundRect(18, this.height - 70, 220, 46, 8); ctx.fill(); ctx.stroke(); ctx.font = '600 9px "Segoe UI", sans-serif'; ctx.fillStyle = '#91b2b4'; ctx.fillText('下一动作', 31, this.height - 49); ctx.font = '700 16px "Segoe UI", sans-serif'; ctx.fillStyle = '#ffc875'; ctx.fillText(`${String(nav.action || 'STRAIGHT').replaceAll('_', ' ')}  ·  ${nav.distance_m == null ? '—' : `${Math.round(nav.distance_m)} m`}`, 31, this.height - 31); ctx.restore();
      ctx.save(); ctx.font = '10px "Segoe UI", sans-serif'; ctx.fillStyle = '#e7eee9cc'; ctx.shadowColor = '#071218'; ctx.shadowBlur = 3; ctx.fillText('© OpenStreetMap contributors', this.width - 166, this.height - 14); ctx.restore();
    }
  }

  const ROUTE_GEO = [[126.631, 45.741], [126.635, 45.744], [126.639, 45.747], [126.643, 45.750], [126.647, 45.753], [126.651, 45.756], [126.655, 45.760]];

  globalThis.NavigationMapRenderer = NavigationMapRenderer;
  globalThis.OfflineRasterNavigationMap = OfflineRasterNavigationMap;
})();
