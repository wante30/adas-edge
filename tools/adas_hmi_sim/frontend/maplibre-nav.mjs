import * as maplibregl from './vendor/maplibre-gl.mjs';

// The npm ESM build expects a sibling worker file. Keep the worker local so
// the simulator does not depend on a CDN and so the same bundle can move to
// the RK3566 image.
maplibregl.setWorkerUrl(new URL('./vendor/maplibre-gl-worker.mjs', import.meta.url).href);

const CENTER = [126.642, 45.752];
const ROUTE = [
  [126.631, 45.741], [126.635, 45.744], [126.639, 45.747],
  [126.643, 45.750], [126.647, 45.753], [126.651, 45.756], [126.655, 45.760],
];
const onlineMap = new URLSearchParams(location.search).has('online-map');
// MapLibre resolves relative tile URLs against the style URL. Because this
// demo supplies an inline style, resolve the local pack against the page URL
// explicitly so the same code works from the PC simulator and on RK3566.
const TILE_TEMPLATE = onlineMap
  ? 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
  : new URL('./tiles/{z}/{x}/{y}.png', location.href).href;
const STYLE = {
  version: 8,
  sources: { osm: { type: 'raster', tiles: [TILE_TEMPLATE], tileSize: 256, attribution: '© OpenStreetMap contributors' } },
  layers: [{ id: 'osm', type: 'raster', source: 'osm', paint: { 'raster-saturation': -0.7, 'raster-contrast': 0.12, 'raster-brightness-min': 0.08, 'raster-brightness-max': 0.72 } }],
};
function lineFeature(coordinates) { return { type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates } }; }
function pointFeature(coordinates) { return { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates } }; }
function interpolate(progress) { const scaled = Math.max(0, Math.min(0.999, progress)) * (ROUTE.length - 1); const index = Math.floor(scaled); const amount = scaled - index; const a = ROUTE[index]; const b = ROUTE[Math.min(ROUTE.length - 1, index + 1)]; return [a[0] + (b[0] - a[0]) * amount, a[1] + (b[1] - a[1]) * amount]; }

class MapLibreNavigationMap {
  constructor(host) {
    this.host = host; this.frame = null; this.loaded = false; this.tileSeen = false; this.fallbackUsed = false;
    this.map = new maplibregl.Map({ container: host, style: STYLE, center: CENTER, zoom: 13.8, pitch: 48, bearing: 18, attributionControl: true, cooperativeGestures: true });
    this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    this.map.on('load', () => { this.loaded = true; this.map.addSource('route', { type: 'geojson', data: lineFeature(ROUTE) }); this.map.addLayer({ id: 'route-glow', type: 'line', source: 'route', paint: { 'line-color': '#32bdfd', 'line-width': 15, 'line-opacity': 0.22, 'line-blur': 1.3 } }); this.map.addLayer({ id: 'route', type: 'line', source: 'route', paint: { 'line-color': '#55d7ff', 'line-width': 5.5, 'line-opacity': 0.98 } }); this.map.addSource('vehicle', { type: 'geojson', data: pointFeature(ROUTE[0]) }); this.map.addLayer({ id: 'vehicle-halo', type: 'circle', source: 'vehicle', paint: { 'circle-radius': 18, 'circle-color': '#35d4ff', 'circle-opacity': 0.24 } }); this.map.addLayer({ id: 'vehicle', type: 'circle', source: 'vehicle', paint: { 'circle-radius': 8, 'circle-color': '#5ee4ff', 'circle-stroke-color': '#05212b', 'circle-stroke-width': 3 } }); this.map.resize(); this.update(this.frame); });
    this.map.on('sourcedata', (event) => { if (event.sourceId === 'osm' && (event.isSourceLoaded || event.sourceDataType === 'content')) this.tileSeen = true; });
    this.map.on('data', (event) => { if (event.sourceId === 'osm' && event.dataType === 'source') this.tileSeen = true; });
    this.map.on('error', (event) => { this.host.dataset.mapError = event.error?.message || 'tile error'; });
    // Keep the PC demo usable when the target has no Internet or the public
    // OSM raster endpoint is unavailable. The same HMI still runs with the
    // local vector fallback, while MapLibre remains the production adapter.
    setTimeout(() => { if (!this.loaded || !this.tileSeen || !this.host.querySelector('.maplibregl-canvas')) this.useFallback(); }, 1800);
  }
  useFallback() {
    if (this.fallbackUsed || !globalThis.NavigationMapRenderer) return;
    this.fallbackUsed = true; this.map.remove(); this.host.innerHTML = '';
    const Renderer = globalThis.OfflineRasterNavigationMap || globalThis.NavigationMapRenderer;
    this.host.dataset.mapMode = Renderer === globalThis.OfflineRasterNavigationMap ? 'offline-osm-raster' : 'local-vector-fallback';
    this.fallback = new Renderer(this.host); this.fallback.update(this.frame);
  }
  resize() { if (this.fallback) this.fallback.resize(); else this.map.resize(); }
  update(frame) { this.frame = frame; if (this.fallback) { this.fallback.update(frame); return; } if (!this.loaded) return; const progress = ((Number(frame?.frame_id) || 0) % 600) / 600; const source = this.map.getSource('vehicle'); if (source) source.setData(pointFeature(interpolate(progress))); }
}

globalThis.MapLibreNavigationMap = MapLibreNavigationMap;
