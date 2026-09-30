(function () {
const $ = (id) => document.getElementById(id);
const dom = { host: $('scene-host'), connection: $('connection'), mode: $('mode-pill'), video: $('video-feed'), videoEmpty: $('video-empty'), videoSize: $('video-size'), frame: $('frame-id'), traffic: $('traffic-state'), lamp: $('traffic-lamp'), action: $('action-state'), warningStrip: $('warning-strip'), warning: $('warning-text'), navArrow: $('nav-arrow'), navAction: $('nav-action'), navDistance: $('nav-distance'), videoFps: $('video-fps'), detectorFps: $('detector-fps'), wsRate: $('ws-rate'), tracks: $('track-count'), sceneTracks: $('scene-track-count'), debug: $('debug-panel'), debugText: $('debug-text') };
let renderer; let lastFrame; let debugVisible = false; let uiFrames = 0; let uiStarted = performance.now();

function setText(node, value) { node.textContent = value; }
function navArrow(action) { return { STRAIGHT: '↑', TURN_LEFT: '↰', TURN_RIGHT: '↱', CHANGE_LEFT: '←', CHANGE_RIGHT: '→', ARRIVE: '⌂' }[action] || '↑'; }
function updateStatus(frame) {
  const traffic = frame.traffic_light?.state || 'UNKNOWN'; const warnings = frame.warnings || []; const metrics = frame.metrics || {};
  setText(dom.mode, String(frame.source || 'mock').toUpperCase()); setText(dom.frame, `FRAME ${String(frame.frame_id).padStart(5, '0')}`); setText(dom.traffic, traffic); setText(dom.action, frame.ego?.action || 'STOP');
  dom.lamp.dataset.state = traffic; dom.warningStrip.classList.toggle('alert', warnings.length > 0 || frame.ego?.action === 'STOP'); setText(dom.warning, warnings.length ? warnings.join(' · ') : 'ROAD CLEAR / MONITORING');
  const nav = frame.navigation || {}; setText(dom.navArrow, navArrow(nav.action)); setText(dom.navAction, String(nav.action || 'STRAIGHT').replaceAll('_', ' ')); setText(dom.navDistance, nav.distance_m == null ? '—' : `${Math.round(nav.distance_m)} m`);
  setText(dom.videoFps, `${Number(metrics.video_fps || 0).toFixed(1)} Hz`); setText(dom.detectorFps, `${Number(metrics.detector_fps || 0).toFixed(1)} Hz`); setText(dom.wsRate, `${Number(metrics.backend_ws_rate || 0).toFixed(1)} Hz`); setText(dom.tracks, `${frame.objects?.length || 0} / ${metrics.track_count || 0}`); if (dom.sceneTracks) setText(dom.sceneTracks, String(metrics.track_count || frame.objects?.length || 0).padStart(2, '0'));
  if (debugVisible) { dom.debugText.textContent = JSON.stringify({ frame_id: frame.frame_id, objects: frame.objects, lanes: frame.lanes, traffic_light: frame.traffic_light, navigation: frame.navigation, warnings: frame.warnings, metrics: frame.metrics }, null, 2); }
}
function receive(message) { if (message.type !== 'scene') return; lastFrame = message.scene; renderer.update(lastFrame); updateStatus(lastFrame); uiFrames += 1; const elapsed = performance.now() - uiStarted; if (elapsed > 1000) { document.body.dataset.uiFps = (uiFrames * 1000 / elapsed).toFixed(0); uiFrames = 0; uiStarted = performance.now(); } if (message.video_jpeg) { dom.video.src = `data:image/jpeg;base64,${message.video_jpeg}`; dom.videoEmpty.hidden = true; dom.videoSize.textContent = '960 × 540'; } }
function connect() { const protocol = location.protocol === 'https:' ? 'wss' : 'ws'; const socket = new WebSocket(`${protocol}://${location.host}/ws`); socket.onopen = () => setText(dom.connection, 'LIVE'); socket.onclose = () => { setText(dom.connection, 'RECONNECTING'); setTimeout(connect, 1000); }; socket.onerror = () => setText(dom.connection, 'RETRYING'); socket.onmessage = (event) => { try { receive(JSON.parse(event.data)); } catch (error) { console.error(error); } }; }
dom.debug.addEventListener('click', () => { debugVisible = false; dom.debug.classList.remove('visible'); }); document.addEventListener('keydown', (event) => { if (event.key.toLowerCase() === 'd') { debugVisible = !debugVisible; dom.debug.classList.toggle('visible', debugVisible); if (lastFrame) updateStatus(lastFrame); } });
try { renderer = new SceneRenderer(dom.host); } catch (error) { dom.host.innerHTML = `<div class="scene-error">WebGL renderer unavailable<br><small>${error.message}</small></div>`; console.error(error); renderer = { update: () => {} }; }
connect();
})();
