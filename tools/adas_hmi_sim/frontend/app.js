(function () {
  const $ = (id) => document.getElementById(id);
  const dom = {
    host: $('scene-host'), navMap: $('nav-map'), connection: $('connection'), mode: $('mode-pill'),
    video: $('video-feed'), videoLeft: $('video-left'), videoRight: $('video-right'), videoRear: $('video-rear'),
    surround: $('surround-strip'), surroundToggle: $('surround-toggle'), videoEmpty: $('video-empty'), videoSize: $('video-size'),
    bevCard: $('bev-card'), bev: $('bev-feed'), sceneSurround: $('scene-surround-feed'), bevCalibration: $('bev-calibration'),
    frame: $('frame-id'), traffic: $('traffic-state'), lamp: $('traffic-lamp'), action: $('action-state'),
    warningStrip: $('warning-strip'), warning: $('warning-text'), navArrow: $('nav-arrow'), navAction: $('nav-action'), navDistance: $('nav-distance'),
    videoFps: $('video-fps'), detectorFps: $('detector-fps'), wsRate: $('ws-rate'), tracks: $('track-count'), sceneTracks: $('scene-track-count'), renderFps: $('render-fps'),
    sourceLabel: $('source-label'), worldLabel: $('world-label'), viewButtons: [...document.querySelectorAll('.view-button')], debug: $('debug-panel'), debugText: $('debug-text'),
  };
  let renderer;
  let mapRenderer;
  let lastFrame;
  let debugVisible = false;
  let uiFrames = 0;
  let uiStarted = performance.now();

  function setText(node, value) { node.textContent = value; }
  function navArrow(action) { return { STRAIGHT: '↑', TURN_LEFT: '↰', TURN_RIGHT: '↱', CHANGE_LEFT: '←', CHANGE_RIGHT: '→', ARRIVE: '⌂' }[action] || '↑'; }

  function activateView(mode) {
    dom.viewButtons.forEach((button) => {
      const active = button.dataset.view === mode;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    renderer.setView(mode);
    if (dom.navMap) dom.navMap.hidden = mode !== 'nav';
    if (dom.sceneSurround) dom.sceneSurround.hidden = mode !== 'bowl' || !dom.sceneSurround.src;
    const labels = { nav: '导航地图', drive: '驾驶感知', bowl: '环视 AVM' };
    setText(dom.worldLabel, labels[mode] || '导航地图');
    document.body.dataset.viewMode = mode;
    if (mapRenderer && mode === 'nav') mapRenderer.resize();
  }

  function updateStatus(frame) {
    const traffic = frame.traffic_light?.state || 'UNKNOWN';
    const warnings = frame.warnings || [];
    const metrics = frame.metrics || {};
    setText(dom.mode, String(frame.source || 'mock').toUpperCase());
    setText(dom.frame, `FRAME ${String(frame.frame_id).padStart(5, '0')}`);
    setText(dom.traffic, traffic);
    setText(dom.action, frame.ego?.action || 'STOP');
    dom.lamp.dataset.state = traffic;
    dom.warningStrip.classList.toggle('alert', warnings.length > 0 || frame.ego?.action === 'STOP');
    setText(dom.warning, warnings.length ? warnings.join(' · ') : 'ROAD CLEAR / MONITORING');
    const nav = frame.navigation || {};
    setText(dom.navArrow, navArrow(nav.action));
    setText(dom.navAction, String(nav.action || 'STRAIGHT').replaceAll('_', ' '));
    setText(dom.navDistance, nav.distance_m == null ? '—' : `${Math.round(nav.distance_m)} m`);
    setText(dom.videoFps, `${Number(metrics.video_fps || 0).toFixed(1)} Hz`);
    setText(dom.detectorFps, `${Number(metrics.detector_fps || 0).toFixed(1)} Hz`);
    setText(dom.wsRate, `${Number(metrics.backend_ws_rate || 0).toFixed(1)} Hz`);
    setText(dom.tracks, `${frame.objects?.length || 0} / ${metrics.track_count || 0}`);
    if (dom.sceneTracks) setText(dom.sceneTracks, String(metrics.track_count || frame.objects?.length || 0).padStart(2, '0'));
    if (debugVisible) {
      dom.debugText.textContent = JSON.stringify({ frame_id: frame.frame_id, objects: frame.objects, lanes: frame.lanes, traffic_light: frame.traffic_light, navigation: frame.navigation, warnings: frame.warnings, metrics: frame.metrics }, null, 2);
    }
  }

  function setAuxCamerasVisible(visible) {
    if (dom.surround) dom.surround.hidden = !visible;
    if (dom.surroundToggle) {
      dom.surroundToggle.setAttribute('aria-expanded', String(visible));
      dom.surroundToggle.classList.toggle('active', visible);
    }
  }

  function receive(message) {
    if (message.type !== 'scene') return;
    lastFrame = message.scene;
    renderer.update(lastFrame);
    if (mapRenderer) mapRenderer.update(lastFrame);
    updateStatus(lastFrame);
    uiFrames += 1;
    const elapsed = performance.now() - uiStarted;
    if (elapsed > 1000) {
      document.body.dataset.uiFps = (uiFrames * 1000 / elapsed).toFixed(0);
      uiFrames = 0;
      uiStarted = performance.now();
    }
    if (dom.renderFps) dom.renderFps.textContent = `RENDER ${document.body.dataset.renderFps || '—'}`;
    if (message.scene?.metadata?.system_mode) dom.sourceLabel.textContent = message.scene.metadata.system_mode.replaceAll('_', ' ');
    if (message.video_jpeg) {
      dom.video.src = `data:image/jpeg;base64,${message.video_jpeg}`;
      dom.videoEmpty.hidden = true;
      dom.videoSize.textContent = message.video_jpegs ? 'FRONT / PRIMARY' : '960 × 540';
    }
    if (message.video_jpegs) {
      for (const [role, element] of [['left', dom.videoLeft], ['right', dom.videoRight], ['rear', dom.videoRear]]) {
        const value = message.video_jpegs[role];
        if (value && element) { element.src = `data:image/jpeg;base64,${value}`; element.hidden = false; }
      }
    }
    if (message.surround_bev_jpeg && dom.bevCard && dom.bev) {
      const metadata = message.scene?.metadata?.surround_bev || {};
      const surroundData = `data:image/jpeg;base64,${message.surround_bev_jpeg}`;
      dom.bev.src = surroundData;
      if (dom.sceneSurround) { dom.sceneSurround.src = surroundData; dom.sceneSurround.hidden = document.body.dataset.viewMode !== 'bowl'; }
      dom.bevCard.hidden = false;
      dom.bevCalibration.textContent = metadata.calibration_mode === 'APPROXIMATE_PLANAR' ? '近似映射' : metadata.projection === 'bowl' ? '标定 3D 曲面' : '标定鸟瞰';
      renderer.updateSurround(message.surround_bev_jpeg, metadata);
      dom.viewButtons.forEach((button) => { if (button.dataset.view === 'bowl') button.disabled = false; });
      document.body.dataset.hasSurround = '1';
      document.body.dataset.surroundMode = metadata.calibration_mode || 'UNKNOWN';
    }
  }

  function connect() {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${protocol}://${location.host}/ws`);
    socket.onopen = () => setText(dom.connection, 'LIVE');
    socket.onclose = () => { setText(dom.connection, 'RECONNECTING'); setTimeout(connect, 1000); };
    socket.onerror = () => setText(dom.connection, 'RETRYING');
    socket.onmessage = (event) => { try { receive(JSON.parse(event.data)); } catch (error) { console.error(error); } };
  }

  dom.debug.addEventListener('click', () => { debugVisible = false; dom.debug.classList.remove('visible'); });
  if (dom.surroundToggle) dom.surroundToggle.addEventListener('click', () => setAuxCamerasVisible(dom.surround.hidden));
  dom.viewButtons.forEach((button) => button.addEventListener('click', () => activateView(button.dataset.view)));
  document.addEventListener('keydown', (event) => {
    if (event.key.toLowerCase() === 'd') { debugVisible = !debugVisible; dom.debug.classList.toggle('visible', debugVisible); if (lastFrame) updateStatus(lastFrame); }
    if (event.key.toLowerCase() === 's') setAuxCamerasVisible(dom.surround.hidden);
  });
  try { renderer = new SceneRenderer(dom.host); } catch (error) { dom.host.innerHTML = `<div class="scene-error">WebGL renderer unavailable<br><small>${error.message}</small></div>`; console.error(error); renderer = { update: () => {}, updateSurround: () => {}, setView: () => {} }; }
  try { const MapRenderer = globalThis.MapLibreNavigationMap || globalThis.NavigationMapRenderer; mapRenderer = new MapRenderer(dom.navMap); } catch (error) { console.error(error); }
  activateView('nav');
  connect();
})();
