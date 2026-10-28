(function () {
const COLORS = { road: 0x9bbdc0, lane: 0xf1f4ea, cyan: 0x36c7ff, amber: 0xffbd70, red: 0xff5964, green: 0x55e398, blue: 0x268df1, marker: 0x8ca9a6 };

function material(color, roughness = 0.5) { return new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.15 }); }

function makeCar(color, scale = 1) {
  const group = new THREE.Group();
  const modelRoot = new THREE.Group(); modelRoot.name = 'apollo-model-root'; group.add(modelRoot);
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.45 * scale, 0.5 * scale, 2.7 * scale), material(color, 0.28));
  body.position.y = 0.48 * scale;
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.06 * scale, 0.43 * scale, 1.25 * scale), material(0x354a57, 0.25));
  cabin.position.set(0, 0.82 * scale, -0.05 * scale);
  modelRoot.add(body, cabin);
  const glass = new THREE.Mesh(new THREE.BoxGeometry(0.9 * scale, 0.22 * scale, 0.72 * scale), new THREE.MeshStandardMaterial({ color: 0x15252f, roughness: 0.12, metalness: 0.35 }));
  glass.position.set(0, 1.04 * scale, 0.02 * scale);
  modelRoot.add(glass);
  const headMaterial = new THREE.MeshBasicMaterial({ color: 0xdff9ff });
  const tailMaterial = new THREE.MeshBasicMaterial({ color: 0xff5964 });
  for (const x of [-0.48, 0.48]) {
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.18 * scale, 0.08 * scale, 0.06 * scale), headMaterial);
    head.position.set(x * scale, 0.58 * scale, 1.36 * scale); modelRoot.add(head);
    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.18 * scale, 0.08 * scale, 0.06 * scale), tailMaterial);
    tail.position.set(x * scale, 0.58 * scale, -1.36 * scale); modelRoot.add(tail);
  }
  for (const x of [-0.78, 0.78]) for (const z of [-0.82, 0.84]) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.22 * scale, 0.22 * scale, 0.12 * scale, 12), material(0x11151a, 0.8));
    wheel.rotation.z = Math.PI / 2; wheel.position.set(x * scale, 0.25 * scale, z * scale); modelRoot.add(wheel);
  }
  return group;
}

function parseMtl(text) {
  const result = {}; let current = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim(); if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\s+/); const key = parts.shift();
    if (key === 'newmtl') { current = parts.join(' '); result[current] = { color: 0x8aa7b8, roughness: 0.52, metalness: 0.08, opacity: 1 }; }
    else if (current && key === 'Kd') { const rgb = parts.map(Number); result[current].color = (Math.round((rgb[0] || 0) * 255) << 16) | (Math.round((rgb[1] || 0) * 255) << 8) | Math.round((rgb[2] || 0) * 255); }
    else if (current && key === 'd') result[current].opacity = Number(parts[0]);
  }
  return result;
}

function parseApolloCar(text, materials) {
  const vertices = []; const groups = new Map(); let active = 'body_color';
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim(); if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\s+/); const key = parts.shift();
    if (key === 'v') {
      const v = parts.slice(0, 3).map(Number); vertices.push(v);
      minX = Math.min(minX, v[0]); maxX = Math.max(maxX, v[0]); minY = Math.min(minY, v[1]); maxY = Math.max(maxY, v[1]); minZ = Math.min(minZ, v[2]); maxZ = Math.max(maxZ, v[2]);
    } else if (key === 'usemtl') active = parts.join(' ');
    else if (key === 'f') {
      const indexes = parts.map(token => Number(token.split('/')[0]) - 1).filter(index => index >= 0 && index < vertices.length);
      if (indexes.length < 3) continue;
      if (!groups.has(active)) groups.set(active, []);
      const target = groups.get(active);
      for (let i = 1; i < indexes.length - 1; i++) target.push(indexes[0], indexes[i], indexes[i + 1]);
    }
  }
  const root = new THREE.Group(); root.name = 'apollo-carviz-vehicle';
  for (const [name, indices] of groups) {
    const positions = [];
    for (const index of indices) {
      const v = vertices[index];
      // Center the upstream model and place the wheels on y=0 before rotating
      // it into the local road coordinate system.
      positions.push(v[0] - (minX + maxX) / 2, v[1] - minY, v[2] - (minZ + maxZ) / 2);
    }
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geometry.computeVertexNormals();
    const spec = materials[name] || { color: 0x70818a, roughness: 0.58, metalness: 0.08, opacity: 1 };
    const meshMaterial = new THREE.MeshStandardMaterial({ color: spec.color, roughness: spec.roughness, metalness: spec.metalness, transparent: spec.opacity < 1, opacity: spec.opacity });
    const mesh = new THREE.Mesh(geometry, meshMaterial); mesh.name = `apollo-${name}`; root.add(mesh);
  }
  root.rotation.y = Math.PI / 2; root.scale.set(0.88, 0.92, 0.88); root.position.y = 0.02;
  return root;
}

function tintApolloCar(root, color) {
  root.traverse(node => {
    if (!node.isMesh || !node.material) return;
    // Three.js clones share materials by default.  Give every vehicle its
    // own material before tinting, otherwise the last detected car recolors
    // the ego vehicle and every other clone as well.
    // Always clone here: Object3D.clone(true) copies userData as well, so a
    // marker on the template would otherwise make later clones reuse its
    // shared material and recolor every vehicle together.
    node.material = node.material.clone();
    const name = String(node.name || '');
    if (name.includes('body_color') || name.includes('Material__19')) node.material.color.set(color);
  });
}

function makePerson(color = 0x3e6370) {
  const group = new THREE.Group();
  const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.32, 0.9, 8), material(color)); torso.position.y = 0.65;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 12, 8), material(0xffd2ad)); head.position.y = 1.28;
  group.add(torso, head); return group;
}

function laneSegment(a, b, style) {
  const length = Math.max(0.1, Math.hypot(b.x - a.x, b.z - a.z));
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.065, 0.025, length), new THREE.MeshBasicMaterial({ color: style === 'solid' ? COLORS.lane : COLORS.cyan, transparent: true, opacity: style === 'unknown' ? 0.25 : 0.9 }));
  mesh.position.set((a.x + b.x) / 2, 0.025, (a.z + b.z) / 2);
  mesh.rotation.y = Math.atan2(b.x - a.x, b.z - a.z); return mesh;
}

class SceneRenderer {
  constructor(host) {
    this.host = host; this.latest = null; this.objects = new Map();
    this.scene = new THREE.Scene(); this.scene.background = null; this.scene.fog = new THREE.Fog(0xa8cdd1, 34, 92);
    this.camera = new THREE.PerspectiveCamera(49, 1, 0.1, 160); this.camera.position.set(0, 5.2, -8.5); this.camera.lookAt(0, 0.5, 16);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1)); this.renderer.outputColorSpace = THREE.SRGBColorSpace; const loading = host.querySelector('.scene-loading'); if (loading) loading.remove(); host.insertBefore(this.renderer.domElement, host.firstChild);
    this.scene.add(new THREE.HemisphereLight(0xe8ffff, 0x64858a, 2.5)); const key = new THREE.DirectionalLight(0xffffff, 2.7); key.position.set(-6, 12, -5); this.scene.add(key);
    this.apolloCarTemplate = null; this.loadApolloCarvizAsset();
    const road = new THREE.Mesh(new THREE.PlaneGeometry(18, 110), material(COLORS.road, 0.95)); road.rotation.x = -Math.PI / 2; road.position.z = 32; this.scene.add(road);
    const shoulder = new THREE.Mesh(new THREE.PlaneGeometry(34, 110), new THREE.MeshStandardMaterial({ color: 0xabc9c8, roughness: 1 })); shoulder.rotation.x = -Math.PI / 2; shoulder.position.y = -0.03; shoulder.position.z = 32; this.scene.add(shoulder);
    this.addDriveRoadMarkings(); this.addCitySilhouette();
    this.addRoadsideDetails();
    this.laneGroup = new THREE.Group(); this.scene.add(this.laneGroup);
    this.ego = makeCar(0x142432, 1.05); this.ego.position.y = 0.04; this.scene.add(this.ego);
    const egoHalo = new THREE.Mesh(new THREE.CircleGeometry(1.65, 48), new THREE.MeshBasicMaterial({ color: COLORS.blue, transparent: true, opacity: 0.3, depthWrite: false })); egoHalo.rotation.x = -Math.PI / 2; egoHalo.position.y = 0.06; this.ego.add(egoHalo);
    this.signal = new THREE.Group(); this.signal.position.set(0, 3.8, 20); this.scene.add(this.signal); this.makeSignal();
    this.resize = () => { const box = host.getBoundingClientRect(); this.renderer.setSize(box.width, box.height, false); this.camera.aspect = box.width / Math.max(1, box.height); this.camera.updateProjectionMatrix(); };
    window.addEventListener('resize', this.resize); this.resize();
    this.drivingNodes = [...this.scene.children].filter(node => !node.isLight && node !== this.ego);
    this.surroundView = new SurroundView(this.scene); this.viewMode = 'drive'; this.renderFrames = 0; this.renderStarted = performance.now(); this.lastAnimationTime = this.renderStarted; this.motionOffset = 0;
    this.animate = () => {
      requestAnimationFrame(this.animate);
      const now = performance.now();
      const dt = Math.min(0.05, Math.max(0, (now - this.lastAnimationTime) / 1000));
      this.lastAnimationTime = now;
      if (this.latest && this.viewMode === 'drive') {
        const steering = Math.max(-18, Math.min(18, Number(this.latest.ego?.steering || 0)));
        const targetX = steering / 18 * 0.72;
        this.ego.position.x += (targetX - this.ego.position.x) * 0.08;
        this.ego.rotation.y += (steering * Math.PI / 180 - this.ego.rotation.y) * 0.08;
        // The ego car stays in the driver's perspective.  Forward movement is
        // represented by the lane dashes and route ribbon flowing toward it,
        // which prevents a frozen cockpit while preserving the usual ADAS
        // ego-centric framing.
        const speed = Math.max(0, Number(this.latest.ego?.speed || 0)) / 3.6;
        if (speed > 0) {
          this.motionOffset = (this.motionOffset + speed * dt * 0.72) % 7;
          this.roadMarkings.position.z = -this.motionOffset;
          this.routeGroup.position.z = -this.motionOffset;
          this.roadside.position.z = -this.motionOffset * 0.65;
        }
      }
      this.renderer.render(this.scene, this.camera); this.renderFrames++;
      if (now - this.renderStarted >= 1000) { document.body.dataset.renderFps = (this.renderFrames * 1000 / (now - this.renderStarted)).toFixed(1); this.renderFrames = 0; this.renderStarted = now; }
    };
    this.animate();
  }
  async loadApolloCarvizAsset() {
    try {
      const [objText, mtlText] = await Promise.all([
        fetch('./vendor/apollo/car.obj').then(response => response.text()),
        fetch('./vendor/apollo/car.mtl').then(response => response.text()),
      ]);
      this.apolloCarTemplate = parseApolloCar(objText, parseMtl(mtlText));
      tintApolloCar(this.apolloCarTemplate, 0xe1e7e9);
      this.replaceVehicleModels();
    } catch (error) { console.warn('Apollo Carviz asset unavailable; using fallback vehicle mesh', error); }
  }
  replaceVehicleModels() {
    if (!this.apolloCarTemplate) return;
    const replace = (group, color, scale = 1) => {
      const root = group.getObjectByName('apollo-model-root'); if (!root) return;
      root.clear(); const model = this.apolloCarTemplate.clone(true); model.scale.multiplyScalar(scale); tintApolloCar(model, color); root.add(model);
    };
    replace(this.ego, 0x172937, 1.02);
    for (const group of this.objects.values()) replace(group, group.userData.modelColor || 0xe1e7e9, group.userData.baseScale || 1);
  }
  addDriveRoadMarkings() {
    this.roadMarkings = new THREE.Group();
    const white = new THREE.MeshBasicMaterial({ color: 0xf5f8ef, transparent: true, opacity: 0.88 });
    const yellow = new THREE.MeshBasicMaterial({ color: 0xffcf52, transparent: true, opacity: 0.95 });
    for (const x of [-6, -2, 2, 6]) for (let z = 7; z < 82; z += 7) { const dash = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.035, 2.8), white); dash.position.set(x, 0.035, z); this.roadMarkings.add(dash); }
    for (const x of [-8.1, 8.1]) { const edge = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.04, 102), yellow); edge.position.set(x, 0.04, 32); this.roadMarkings.add(edge); }
    this.routeGroup = new THREE.Group();
    const routeMaterial = new THREE.MeshBasicMaterial({ color: 0x168ff1, transparent: true, opacity: 0.48, depthWrite: false });
    const routeHighlight = new THREE.MeshBasicMaterial({ color: 0x55c9ff, transparent: true, opacity: 0.72, depthWrite: false });
    for (let z = 3; z < 66; z += 2.25) { const x = Math.sin(z / 15) * 0.45; const nextX = Math.sin((z + 2.25) / 15) * 0.45; const segment = new THREE.Mesh(new THREE.BoxGeometry(1.55, 0.035, 2.6), routeMaterial); segment.position.set(x, 0.065, z); segment.rotation.y = Math.atan2(nextX - x, 2.25); this.routeGroup.add(segment); }
    for (let z = 11; z < 60; z += 9) { const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.52, 3), routeHighlight); arrow.rotation.x = -Math.PI / 2; arrow.position.set(Math.sin(z / 15) * 0.45, 0.09, z); this.routeGroup.add(arrow); }
    this.scene.add(this.roadMarkings, this.routeGroup);
  }
  addCitySilhouette() {
    const skyline = new THREE.Group();
    const colors = [0x9fc3c6, 0x8fb7bc, 0xb3cdd0];
    const buildings = [[-14, 5, 4], [-10, 8, 5], [-6, 4, 3], [6, 6, 4], [10, 10, 5], [15, 5, 4]];
    buildings.forEach(([x, h, w], index) => { const building = new THREE.Mesh(new THREE.BoxGeometry(w, h, 2.8), material(colors[index % colors.length], 0.9)); building.position.set(x, h / 2 - 0.04, 58 + (index % 2) * 3); skyline.add(building); });
    this.scene.add(skyline);
  }
  addRoadsideDetails() {
    this.roadside = new THREE.Group();
    const postMaterial = new THREE.MeshStandardMaterial({ color: COLORS.marker, roughness: 0.7, metalness: 0.1 });
    const reflectorMaterial = new THREE.MeshBasicMaterial({ color: COLORS.cyan });
    for (const z of [7, 14, 23, 35, 50, 67]) {
      for (const side of [-1, 1]) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.48, 0.055), postMaterial);
        post.position.set(side * 7.7, 0.24, z);
        const reflector = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.07, 0.025), reflectorMaterial);
        reflector.position.set(side * 7.67, 0.39, z - 0.03);
        this.roadside.add(post, reflector);
      }
    }
    this.scene.add(this.roadside);
  }
  makeSignal() {
    const pole = new THREE.Mesh(new THREE.BoxGeometry(0.08, 3.8, 0.08), material(0x55616b)); pole.position.y = -1.8; this.signal.add(pole);
    const housing = new THREE.Mesh(new THREE.BoxGeometry(0.72, 1.78, 0.36), material(0x11161b, 0.6)); housing.position.y = 0.25; this.signal.add(housing);
    this.signal.lamps = ['RED', 'YELLOW', 'GREEN'].map((_, index) => { const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), new THREE.MeshBasicMaterial({ color: 0x273038 })); lamp.position.set(0, 0.78 - index * 0.52, -0.21); this.signal.add(lamp); return lamp; });
  }
  updateSignal(state) { const colors = { RED: COLORS.red, YELLOW: COLORS.amber, GREEN: COLORS.green }; this.signal.lamps.forEach((lamp, i) => { const key = ['RED', 'YELLOW', 'GREEN'][i]; lamp.material.color.set(key === state ? colors[key] : 0x273038); }); }
  updateLanes(lanes) { this.laneGroup.clear(); for (const lane of Object.values(lanes || {})) { const points = lane.points || []; for (let i = 1; i < points.length; i++) { if (lane.style === 'dashed' && i % 2 === 0) continue; this.laneGroup.add(laneSegment(points[i - 1], points[i], lane.style)); } } }
  updateObjects(items) {
    const live = new Set();
    for (const item of items || []) {
      live.add(item.id); let group = this.objects.get(item.id);
      if (!group) {
        const scale = item.class_name === 'bus' || item.class_name === 'truck' ? 1.28 : item.class_name === 'person' ? 0.75 : 1;
        const palette = [0xe7ece8, 0xb9c7cf, 0x8ea7b6, 0xd4d8d4, 0x6f889a];
        const modelColor = palette[Math.abs(Number(item.id) || 0) % palette.length];
        const personColors = [0x3e6370, 0x6f4f5d, 0x7a6a43, 0x304c5a];
        group = item.class_name === 'person' ? makePerson(personColors[Math.abs(Number(item.id) || 0) % personColors.length]) : makeCar(modelColor, scale);
        group.userData.baseScale = scale; group.userData.modelColor = modelColor;
        this.objects.set(item.id, group); this.scene.add(group);
        if (this.apolloCarTemplate && item.class_name !== 'person') this.replaceVehicleModels();
      }
      group.position.set(item.world.x, 0.05, item.world.z); group.scale.setScalar(Math.max(0.2, item.alpha || 1)); group.visible = item.alpha > 0.02;
    }
    for (const [id, group] of this.objects) if (!live.has(id)) { this.scene.remove(group); this.objects.delete(id); }
  }
  setView(mode) {
    this.viewMode = mode;
    const drive = mode === 'drive';
    const navigation = mode === 'nav';
    this.drivingNodes.forEach(node => { node.visible = drive; });
    this.objects.forEach(node => { node.visible = drive; });
    this.ego.visible = drive;
    // The HMI uses the received BEV JPEG as the primary map layer.  Keeping
    // the Three.js bowl hidden here avoids a blank CanvasTexture on browsers
    // that resize the source image while the WebSocket is still streaming;
    // calibrated bowl projections are already encoded in that JPEG.
    this.surroundView.group.visible = false;
    this.scene.fog = drive ? new THREE.Fog(0xa8cdd1, 34, 92) : null;
    this.ego.rotation.y = 0; this.ego.position.x = 0; this.ego.scale.setScalar(1);
    this.camera.up.set(0, 1, 0);
    if (drive) { this.camera.position.set(0, 5.2, -8.5); this.camera.lookAt(0, 0.5, 16); }
    else if (navigation) { this.camera.position.set(0, 13, -0.01); this.camera.lookAt(0, 0, 0); }
    else if (mode === 'top') { this.camera.position.set(0, 13, -0.01); this.camera.lookAt(0, 0, 0); }
    else { this.camera.position.set(7.5, 8, -7.5); this.camera.lookAt(0, 0, 0); }
    document.body.dataset.viewMode = mode;
  }
  updateSurround(jpeg, metadata) { this.surroundView.update(jpeg, metadata); }
  update(frame) { this.latest = frame; this.updateSignal(frame.traffic_light?.state || 'UNKNOWN'); this.updateLanes(frame.lanes); this.updateObjects(frame.objects); if (this.viewMode !== 'drive') this.objects.forEach(node => { node.visible = false; }); }
}
globalThis.SceneRenderer = SceneRenderer;
})();
