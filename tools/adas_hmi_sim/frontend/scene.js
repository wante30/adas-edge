(function () {
const COLORS = { road: 0x252c32, lane: 0xdde6df, cyan: 0x65d8df, amber: 0xffbd70, red: 0xff5964, green: 0x55e398, blue: 0x7ba8ff, marker: 0x8ca9a6 };

function material(color, roughness = 0.5) { return new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.15 }); }

function makeCar(color, scale = 1) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.45 * scale, 0.5 * scale, 2.7 * scale), material(color, 0.28));
  body.position.y = 0.48 * scale;
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.06 * scale, 0.43 * scale, 1.25 * scale), material(0x354a57, 0.25));
  cabin.position.set(0, 0.82 * scale, -0.05 * scale);
  group.add(body, cabin);
  const glass = new THREE.Mesh(new THREE.BoxGeometry(0.9 * scale, 0.22 * scale, 0.72 * scale), new THREE.MeshStandardMaterial({ color: 0x15252f, roughness: 0.12, metalness: 0.35 }));
  glass.position.set(0, 1.04 * scale, 0.02 * scale);
  group.add(glass);
  const headMaterial = new THREE.MeshBasicMaterial({ color: 0xdff9ff });
  const tailMaterial = new THREE.MeshBasicMaterial({ color: 0xff5964 });
  for (const x of [-0.48, 0.48]) {
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.18 * scale, 0.08 * scale, 0.06 * scale), headMaterial);
    head.position.set(x * scale, 0.58 * scale, 1.36 * scale); group.add(head);
    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.18 * scale, 0.08 * scale, 0.06 * scale), tailMaterial);
    tail.position.set(x * scale, 0.58 * scale, -1.36 * scale); group.add(tail);
  }
  for (const x of [-0.78, 0.78]) for (const z of [-0.82, 0.84]) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.22 * scale, 0.22 * scale, 0.12 * scale, 12), material(0x11151a, 0.8));
    wheel.rotation.z = Math.PI / 2; wheel.position.set(x * scale, 0.25 * scale, z * scale); group.add(wheel);
  }
  return group;
}

function makePerson() {
  const group = new THREE.Group();
  const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.32, 0.9, 8), material(COLORS.amber)); torso.position.y = 0.65;
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
    this.scene = new THREE.Scene(); this.scene.background = null; this.scene.fog = new THREE.Fog(0x16282d, 32, 82);
    this.camera = new THREE.PerspectiveCamera(49, 1, 0.1, 160); this.camera.position.set(0, 5.2, -8.5); this.camera.lookAt(0, 0.5, 16);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1)); this.renderer.outputColorSpace = THREE.SRGBColorSpace; const loading = host.querySelector('.scene-loading'); if (loading) loading.remove(); host.insertBefore(this.renderer.domElement, host.firstChild);
    this.scene.add(new THREE.HemisphereLight(0xa7c5ce, 0x151c22, 1.8)); const key = new THREE.DirectionalLight(0xc9e9ff, 2.2); key.position.set(-6, 10, -5); this.scene.add(key);
    const road = new THREE.Mesh(new THREE.PlaneGeometry(16, 110), material(COLORS.road, 0.95)); road.rotation.x = -Math.PI / 2; road.position.z = 32; this.scene.add(road);
    const shoulder = new THREE.Mesh(new THREE.PlaneGeometry(28, 110), new THREE.MeshStandardMaterial({ color: 0x11181c, roughness: 1 })); shoulder.rotation.x = -Math.PI / 2; shoulder.position.y = -0.03; shoulder.position.z = 32; this.scene.add(shoulder);
    this.addRoadsideDetails();
    this.laneGroup = new THREE.Group(); this.scene.add(this.laneGroup);
    this.ego = makeCar(COLORS.cyan, 1.05); this.ego.position.y = 0.04; this.scene.add(this.ego);
    this.signal = new THREE.Group(); this.signal.position.set(0, 3.8, 20); this.scene.add(this.signal); this.makeSignal();
    this.resize = () => { const box = host.getBoundingClientRect(); this.renderer.setSize(box.width, box.height, false); this.camera.aspect = box.width / Math.max(1, box.height); this.camera.updateProjectionMatrix(); };
    window.addEventListener('resize', this.resize); this.resize();
    this.animate = () => { requestAnimationFrame(this.animate); if (this.latest) this.ego.rotation.y += ((this.latest.ego.steering || 0) * Math.PI / 180 - this.ego.rotation.y) * 0.08; this.renderer.render(this.scene, this.camera); };
    this.animate();
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
      if (!group) { const scale = item.class_name === 'bus' || item.class_name === 'truck' ? 1.28 : item.class_name === 'person' ? 0.75 : 1; group = item.class_name === 'person' ? makePerson() : makeCar(item.class_name === 'car' ? 0x6d86a4 : 0x907e63, scale); group.userData.baseScale = scale; this.objects.set(item.id, group); this.scene.add(group); }
      group.position.set(item.world.x, 0.05, item.world.z); group.scale.setScalar(Math.max(0.2, item.alpha || 1)); group.visible = item.alpha > 0.02;
    }
    for (const [id, group] of this.objects) if (!live.has(id)) { this.scene.remove(group); this.objects.delete(id); }
  }
  update(frame) { this.latest = frame; this.updateSignal(frame.traffic_light?.state || 'UNKNOWN'); this.updateLanes(frame.lanes); this.updateObjects(frame.objects); }
}
globalThis.SceneRenderer = SceneRenderer;
})();
