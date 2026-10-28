/* Bowl projection adapted from Open-3D-Surround-View (Nick Pai, MIT).
 * Geometry uses the SAME rounded-rectangle height equation as the LUT.
 * Coordinates: ISO X forward/Y left/Z up -> Three.js (-Y, Z, X).
 */
(function () {
  class SurroundView {
    constructor(scene) {
      this.group = new THREE.Group(); this.group.visible = false; scene.add(this.group);
      this.canvas = document.createElement('canvas'); this.canvas.width = 320; this.canvas.height = 320;
      this.texture = new THREE.CanvasTexture(this.canvas);
      this.texture.colorSpace = THREE.SRGBColorSpace;
      this.texture.generateMipmaps = false;
      this.texture.minFilter = THREE.LinearFilter;
      this.material = new THREE.MeshBasicMaterial({ map: this.texture, side: THREE.DoubleSide });
      this.image = new Image(); this.busy = false; this.key = '';
      this.image.onload = () => {
        if (this.canvas.width !== this.image.naturalWidth || this.canvas.height !== this.image.naturalHeight) {
          this.canvas.width = this.image.naturalWidth; this.canvas.height = this.image.naturalHeight;
        }
        this.canvas.getContext('2d').drawImage(this.image, 0, 0);
        this.texture.needsUpdate = true; this.busy = false;
      };
      this.image.onerror = () => { this.busy = false; };
    }
    configure(metadata) {
      const key = JSON.stringify([metadata.projection, metadata.extent_m, metadata.bowl]);
      if (this.key === key) return;
      this.key = key;
      if (this.mesh) { this.group.remove(this.mesh); this.mesh.geometry.dispose(); }
      const bowl = metadata.bowl || { flat_rect_x: 2.5, flat_rect_y: 2.5, flat_margin: 1.5, steepness: 0.5, max_radius: 4.8 }; const extent = metadata.extent_m || 10; const n = 64;
      const vertices = [], uv = [], indices = [];
      const heightAt = (x, y) => {
        if (metadata.projection !== 'bowl') return 0;
        const distance = Math.hypot(Math.max(Math.abs(x) - bowl.flat_rect_x, 0), Math.max(Math.abs(y) - bowl.flat_rect_y, 0));
        return Math.max(distance - bowl.flat_margin, 0) ** 2 * bowl.steepness;
      };
      for (let row = 0; row <= n; row++) for (let col = 0; col <= n; col++) {
        const x = extent / 2 - row * extent / n, y = extent / 2 - col * extent / n;
        vertices.push(-y, heightAt(x, y), x); uv.push(col / n, 1 - row / n);
      }
      for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
        const x = extent / 2 - (row + 0.5) * extent / n, y = extent / 2 - (col + 0.5) * extent / n;
        if (Math.hypot(x, y) > bowl.max_radius) continue;
        const a = row * (n + 1) + col, b = a + 1, c = a + n + 1, d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); geometry.setIndex(indices); geometry.computeVertexNormals();
      this.mesh = new THREE.Mesh(geometry, this.material); this.group.add(this.mesh);
    }
    update(jpeg, metadata) {
      this.configure(metadata);
      if (!this.busy) { this.busy = true; this.image.src = `data:image/jpeg;base64,${jpeg}`; }
    }
  }
  globalThis.SurroundView = SurroundView;
})();
