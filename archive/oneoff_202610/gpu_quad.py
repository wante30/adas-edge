from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
old='''void quad(std::vector<float> &v, float x1, float z1, float x2, float z2,
          float y, float r, float g, float b) {
    float x3 = x2, z3 = z2, x4 = x1, z4 = z1;
    pushVertex(v, x1, y, z1, r, g, b); pushVertex(v, x2, y, z2, r, g, b);
    pushVertex(v, x3, y, z3, r, g, b);
    pushVertex(v, x1, y, z1, r, g, b); pushVertex(v, x3, y, z3, r, g, b);
    pushVertex(v, x4, y, z4, r, g, b);
}'''
new='''void quad(std::vector<float> &v, float x1, float z1, float x2, float z2,
          float y, float r, float g, float b) {
    // x1/z1 and x2/z2 are opposite corners; build the four actual corners.
    const float xa = x1, za = z1, xb = x2, zb = z2;
    pushVertex(v, xa, y, za, r, g, b); pushVertex(v, xb, y, za, r, g, b);
    pushVertex(v, xb, y, zb, r, g, b);
    pushVertex(v, xa, y, za, r, g, b); pushVertex(v, xb, y, zb, r, g, b);
    pushVertex(v, xa, y, zb, r, g, b);
}'''
if old not in s: raise SystemExit('quad marker')
s=s.replace(old,new)
# Qt FBO is presented with the opposite vertical origin; flip the projection.
s=s.replace('''        projection.perspective(55.0f, aspect, 0.08f, 180.0f);
        QMatrix4x4 view;''','''        projection.perspective(55.0f, aspect, 0.08f, 180.0f);
        projection.scale(1.0f, -1.0f, 1.0f);
        QMatrix4x4 view;''')
p.write_text(s,encoding='utf-8')
print('quad/flip patched')
