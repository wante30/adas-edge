from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
# modelCar signature + z scale
s=s.replace('''void modelCar(std::vector<float> &v, const QVector<ModelFace> &faces, float x, float z,
              float scale, bool ego) {''','''void modelCar(std::vector<float> &v, const QVector<ModelFace> &faces, float x, float z,
              float scale, float scaleZ, bool ego) {''')
s=s.replace('''    const float cx = 1.428f, cz = .029f;
    const int step = ego ? 1 : 2;''','''    const float cx = 1.428f, cz = .029f;
    const int step = ego ? 2 : 4;''')
s=s.replace('''            const float wx = x + (zs[k] - cz) * scale;
            const float wz = z + (xs[k] - cx) * scale;''','''            const float wx = x + (zs[k] - cz) * scaleZ;
            const float wz = z + (xs[k] - cx) * scale;''')
# branch all non-person model
s=s.replace('''    if (!truck && !bus && !person && g_modelFaces) {
        const float scale = d / 4.93f;
        modelCar(v, *g_modelFaces, x, z, scale, ego);
        return;
    }''','''    if (!person && g_modelFaces) {
        modelCar(v, *g_modelFaces, x, z, d / 4.93f, w / 2.11f, ego);
        return;
    }''')
# demo lanes/z more visible
s=s.replace('''        const float lanes[] = {-3.2f, 1.9f, 0.0f, 3.7f, -1.9f};''','''        const float lanes[] = {-2.2f, 2.2f, 0.0f, 3.0f, -3.0f};''')
# environment before ground
s=s.replace('''        // Ground, road bed, path, lane lines, grid. Static LIVE geometry is
        // intentionally not moved without ego motion input.
        quad(v, -24, -8, 24, 92, 0.0f, .045f, .060f, .070f);
        quad(v, -2.0f, 0, 2.0f, 82, 0.02f, .075f, .28f, .43f);
        // road shoulders
        quad(v, -2.45f, 0, -2.10f, 82, .028f, .20f, .24f, .26f);
        quad(v, 2.10f, 0, 2.45f, 82, .028f, .20f, .24f, .26f);
        quad(v, -2.16f, 0, -2.10f, 82, .031f, .62f, .72f, .76f);
        quad(v, 2.10f, 0, 2.16f, 82, .031f, .62f, .72f, .76f);
        for (int i = 0; i < 22; ++i) {
            const float z0 = i * 4.0f;
            quad(v, -.045f, z0 + .8f, .045f, z0 + 2.4f, .033f, .48f, .54f, .57f);
        }
        for (int i = 0; i < 8; ++i) {
            const float z = 8.0f + i * 9.0f;
            quad(v, -22, z, -3.2f, z + .06f, .022f, .085f, .11f, .13f);
            quad(v, 3.2f, z, 22, z + .06f, .022f, .085f, .11f, .13f);
        }
        for (int side = -1; side <= 1; side += 2) {
            for (int i = 0; i < 7; ++i) {
                const float x = side * (4.0f + i * 3.0f);
                quad(v, x, 0, x + .05f, 88, .025f, .065f, .085f, .10f);
            }
        }''','''        // Ground, grass shoulders, road bed, path, lane lines, grid.
        quad(v, -30, -8, 30, 96, 0.0f, .035f, .055f, .060f);
        quad(v, -28, -6, -4.2f, 92, .006f, .055f, .085f, .075f);
        quad(v, 4.2f, -6, 28, 92, .006f, .055f, .085f, .075f);
        quad(v, -4.1f, 0, 4.1f, 88, .018f, .105f, .125f, .135f);
        quad(v, -3.15f, 0, 3.15f, 86, .024f, .070f, .27f, .42f);
        // road shoulders
        quad(v, -3.75f, 0, -3.15f, 86, .028f, .22f, .25f, .26f);
        quad(v, 3.15f, 0, 3.75f, 86, .028f, .22f, .25f, .26f);
        quad(v, -3.22f, 0, -3.15f, 86, .031f, .62f, .72f, .76f);
        quad(v, 3.15f, 0, 3.22f, 86, .031f, .62f, .72f, .76f);
        for (int i = 0; i < 22; ++i) {
            const float z0 = i * 4.0f;
            quad(v, -.055f, z0 + .8f, .055f, z0 + 2.4f, .033f, .48f, .54f, .57f);
        }
        for (int i = 0; i < 8; ++i) {
            const float z = 8.0f + i * 9.0f;
            quad(v, -28, z, -4.2f, z + .06f, .022f, .085f, .11f, .13f);
            quad(v, 4.2f, z, 28, z + .06f, .022f, .085f, .11f, .13f);
        }
        for (int side = -1; side <= 1; side += 2) {
            for (int i = 0; i < 7; ++i) {
                const float x = side * (5.0f + i * 3.0f);
                quad(v, x, 0, x + .05f, 88, .025f, .065f, .085f, .10f);
            }
        }
        // distant city silhouette; purely decorative DEMO/LIVE background
        for (int i = 0; i < 11; ++i) {
            const float x = -24.0f + i * 4.8f;
            const float h = 3.0f + float((i * 7) % 9);
            box3(v, x, 0.0f, 82.0f + (i % 3) * 2.5f, 3.2f, h, 2.0f,
                 .045f, .065f, .075f, false);
        }''')
p.write_text(s,encoding='utf-8')
print('model/environment patched')
