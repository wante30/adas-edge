from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
# live real 3D meshes instead of billboards
s=s.replace('''        for (const LiveCar &car : live_) addBillboard(car.x, car.z, car.cls);''','''        for (const LiveCar &car : live_) carMesh(v, car.x, car.z, car.cls, false);''')
# wider ground and horizon/backdrop additions in buildScene
s=s.replace('''        quad(v, -32, -8, 32, 100, 0.0f, .035f, .055f, .060f);
        quad(v, -30, -6, -7.8f, 96, .006f, .055f, .085f, .075f);
        quad(v, 7.8f, -6, 30, 96, .006f, .055f, .085f, .075f);''','''        quad(v, -90, -12, 90, 120, 0.0f, .030f, .046f, .050f);
        quad(v, -80, -8, -7.8f, 112, .006f, .045f, .070f, .062f);
        quad(v, 7.8f, -8, 80, 112, .006f, .045f, .070f, .062f);
        // Dark distant sky wall; buildings are only schematic background.
        box3(v, 0, 0, 116, 170, 34, 2, .020f, .032f, .044f, false);''')
# add closer side buildings before city silhouette? after for city loop maybe
s=s.replace('''        for (int i = 0; i < 13; ++i) {
            const float x = -30.0f + i * 5.0f;
            const float h = 3.0f + float((i * 7) % 9);
            box3(v, x, 0.0f, 86.0f + (i % 3) * 2.5f, 3.4f, h, 2.0f,
                 .045f, .065f, .075f, false);
        }''','''        for (int i = 0; i < 13; ++i) {
            const float x = -30.0f + i * 5.0f;
            const float h = 3.0f + float((i * 7) % 9);
            box3(v, x, 0.0f, 86.0f + (i % 3) * 2.5f, 3.4f, h, 2.0f,
                 .045f, .065f, .075f, false);
        }
        for (int side = -1; side <= 1; side += 2) {
            for (int i = 0; i < 4; ++i) {
                box3(v, side * (14.0f + i * 3.5f), 0.0f, 26.0f + i * 13.0f,
                     4.0f + i * 1.2f, 4.0f + (i % 3) * 3.0f, 4.0f,
                     .035f, .052f, .062f, false);
            }
        }''')
p.write_text(s,encoding='utf-8')
print('3d traffic/wide environment')
