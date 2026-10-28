from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
# camera centered on ego lane
s=s.replace('view.lookAt(QVector3D(0.0f, 8.2f, -13.5f), QVector3D(0.0f, 0.2f, 18.0f), QVector3D(0, 1, 0));',
            'view.lookAt(QVector3D(-1.75f, 8.2f, -13.5f), QVector3D(-1.75f, 0.2f, 18.0f), QVector3D(0, 1, 0));')
# LIVE spread across four lanes, still uncalibrated relative
s=s.replace('const float tx = std::clamp(float((cx - .5) * 7.0), -5.4f, 5.4f);',
            'const float tx = std::clamp(float(-1.75 + (cx - .5) * 14.0), -6.8f, 6.8f);')
# demo lanes and count
s=s.replace('''        const float lanes[] = {-2.4f, 2.4f, 0.0f, 3.1f, -3.1f, 1.2f};
        for (int i = 0; i < 6; ++i) {''','''        const float lanes[] = {-5.25f, -1.75f, 1.75f, 5.25f, -5.25f, 1.75f};
        for (int i = 0; i < 6; ++i) {''')
start=s.index('        // Ground, grass shoulders, road bed, path, lane lines, grid.')
end=s.index('        if (demo_) addDemo(v); else addLive(v);', start)
new='''        // Four same-direction lanes, 3.5 m each. The FSD blue path is only
        // the ego lane (second from the left), not the whole road.
        const float roadHalf = 7.0f;
        const float egoX = -1.75f;
        quad(v, -32, -8, 32, 100, 0.0f, .035f, .055f, .060f);
        quad(v, -30, -6, -7.8f, 96, .006f, .055f, .085f, .075f);
        quad(v, 7.8f, -6, 30, 96, .006f, .055f, .085f, .075f);
        quad(v, -7.5f, 0, 7.5f, 92, .018f, .105f, .125f, .135f);
        quad(v, -7.25f, 0, 7.25f, 92, .026f, .105f, .115f, .120f);
        // Ego-lane FSD path.
        quad(v, egoX - 1.55f, 0, egoX + 1.55f, 90, .028f, .060f, .25f, .40f);
        // Solid road edges.
        quad(v, -7.15f, 0, -7.06f, 90, .031f, .65f, .73f, .76f);
        quad(v, 7.06f, 0, 7.15f, 90, .031f, .65f, .73f, .76f);
        // Dashed lane boundaries at -3.5, 0, +3.5.
        for (float laneX : {-3.5f, 0.0f, 3.5f}) {
            for (int i = 0; i < 22; ++i) {
                const float z0 = i * 4.0f + .7f;
                quad(v, laneX - .045f, z0, laneX + .045f, z0 + 2.35f, .032f, .48f, .55f, .58f);
            }
        }
        // Roadside grid and city silhouettes.
        for (int i = 0; i < 8; ++i) {
            const float z = 8.0f + i * 9.0f;
            quad(v, -30, z, -7.6f, z + .06f, .022f, .085f, .11f, .13f);
            quad(v, 7.6f, z, 30, z + .06f, .022f, .085f, .11f, .13f);
        }
        for (int side = -1; side <= 1; side += 2) {
            for (int i = 0; i < 8; ++i) {
                const float x = side * (9.0f + i * 3.0f);
                quad(v, x, 0, x + .05f, 92, .025f, .065f, .085f, .10f);
            }
        }
        for (int i = 0; i < 13; ++i) {
            const float x = -30.0f + i * 5.0f;
            const float h = 3.0f + float((i * 7) % 9);
            box3(v, x, 0.0f, 86.0f + (i % 3) * 2.5f, 3.4f, h, 2.0f,
                 .045f, .065f, .075f, false);
        }
'''
s=s[:start]+new+s[end:]
p.write_text(s,encoding='utf-8')
print('four lane road installed')
