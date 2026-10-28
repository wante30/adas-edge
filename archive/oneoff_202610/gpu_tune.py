from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    // lamps: neutral amber/white, never hazard red
    box3(v, x - w * .31f, h * .40f, z - d * .505f, .22f, .08f, .03f, .92f, .82f, .56f);
    box3(v, x + w * .31f, h * .40f, z - d * .505f, .22f, .08f, .03f, .92f, .82f, .56f);
    if (ego) {
        box3(v, x, h * .84f, z + d * .02f, w * .55f, .035f, d * .21f, .38f, .68f, .78f);
    }''','''    if (!person && !truck && !bus) {
        // Rear glass, trunk lip and bumper make the chase view read as a car.
        box3(v, x, h * .72f, z - d * .505f, w * .58f, h * .25f, .035f, .035f, .065f, .085f);
        box3(v, x, h * .52f, z - d * .50f, w * .94f, .09f, .11f, r * .75f, g * .75f, b * .75f);
        box3(v, x, h * .20f, z - d * .50f, w * .98f, .11f, .10f, .08f, .09f, .10f);
        box3(v, x, h * .88f, z - d * .04f, w * .80f, .035f, d * .18f, r * .7f, g * .7f, b * .7f);
    }
    // lamps: neutral amber/white, never hazard red
    box3(v, x - w * .31f, h * .40f, z - d * .51f, .24f, .10f, .03f, .92f, .82f, .56f);
    box3(v, x + w * .31f, h * .40f, z - d * .51f, .24f, .10f, .03f, .92f, .82f, .56f);
    if (ego) {
        box3(v, x, h * .84f, z + d * .02f, w * .55f, .035f, d * .21f, .38f, .68f, .78f);
    }''')
s=s.replace('''        const float z = 8.0f + std::fmod(10.0f + i * 11.0f + phase * 58.0f, 58.0f);
        carMesh(v, lanes[i], z, QString::fromLatin1(classes[i]), false);''','''        const float z = 5.0f + std::fmod(4.0f + i * 7.0f + phase * 34.0f, 34.0f);
        carMesh(v, lanes[i], z, QString::fromLatin1(classes[i]), false);''')
s=s.replace('''        projection.perspective(55.0f, aspect, 0.08f, 180.0f);
        projection.scale(1.0f, -1.0f, 1.0f);
        QMatrix4x4 view;
        view.lookAt(QVector3D(0.0f, 6.5f, -10.5f), QVector3D(0.0f, 0.4f, 23.0f), QVector3D(0, 1, 0));''','''        projection.perspective(52.0f, aspect, 0.08f, 180.0f);
        projection.scale(1.0f, -1.0f, 1.0f);
        QMatrix4x4 view;
        view.lookAt(QVector3D(0.0f, 8.2f, -13.5f), QVector3D(0.0f, 0.2f, 18.0f), QVector3D(0, 1, 0));''')
p.write_text(s,encoding='utf-8')
print('scene tuning patched')
