from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
# insert quad3/tri3 before carMesh
marker='''static QVector<ModelFace> *g_modelFaces = nullptr;

void carMesh'''
new='''static QVector<ModelFace> *g_modelFaces = nullptr;

void tri3(std::vector<float> &v, const QVector3D &a, const QVector3D &b,
          const QVector3D &c, float r, float g, float bl) {
    pushVertex(v, a.x(), a.y(), a.z(), r, g, bl);
    pushVertex(v, b.x(), b.y(), b.z(), r, g, bl);
    pushVertex(v, c.x(), c.y(), c.z(), r, g, bl);
}

void quad3(std::vector<float> &v, const QVector3D &a, const QVector3D &b,
           const QVector3D &c, const QVector3D &d, float r, float g, float bl) {
    tri3(v, a, b, c, r, g, bl);
    tri3(v, a, c, d, r, g, bl);
}

void carMesh'''
if marker not in s: raise SystemExit('insert marker')
s=s.replace(marker,new,1)
# replace car branch cabin box and glass with tapered prism
old='''    } else {
        box3(v, x, .18f, z, w, h * .48f, d, r, g, b);
        box3(v, x, h * .60f, z - d * .03f, w * .82f, h * .46f, d * .58f, r * 1.05f, g * 1.05f, b * 1.05f);
        // glass planes, kept as thin dark boxes on the visible rear/side
        box3(v, x, h * .67f, z - d * .31f, w * .62f, h * .22f, .025f, .04f, .07f, .09f);
        box3(v, x - w * .41f, h * .65f, z - d * .02f, .025f, h * .22f, d * .34f, .05f, .09f, .11f);
        box3(v, x + w * .41f, h * .65f, z - d * .02f, .025f, h * .22f, d * .34f, .05f, .09f, .11f);
    }'''
new='''    } else {
        const float y0 = h * .58f, y1 = h * 1.28f;
        const float lw = w * .82f, uw = w * .56f;
        const float ld = d * .58f, ud = d * .40f;
        const float zl = z - ld * .5f, zrf = z + ld * .5f;
        const float zu = z - ud * .18f, zuf = z + ud * .5f;
        // lower body and tapered cabin/roof
        box3(v, x, .18f, z, w, h * .48f, d, r, g, b);
        QVector3D bl(x - lw * .5f, y0, zl), br(x + lw * .5f, y0, zl);
        QVector3D fl(x - lw * .5f, y0, zrf), fr(x + lw * .5f, y0, zrf);
        QVector3D ubl(x - uw * .5f, y1, zu), ubr(x + uw * .5f, y1, zu);
        QVector3D ufl(x - uw * .5f, y1, zuf), ufr(x + uw * .5f, y1, zuf);
        quad3(v, bl, br, ubr, ubl, r * .85f, g * .85f, b * .85f);   // rear slope
        quad3(v, fl, fr, ufr, ufl, r * .95f, g * .95f, b * .95f);   // front slope
        quad3(v, bl, ubl, ufl, fl, r, g, b);                        // left side
        quad3(v, br, ubr, ufr, fr, r, g, b);                        // right side
        quad3(v, ubl, ubr, ufr, ufl, r * 1.10f, g * 1.10f, b * 1.10f); // roof
        // rear/side glass, slightly inset
        const float gy0 = y0 + h * .10f, gy1 = y1 - h * .06f;
        tri3(v, QVector3D(x - lw * .34f, gy0, zl + .035f),
             QVector3D(x + lw * .34f, gy0, zl + .035f),
             QVector3D(x + uw * .32f, gy1, zu + .02f), .025f, .055f, .075f);
        tri3(v, QVector3D(x - lw * .34f, gy0, zl + .035f),
             QVector3D(x + uw * .32f, gy1, zu + .02f),
             QVector3D(x - uw * .32f, gy1, zu + .02f), .025f, .055f, .075f);
        box3(v, x - lw * .505f, y0 + .12f, z - d * .09f, .022f, h * .26f, d * .24f, .04f, .08f, .10f);
        box3(v, x + lw * .505f, y0 + .12f, z - d * .09f, .022f, h * .26f, d * .24f, .04f, .08f, .10f);
        // hood/trunk planes give the body a car silhouette
        quad3(v, QVector3D(x - w * .48f, h * .66f, z + d * .18f),
              QVector3D(x + w * .48f, h * .66f, z + d * .18f),
              QVector3D(x + uw * .5f, y1, zuf),
              QVector3D(x - uw * .5f, y1, zuf), r * .92f, g * .92f, b * .92f);
    }'''
if old not in s: raise SystemExit('car branch marker')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('tapered car body')
