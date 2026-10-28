from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
start=s.index('        billboard_program_ = std::make_unique<QOpenGLShaderProgram>();')
end=s.index('    }\n\n    void addBillboard', start)
s=s[:start]+'''        // The billboard traffic path is retired: real targets are now 3D
        // meshes. Do not create its texture/VAO/program.
'''+s[end:]
s=s.replace('''        drawBillboards(mvp);
        checkGl("billboards");
''','''        // drawBillboards() retired with the 2D traffic path.
''',1)
p.write_text(s,encoding='utf-8')
print('billboard init removed')
