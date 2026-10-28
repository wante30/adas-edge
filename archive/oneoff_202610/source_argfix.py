from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    const QString pciePgvFile = stringArg(args, "--pcie-pgv-file", QStringLiteral("/dev/shm/pango-video/latest.pgv"));
    const QString localPgvFile = stringArg(args, "--local-pgv-file", QStringLiteral("/dev/shm/pgv_local/latest.pgv"));''','''    QString pciePgvFile = stringArg(args, "--pcie-pgv-file");
    if (pciePgvFile.isEmpty()) pciePgvFile = QStringLiteral("/dev/shm/pango-video/latest.pgv");
    QString localPgvFile = stringArg(args, "--local-pgv-file");
    if (localPgvFile.isEmpty()) localPgvFile = QStringLiteral("/dev/shm/pgv_local/latest.pgv");''',1)
p.write_text(s,encoding='utf-8')
print('default fix')
