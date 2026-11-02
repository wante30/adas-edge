"""A/B redundant QML offscreen layers, retaining all UI contents and features."""
import json,subprocess,time,os
from pathlib import Path
import profile_system as profile
root=profile.ROOT
qml=root/"hmi/diagnostic_direct_layers.qml"
qml.write_text((root/"hmi/main.qml").read_text().replace("layer.enabled: true","layer.enabled: false"),encoding="utf-8")
env=dict(os.environ,QT_QML_FILE=str(qml),QT_GUI_PROFILE="1")
subprocess.run(["python3",str(root/"tools/test_session.py"),"start","--build","build-stage4","--label","direct_layers","--ai-hz","0","--control-mode","HARDWARE"],env=env,check=True)
time.sleep(25);profile.sample("direct_layers",40)
