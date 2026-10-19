import QtQuick 2.12
import QtGpuProbe 1.0

Item {
    id: root
    width: 1280
    height: 720
    property int viewMode: 0

    function trafficLabel(code) {
        if (code === "RED") return "红灯"
        if (code === "GREEN") return "绿灯"
        if (code === "YELLOW") return "黄灯"
        return "未知"
    }

    Rectangle {
        anchors.fill: parent
        color: "#070d15"
    }

    Rectangle {
        id: topBar
        x: 20; y: 16; width: parent.width - 40; height: 60
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#101f30" }
            GradientStop { position: 1.0; color: "#0a1420" }
        }
        border.color: "#28637c"; border.width: 1; radius: 12
        Text { x: 22; y: 9; text: "◉  ADAS  智能驾驶感知系统"; color: "#ecf7ff"; font.pixelSize: 22; font.family: "Noto Sans CJK SC" }
        Text { x: 22; y: 37; text: "Qt Quick / QML + C++ / RGB565 GPU 探针"; color: "#7e9bad"; font.pixelSize: 11 }
        Row {
            x: 420; y: 11; spacing: 10
            Repeater {
                model: ["单路检测", "四路检测", "空间视图"]
                delegate: Rectangle {
                    width: 112; height: 38; radius: 18
                    color: root.viewMode === index ? "#176c91" : "#122334"
                    border.color: root.viewMode === index ? "#5ce0ff" : "#2a4b60"
                    Text { anchors.centerIn: parent; text: modelData; color: "#e9f8ff"; font.pixelSize: 13 }
                    MouseArea { anchors.fill: parent; onClicked: root.viewMode = index }
                }
            }
        }
        Text { x: parent.width - 240; y: 12; text: video.hardwareAccelerated ? "GPU  硬件" : "GPU  待验证"; color: video.hardwareAccelerated ? "#52e4bc" : "#ffcf5a"; font.pixelSize: 13 }
        Text { x: parent.width - 240; y: 35; text: "渲染回调 " + video.renderFps.toFixed(1) + "/s  ·  " + video.renderer; color: "#8eb1c3"; font.pixelSize: 10; elide: Text.ElideRight; width: 225 }
    }

    Rectangle {
        id: trafficCard
        x: 20; y: 92; width: 282; height: 186; radius: 14
        color: "#102131"; border.color: "#2c94ba"; border.width: 1
        Text { x: 18; y: 14; text: "交通灯状态"; color: "#9db4c2"; font.pixelSize: 16 }
        Rectangle { x: 18; y: 51; width: 48; height: 94; radius: 10; color: "#071018"; border.color: "#3d6678" }
        Column { x: 33; y: 62; spacing: 7
            Repeater { model: ["#e34a58", "#e1b243", "#43e0ae"]
                delegate: Rectangle {
                    width: 18; height: 18; radius: 9; color: modelData; opacity: index === 2 ? 0.78 : 0.28
                    SequentialAnimation on opacity {
                        running: index === 2
                        loops: Animation.Infinite
                        NumberAnimation { to: 1.0; duration: 700 }
                        NumberAnimation { to: 0.62; duration: 700 }
                    }
                }
            }
        }
        Text {
            x: 84; y: 54; text: root.trafficLabel(video.aiTraffic)
            color: video.aiTraffic === "GREEN" ? "#55e7b5" : video.aiTraffic === "RED" ? "#ff6575" : video.aiTraffic === "YELLOW" ? "#f3c35b" : "#9ab4c1"
            font.pixelSize: 25
        }
        Text { x: 84; y: 90; width: 176; text: video.aiStatus; color: "#9ab4c1"; font.pixelSize: 10; elide: Text.ElideRight }
        Rectangle { x: 84; y: 120; width: 178; height: 1; color: "#285164" }
        Text { x: 18; y: 153; text: "实时检测框  " + video.aiDetectionCount; color: "#b5c9d3"; font.pixelSize: 11 }
        Text { x: 194; y: 14; text: video.aiResultsReceived > 0 ? "MVA2 实时" : "等待 AI"; color: "#f5c75f"; font.pixelSize: 10 }
    }

    Rectangle {
        id: videoFrame
        x: 320; y: 92; width: 640; height: 398; radius: 14
        visible: root.viewMode === 0
        color: "#05090e"; border.color: "#286981"; border.width: 1
        VideoItem { id: video; objectName: "videoProbeItem"; anchors.fill: parent; anchors.margins: 4 }
        Repeater {
            model: video.aiBoxes
            delegate: Item {
                x: video.x + modelData.x * video.width / 1280
                y: video.y + modelData.y * video.height / 720
                width: modelData.width * video.width / 1280
                height: modelData.height * video.height / 720
                Rectangle { anchors.fill: parent; color: "transparent"; border.color: modelData.color; border.width: 2 }
                Rectangle {
                    y: -18; width: label.implicitWidth + 8; height: 16
                    color: modelData.color; opacity: 0.92
                    Text { id: label; anchors.centerIn: parent; text: modelData.label; color: "#ffffff"; font.pixelSize: 9 }
                }
            }
        }
        Rectangle { x: 16; y: 14; width: 190; height: 34; radius: 17; color: "#152636"; opacity: 0.92
            Text { anchors.centerIn: parent; text: root.viewMode === 1 ? "四路预览 / GPU纹理" : root.viewMode === 2 ? "空间视图 / 预览" : "前视感知 / RGB565"; color: "#e9f7ff"; font.pixelSize: 13 }
        }
        Text { x: 18; y: height - 30; text: "frame_id  " + video.frameId + "   ·   latest-only"; color: "#85bacb"; font.pixelSize: 11 }
        Text { x: width - 190; y: height - 30; text: "上传 " + video.uploadMs.toFixed(2) + " ms"; color: "#85bacb"; font.pixelSize: 11 }
    }

    Rectangle {
        x: 320; y: 92; width: 640; height: 398; radius: 14
        visible: root.viewMode === 1
        color: "#05090e"; border.color: "#286981"; border.width: 1
        Grid {
            anchors.fill: parent; anchors.margins: 10; spacing: 8; columns: 2
            Repeater {
                model: ["前视 / RGB565 实时", "左视 / 预览", "后视 / 预览", "右视 / 预览"]
                delegate: Rectangle {
                    width: 306; height: 184; radius: 8
                    gradient: Gradient {
                        GradientStop { position: 0.0; color: "#10283b" }
                        GradientStop { position: 0.56; color: "#16212a" }
                        GradientStop { position: 1.0; color: "#08121d" }
                    }
                    border.color: index === 0 ? "#53dfff" : "#2b718d"; border.width: 1
                    Text { x: 12; y: 10; text: modelData; color: "#e4f4fa"; font.pixelSize: 12 }
                    Rectangle { x: parent.width / 2 - 3; y: 62; width: 6; height: 96; color: "#eff0d0"; opacity: 0.75 }
                    Rectangle { x: parent.width / 2 - 62; y: 95; width: 124; height: 2; color: "#7cb0bb"; rotation: -12; opacity: 0.45 }
                    Text { x: 12; y: parent.height - 24; text: index === 0 ? "frame_id " + video.frameId : "DESIGN PREVIEW"; color: "#88b3c3"; font.pixelSize: 10 }
                    Text { x: parent.width - 70; y: parent.height - 24; text: "预览数据"; color: "#f2c45c"; font.pixelSize: 9 }
                }
            }
        }
    }

    Rectangle {
        x: 320; y: 92; width: 640; height: 398; radius: 14
        visible: root.viewMode === 2
        color: "#081523"; border.color: "#286981"; border.width: 1
        Text { x: 24; y: 20; text: "ADAS 空间示意"; color: "#e3f3fa"; font.pixelSize: 18 }
        Text { x: 24; y: 50; text: "设计预览 · 未接入标定与真实目标"; color: "#f2c45c"; font.pixelSize: 11 }
        Canvas {
            anchors.fill: parent; anchors.margins: 60
            onPaint: {
                var p = getContext("2d"); p.clearRect(0,0,width,height); p.strokeStyle="#38bfe8"; p.lineWidth=4;
                p.beginPath(); p.moveTo(width*0.43,height); p.lineTo(width*0.18,height*0.12); p.moveTo(width*0.57,height); p.lineTo(width*0.82,height*0.12); p.stroke();
                p.fillStyle="#173c56"; p.fillRect(width*0.44,height*0.72,width*0.12,height*0.18); p.strokeRect(width*0.44,height*0.72,width*0.12,height*0.18);
                p.fillStyle="#ff4c69"; p.fillRect(width*0.465,height*0.24,width*0.07,height*0.09); p.fillStyle="#269be0"; p.fillRect(width*0.22,height*0.48,width*0.08,height*0.1); p.fillRect(width*0.70,height*0.48,width*0.08,height*0.1);
            }
        }
    }

    Rectangle {
        x: 978; y: 92; width: 282; height: 154; radius: 14
        color: "#0d263b"; border.color: "#2b9dcd"; border.width: 1
        Text { x: 18; y: 14; text: "导航辅助"; color: "#a5bac7"; font.pixelSize: 16 }
        Text { x: 18; y: 50; text: "↗"; color: "#43c8ff"; font.pixelSize: 50 }
        Text { x: 98; y: 54; width: 165; text: video.aiNavigation; color: "#edf8ff"; font.pixelSize: 15; elide: Text.ElideRight }
        Text { x: 98; y: 88; text: video.aiResultsReceived > 0 ? "OCR 已接入" : "等待 OCR"; color: "#d5e7ef"; font.pixelSize: 13 }
        Text { x: 98; y: 119; width: 165; text: video.aiStatus; color: "#8faab8"; font.pixelSize: 9; elide: Text.ElideRight }
        Text { x: 204; y: 14; text: "AI状态"; color: "#f5c75f"; font.pixelSize: 10 }
    }

    Rectangle {
        x: 978; y: 258; width: 282; height: 152; radius: 14
        color: "#351321"; border.color: "#e64c67"; border.width: 1
        Text { x: 18; y: 14; text: "⚠  危险事件"; color: "#ff6680"; font.pixelSize: 16 }
        Column { x: 18; y: 48; spacing: 0
            Repeater { model: video.aiHazards.length > 0 ? video.aiHazards : ["等待分析结果"]
                delegate: Rectangle { width: 245; height: 32; color: "transparent"; border.color: "#713146"; border.width: index === 0 ? 0 : 1
                    Text { anchors.verticalCenter: parent.verticalCenter; text: "•  " + modelData; color: "#ffdce2"; font.pixelSize: 12; elide: Text.ElideRight; width: 230 }
                }
            }
        }
        Text { x: 205; y: 130; text: video.aiResultsReceived > 0 ? "MVA2" : "等待"; color: "#f5c75f"; font.pixelSize: 10 }
    }

    Rectangle {
        x: 978; y: 422; width: 282; height: 68; radius: 14
        color: "#0e2130"; border.color: "#2b789b"; border.width: 1
        Text { x: 18; y: 12; text: "车道状态"; color: "#bed2dc"; font.pixelSize: 15 }
        Text { x: 18; y: 39; text: "左侧线  ▬    当前车道：直行    ▬  右侧线"; color: "#79d8e9"; font.pixelSize: 11 }
    }

    Row {
        x: 20; y: 510; spacing: 12
        Repeater { model: ["前视", "左视", "后视", "导航"]
            delegate: Rectangle { width: 196; height: 118; radius: 12; color: "#0d1824"; border.color: index === 0 ? "#56dcff" : "#23576d"; border.width: 1
                Rectangle { x: 8; y: 8; width: parent.width - 16; height: 74; color: "#05090e"; border.color: "#1f526a" }
                Text { x: 12; y: 88; text: (index + 1) + "  " + modelData; color: "#e3f2f8"; font.pixelSize: 13 }
                Text { x: 12; y: 103; text: index === 0 ? "HDMI-IN" : "预览通道"; color: "#7c9cac"; font.pixelSize: 10 }
            }
        }
    }

    Rectangle {
        x: 832; y: 510; width: 428; height: 118; radius: 14
        color: "#0c2032"; border.color: "#2f82a4"; border.width: 1
        Text { x: 18; y: 13; text: "ADAS 空间示意"; color: "#e3f3fa"; font.pixelSize: 15 }
        Text { x: width - 88; y: 14; text: "预览数据"; color: "#f5c75f"; font.pixelSize: 10 }
        Canvas { anchors.fill: parent; anchors.margins: 8; onPaint: {
                var p = getContext("2d"); p.clearRect(0,0,width,height); p.strokeStyle="#38bfe8"; p.lineWidth=2;
                p.beginPath(); p.moveTo(214,100); p.lineTo(170,20); p.moveTo(238,100); p.lineTo(282,20); p.stroke();
                p.fillStyle="#173c56"; p.fillRect(205,66,42,30); p.strokeRect(205,66,42,30);
                p.fillStyle="#ff4c69"; p.fillRect(210,28,30,18); p.fillStyle="#269be0"; p.fillRect(152,44,28,17); p.fillRect(278,44,28,17);
            }
        }
        Text { x: 18; y: 91; text: "车辆 12  行人 3  非机动车 1   ·   空间示意 · 预览"; color: "#84b8c8"; font.pixelSize: 10 }
    }

    Text { x: 20; y: 686; text: "OpenGL ES  ·  " + video.glVersion + "  ·  EGL " + video.eglVendor; color: "#6f8e9d"; font.pixelSize: 10; elide: Text.ElideRight; width: 820 }
    Text { x: 1000; y: 686; text: "RGB565 1280×720  |  物理 scanout 未测量"; color: "#f5c75f"; font.pixelSize: 10 }
}
