import QtQuick 2.12
import QtGpuProbe 1.0
import "components"

Item {
    id: root
    width: 1280
    height: 720
    focus: true
    Rectangle {
        id: fpgaEntry
        x: 740; y: 76; width: 112; height: 24; z: 900
        color: "#0B2630"; border.color: root.cyan; border.width: 0.8; radius: 2
        Text { anchors.centerIn: parent; text: "FPGA CTRL"; color: root.cyanHot; font.family: root.num; font.pixelSize: 7 }
        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: root.fpgaOpen = true }
    }

    property string cjk: "Noto Sans CJK SC"
    property string num: "DejaVu Sans"
    property color ink: "#EAFBFF"
    property color inkSoft: "#C4E1E8"
    property color muted: "#7F9BA9"
    property color cyan: "#3CE3FF"
    property color cyanHot: "#8CF5FF"
    property color mint: "#45F3B0"
    property color amber: "#FFBC5C"
    property color red: "#FF5064"
    property color line: "#1B4052"
    property string inputLayout: (typeof pgvInputLayout !== "undefined" ? pgvInputLayout : "front")
    // The user wants the top-right quadrant (Q2) as the default main view.
    property int roadQuadrant: (typeof pgvRoadQuadrant !== "undefined" ? pgvRoadQuadrant : 1)
    property int navQuadrant: 1
    property int viewMode: (typeof initialViewMode !== "undefined" ? initialViewMode : 0)
    property bool autoCycle: (typeof autoCycleCapture !== "undefined" ? autoCycleCapture : false)
    property var now: new Date()
    property bool hudTestOpen: false
    property bool fpgaOpen: false
    property bool videoQsg: typeof videoQsgMode !== "undefined" ? videoQsgMode : false
    property var fpgaControl: fpgaIsp
    property bool autoHudSend: true
    property bool sceneEnabled: (typeof sceneEnabled !== "undefined" ? sceneEnabled : true)
    property int trackingMode: 2  // 0=RAW, 1=FIX4, 2=TIME
    property int testTraffic: 1
    property int testLaneLeft: 0
    property int testLaneRight: 0
    property var hudFlags: ({
        perception: true,
        pedestrian: false,
        staticTarget: false,
        overspeed: false,
        lka: false,
        legacy: true
    })

    HudControl { id: pcieHud }

    onRoadQuadrantChanged: if (video) video.aiRoadQuadrant = roadQuadrant
    onNavQuadrantChanged: if (video) video.aiNavigationQuadrant = navQuadrant
    function attachTeslaScene() {
        if (!teslaLive) return
        var target = (viewMode === 2) ? spatialSceneHost : cardSceneHost
        if (teslaLive.parent !== target) {
            teslaLive.parent = target
            teslaLive.anchors.fill = target
        }
        teslaLive.visible = (viewMode === 0 || viewMode === 2)
    }

    onViewModeChanged: {
        if (video) video.spatialViewActive = (viewMode === 2)
        attachTeslaScene()
    }

    Component.onCompleted: {
        video.aiRoadQuadrant = roadQuadrant
        video.aiNavigationQuadrant = navQuadrant
        attachTeslaScene()
    }

    // Real snapshots from the current PCIe 2x2 frame. The large viewport is
    // live; these dock thumbnails are snapshots, labelled SNAP.
    property var cameras: [
        { name: "前视", source: "PCIE Q1", image: "assets/cam_front.png" },
        { name: "导航", source: "PCIE Q2", image: "assets/cam_upper_right.png" },
        { name: "车内", source: "PCIE Q3", image: "assets/cam_lower_left.png" },
        { name: "路口", source: "PCIE Q4", image: "assets/cam_hdmi.png" }
    ]

    Timer {
        interval: 1000
        running: true
        repeat: true
        onTriggered: root.now = new Date()
    }

    // Capture-only helper; the normal UI never auto-switches.
    Timer {
        interval: 5000
        running: root.autoCycle
        repeat: true
        onTriggered: root.viewMode = (root.viewMode + 1) % 3
    }

    // The AI traffic/pedestrian snapshot is transmitted to the FPGA over the
    // root BAR1 bridge every 500 ms. Other V2 fields are manual-test only.
    Timer {
        interval: 500
        running: root.autoHudSend
        repeat: true
        onTriggered: root.sendAiHud()
    }

    Keys.onPressed: {
        if (event.key === Qt.Key_1) { root.viewMode = 0; event.accepted = true }
        else if (event.key === Qt.Key_2) { root.viewMode = 1; event.accepted = true }
        else if (event.key === Qt.Key_3) { root.viewMode = 2; event.accepted = true }
        else if (event.key === Qt.Key_T) { root.hudTestOpen = !root.hudTestOpen; event.accepted = true }
        else if (event.key === Qt.Key_I) { root.fpgaOpen = !root.fpgaOpen; event.accepted = true }
        else if (event.key === Qt.Key_L) { root.trackingMode = (root.trackingMode + 1) % 3; event.accepted = true }
        else if (event.key === Qt.Key_Escape) { root.hudTestOpen = false; root.fpgaOpen = false; event.accepted = true }
        else if (event.key === Qt.Key_Q) { Qt.quit() }
    }

    function toggleHudFlag(key) {
        var old = root.hudFlags, next = ({})
        for (var name in old) next[name] = old[name]
        next[key] = !next[key]
        root.hudFlags = next
    }
    function sendAiHud() {
        var counts = video.aiDetectionCounts || ({})
        var traffic = video.aiTraffic === "RED" ? 1 : (video.aiTraffic === "GREEN" ? 2 : 0)
        var pedestrian = (counts["person"] || 0) > 0
        pcieHud.sendPerception(true, traffic, pedestrian, false, false, 0, 0,
                                false, false, true, 0x101)
    }
    function sendTestHud() {
        var f = root.hudFlags
        pcieHud.sendPerception(f.perception, root.testTraffic, f.pedestrian,
                                f.staticTarget, f.overspeed, root.testLaneLeft,
                                root.testLaneRight, f.lka, false, f.legacy, 0x101)
    }
    function laneText(value) {
        return value === 2 ? "虚线" : (value === 1 ? "实线" : "无")
    }

    function sourceIsPcie() { return video.sourceMode === "pgv-live" }
    function sourceLabel() {
        if (!video.sourceOnline) return "视频等待"
        if (video.sourceMode === "pgv-live") return "PCIe LIVE"
        if (video.sourceMode === "pgv-fixture") return "PGVF TEST"
        if (video.sourceMode === "pgv-local") return "LOCAL REPLAY"
        if (video.sourceMode === "local-raw") return "LOCAL RAW"
        if (video.sourceMode === "replay") return "REPLAY"
        return "DEMO"
    }
    function sourceColor() {
        if (!video.sourceOnline) return ambertone()
        return video.sourceMode === "pgv-live" ? mint : amber
    }
    function ambertone() { return amber }

    function detectionMap() { return video.aiDetectionCounts ? video.aiDetectionCounts : ({}) }
    function countOf(name) { var m = detectionMap(); return m && m[name] !== undefined ? m[name] : 0 }
    function vehicleCount() {
        if (!video.aiOnline) return "--"
        return (countOf("car") + countOf("bus") + countOf("truck") + countOf("motorcycle"))
    }
    function personCount() {
        if (!video.aiOnline) return "--"
        return countOf("person")
    }
    function cycleCount() {
        if (!video.aiOnline) return "--"
        return countOf("bicycle")
    }
    function totalDetections() {
        if (!video.aiOnline) return "--"
        var m = detectionMap(), total = 0
        for (var k in m) total += m[k]
        return total
    }
    function trafficConfidence() {
        if (!video.aiOnline) return "--"
        var boxes = video.aiBoxes || [], best = 0
        for (var i = 0; i < boxes.length; ++i) {
            if (boxes[i].className === "traffic light")
                best = Math.max(best, boxes[i].confidence || 0)
        }
        return best > 0 ? Math.round(best * 100) + "%" : "--"
    }
    function trafficText() {
        if (!video.aiOnline) return "等待识别"
        if (video.aiTraffic === "GREEN") return "绿灯"
        if (video.aiTraffic === "RED") return "红灯"
        if (video.aiTraffic === "YELLOW") return "黄灯"
        return "未识别"
    }
    function trafficTone() {
        if (!video.aiOnline) return muted
        if (video.aiTraffic === "GREEN") return mint
        if (video.aiTraffic === "RED") return red
        if (video.aiTraffic === "YELLOW") return amber
        return muted
    }
    function trafficLamp(index) {
        if (!video.aiOnline) return "#1A3038"
        if (video.aiTraffic === "RED" && index === 0) return red
        if (video.aiTraffic === "YELLOW" && index === 1) return amber
        if (video.aiTraffic === "GREEN" && index === 2) return mint
        return "#1A3038"
    }
    function signalChip() {
        if (!video.aiOnline) return "等待 AI"
        if (video.aiTraffic === "GREEN") return "通行"
        if (video.aiTraffic === "RED") return "停车"
        if (video.aiTraffic === "YELLOW") return "注意"
        return "未识别"
    }

    function navValid() {
        return video.aiOnline && video.aiNavigationState === "VALID"
                && ((video.aiNavigationRoad && video.aiNavigationRoad.length > 0)
                    || video.aiNavigationDistanceM >= 0)
    }
    function navDirectionCode() {
        if (!navValid()) return "none"
        return video.aiNavigationDirection || "none"
    }
    function navDirection() {
        if (!navValid()) return "等待"
        if (video.aiNavigationDirection === "left") return "左转"
        if (video.aiNavigationDirection === "right") return "右转"
        if (video.aiNavigationDirection === "forward") return "直行"
        if (video.aiNavigationDirection === "uturn") return "掉头"
        return "导航"
    }
    function navDistance() {
        if (!navValid() || video.aiNavigationDistanceM < 0) return "—"
        if (video.aiNavigationDistanceM === 0) return "现在"
        return Math.round(video.aiNavigationDistanceM)
    }
    function navRoad() {
        if (navValid() && video.aiNavigationRoad) return video.aiNavigationRoad
        if (!video.aiOnline) return "AI 未连接"
        if (video.aiNavigationState === "STOPPED") return "导航已停止更新"
        if (video.aiNavigationState === "STALE") return "导航暂不可读"
        if (video.aiNavigationState === "UNREADABLE") return "导航暂不可读"
        return "等待 Q2 OCR"
    }
    function navMeta() {
        if (navValid()) {
            return "OCR Q" + (navQuadrant + 1) + " · "
                    + (navDirectionCode() === "none" ? "导航卡片" : navDirection())
        }
        if (!video.aiOnline) return "AI 未连接"
        if (video.aiNavigationState === "UNREADABLE") return "导航暂不可读 · Q" + (navQuadrant + 1)
        if (video.aiNavigationState === "STALE") return "导航过期 · Q" + (navQuadrant + 1)
        return "等待导航 OCR · Q" + (navQuadrant + 1)
    }

    function hazardLabel(kind) {
        if (kind === "LEAD_RAPID_CLOSING") return "前车快速接近"
        if (kind === "PEDESTRIAN_CROSSING_PATH") return "行人进入预测路径"
        if (kind === "PEDESTRIAN_IN_PATH") return "前方行人"
        if (kind === "RED_SIGNAL_CONFIRMED") return "确认红灯 · 注意信号"
        if (kind === "LEAD_BRAKE_LAMP_RISE") return "前车制动灯变化"
        if (kind === "SUSPECTED_BRAKING_WITH_CLOSING") return "制动灯变化且快速接近"
        return "感知风险"
    }
    function hazardRows() {
        var rows = []
        if (video.aiOnline) {
            var hs = (video.aiHazards || []).slice()
            var priority = { "RED_SIGNAL_CONFIRMED": 0, "PEDESTRIAN_CROSSING_PATH": 1,
                             "PEDESTRIAN_IN_PATH": 1, "LEAD_RAPID_CLOSING": 2,
                             "SUSPECTED_BRAKING_WITH_CLOSING": 3, "LEAD_BRAKE_LAMP_RISE": 4 }
            hs.sort(function(a, b) { var pa = priority[a]; var pb = priority[b]; if (pa === undefined) pa = 9; if (pb === undefined) pb = 9; return pa - pb })
            for (var i = 0; i < hs.length && i < 3; ++i)
                rows.push({ label: hazardLabel(hs[i]), live: true, value: "实时" })
            return rows
        }
        return [ { label: "AI 源失联", live: false, value: "无数据" } ]
    }
    function hazardBadge() {
        if (!video.aiOnline) return "离线"
        var n = (video.aiHazards || []).length
        return n > 0 ? (n + " 项实时") : "监控中"
    }
    function hazardFooter() {
        if (!video.aiOnline) return "AI sidecar 未连接"
        if ((video.aiHazards || []).length === 0) return "AI 实时分析 · 更新 " + video.aiResultAgeMs + " ms"
        return "AI 帧龄 " + video.aiResultAgeMs + " ms · 实时分析"
    }
    function previewWarningVisible() {
        return video.aiOnline && (video.aiHazards || []).length === 0
    }
    function primaryHazard() {
        var hs = video.aiHazards || []
        return hs.length > 0 ? hazardLabel(hs[0]) : "未检测到危险事件"
    }

    function targetTone(name) {
        if (name === "person" || name === "bicycle") return mint
        if (name === "traffic light") return amber
        return cyan
    }
    function compactName(name) {
        if (!name) return "OBJECT"
        if (name === "traffic light") return "SIGNAL"
        return String(name).toUpperCase()
    }
    function targetFrame(name) {
        if (name === "person" || name === "bicycle") return "assets/target_mint.png"
        if (name === "traffic light") return "assets/target_amber.png"
        return "assets/target_cyan.png"
    }

    function metricLabel(index) {
        return ["PCIE", "视频", "AI", "NPU", "GPU"][index]
    }
    function metricValue(index) {
        if (index === 0) return video.sourceOnline ? "在线" : "断开"
        if (index === 1) return video.sourceFps.toFixed(0) + " FPS"
        if (index === 2) return video.aiOnline ? video.aiFps.toFixed(1) + " FPS" : "等待"
        if (index === 3) return Math.round(video.npuLoadPercent) + "%"
        return Math.round(video.gpuLoadPercent) + "%"
    }
    function metricColor(index) {
        if (index === 0) return video.sourceOnline ? mint : amber
        return index < 3 ? ink : inkSoft
    }
    function inputSummary() {
        if (video.sourceMode === "pgv-live")
            return inputLayout === "quad" ? ("PCIe 主视 Q" + (roadQuadrant + 1)) : "PCIe 前视"
        if (video.sourceMode === "pgv-local") return "本地回放 Q" + (roadQuadrant + 1)
        if (video.sourceMode === "local-raw") return "本地直读 Q" + (roadQuadrant + 1)
        if (video.sourceMode === "pgv-fixture") return "PGVF 夹具"
        if (video.sourceMode === "replay") return "前视 (RGB565)"
        return "模拟视频源"
    }
    function quadName(index) {
        return ["PCIe Q1 · 前视", "PCIe Q2 · 导航", "PCIe Q3 · 车内", "PCIe Q4 · 路口"][index]
    }
    function clockText() { return Qt.formatDateTime(now, "HH:mm") }
    function dateText() { return Qt.formatDateTime(now, "yyyy/MM/dd") }

    // ------------------------------------------------------------------ base
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#06131C" }
            GradientStop { position: 0.48; color: "#03090E" }
            GradientStop { position: 1.0; color: "#02070B" }
        }
    }

    // Subtle engineering grid, kept static and cheap.
    Canvas {
        anchors.fill: parent
        opacity: 0.7
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            ctx.strokeStyle = "rgba(70,145,160,0.045)"
            ctx.lineWidth = 1
            for (var x = 0; x <= width; x += 48) {
                ctx.beginPath(); ctx.moveTo(x, 64); ctx.lineTo(x, height); ctx.stroke()
            }
            for (var y = 80; y <= height; y += 48) {
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke()
            }
            ctx.strokeStyle = "rgba(66,167,186,0.05)"
            ctx.beginPath(); ctx.moveTo(0, 684); ctx.lineTo(width, 620); ctx.stroke()
        }
    }

    // ----------------------------------------------------------------- header
    Rectangle {
        id: header
        x: 0; y: 0; width: parent.width; height: 64
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#081823" }
            GradientStop { position: 0.72; color: "#051019" }
            GradientStop { position: 1.0; color: "#030B11" }
        }
    }
    Rectangle { x: 0; y: 0; width: 3; height: 64; color: cyan }
    Rectangle {
        x: 0; y: 63; width: parent.width; height: 1
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "#8CF5FF" }
            GradientStop { position: 0.35; color: "#1B6377" }
            GradientStop { position: 0.76; color: "#255E6E" }
            GradientStop { position: 1.0; color: "#65CFE1" }
        }
    }

    Item {
        id: brandBadge
        x: 20; y: 14; width: 38; height: 38
        HudPanel {
            anchors.fill: parent
            cut: 6
            accent: root.cyan
            fillTop: "#0B3445"
            fillBottom: "#05151E"
            fillOpacity: 0.88
            glow: true
            showTicks: false
        }
        Image {
            anchors.centerIn: parent
            width: 28; height: 22
            source: "assets/icon_car.png"
            fillMode: Image.PreserveAspectFit
            smooth: true
        }
    }
    Text {
        x: 68; y: 0; width: 84; height: 35
        text: "ADAS"
        color: root.ink
        font.family: root.num
        font.pixelSize: 26
        font.weight: Font.Bold
        font.letterSpacing: 1.5
        verticalAlignment: Text.AlignVCenter
    }
    Rectangle { x: 154; y: 12; width: 1; height: 38; color: "#27515E" }
    Text {
        x: 164; y: 5; width: 168; height: 24
        text: "智能驾驶感知系统"
        color: root.ink
        font.family: root.cjk
        font.pixelSize: 17
        font.weight: Font.DemiBold
        verticalAlignment: Text.AlignVCenter
    }
    Text {
        x: 166; y: 29; width: 176; height: 12
        text: "MULTI-SENSOR  ·  PERCEPTION"
        color: "#7BABB8"
        font.family: root.num
        font.pixelSize: 6
        font.letterSpacing: 1.3
        opacity: 0.9
    }
    Rectangle {
        x: 313; y: 34; width: 4; height: 4; radius: 2
        color: root.mint
        opacity: 0.65 + 0.35 * Math.sin(root.now.getSeconds() * 1.7)
    }
    Text {
        x: 321; y: 29; text: "LIVE"
        color: "#7195A2"; font.family: root.num; font.pixelSize: 6; font.weight: Font.DemiBold
        font.letterSpacing: 1.0
    }

    // Mode tabs.
    Rectangle {
        id: tabTrack
        x: 348; y: 15; width: 270; height: 34; radius: 7
        color: "#06131B"
        border.color: "#1A4051"; border.width: 1
    }
    Rectangle {
        id: tabHighlight
        x: 349 + root.viewMode * 90
        y: 16; width: 88; height: 32; radius: 6
        color: "#0B3C50"
        border.color: root.cyan; border.width: 1
        Behavior on x { NumberAnimation { duration: 150; easing.type: Easing.OutQuad } }
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#103E50" }
            GradientStop { position: 1.0; color: "#08222E" }
        }
    }
    Row {
        x: 348; y: 15
        Repeater {
            model: ["单路检测", "四路检测", "空间视图"]
            delegate: Item {
                width: 90; height: 34
                Rectangle {
                    anchors.centerIn: parent
                    width: 3; height: 3; radius: 1.5
                    color: index === root.viewMode ? root.cyan : "#507983"
                    opacity: index === root.viewMode ? 0.95 : 0.55
                    x: 12
                }
                Text {
                    anchors.centerIn: parent
                    text: modelData
                    color: index === root.viewMode ? root.ink : "#7698A5"
                    font.family: root.cjk
                    font.pixelSize: 11
                    font.weight: index === root.viewMode ? Font.DemiBold : Font.Normal
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.viewMode = index
                }
            }
        }
    }

    // Input source chip. It is a status chip, not a fake dropdown.
    Rectangle {
        x: 626; y: 15; width: 208; height: 34; radius: 7
        color: "#06141D"
        border.color: "#214A5C"; border.width: 1
        Image {
            x: 11; y: 10; width: 14; height: 13
            source: "assets/icon_camera.png"; fillMode: Image.PreserveAspectFit
        }
        Text {
            x: 32; y: 3; text: "输入源"
            color: "#6B919D"; font.family: root.cjk; font.pixelSize: 7
        }
        Text {
            x: 32; y: 14; text: root.inputSummary()
            color: root.inkSoft; font.family: root.cjk; font.pixelSize: 10; font.weight: Font.DemiBold
        }

        Rectangle { x: 174; y: 9; width: 1; height: 17; color: "#214A5C" }
        Text {
            x: 179; y: 10; text: root.inputLayout === "quad" ? "2×2" : "1×1"
            color: "#84A9B4"; font.family: root.num; font.pixelSize: 6
        }
        Rectangle {
            x: 198; y: 14; width: 5; height: 5; radius: 2.5
            color: video.sourceOnline ? root.mint : root.amber
        }
    }

    // Five compact telemetry cells.
    Item {
        x: 850; y: 8; width: 318; height: 48
        Repeater {
            model: 5
            delegate: Item {
                x: index * 63.5; width: 63.5; height: 48
                Rectangle {
                    x: 0; y: 13; width: 1; height: 30; color: "#1A3C4C"
                    visible: index > 0
                }
                Text {
                    x: 9; y: 2; text: root.metricLabel(index)
                    color: "#708F9B"; font.family: root.num; font.pixelSize: 7; font.letterSpacing: 0.8
                }
                Text {
                    x: 9; y: 16; width: 54; height: 20
                    text: root.metricValue(index)
                    color: root.metricColor(index)
                    font.family: root.num; font.pixelSize: index === 1 ? 9 : 10
                    font.weight: Font.DemiBold
                    elide: Text.ElideRight
                }
            }
        }
    }

    Text {
        x: 1176; y: 6; width: 84; height: 28
        text: root.clockText()
        color: root.ink
        font.family: root.num; font.pixelSize: 19; font.weight: Font.DemiBold
        horizontalAlignment: Text.AlignRight
        verticalAlignment: Text.AlignVCenter
    }
    Text {
        x: 1176; y: 35; width: 84; height: 15
        text: root.dateText()
        color: "#6E929F"; font.family: root.num; font.pixelSize: 7
        horizontalAlignment: Text.AlignRight
        verticalAlignment: Text.AlignVCenter
    }
    Rectangle { x: 1166; y: 15; width: 1; height: 34; color: "#224653" }

    // ------------------------------------------------------------------- main
    Item {
        id: mainView
        x: 20; y: 77; width: 967; height: 460
        clip: true

        Rectangle { anchors.fill: parent; color: "#02070B" }

        VideoItem {
            id: video
            objectName: "videoProbeItem"
            x: 0
            y: (mainView.height - height) / 2
            width: mainView.width
            height: width * 9.0 / 16.0
            visible: root.viewMode !== 2 && !root.videoQsg
            qsgTarget: qsgVideo
            sourceViewport: {
                if (root.viewMode === 1 || root.inputLayout !== "quad")
                    return Qt.rect(0, 0, 1, 1)
                var q = root.roadQuadrant
                return Qt.rect((q % 2) * 0.5, Math.floor(q / 2) * 0.5, 0.5, 0.5)
            }
        }

        QsgVideoItem {
            id: qsgVideo
            x: video.x
            y: video.y
            width: video.width
            height: video.height
            visible: root.viewMode !== 2 && root.videoQsg
            source: video
        }

        Item {
            id: spatialSceneHost
            x: 0; y: 0; width: mainView.width; height: mainView.height
            visible: root.viewMode === 2
        }

        // Readability gradients.
        Rectangle {
            x: 0; y: 0; width: parent.width; height: 118
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#D901080C" }
                GradientStop { position: 1.0; color: "#0001080C" }
            }
        }
        Rectangle {
            x: 0; y: parent.height - 112; width: parent.width; height: 112
            gradient: Gradient {
                GradientStop { position: 0.0; color: "#0001080C" }
                GradientStop { position: 1.0; color: "#D901080C" }
            }
        }

        // AI target layer, mapped from the 1280x720 AI display space onto
        // the live front view. Simple QML primitives render faster than a
        // continuously repainted Canvas on this Mali/X11 path.
        Item {
            anchors.fill: parent
            clip: true
            visible: root.viewMode === 0 && video.aiOnline && video.bboxCount > 0
            Repeater {
                model: video.bboxModel
                delegate: Item {
                    property color tone: root.targetTone(model.className)
                    property bool hot: model.primary
                    property real predictSeconds: {
                        if (root.trackingMode === 0) return 0
                        var maxSeconds = video.maxTrackPredictionMs / 1000
                        if (root.trackingMode === 1) {
                            var lagFrames = Math.max(0, Number(video.displayFrameId)
                                                        - Number(model.frameId || video.displayFrameId))
                            return Math.min(maxSeconds, Math.min(4, lagFrames)
                                                       / Math.max(1, video.sourceFps))
                        }
                        var displayNs = Number(video.displayTimestampNs)
                        var detectNs = Number(model.detectTimestampNs)
                        if (displayNs > 0 && detectNs > 0)
                            return Math.min(maxSeconds, Math.max(0, (displayNs - detectNs) / 1e9))
                        var lag = Math.max(0, Number(video.displayFrameId)
                                              - Number(model.frameId || video.displayFrameId))
                        return Math.min(maxSeconds, lag / Math.max(1, video.sourceFps))
                    }
                    property real predictedWidth: Math.max(1, model.width + (model.vw || 0) * predictSeconds)
                    property real predictedHeight: Math.max(1, model.height + (model.vh || 0) * predictSeconds)
                    x: (model.x + (model.vx || 0) * predictSeconds) / 1280 * video.width
                    y: (model.y + (model.vy || 0) * predictSeconds) / 720 * video.height + video.y
                    width: predictedWidth / 1280 * video.width
                    height: predictedHeight / 720 * video.height
                    visible: width > 8 && height > 8
                    opacity: model.state === "predicted" ? 0.42
                             : (model.state === "tentative" ? 0.62 : 1.0)
                    z: hot ? 4 : 2

                    Rectangle {
                        anchors.fill: parent
                        color: Qt.rgba(tone.r, tone.g, tone.b, hot ? 0.055 : 0.018)
                    }
                    BorderImage {
                        anchors.fill: parent
                        source: root.targetFrame(model.className)
                        border.left: 14
                        border.top: 14
                        border.right: 14
                        border.bottom: 14
                        horizontalTileMode: BorderImage.Stretch
                        verticalTileMode: BorderImage.Stretch
                        smooth: true
                        opacity: hot ? 1.0 : 0.82
                    }
                    Rectangle {
                        anchors.fill: parent
                        color: "transparent"
                        border.width: hot ? 1.2 : 0.8
                        border.color: Qt.rgba(tone.r, tone.g, tone.b, hot ? 0.62 : 0.34)
                    }

                    Rectangle {
                        id: boxLabel
                        x: 0; y: -19
                        width: Math.min(190, 14 + labelText.implicitWidth)
                        height: 18
                        color: "#E8020C12"
                        border.color: Qt.rgba(tone.r, tone.g, tone.b, 0.5)
                        border.width: 0.8
                        Rectangle { x: 0; y: 0; width: 2.2; height: parent.height; color: tone }
                        Text {
                            id: labelText
                            x: 7; anchors.verticalCenter: parent.verticalCenter
                            text: root.compactName(model.className) + "  "
                                  + Math.round((model.confidence || 0) * 100) + "%"
                            color: Qt.rgba(tone.r, tone.g, tone.b, 0.95)
                            font.family: root.num; font.pixelSize: 9; font.weight: Font.DemiBold
                        }
                    }
                }
            }
        }

        // Frame border and status chip.
        Rectangle {
            anchors.fill: parent
            color: "transparent"
            border.color: "#296579"; border.width: 1
            radius: 4
        }
        Canvas {
            anchors.fill: parent
            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                var c = 14
                ctx.fillStyle = "#02070B"
                ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(c, 0); ctx.lineTo(0, c); ctx.closePath(); ctx.fill()
                ctx.beginPath(); ctx.moveTo(width, height); ctx.lineTo(width - c, height); ctx.lineTo(width, height - c); ctx.closePath(); ctx.fill()
                ctx.strokeStyle = "rgba(60,227,255,0.38)"
                ctx.lineWidth = 1
                ctx.beginPath(); ctx.moveTo(0, 34); ctx.lineTo(0, c); ctx.lineTo(c, 0); ctx.lineTo(34, 0); ctx.stroke()
                ctx.beginPath(); ctx.moveTo(width, height - 34); ctx.lineTo(width, height - c); ctx.lineTo(width - c, height); ctx.lineTo(width - 34, height); ctx.stroke()
            }
            onWidthChanged: requestPaint()
            onHeightChanged: requestPaint()
        }

        Rectangle {
            x: parent.width - 163; y: 12; width: 151; height: 23
            color: "#B8020D13"
            border.color: "#2C5563"; border.width: 0.8
            Rectangle {
                x: 9; y: 9; width: 4; height: 4; radius: 2
                color: root.sourceColor()
            }
            Text {
                x: 18; y: 3
                text: root.sourceLabel()
                color: root.sourceColor()
                font.family: root.num; font.pixelSize: 7; font.weight: Font.DemiBold
            }
            Text {
                anchors.right: parent.right; anchors.rightMargin: 8; y: 3
                text: video.sourceOnline ? video.sourceFps.toFixed(0) + " FPS" : "NO SIGNAL"
                color: "#91B8C4"; font.family: root.num; font.pixelSize: 7
            }
        }

        Rectangle {
            x: parent.width - 108; y: 44; width: 96; height: 23; radius: 2
            color: "#C0031118"; border.color: "#355E6C"; border.width: 0.7
            Rectangle {
                x: 8; y: 9; width: 5; height: 5; radius: 2.5
                color: pcieHud.bridgeOnline ? root.mint : root.amber
            }
            Text {
                x: 18; y: 4; text: "PCIe 信号测试"
                color: pcieHud.bridgeOnline ? root.cyanHot : root.amber
                font.family: root.cjk; font.pixelSize: 7
            }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: {
                    root.autoHudSend = false
                    root.hudTestOpen = true
                }
            }
        }

        Rectangle {
            x: parent.width - 202; y: 44; width: 84; height: 23; radius: 2
            color: "#C0031118"; border.color: root.trackingMode === 2 ? "#3F7059" : "#5A4A2C"
            border.width: 0.7
            Text {
                anchors.centerIn: parent
                text: root.trackingMode === 2 ? "TRK TIME"
                                       : (root.trackingMode === 1 ? "TRK FIX4" : "TRK RAW")
                color: root.trackingMode === 2 ? root.mint
                       : (root.trackingMode === 1 ? root.cyanHot : root.amber)
                font.family: root.num; font.pixelSize: 6
            }
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                onClicked: root.trackingMode = (root.trackingMode + 1) % 3
            }
        }

        // Single mode: signal card.
        Item {
            id: signalCard
            visible: root.viewMode === 0
            layer.enabled: true
            layer.smooth: true
            x: 16; y: 16; width: 244; height: 150
            HudPanel {
                anchors.fill: parent
                cut: 12
                accent: root.mint
                fillTop: "#0A2028"
                fillBottom: "#040E14"
                fillOpacity: 0.94
                borderOpacity: 0.62
            }
            Rectangle {
                x: 1; y: 25; width: 1.5; height: 58; color: Qt.rgba(root.mint.r, root.mint.g, root.mint.b, 0.45)
            }
            // Vertical signal lamp.
            Rectangle {
                x: 14; y: 15; width: 25; height: 66; radius: 8
                color: "#07161A"
                border.color: "#285B55"; border.width: 0.8
                Repeater {
                    model: 3
                    delegate: Rectangle {
                        x: 6; y: 10 + index * 17; width: 13; height: 13; radius: 6.5
                        color: root.trafficLamp(index)
                        border.color: Qt.rgba(root.trafficTone().r, root.trafficTone().g, root.trafficTone().b, 0.28)
                        border.width: 0.8
                    }
                }
            }
            Text {
                x: 51; y: 12; text: "信号灯"
                color: "#8FB3BD"; font.family: root.cjk; font.pixelSize: 8
            }
            Text {
                x: 51; y: 25; width: 102; height: 28
                text: root.trafficText()
                color: root.trafficTone()
                font.family: root.cjk; font.pixelSize: 18; font.weight: Font.DemiBold
                verticalAlignment: Text.AlignVCenter
            }
            Rectangle {
                x: parent.width - 52; y: 17; width: 38; height: 17
                color: "#241E0C"
                border.color: Qt.rgba(root.trafficTone().r, root.trafficTone().g, root.trafficTone().b, 0.42)
                border.width: 0.7
                radius: 2
                Text {
                    anchors.centerIn: parent
                    text: root.signalChip()
                    color: root.trafficTone(); font.family: root.cjk; font.pixelSize: 7
                }
            }
            Text {
                x: 51; y: 55; width: 76; height: 13; text: video.aiOnline ? "实时识别" : "AI 未连接"
                color: "#7E9DA7"; font.family: root.cjk; font.pixelSize: 7
            }
            Text {
                x: parent.width - 105; y: 55; width: 91; height: 13
                text: "置信度 " + root.trafficConfidence()
                color: "#9AB8C0"; font.family: root.num; font.pixelSize: 7
                horizontalAlignment: Text.AlignRight
            }
            Rectangle { x: 13; y: 91; width: parent.width - 26; height: 1; color: "#1C3B42" }
            Row {
                x: 14; y: 99; spacing: 0
                Repeater {
                    model: [
                        { icon: "assets/icon_car.png", value: root.vehicleCount(), name: "车辆" },
                        { icon: "assets/icon_person.png", value: root.personCount(), name: "行人" },
                        { icon: "assets/icon_bike.png", value: root.cycleCount(), name: "非机动车" }
                    ]
                    delegate: Item {
                        width: 72; height: 38
                        Image {
                            x: 0; y: 1; width: 13; height: 13
                            source: modelData.icon; fillMode: Image.PreserveAspectFit
                        }
                        Text {
                            x: 17; y: -2; text: modelData.value
                            color: root.ink; font.family: root.num; font.pixelSize: 13; font.weight: Font.DemiBold
                        }
                        Text {
                            x: 17; y: 16; text: modelData.name
                            color: "#89A5AE"; font.family: root.cjk; font.pixelSize: 7
                        }
                    }
                }
            }
        }

        // Single mode: a real hazard warning or the neutral monitoring strip.
        Rectangle {
            id: warningStrip
            visible: root.viewMode === 0
            x: parent.width / 2 - 181; y: parent.height - 37
            width: 362; height: 27; radius: 3
            color: (video.aiOnline && (video.aiHazards || []).length > 0)
                   ? "#B326111A" : "#B308171E"
            border.width: 0.8
            border.color: (video.aiOnline && (video.aiHazards || []).length > 0)
                          ? "#B9FF5064" : "#4A2A5B66"
            Image {
                x: 10; y: 6; width: 15; height: 15
                source: "assets/icon_warning.png"; fillMode: Image.PreserveAspectFit
                opacity: (video.aiOnline && (video.aiHazards || []).length > 0) ? 1 : 0.5
            }
            Text {
                x: 32; y: 4; width: 260; height: 19
                text: (video.aiOnline && (video.aiHazards || []).length > 0)
                      ? root.primaryHazard() : (video.aiOnline ? "感知监控中 · 未检测到危险事件" : "AI sidecar 未连接")
                color: (video.aiOnline && (video.aiHazards || []).length > 0) ? "#FFE7E9" : "#91B6C0"
                font.family: root.cjk; font.pixelSize: 9; font.weight: Font.DemiBold
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }
            Text {
                anchors.right: parent.right; anchors.rightMargin: 10
                y: 5; text: (video.aiOnline && (video.aiHazards || []).length > 0) ? "实时" : "PGVF"
                color: (video.aiOnline && (video.aiHazards || []).length > 0) ? "#FF9BA3" : "#6E939E"
                font.family: root.num; font.pixelSize: 6
            }
        }

        // Single mode: bottom-left telemetry.
        Rectangle {
            visible: root.viewMode === 0
            x: 14; y: parent.height - 22; width: 250; height: 14; radius: 2
            color: "#B8031118"
            Text {
                x: 8; y: 1
                text: "FUSION " + root.totalDetections() + "  ·  PCIE RGB565  ·  AI " + (video.aiOnline ? "LIVE" : "OFF")
                color: "#7099A4"; font.family: root.num; font.pixelSize: 6
            }
        }

        // Quad mode labels.
        Repeater {
            model: 4
            delegate: Rectangle {
                visible: root.viewMode === 1
                x: (index % 2 === 0 ? 10 : parent.width - 124)
                y: (index === 1 ? 40 : (index === 0 ? 10 : parent.height / 2 + 8))
                width: 114; height: 19; radius: 2
                color: "#C0031118"
                border.color: index === root.roadQuadrant ? "#643CE3FF" : "#352F5662"
                border.width: 0.7
                Text {
                    x: 8; anchors.verticalCenter: parent.verticalCenter
                    text: root.quadName(index)
                    color: index === root.roadQuadrant ? root.cyanHot : root.inkSoft
                    font.family: root.cjk; font.pixelSize: 7
                }
                Rectangle {
                    anchors.right: parent.right; anchors.rightMargin: 7
                    anchors.verticalCenter: parent.verticalCenter
                    width: 4; height: 4; radius: 2
                    color: root.mint
                }
            }
        }
        Rectangle {
            visible: root.viewMode === 1
            x: parent.width / 2 - 96; y: 10; width: 192; height: 20; radius: 2
            color: "#D504131B"; border.color: "#365D6D"; border.width: 0.6
            Text {
                anchors.centerIn: parent
                text: "四路环视  ·  PGVF 同步 60 Hz"
                color: "#A4C8D1"; font.family: root.cjk; font.pixelSize: 7
            }
        }

        // Spatial mode overlay.
        Item {
            id: spatialOverlay
            anchors.fill: parent
            visible: root.viewMode === 2
            Text {
                x: 24; y: 16; text: "空间视图"
                color: root.ink; font.family: root.cjk; font.pixelSize: 20; font.weight: Font.DemiBold
            }
            Text {
                x: 25; y: 43; text: "BEV  ·  OCCUPANCY  ·  FUSION  /  RANGE 80 m"
                color: "#76AAB7"; font.family: root.num; font.pixelSize: 7; font.letterSpacing: 0.7
            }
            Row {
                x: parent.width - 196; y: 17; spacing: 5
                Repeater {
                    model: [
                        { name: "BEV", tone: root.cyan },
                        { name: "OCC", tone: root.mint },
                        { name: "VRU", tone: root.amber }
                    ]
                    delegate: Rectangle {
                        width: 60; height: 19; radius: 2
                        color: "#B3031118"; border.color: "#3A51606B"; border.width: 0.6
                        Rectangle { x: 8; anchors.verticalCenter: parent.verticalCenter; width: 4; height: 4; radius: 2; color: modelData.tone }
                        Text { x: 17; anchors.verticalCenter: parent.verticalCenter; text: modelData.name; color: modelData.tone; font.family: root.num; font.pixelSize: 6 }
                    }
                }
            }
            Item {
                x: parent.width - 242; y: 72; width: 218; height: 91
                HudPanel {
                    anchors.fill: parent
                    cut: 8; accent: root.red; fillTop: "#1D0C12"; fillBottom: "#080B0F"
                    fillOpacity: 0.92; borderOpacity: 0.72; glow: true
                }
                Image { x: 12; y: 12; width: 14; height: 14; source: "assets/icon_warning.png" }
                Text {
                    x: 32; y: 8; text: video.aiOnline ? ("感知目标  " + root.totalDetections()) : "等待 AI"
                    color: "#FFB4BA"; font.family: root.cjk; font.pixelSize: 8
                }
                Text {
                    x: 12; y: 32; text: video.aiOnline ? root.primaryHazard() : "AI sidecar 未连接"
                    color: root.ink; font.family: root.cjk; font.pixelSize: 12; font.weight: Font.DemiBold
                }
                Text {
                    x: 12; y: 57; text: "空间示意 · 未标定"; color: root.amber
                    font.family: root.cjk; font.pixelSize: 7
                }
                Text {
                    x: 12; y: 73; text: "not a calibrated lane model" ; color: "#7D7478"
                    font.family: root.num; font.pixelSize: 6
                }
            }
            Rectangle {
                x: parent.width - 242; y: 176; width: 218; height: 103; radius: 4
                color: "#B3051219"; border.color: "#254B58"; border.width: 0.7
                Text {
                    x: 12; y: 9; text: "TESLA-BEVCAM-CLONE"
                    color: root.cyanHot; font.family: root.num; font.pixelSize: 7; font.weight: Font.DemiBold
                }
                Text {
                    x: 12; y: 27; text: "BEV 空间场景"
                    color: root.inkSoft; font.family: root.cjk; font.pixelSize: 10; font.weight: Font.DemiBold
                }
                Rectangle { x: 12; y: 47; width: 194; height: 1; color: "#1D3B46" }
                Text {
                    x: 12; y: 55; text: "LIVE TRACKS · RELATIVE_ONLY"
                    color: root.amber; font.family: root.cjk; font.pixelSize: 6
                }
                Text {
                    x: 12; y: 72; text: "真实 Track ID · 非标定相对空间"
                    color: "#7E9FA9"; font.family: root.cjk; font.pixelSize: 6
                }
            }
            Text {
                x: 24; y: parent.height - 32; text: "EGO   ·   " + root.totalDetections() + " TARGETS   ·   CLONE VISUAL  ·   NON-CALIBRATED"
                color: "#8FBCC8"; font.family: root.num; font.pixelSize: 7
            }
        }
    }

    // ----------------------------------------------------------------- sidebar
    Item {
        id: navCard
        x: 1000
        layer.enabled: true
        layer.smooth: true; y: 77; width: 260; height: 152
        HudPanel {
            anchors.fill: parent
            cut: 11; accent: root.cyan
            fillTop: "#0A1D28"; fillBottom: "#040F17"
            fillOpacity: 0.95; borderOpacity: 0.58
        }
        Text {
            x: 14; y: 9; text: "路径规划"
            color: "#9EC9D3"; font.family: root.cjk; font.pixelSize: 9; font.weight: Font.DemiBold
        }
        Text {
            x: 73; y: 10; text: "NAVIGATION"
            color: "#517F8B"; font.family: root.num; font.pixelSize: 6; font.letterSpacing: 0.9
        }
        Rectangle {
            x: parent.width - 75; y: 9; width: 61; height: 16; radius: 2
            color: root.navValid() ? "#0B302A" : "#2B220E"
            border.color: root.navValid() ? "#436B56" : "#6B552A"; border.width: 0.6
            Text {
                anchors.centerIn: parent
                text: root.navValid() ? "AI 实时"
                      : (video.aiNavigationState === "STALE" ? "已过期"
                      : (video.aiNavigationState === "UNREADABLE" ? "暂不可读" : "等待"))
                color: root.navValid() ? root.mint : root.amber
                font.family: root.cjk; font.pixelSize: 6
            }
        }
        Image {
            x: 13; y: 30; width: 62; height: 62
            source: "assets/route_turn.png"
            fillMode: Image.PreserveAspectFit
            visible: root.navValid() && root.navDirectionCode() === "right"
        }
        Item {
            x: 13; y: 30; width: 62; height: 62
            visible: root.navValid() && root.navDirectionCode() !== "right"
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                y: 8; text: "OCR"; color: root.cyan; font.family: root.num
                font.pixelSize: 13; font.weight: Font.DemiBold
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                y: 28; text: "Q" + (root.navQuadrant + 1); color: "#79A8B3"
                font.family: root.num; font.pixelSize: 7
            }
            Rectangle {
                anchors.horizontalCenter: parent.horizontalCenter
                y: 44; width: 32; height: 1; color: "#2D5360"
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                y: 49; text: "LIVE"; color: root.mint; font.family: root.num; font.pixelSize: 6
            }
        }
        Text {
            x: 81; y: 25; width: 56; height: 30
            text: root.navDistance()
            color: root.ink; font.family: root.num; font.pixelSize: 28; font.weight: Font.Bold
            verticalAlignment: Text.AlignVCenter
        }
        Text {
            x: 133; y: 38; text: "m"
            color: root.muted; font.family: root.num; font.pixelSize: 8
            visible: root.navValid() && video.aiNavigationDistanceM > 0
        }
        Text {
            x: 153; y: 31; width: 84; height: 24
            text: root.navDirection()
            color: root.cyanHot; font.family: root.cjk; font.pixelSize: 17; font.weight: Font.DemiBold
            verticalAlignment: Text.AlignVCenter
        }
        Text {
            x: 82; y: 58; width: 158; height: 17
            text: root.navRoad()
            color: root.inkSoft; font.family: root.cjk; font.pixelSize: 10
            elide: Text.ElideRight
        }
        Rectangle { x: 14; y: 100; width: parent.width - 28; height: 1; color: "#1A3B47" }
        Text {
            x: 14; y: 106; width: 232; height: 13
            text: root.navMeta()
            color: root.navValid() ? "#77A8B2" : "#A58E58"
            font.family: root.cjk; font.pixelSize: 6
        }
        Text {
            x: 14; y: 124; text: "导航源"
            color: "#7598A3"; font.family: root.cjk; font.pixelSize: 6
        }
        Row {
            x: 57; y: 121; spacing: 3
            Repeater {
                model: 4
                delegate: Rectangle {
                    width: 36; height: 18; radius: 2
                    color: index === root.navQuadrant ? "#163B47" : "#07161D"
                    border.width: 0.7
                    border.color: index === root.navQuadrant ? root.amber : "#284752"
                    Text {
                        anchors.centerIn: parent
                        text: "Q" + (index + 1)
                        color: index === root.navQuadrant ? root.amber : "#7F9BA5"
                        font.family: root.num; font.pixelSize: 6
                    }
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: root.navQuadrant = index
                    }
                }
            }
        }
    }

    Item {
        id: hazardCard
        x: 1000
        layer.enabled: true
        layer.smooth: true; y: 241; width: 260; height: 142
        HudPanel {
            anchors.fill: parent
            cut: 11; accent: root.red
            fillTop: "#1D0B11"; fillBottom: "#080B0F"
            fillOpacity: 0.96; borderOpacity: 0.64; glow: true
        }
        Image { x: 13; y: 10; width: 15; height: 15; source: "assets/icon_warning.png" }
        Text {
            x: 34; y: 6; text: "危险事件"
            color: "#FFE7E9"; font.family: root.cjk; font.pixelSize: 10; font.weight: Font.DemiBold
        }
        Text {
            x: 96; y: 10; text: "HAZARD"
            color: "#A76D73"; font.family: root.num; font.pixelSize: 6; font.letterSpacing: 0.9
        }
        Rectangle {
            x: parent.width - 73; y: 8; width: 59; height: 16; radius: 2
            color: "#2E1119"; border.color: "#65303A"; border.width: 0.6
            Text {
                anchors.centerIn: parent
                text: root.hazardBadge()
                color: "#E3A1A8"; font.family: root.cjk; font.pixelSize: 6
            }
        }
        Rectangle { x: 13; y: 31; width: parent.width - 26; height: 1; color: "#3A1E24" }
        Item {
            visible: video.aiOnline && root.hazardRows().length === 0
            x: 0; y: 34; width: parent.width; height: 89
            Image {
                anchors.horizontalCenter: parent.horizontalCenter
                y: 10; width: 26; height: 26
                source: "assets/icon_warning.png"
                opacity: 0.28
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                y: 42; text: "未检测到危险事件"
                color: "#E2BFC2"; font.family: root.cjk; font.pixelSize: 9
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                y: 60; text: "AI 实时监控中"
                color: "#A6797E"; font.family: root.cjk; font.pixelSize: 6
            }
        }
        Repeater {
            model: root.hazardRows()
            delegate: Item {
                x: 10; y: 37 + index * 28; width: parent.width - 20; height: 25
                Rectangle {
                    x: 0; y: 2; width: 2; height: 17; color: root.red
                    opacity: modelData.live ? 0.75 : 0.25
                }
                Image {
                    x: 9; y: 4; width: 12; height: 12
                    source: "assets/icon_warning.png"
                    opacity: modelData.live ? 0.9 : 0.35
                }
                Text {
                    x: 28; y: 1; width: 176; height: 17
                    text: modelData.label
                    color: modelData.live ? "#F5DFE1" : "#8F7376"
                    font.family: root.cjk; font.pixelSize: 8
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
                Text {
                    anchors.right: parent.right; y: 2
                    text: modelData.value
                    color: modelData.live ? "#FF9DA5" : "#69585A"
                    font.family: root.num; font.pixelSize: 6
                }
                Rectangle { x: 28; anchors.bottom: parent.bottom; width: parent.width - 28; height: 1; color: "#332027" }
            }
        }
        Text {
            x: 14; y: parent.height - 17
            text: root.hazardFooter()
            color: "#B8848A"; font.family: root.cjk; font.pixelSize: 6
            elide: Text.ElideRight; width: parent.width - 28
        }
    }

    Item {
        id: laneCard
        x: 1000
        layer.enabled: true
        layer.smooth: true; y: 395; width: 260; height: 142
        HudPanel {
            anchors.fill: parent
            cut: 11; accent: root.cyan
            fillTop: "#091D27"; fillBottom: "#040F16"
            fillOpacity: 0.95; borderOpacity: 0.52
        }
        Image { x: 13; y: 10; width: 14; height: 14; source: "assets/icon_lane.png" }
        Text {
            x: 34; y: 6; text: "车道状态"
            color: root.inkSoft; font.family: root.cjk; font.pixelSize: 9; font.weight: Font.DemiBold
        }
        Rectangle {
            x: parent.width - 72; y: 8; width: 58; height: 16; radius: 2
            color: "#0B1B22"; border.color: "#355E6C"; border.width: 0.6
            Text {
                anchors.centerIn: parent; text: "未接入"; color: root.amber
                font.family: root.cjk; font.pixelSize: 6
            }
        }
        Image {
            x: 12; y: 33; width: 78; height: 96
            source: "assets/lane_diagram.png"
            fillMode: Image.PreserveAspectFit
        }
        Text { x: 104; y: 35; text: "当前车道"; color: "#78949E"; font.family: root.cjk; font.pixelSize: 7 }
        Text { x: 178; y: 35; text: "—"; color: "#78949E"; font.family: root.cjk; font.pixelSize: 8; font.weight: Font.DemiBold }
        Rectangle { x: 104; y: 53; width: parent.width - 118; height: 1; color: "#1A343D" }
        Text { x: 104; y: 59; text: "左侧车道"; color: "#78949E"; font.family: root.cjk; font.pixelSize: 7 }
        Text { x: 178; y: 59; text: "—"; color: "#78949E"; font.family: root.cjk; font.pixelSize: 8; font.weight: Font.DemiBold }
        Rectangle { x: 104; y: 77; width: parent.width - 118; height: 1; color: "#1A343D" }
        Text { x: 104; y: 83; text: "右侧车道"; color: "#78949E"; font.family: root.cjk; font.pixelSize: 7 }
        Text { x: 178; y: 83; text: "—"; color: "#78949E"; font.family: root.cjk; font.pixelSize: 8; font.weight: Font.DemiBold }
        Text {
            x: 104; y: parent.height - 23; width: parent.width - 118
            text: "车道信息未接入"
            color: "#8A9AA0"; font.family: root.cjk; font.pixelSize: 6
        }
    }

    // --------------------------------------------------------------- bottom dock
    Repeater {
        model: root.cameras
        delegate: Item {
            x: 20 + index * 203; y: 547; width: 193; height: 145
            layer.enabled: true
            layer.smooth: true
            Rectangle {
                anchors.fill: parent
                radius: 5
                color: "#040D13"
                border.width: index === root.roadQuadrant ? 1.2 : 0.8
                border.color: index === root.roadQuadrant ? root.cyan : "#254754"
                clip: true
                Image {
                    anchors.fill: parent
                    source: modelData.image
                    fillMode: Image.PreserveAspectCrop
                    smooth: true
                    opacity: index === root.roadQuadrant ? 1 : 0.78
                }
                Rectangle {
                    anchors.fill: parent
                    gradient: Gradient {
                        GradientStop { position: 0.0; color: "#23040B11" }
                        GradientStop { position: 0.55; color: "#00040B11" }
                        GradientStop { position: 1.0; color: "#E6040B11" }
                    }
                }
                Rectangle {
                    x: 9; y: 9; width: 31; height: 13; radius: 2
                    color: "#B4020D13"
                    Text {
                        anchors.centerIn: parent
                        text: index === root.roadQuadrant ? "MAIN"
                              : (index === root.navQuadrant ? "NAV" : "SNAP")
                        color: index === root.roadQuadrant ? root.mint
                               : (index === root.navQuadrant ? root.amber : "#81A7B2")
                        font.family: root.num; font.pixelSize: 5; font.weight: Font.DemiBold
                    }
                }
                Rectangle {
                    anchors.right: parent.right; anchors.rightMargin: 8; y: 10
                    width: 4; height: 4; radius: 2
                    color: index === root.roadQuadrant ? root.mint : "#6A8993"
                }
                Rectangle {
                    x: 10; y: 104; width: 23; height: 19; radius: 2
                    color: index === root.roadQuadrant ? root.cyan : "#0D2B36"
                    Text {
                        anchors.centerIn: parent
                        text: "0" + (index + 1)
                        color: index === root.roadQuadrant ? "#031118" : "#9AC5CE"
                        font.family: root.num; font.pixelSize: 7; font.weight: Font.Bold
                    }
                }
                Text {
                    x: 39; y: 102; width: 100; height: 17
                    text: modelData.name
                    color: root.ink; font.family: root.cjk; font.pixelSize: 9; font.weight: Font.DemiBold
                }
                Text {
                    x: 39; y: 119; width: 133; height: 12
                    text: modelData.source
                    color: "#80A7B1"; font.family: root.num; font.pixelSize: 6
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        root.roadQuadrant = index
                        root.viewMode = 0
                    }
                }
            }
        }
    }

    Item {
        id: spatialCard
        x: 833
        layer.enabled: true
        layer.smooth: true; y: 547; width: 427; height: 145
        visible: root.viewMode === 0
        HudPanel {
            anchors.fill: parent
            cut: 11; accent: root.cyan
            fillTop: "#091D27"; fillBottom: "#040F16"
            fillOpacity: 0.96; borderOpacity: 0.58
        }
        Image { x: 14; y: 11; width: 15; height: 15; source: "assets/icon_radar.png" }
        Text {
            x: 35; y: 7; text: "Tesla BEV 空间"
            color: root.ink; font.family: root.cjk; font.pixelSize: 10; font.weight: Font.DemiBold
        }
        Rectangle {
            x: parent.width - 72; y: 9; width: 57; height: 16; radius: 2
            color: "#0B302A"; border.color: "#3F7059"; border.width: 0.6
            Text { anchors.centerIn: parent; text: "LIVE"; color: root.mint; font.family: root.num; font.pixelSize: 6 }
        }
        Rectangle { x: 14; y: 31; width: parent.width - 28; height: 1; color: "#1A3942" }
        Item {
            id: cardSceneHost
            x: 14; y: 37; width: parent.width - 28; height: 88
            visible: root.viewMode === 0
            TeslaSceneItem {
                id: teslaLive
                anchors.fill: parent
                boxes: video.aiBoxes
                online: video.aiOnline
                visible: root.sceneEnabled
                sourceKey: video.sourceMode
                sourceFrameId: video.displayFrameId
                sourceTimestampNs: video.displayTimestampNs
            }
        }
        Text {
            x: 16; y: parent.height - 15
            text: "LIVE · Q1 · F" + video.displayFrameId + " · " + video.aiDetectionCount + " targets · RELATIVE_ONLY"
            color: "#7FB3B8"; font.family: root.cjk; font.pixelSize: 6
        }
    }

    // ------------------------------------------------------------ signal test
    Item {
        anchors.fill: parent
        visible: root.hudTestOpen
        z: 300

        Rectangle {
            anchors.fill: parent
            color: "#C902070B"
            MouseArea { anchors.fill: parent; onClicked: root.hudTestOpen = false }
        }

        Item {
            x: 240; y: 118; width: 800; height: 486
            HudPanel {
                anchors.fill: parent
                cut: 14; accent: root.cyan
                fillTop: "#0A1D28"; fillBottom: "#030C12"
                fillOpacity: 0.98; borderOpacity: 0.7; glow: true
            }
            MouseArea { anchors.fill: parent; onClicked: {} }

            Text {
                x: 24; y: 16; text: "PCIe HUD V2 信号测试"
                color: root.ink; font.family: root.cjk; font.pixelSize: 15; font.weight: Font.DemiBold
            }
            Text {
                x: 280; y: 20; text: "BAR1 +0x638 / +0x644 · root bridge 38262"
                color: "#6F95A0"; font.family: root.num; font.pixelSize: 7
            }
            Rectangle { x: 24; y: 47; width: parent.width - 48; height: 1; color: "#23434E" }
            Rectangle {
                x: parent.width - 92; y: 14; width: 70; height: 22; radius: 2
                color: pcieHud.bridgeOnline ? "#0B302A" : "#2B220E"
                border.color: pcieHud.bridgeOnline ? "#3F7059" : "#6B552A"; border.width: 0.7
                Text {
                    anchors.centerIn: parent
                    text: pcieHud.bridgeOnline ? "BRIDGE ON" : "BRIDGE OFF"
                    color: pcieHud.bridgeOnline ? root.mint : root.amber
                    font.family: root.num; font.pixelSize: 6
                }
            }
            Text {
                x: parent.width - 30; y: 16; text: "×"
                color: "#89A9B3"; font.family: root.num; font.pixelSize: 18
                MouseArea { anchors.fill: parent; anchors.margins: -8; onClicked: root.hudTestOpen = false }
            }

            Text { x: 24; y: 63; text: "红绿灯"; color: "#8FB0BA"; font.family: root.cjk; font.pixelSize: 8 }
            Repeater {
                model: [{ name: "熄灭", value: 0 }, { name: "红灯", value: 1 }, { name: "绿灯", value: 2 }]
                delegate: Rectangle {
                    x: 90 + index * 92; y: 59; width: 82; height: 24; radius: 2
                    color: root.testTraffic === modelData.value ? "#163B47" : "#07161D"
                    border.width: 0.8
                    border.color: root.testTraffic === modelData.value ? root.cyan : "#284752"
                    Text {
                        anchors.centerIn: parent
                        text: modelData.name
                        color: root.testTraffic === modelData.value ? root.cyanHot : "#8AA5AE"
                        font.family: root.cjk; font.pixelSize: 8
                    }
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: root.testTraffic = modelData.value
                    }
                }
            }
            Text {
                x: 380; y: 66; text: "协议只有 OFF/RED/GREEN；黄灯无字段"
                color: "#6E8E98"; font.family: root.cjk; font.pixelSize: 6
            }

            Repeater {
                model: [
                    { key: "perception", label: "感知使能 (perception_enable)" },
                    { key: "pedestrian", label: "行人 (pedestrian)" },
                    { key: "staticTarget", label: "静止目标 (static_target)" },
                    { key: "overspeed", label: "超速 (overspeed)" },
                    { key: "lka", label: "LKA 警告 (lka_warning)" },
                    { key: "legacy", label: "HUD/ADAS 使能 (0x608)" }
                ]
                delegate: Item {
                    x: 24 + (index % 2) * 378
                    y: 100 + Math.floor(index / 2) * 42
                    width: 360; height: 32
                    Text {
                        x: 0; y: 4; text: modelData.label
                        color: "#A9C5CD"; font.family: root.cjk; font.pixelSize: 8
                    }
                    Rectangle {
                        x: 282; y: 3; width: 56; height: 22; radius: 2
                        color: root.hudFlags[modelData.key] ? "#0C3A33" : "#101C22"
                        border.color: root.hudFlags[modelData.key] ? root.mint : "#33505B"; border.width: 0.8
                        Text {
                            anchors.centerIn: parent
                            text: root.hudFlags[modelData.key] ? "ON" : "OFF"
                            color: root.hudFlags[modelData.key] ? root.mint : "#718C95"
                            font.family: root.num; font.pixelSize: 7
                        }
                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: root.toggleHudFlag(modelData.key)
                        }
                    }
                }
            }

            Text { x: 24; y: 232; text: "车道"; color: "#8FB0BA"; font.family: root.cjk; font.pixelSize: 8 }
            Text { x: 90; y: 235; text: "左 " + root.laneText(root.testLaneLeft); color: root.cyanHot; font.family: root.cjk; font.pixelSize: 8 }
            Rectangle {
                x: 156; y: 229; width: 52; height: 22; radius: 2
                color: "#163B47"; border.color: "#3A6775"; border.width: 0.7
                Text { anchors.centerIn: parent; text: "切换"; color: root.inkSoft; font.family: root.cjk; font.pixelSize: 7 }
                MouseArea { anchors.fill: parent; onClicked: root.testLaneLeft = (root.testLaneLeft + 1) % 3 }
            }
            Text { x: 240; y: 235; text: "右 " + root.laneText(root.testLaneRight); color: root.cyanHot; font.family: root.cjk; font.pixelSize: 8 }
            Rectangle {
                x: 306; y: 229; width: 52; height: 22; radius: 2
                color: "#163B47"; border.color: "#3A6775"; border.width: 0.7
                Text { anchors.centerIn: parent; text: "切换"; color: root.inkSoft; font.family: root.cjk; font.pixelSize: 7 }
                MouseArea { anchors.fill: parent; onClicked: root.testLaneRight = (root.testLaneRight + 1) % 3 }
            }
            Text {
                x: 390; y: 235; text: "时间控制未启用：板端时钟未同步"
                color: "#8B7B55"; font.family: root.cjk; font.pixelSize: 6
            }

            Rectangle { x: 24; y: 274; width: parent.width - 48; height: 1; color: "#23434E" }
            Text {
                x: 24; y: 287; width: parent.width - 48; height: 34
                text: "识别信号自动发送：红/绿灯取 AI traffic，行人取 person 检测；每 500 ms 一次。\n"
                      + "手动测试可勾选其它字段；HUD/ADAS 使能通过 legacy 0x608 保持可见。"
                color: "#8EAAB3"; font.family: root.cjk; font.pixelSize: 7; lineHeight: 1.3
            }
            Text {
                x: 24; y: 334; width: parent.width - 48; height: 20
                text: "状态: " + pcieHud.bridgeStatus + "  ·  applied seq " + pcieHud.lastAppliedSeq
                      + "  ·  error " + pcieHud.lastError
                color: pcieHud.bridgeOnline ? root.mint : root.amber
                font.family: root.num; font.pixelSize: 7
            }
            Text {
                x: 24; y: 354; width: parent.width - 48; height: 20
                text: "回复: " + (pcieHud.lastReply.length > 160 ? pcieHud.lastReply.substring(0, 160) + "..." : pcieHud.lastReply)
                color: "#6E8C95"; font.family: root.num; font.pixelSize: 6
                elide: Text.ElideRight
            }

            Rectangle {
                x: 24; y: 390; width: 176; height: 34; radius: 3
                color: "#0C3C50"; border.color: root.cyan; border.width: 1
                Text { anchors.centerIn: parent; text: "发送一次"; color: root.cyanHot; font.family: root.cjk; font.pixelSize: 10; font.weight: Font.DemiBold }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.sendTestHud()
                }
            }
            Rectangle {
                x: 214; y: 390; width: 176; height: 34; radius: 3
                color: root.autoHudSend ? "#0C3A33" : "#101C22"
                border.color: root.autoHudSend ? root.mint : "#33505B"; border.width: 1
                Text { anchors.centerIn: parent; text: root.autoHudSend ? "AI 自动发送 ON" : "AI 自动发送 OFF"; color: root.autoHudSend ? root.mint : "#8AA5AE"; font.family: root.cjk; font.pixelSize: 9 }
                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: root.autoHudSend = !root.autoHudSend }
            }
            Rectangle {
                x: 404; y: 390; width: 100; height: 34; radius: 3
                color: "#07161D"; border.color: "#33505B"; border.width: 0.8
                Text { anchors.centerIn: parent; text: "关闭"; color: "#9CB5BE"; font.family: root.cjk; font.pixelSize: 9 }
                MouseArea { anchors.fill: parent; onClicked: root.hudTestOpen = false }
            }
            Text {
                x: 520; y: 397; text: "T 打开/关闭 · Esc 关闭"
                color: "#5F7E88"; font.family: root.cjk; font.pixelSize: 6
            }
        }
    }

    // ------------------------------------------------------------------ footer
    Rectangle { x: 20; y: 696; width: 1240; height: 1; color: "#142E3A" }
    Text {
        x: 20; y: 701; height: 15; text: "PCIE RGB565 1280×720  ·  Mali-G52  ·  GLES 3.2  ·  " + video.sourceFps.toFixed(0) + " FPS IN"
        color: "#7195A0"; font.family: root.num; font.pixelSize: 6
    }
    Text {
        x: 500; y: 701; width: 550; height: 15
        text: "1 / 2 / 3 切换视图   ·   Q 退出"
        color: "#6F909B"; font.family: root.cjk; font.pixelSize: 7
        horizontalAlignment: Text.AlignHCenter
    }
    Text {
        x: 1036; y: 701; width: 224; height: 15
        text: video.aiOnline ? ("AI SIDECAR LIVE  ·  " + video.aiFps.toFixed(1) + " FPS") : "AI SIDECAR OFFLINE"
        color: video.aiOnline ? "#74A7A8" : "#A98A50"
        font.family: root.num; font.pixelSize: 6
        horizontalAlignment: Text.AlignRight
    }

    Loader {
        id: protocolLoader
        anchors.fill: parent
        active: root.hudTestOpen
        source: "ProtocolTestPanel.qml"
        z: 500
        onLoaded: {
            item.hud = pcieHud
            item.closeRequested.connect(function() { root.hudTestOpen = false })
        }
    }
    Loader {
        anchors.fill: parent
        active: root.fpgaOpen
        source: "FpgaControlPanel.qml"
        z: 520
        onLoaded: {
            item.control = root.fpgaControl
            item.closeRequested.connect(function() { root.fpgaOpen = false })
        }
    }
}
