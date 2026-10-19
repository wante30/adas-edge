import QtQuick 2.12

// Compact live BEV projection for the selected main camera. It is deliberately
// a schematic projection, not a calibrated ground-plane transform: the purpose
// is to reflect the current AI detections and selected quadrant in the dock.
Item {
    id: bev
    property var boxes: []
    property int activeQuadrant: 1
    property bool online: false

    function targetTone(name) {
        if (name === "person" || name === "bicycle") return "#45F3B0"
        if (name === "traffic light") return "#FFBC5C"
        return "#3CE3FF"
    }

    Canvas {
        id: canvas
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            if (width < 20 || height < 20) return
            var w = width, h = height, cx = w * 0.5
            var horizon = h * 0.20, bottom = h - 5

            ctx.fillStyle = "#031017"
            ctx.fillRect(0, 0, w, h)

            // Perspective road.
            ctx.beginPath()
            ctx.moveTo(cx - 13, horizon)
            ctx.lineTo(cx + 13, horizon)
            ctx.lineTo(w - 4, bottom)
            ctx.lineTo(4, bottom)
            ctx.closePath()
            ctx.fillStyle = "#091D25"
            ctx.fill()
            ctx.strokeStyle = "rgba(66,112,125,0.55)"
            ctx.lineWidth = 1
            ctx.stroke()

            // Lane rails.
            ctx.strokeStyle = "rgba(207,231,235,0.68)"
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(cx - 6, horizon); ctx.lineTo(cx - w * 0.36, bottom)
            ctx.moveTo(cx + 6, horizon); ctx.lineTo(cx + w * 0.36, bottom)
            ctx.stroke()

            // Dashed center lane.
            ctx.strokeStyle = "rgba(126,158,166,0.55)"
            ctx.lineWidth = 0.8
            for (var s = 0; s < 7; ++s) {
                var t0 = 0.18 + s * 0.115
                var t1 = t0 + 0.055
                var y0 = horizon + (bottom - horizon) * t0
                var y1 = horizon + (bottom - horizon) * t1
                var a = 1 - t0 * 0.7
                ctx.beginPath()
                ctx.moveTo(cx - 2 + (1 - a) * 2, y0)
                ctx.lineTo(cx - 2 + (1 - a) * 2, y1)
                ctx.stroke()
            }

            // Ego lane / path.
            ctx.beginPath()
            ctx.moveTo(cx - 4, horizon + 2)
            ctx.lineTo(cx + 4, horizon + 2)
            ctx.lineTo(cx + w * 0.15, bottom)
            ctx.lineTo(cx - w * 0.15, bottom)
            ctx.closePath()
            ctx.fillStyle = "rgba(38,151,255,0.32)"
            ctx.fill()
            ctx.strokeStyle = "rgba(90,190,255,0.55)"
            ctx.lineWidth = 0.8
            ctx.stroke()

            // Radar arcs.
            ctx.strokeStyle = "rgba(60,227,255,0.14)"
            for (var r = 1; r <= 3; ++r) {
                ctx.beginPath()
                ctx.arc(cx, bottom, r * w * 0.16, Math.PI, 2 * Math.PI)
                ctx.stroke()
            }

            // Ego car.
            ctx.fillStyle = "#08161D"
            ctx.strokeStyle = "#55E6F4"
            ctx.lineWidth = 0.9
            ctx.fillRect(cx - 5, bottom - 15, 10, 17)
            ctx.strokeRect(cx - 5, bottom - 15, 10, 17)

            // Detections projected from the selected main camera.
            var list = bev.boxes || []
            for (var i = 0; i < list.length && i < 10; ++i) {
                var b = list[i]
                var hFrac = Math.max(0.01, Math.min(0.45, b.height / 720))
                var p = Math.max(0.20, Math.min(0.92, 0.18 + hFrac * 2.35))
                var y = horizon + (bottom - horizon) * p
                var nx = (b.x + b.width * 0.5) / 1280
                var x = cx + (nx - 0.5) * w * 0.86 * p
                var bw = 4 + p * 17
                var bh = 3 + p * 11
                var tone = targetTone(b.className)
                ctx.globalAlpha = 0.82
                ctx.fillStyle = "rgba(5,13,18,0.8)"
                ctx.fillRect(x - bw / 2, y - bh, bw, bh)
                ctx.strokeStyle = tone
                ctx.lineWidth = i === 0 ? 1.2 : 0.8
                ctx.strokeRect(x - bw / 2 + 0.5, y - bh + 0.5, bw - 1, bh - 1)
                ctx.fillStyle = tone
                ctx.globalAlpha = 0.9
                ctx.fillRect(x - bw / 2, y - bh, Math.min(4, bw * 0.4), 1.5)
                ctx.globalAlpha = 1.0
            }

            if (!bev.online) {
                ctx.fillStyle = "rgba(3,16,23,0.55)"
                ctx.fillRect(0, 0, w, h)
                ctx.fillStyle = "#8A7A55"
                ctx.font = "8px Noto Sans CJK SC"
                ctx.fillText("AI 未连接", 10, h - 8)
            }
        }
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
    }

    onBoxesChanged: canvas.requestPaint()
    onOnlineChanged: canvas.requestPaint()
    onActiveQuadrantChanged: canvas.requestPaint()
    Component.onCompleted: canvas.requestPaint()
}
