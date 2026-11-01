import QtQuick 2.12

// A static cut-corner glass card. Canvas is used only here; it paints on size
// or colour changes and is cached by the scene graph otherwise.
Item {
    id: panel

    property color accent: "#3CE3FF"
    property color fillTop: "#0A1E29"
    property color fillBottom: "#040E16"
    property real fillOpacity: 0.94
    property real borderOpacity: 0.46
    property real cut: 12
    property bool glow: false
    property bool showTicks: true

    Canvas {
        id: canvas
        anchors.fill: parent

        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            if (width < 4 || height < 4)
                return

            var c = Math.min(panel.cut, width * 0.22, height * 0.22)
            var w = width
            var h = height

            function path() {
                ctx.beginPath()
                ctx.moveTo(c, 0)
                ctx.lineTo(w, 0)
                ctx.lineTo(w, h - c)
                ctx.lineTo(w - c, h)
                ctx.lineTo(0, h)
                ctx.lineTo(0, c)
                ctx.closePath()
            }

            var fill = ctx.createLinearGradient(0, 0, w, h)
            fill.addColorStop(0, Qt.rgba(panel.fillTop.r, panel.fillTop.g, panel.fillTop.b, panel.fillOpacity))
            fill.addColorStop(0.62, Qt.rgba(panel.fillTop.r * 0.76, panel.fillTop.g * 0.78, panel.fillTop.b * 0.82, panel.fillOpacity))
            fill.addColorStop(1, Qt.rgba(panel.fillBottom.r, panel.fillBottom.g, panel.fillBottom.b, panel.fillOpacity))

            path()
            ctx.fillStyle = fill
            ctx.fill()

            // Glass sheen across the upper third.
            var sheen = ctx.createLinearGradient(0, 0, w, 0)
            sheen.addColorStop(0, "rgba(210,250,255,0.00)")
            sheen.addColorStop(0.35, "rgba(210,250,255,0.035)")
            sheen.addColorStop(1, "rgba(210,250,255,0.00)")
            ctx.fillStyle = sheen
            ctx.fillRect(c, 0, w - c, Math.min(h * 0.34, 54))

            if (panel.glow) {
                ctx.lineWidth = 8
                ctx.strokeStyle = Qt.rgba(panel.accent.r, panel.accent.g, panel.accent.b, 0.035)
                path()
                ctx.stroke()
                ctx.lineWidth = 3.5
                ctx.strokeStyle = Qt.rgba(panel.accent.r, panel.accent.g, panel.accent.b, 0.09)
                path()
                ctx.stroke()
            }

            ctx.lineWidth = 1
            ctx.strokeStyle = Qt.rgba(panel.accent.r, panel.accent.g, panel.accent.b, panel.borderOpacity)
            path()
            ctx.stroke()

            ctx.strokeStyle = "rgba(64,124,143,0.25)"
            ctx.beginPath()
            ctx.moveTo(c + 2, h)
            ctx.lineTo(w, h)
            ctx.stroke()

            // Narrow luminous lip on the top-left edge.
            var lip = ctx.createLinearGradient(c, 0, w * 0.65, 0)
            lip.addColorStop(0, Qt.rgba(panel.accent.r, panel.accent.g, panel.accent.b, 0.60))
            lip.addColorStop(0.5, Qt.rgba(panel.accent.r, panel.accent.g, panel.accent.b, 0.08))
            lip.addColorStop(1, Qt.rgba(panel.accent.r, panel.accent.g, panel.accent.b, 0.00))
            ctx.strokeStyle = lip
            ctx.lineWidth = 1.2
            ctx.beginPath()
            ctx.moveTo(c, 0.6)
            ctx.lineTo(w * 0.72, 0.6)
            ctx.stroke()

            if (panel.showTicks) {
                ctx.strokeStyle = Qt.rgba(panel.accent.r, panel.accent.g, panel.accent.b, 0.28)
                ctx.lineWidth = 1
                ctx.beginPath()
                ctx.moveTo(14, h + 2.5)
                ctx.lineTo(30, h + 2.5)
                ctx.moveTo(w - 30, -2.5)
                ctx.lineTo(w - 14, -2.5)
                ctx.stroke()
            }
        }

        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
    }

    onAccentChanged: canvas.requestPaint()
    onFillTopChanged: canvas.requestPaint()
    onFillBottomChanged: canvas.requestPaint()
    onFillOpacityChanged: canvas.requestPaint()
    onBorderOpacityChanged: canvas.requestPaint()
    onCutChanged: canvas.requestPaint()
    onGlowChanged: canvas.requestPaint()
    Component.onCompleted: canvas.requestPaint()
}
