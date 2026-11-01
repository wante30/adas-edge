import QtQuick 2.12
Item {
    id: panel
    property var hud
    signal closeRequested
    onHudChanged: if (hud) hud.queryProtocol()
    Rectangle { anchors.fill: parent; color: "#E6030A0E" }
    Rectangle { x: 40; y: 40; width: 1200; height: 640; color: "#071820"; border.color: "#2A6576" }
    Text { x: 64; y: 64; text: "PCIe COMPLETE PROTOCOL TEST"; color: "#8CF5FF"; font.family: "DejaVu Sans"; font.pixelSize: 16 }
    Column {
        x: 64; y: 100; width: 1152; height: 420; spacing: 2
        Repeater {
            model: hud ? hud.protocolLines : []
            delegate: Text {
                width: 1152
                text: modelData
                color: "#B7D4DC"; font.family: "DejaVu Sans"; font.pixelSize: 6
            }
        }
    }
    Rectangle { x: 1100; y: 56; width: 110; height: 34; color: "#2B1016"; border.color: "#FF5064"
        Text { anchors.centerIn: parent; text: "CLOSE"; color: "#FF8A96"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
        MouseArea { anchors.fill: parent; onClicked: panel.closeRequested() }
    }
}
