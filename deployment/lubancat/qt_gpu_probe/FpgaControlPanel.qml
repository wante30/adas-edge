import QtQuick 2.12

Item {
    id: panel
    property var control
    signal closeRequested

    Rectangle { anchors.fill: parent; color: "#E6030A0E" }
    Rectangle {
        x: 40; y: 40; width: 1200; height: 640
        color: "#071820"; border.color: "#2A6576"; border.width: 1
    }
    Text {
        x: 64; y: 58; text: "FPGA IMAGE / ALGORITHM CONTROL"
        color: "#8CF5FF"; font.family: "DejaVu Sans"; font.pixelSize: 15
    }
    Text {
        x: 64; y: 84; text: "QUERY → validate → shadow → APPLY → status → QUERY"
        color: "#6F95A0"; font.family: "DejaVu Sans"; font.pixelSize: 7
    }
    Rectangle {
        x: 64; y: 102; width: 1152; height: 1; color: "#23434E"
    }
    Text {
        x: 64; y: 114; width: 1120; height: 390
        text: control ? control.summary : "NO SERVICE"
        color: "#B7D4DC"; font.family: "DejaVu Sans"; font.pixelSize: 7
        wrapMode: Text.WrapAnywhere
    }
    Row {
        x: 64; y: 520; spacing: 12
        Rectangle { width: 125; height: 36; color: "#0B302A"; border.color: "#45F3B0"
            Text { anchors.centerIn: parent; text: "READ"; color: "#45F3B0"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
            MouseArea { anchors.fill: parent; onClicked: if (panel.control) panel.control.query() }
        }
        Rectangle { width: 145; height: 36; color: "#0B2630"; border.color: "#3CE3FF"
            Text { anchors.centerIn: parent; text: "ISP RESET"; color: "#8CF5FF"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
            MouseArea { anchors.fill: parent; onClicked: if (panel.control) panel.control.applyIspReset() }
        }
        Rectangle { width: 135; height: 36; color: "#0B2630"; border.color: "#3CE3FF"
            Text { anchors.centerIn: parent; text: "MASTER ON"; color: "#8CF5FF"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
            MouseArea { anchors.fill: parent; onClicked: if (panel.control) panel.control.applyIspMaster(true) }
        }
        Rectangle { width: 135; height: 36; color: "#101C22"; border.color: "#33505B"
            Text { anchors.centerIn: parent; text: "MASTER OFF"; color: "#9CB5BE"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
            MouseArea { anchors.fill: parent; onClicked: if (panel.control) panel.control.applyIspMaster(false) }
        }
    }
    Row {
        x: 64; y: 568; spacing: 12
        Rectangle { width: 150; height: 34; color: "#0B302A"; border.color: "#45F3B0"
            Text { anchors.centerIn: parent; text: "ALGO DEHAZE"; color: "#45F3B0"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
            MouseArea { anchors.fill: parent; onClicked: if (panel.control) panel.control.applyAlgo(true, false, false, false) }
        }
        Rectangle { width: 170; height: 34; color: "#0B2630"; border.color: "#3CE3FF"
            Text { anchors.centerIn: parent; text: "ALGO BLUR"; color: "#8CF5FF"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
            MouseArea { anchors.fill: parent; onClicked: if (panel.control) panel.control.applyAlgo(false, true, true, false) }
        }
        Rectangle { width: 150; height: 34; color: "#0B2630"; border.color: "#3CE3FF"
            Text { anchors.centerIn: parent; text: "ALGO EDGE"; color: "#8CF5FF"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
            MouseArea { anchors.fill: parent; onClicked: if (panel.control) panel.control.applyAlgo(false, false, false, true) }
        }
        Rectangle { width: 150; height: 34; color: "#2B1016"; border.color: "#FF5064"
            Text { anchors.centerIn: parent; text: "CLOSE"; color: "#FF8A96"; font.family: "DejaVu Sans"; font.pixelSize: 9 }
            MouseArea { anchors.fill: parent; onClicked: panel.closeRequested() }
        }
    }
    Text {
        x: 64; y: 625; text: "STATUS: " + (control ? control.status : "NO SERVICE")
        color: "#8EAAB3"; font.family: "DejaVu Sans"; font.pixelSize: 7
    }
}
