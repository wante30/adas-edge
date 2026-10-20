import QtQuick 2.12

// The original Tesla-BevCam-Clone visual asset, shared by the compact dock
// and the enlarged spatial page. It is an annotated DEMO animation, not a
// live-oriented 3D reconstruction.
Item {
    id: view
    property bool compact: false

    AnimatedImage {
        anchors.fill: parent
        source: "qrc:/assets/tesla_bev_demo.gif"
        fillMode: view.compact ? Image.PreserveAspectCrop : Image.PreserveAspectCrop
        playing: view.visible
        cache: true
        smooth: true
        asynchronous: false
    }
}
