from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=p.read_text(encoding='utf-8')
s=s.replace('''        if (kind === "LEAD_RAPID_CLOSING") return "前车快速接近"
        if (kind === "PEDESTRIAN_CROSSING_PATH") return "行人进入预测路径"
        if (kind === "LEAD_BRAKE_LAMP_RISE") return "前车制动灯变化"
        if (kind === "SUSPECTED_BRAKING_WITH_CLOSING") return "制动灯变化且快速接近"
        return "感知风险"''','''        if (kind === "LEAD_RAPID_CLOSING") return "前车快速接近"
        if (kind === "PEDESTRIAN_CROSSING_PATH") return "行人进入预测路径"
        if (kind === "PEDESTRIAN_IN_PATH") return "前方行人"
        if (kind === "RED_SIGNAL_CONFIRMED") return "确认红灯 · 注意信号"
        if (kind === "LEAD_BRAKE_LAMP_RISE") return "前车制动灯变化"
        if (kind === "SUSPECTED_BRAKING_WITH_CLOSING") return "制动灯变化且快速接近"
        return "感知风险"''')
s=s.replace('''        if (video.aiOnline) {
            var hs = video.aiHazards || []
            for (var i = 0; i < hs.length && i < 3; ++i)
                rows.push({ label: hazardLabel(hs[i]), live: true, value: "实时" })
            return rows
        }
        return [
            { label: "前车距离较近", live: false, value: "—" },
            { label: "右侧行人注意", live: false, value: "—" },
            { label: "无效变道风险", live: false, value: "—" }
        ]''','''        if (video.aiOnline) {
            var hs = (video.aiHazards || []).slice()
            var priority = { "RED_SIGNAL_CONFIRMED": 0, "PEDESTRIAN_CROSSING_PATH": 1,
                             "PEDESTRIAN_IN_PATH": 1, "LEAD_RAPID_CLOSING": 2,
                             "SUSPECTED_BRAKING_WITH_CLOSING": 3, "LEAD_BRAKE_LAMP_RISE": 4 }
            hs.sort(function(a, b) { return (priority[a] ?? 9) - (priority[b] ?? 9) })
            for (var i = 0; i < hs.length && i < 3; ++i)
                rows.push({ label: hazardLabel(hs[i]), live: true, value: "实时" })
            return rows
        }
        return [ { label: "AI 源失联", live: false, value: "无数据" } ]''')
s=s.replace('''    function hazardBadge() {
        if (!video.aiOnline) return "设计预览"''','''    function hazardBadge() {
        if (!video.aiOnline) return "离线"''')
s=s.replace('''        if (navValid() && video.aiNavigationRoad) return video.aiNavigationRoad
        if (!video.aiOnline) return "AI 未连接"
        return "等待 Q2 OCR"''','''        if (navValid() && video.aiNavigationRoad) return video.aiNavigationRoad
        if (!video.aiOnline) return "AI 未连接"
        if (video.aiNavigationState === "STOPPED") return "导航已停止更新"
        if (video.aiNavigationState === "STALE") return "导航暂不可读"
        if (video.aiNavigationState === "UNREADABLE") return "导航暂不可读"
        return "等待 Q2 OCR"''')
s=s.replace('''        if (!video.aiOnline) return "AI 未连接"
        return "等待导航 OCR · Q" + (navQuadrant + 1)''','''        if (!video.aiOnline) return "AI 未连接"
        if (video.aiNavigationState === "UNREADABLE") return "导航暂不可读 · Q" + (navQuadrant + 1)
        if (video.aiNavigationState === "STALE") return "导航过期 · Q" + (navQuadrant + 1)
        return "等待导航 OCR · Q" + (navQuadrant + 1)''')
s=s.replace('''                text: root.navValid() ? "AI 实时" : "设计预览"
                color: root.navValid() ? root.mint : root.amber''','''                text: root.navValid() ? "AI 实时"
                      : (video.aiNavigationState === "STALE" ? "已过期"
                      : (video.aiNavigationState === "UNREADABLE" ? "暂不可读" : "等待"))
                color: root.navValid() ? root.mint : root.amber''')
# lane card neutral
start=s.index('id: laneCard')
end=s.index('    // --------------------------------------------------------------- bottom dock', start)
block=s[start:end]
block=block.replace('text: "设计预览"','text: "未接入"')
block=block.replace('color: "#2B220E"; border.color: "#6A542A"; border.width: 0.6','color: "#0B1B22"; border.color: "#355E6C"; border.width: 0.6')
block=block.replace('text: "当前车道"; color: "#78949E"','text: "当前车道"; color: "#78949E"')
block=block.replace('text: "直行"; color: root.mint','text: "—"; color: "#78949E"')
block=block.replace('text: "左侧车道"; color: "#78949E"','text: "左侧车道"; color: "#78949E"')
block=block.replace('text: "可变道"; color: "#9FE4D0"','text: "—"; color: "#78949E"')
block=block.replace('text: "右侧车道"; color: "#78949E"','text: "右侧车道"; color: "#78949E"')
block=block.replace('text: "禁止变道"; color: root.red','text: "—"; color: "#78949E"')
block=block.replace('text: "车道识别未接入 · 仅示意"','text: "车道信息未接入"')
block=block.replace('color: "#9A8452"','color: "#8A9AA0"')
s=s[:start]+block+s[end:]
p.write_text(s,encoding='utf-8')
print('qml patched')
