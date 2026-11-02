"""Physical camera identity stays separate from the legacy ROAD_Q1 mode name."""
def quadrant(context):
    return int(context[3]) if len(context) > 3 else 0

def source_id(context):
    return "Q%d" % (quadrant(context) + 1)

def local_detection(detection, context):
    result = dict(detection)
    if context[0] == "MOSAIC_FULL":
        q = quadrant(context)
        ox, oy = (q % 2) * 640, (q // 2) * 360
        x1,y1,x2,y2 = detection["bbox"]
        result["bbox"] = [x1-ox,y1-oy,x2-ox,y2-oy]
    return result
