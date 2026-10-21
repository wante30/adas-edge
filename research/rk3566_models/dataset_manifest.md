# 训练数据 manifest 约定

`tools/audit_adas_dataset_manifest.py` 是**声明性前置审计**，可检查11类标注覆盖声称与视频级拆分泄漏；不能读图查框、不能证明人工作业标注正确，也不能自动解决部分标注导致的背景损失。

必填 CSV 字段：
- `sample_id`: 全局唯一帧ID（建议 `dataset:sequence:frame`）
- `dataset`: 源数据集；必须具体到BDD/DTLD等
- `sequence_id`: 原视频或同一连续采集行程ID，**同一序列不得跨train/val/test**
- `split`: train/val/test
- `classes_exhaustive`: 该图片已逐个完整标注的类别，使用 `|` 分隔；只有核对过“此类出现的目标全被标注”才能声明，不是此类恰好出现过就填写
- `license`: 经核验的数据许可标识/内部记录
- `source_uri`: 素材来源URI或路径（切勿泄漏授权链接）

例子（假设该图片确实对所有11类做了完整标注）：

```csv
sample_id,dataset,sequence_id,split,classes_exhaustive,license,source_uri
sample-a,internal,trip-01,train,person|bicycle|motorcycle|car|bus|truck|red_light|yellow_light|green_light|traffic_light_other|crosswalk,operator_verified,local://clipA/frame0001
sample-b,internal,trip-02,val,person|bicycle|motorcycle|car|bus|truck|red_light|yellow_light|green_light|traffic_light_other|crosswalk,operator_verified,local://clipB/frame0001
```

示例仅演示格式，不代表对应图片确实经过标注。

运行：
```bash
python tools/audit_adas_dataset_manifest.py --manifest data/adas_model_manifest.csv --output reports/manifest_audit.json
pytest -q tests/test_adas_dataset_manifest.py
```

注意：单一来源只标6类时，这个严格11类审计会FAIL，**这是预期行为**。不能在普通检测器训练中把“未标4种交通灯和crosswalk”偷偷当背景。未来如果开发出可验证的class-aware ignore策略，可另设独立数据路径与测试。
