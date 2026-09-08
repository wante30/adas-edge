# Public Road Video Dataset Survey

调查日期：2026-09-06。目标是为 320/416 detector 验收获取 6–10 段短视频，而不是训练集。网页可访问性按本次 HTTP/官方页面检查记录；下载权限和许可证以数据集当前条款为准。

## Candidate Table

| Dataset | Video | Night | Traffic Light | Pedestrian | Intersection | Obstacle | Size | Partial Download | License | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| BDD100K | YES, 100K×约40 s | YES (diverse conditions claimed; exact ratio UNKNOWN) | Labels include traffic light | YES | YES/likely, exact clip index requires metadata | Road-object detection task; cone/obstacle coverage UNKNOWN | 100K videos, >1000 h; total download size UNKNOWN | Official account/download workflow; arbitrary single-clip URL UNKNOWN | Must accept current BDD terms; redistribution/demo scope requires confirmation | Official GitHub/docs accessible; download not anonymously verified |
| UA-DETRAC | YES, traffic-camera sequences | Lighting/weather variation; night ratio UNKNOWN | NO dedicated class confirmed | NO dedicated pedestrian labels | Traffic junction/road scenes | Vehicle-focused; no cone class confirmed | Official page describes 100 videos/traffic surveillance; exact current archive size UNKNOWN | Download/request workflow, single clip availability UNKNOWN | Academic/research terms must be checked on current page | Official University at Albany page accessible |
| KITTI Raw/Tracking | YES, stereo driving sequences | Mostly daylight; night coverage UNKNOWN | No traffic-light labels in benchmark annotations | Limited/varies by task | Urban driving/intersections present | No obstacle/cone labels | Raw data is multi-GB; exact current total UNKNOWN | Individual drive archives are listed; small subset possible | KITTI non-commercial research terms | Official cvlibs pages accessible; not ideal for class coverage |
| Cityscapes | Short driving sequences exist, but public benchmark is primarily frames | YES/limited | Traffic-light/sign labels exist in images | YES | YES | No dedicated road-obstacle video class | 5,000 fine images + sequence data; exact video archive size UNKNOWN | Registration/download; frame extraction is easier than clip selection | Cityscapes license/registration terms | Official site accessible; video suitability limited |
| nuScenes | YES, 20 s scenes from driving logs | YES | Traffic-light/sign/object categories; exact traffic-light annotation semantics vary | YES | YES | Cones/barriers included as object classes | 1,000 scenes, 20 s each; full data large | Download requires account/terms; scene-level subset possible | nuScenes terms require acceptance; research/demo use must follow them | Official site accessible; high-value but heavy and multi-camera format |
| Waymo Open Dataset | YES, segmented driving clips | YES | Traffic-light/sign and road-object labels | YES | YES | Cones/obstacles partly covered | Very large; exact current download size depends on split | Cloud/object-level access and account required | Waymo Dataset License; commercial/demo restrictions require review | Official site availability and current quota need account check |
| Argoverse 2 | Sensor logs with camera streams, not simple MP4 clips | YES/UNKNOWN | Traffic-light labels UNKNOWN | YES | YES | 3D objects include vehicles; cone coverage UNKNOWN | Large; split-dependent | Select log possible after registration | Argoverse 2 license | Official site accessible; preprocessing required |
| AI City Challenge | Video tracks exist, often fixed cameras/traffic surveillance | Scenario-dependent | Scenario-dependent | Scenario-dependent | YES | Scenario-dependent | Challenge releases are split-specific and can be large | Usually registration/request; exact sample links UNKNOWN | Challenge-specific terms | Official site returned 403 in this environment; availability UNKNOWN |

## Detailed Notes

### BDD100K (primary candidate)

Official sources: [BDD100K GitHub](https://github.com/bdd100k/bdd100k), [BDD100K homepage](https://www.bdd100k.com/), [BDD100K documentation](https://doc.bdd100k.com/), and the CVPR paper linked by the repository. The official README states 100K driving videos, each 40 seconds, high resolution, over 1,000 hours, and ten tasks including road-object detection, lane detection, and tracking. This directly matches dashcam-style acceptance better than fixed-camera datasets.

The official repository does not publish an anonymous per-video download URL in the README. Current access flow, account requirement, exact train/val video counts, and whether a user can select arbitrary validation IDs must be confirmed through the BDD portal after login. Do not scrape or bypass that flow. BDD is the best primary source if the team can accept its current terms and obtain a small set of video IDs.

### UA-DETRAC (supplementary candidate)

Official source: [University at Albany CNSE/CV lab page](https://www.albany.edu/cnse/research/computer-vision-machine-learning-lab), which is the current destination of the historical `detrac-db.rit.albany.edu` URL. It is useful for dense traffic and vehicle scale/occlusion stress, but it is traffic surveillance footage rather than moving-camera dashcam. It should supplement BDD only for vehicle robustness; it cannot validate pedestrian or traffic-light behavior without additional labels.

### KITTI Raw/Tracking

Official source: [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php). The page remains reachable and lists individual driving recordings, making a small drive-level download plausible. It is useful as a clean daytime moving-camera sanity check, but its labels and scene distribution do not cover the requested competition class set comprehensively. Treat license and redistribution restrictions as binding.

### Other candidates

Cityscapes is mainly an image/sequence benchmark and requires registration. nuScenes, Waymo and Argoverse are technically rich but have large archives, account/terms requirements, and preprocessing overhead; they are poor choices for an immediate <2 GB MP4 acceptance pack. AI City availability could not be verified because the official site returned HTTP 403 during this check.

## Recommended Choice

**Primary dataset: BDD100K.** It is the only candidate here whose official description directly provides large-scale, 40-second driving videos with diverse weather/environment and road-object tasks. Use a manually selected, license-compliant validation subset after portal access is granted.

**Supplementary dataset: UA-DETRAC** for dense traffic and vehicle occlusion, or KITTI Raw if moving-camera consistency is more important. UA-DETRAC is recommended for the requested stress case, with the explicit limitation that it does not replace BDD for person/traffic-light evaluation.

A public Hugging Face validation-only mirror was downloaded for image-level pre-acceptance: `hirundo-io/bdd100k-validation-only`, 574,910,510 bytes, 10,000 validation images plus CSV boxes. It is image-only and does not satisfy the original video requirement. The mirror carries a BSD 3-Clause file, while BDD100K source-data terms and demonstration rights still require review. No BDD100K original video was downloaded. Exact video clip IDs, direct URLs, night/intersection ratios, and video demonstration rights remain UNKNOWN until portal access is available.
