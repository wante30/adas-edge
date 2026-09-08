# Competition evaluation download plan

目标：下载 6–10 段、每段 20–60 秒、总量优先小于 2 GB（最好小于 1 GB）的真实道路视频。当前不下载完整数据集。

## Proposed minimal set

| File | Source | Clip/video ID | Official acquisition | Expected size | Scenario |
|---|---|---|---|---|---|
| `01_normal_road.mp4` | BDD100K | Select one validation video ID after portal access | Log in/accept BDD terms, select ID, download original clip | UNKNOWN; estimate 50–250 MB | normal road |
| `02_intersection.mp4` | BDD100K | Select one validation ID with intersection metadata | Same as above | UNKNOWN | intersection daylight |
| `03_traffic_light.mp4` | BDD100K | Select one ID with traffic-light labels | Same as above | UNKNOWN | traffic light |
| `04_pedestrian.mp4` | BDD100K | Select one ID with person/crossing content | Same as above | UNKNOWN | pedestrian crossing |
| `05_dense_traffic.mp4` | BDD100K or UA-DETRAC | BDD ID preferred; UA-DETRAC sequence fallback | Use official portal/archive; no direct-link bypass | UNKNOWN | dense traffic |
| `06_dashcam.mp4` | BDD100K | Select moving-camera ID | Same as above | UNKNOWN | dashcam |
| `07_night.mp4` | BDD100K | Select night ID after metadata inspection | Same as above | UNKNOWN | night road |
| `08_obstacle.mp4` | BDD100K/UA-DETRAC | Only if labels/content confirm cone or obstacle | Verify terms and clip metadata first | UNKNOWN | road obstacle |

## Acquisition gate

Before downloading, record the portal URL, account/registration requirement, accepted license terms, exact video ID, source resolution/FPS, duration, and byte size. If the portal only offers bulk archives or requires manual application, stop and report the required human step. Do not use random YouTube clips as formal benchmark material.

Place approved files under `data/competition_eval/raw/`, preserve originals, and add one row per file to `data/competition_eval/manifest.csv` with scenario, lighting, camera motion, and notes. Only after the manifest is complete should the same files be run through the 320 and 416 evaluator.

Current status: **NOT DOWNLOADED**. Missing prerequisite is authorized portal access and exact clip selection; no credentials or license acceptance were bypassed.

## 2026-09-07 update

The downloaded Hugging Face BDD100K validation-only mirror is image-only (10,000 images + CSV), so it is recorded as an external image holdout and not added as a video row in data/competition_eval/manifest.csv. Original BDD100K driving videos still require official portal access and license acceptance. No video benchmark is claimed from the mirror.
