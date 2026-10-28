# Native ego-vehicle illustration

`ego_car_rear.png` is a static vehicle symbol for the competition cockpit. It shows a complete silver-gray vehicle from above and behind, with its front pointing toward the top of the display.

It is **not** a sensor estimate of ego position, heading, speed, lane choice, or a detected vehicle model. The UI must keep that distinction explicit.

## Source and license

The mesh and material definition are copied without geometry changes from the existing local `@dreamview/dreamview-carviz` **1.1.4** package, `assets/models/car.obj` and `car.mtl`. The package's MIT license is reproduced in full in `CARVIZ_LICENSE.txt`; it states **Copyright (C) 2017–2023 hxj1234**. The package has no separate license declaration for this mesh. This is package-level provenance, not a claim of original vehicle-model authorship or an endorsement by an automobile manufacturer.

`ego_car_source.json` records file hashes, package version and modifications. The silver-gray material colors, studio illumination and soft ground shadow were produced by the offline renderer in `../ego_asset.py`; no remote model or generated vehicle image was downloaded.

## Rebuild and use

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe tools/competition_console/ego_asset.py
```

UI code:

```python
from tools.competition_console.ego_asset import load_ego_asset
car = load_ego_asset((300, 320))
canvas.alpha_composite(car, (75, 355))
```

Normal UI rendering only loads the bundled PNG and caches each requested size. Mesh parsing and rasterization run only when the rebuild command is invoked. Do not modify the image returned by the cached loader.
