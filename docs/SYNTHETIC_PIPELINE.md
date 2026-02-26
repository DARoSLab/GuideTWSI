# Synthetic Data Generation Pipeline

A comprehensive guide for generating photorealistic synthetic tactile walking surface indicator (TWSI) data using Unreal Engine 4 (UE4) and Microsoft AirSim.

## Overview

Our synthetic data pipeline generates high-quality, automatically annotated data of truncated domes in diverse sidewalk environments. The pipeline produces:

- RGB images at 1920×1080 resolution
- Pixel-wise semantic and instance segmentation masks
- Depth maps
- 2D bounding boxes (COCO-style)
- Camera intrinsic parameters

This approach generates data at a fraction of the cost of manual collection while providing perfect ground truth labels.

## Prerequisites

### Software

| Software | Version | Purpose |
|----------|---------|---------|
| Unreal Engine 4 | 4.27 | 3D rendering engine |
| AirSim | Latest | Simulation plugin for UE4 |
| Python | 3.10+ | Scripting and automation |
| `airsim` (pip) | Latest | Python API for AirSim |
| OpenCV | 4.8+ | Image processing |
| NumPy | 1.24+ | Array operations |

### Hardware

- GPU: NVIDIA RTX 3060 or better (for UE4 rendering)
- RAM: 32 GB recommended
- Storage: 100+ GB for environments and generated data

## Setup Guide

### Step 1: Install Unreal Engine 4

1. Download and install the [Epic Games Launcher](https://www.unrealengine.com/download)
2. Install UE4 version 4.27 through the launcher
3. Verify installation by creating and running a blank project

### Step 2: Build AirSim Plugin

```bash
# Clone AirSim
git clone https://github.com/microsoft/AirSim.git
cd AirSim

# Build on Linux
./setup.sh
./build.sh

# Build on Windows
build.cmd
```

Follow the [AirSim documentation](https://microsoft.github.io/AirSim/) for detailed platform-specific instructions.

### Step 3: Configure AirSim

Copy the provided settings file to the AirSim configuration directory:

```bash
# Linux
mkdir -p ~/Documents/AirSim
cp data_generation/airsim_settings.json ~/Documents/AirSim/settings.json

# Windows
copy data_generation\airsim_settings.json %USERPROFILE%\Documents\AirSim\settings.json
```

Key settings in `airsim_settings.json`:
- `SimMode`: "Multirotor" — drone-based data collection
- Camera resolution: 1920×1080 at 90° FOV
- Segmentation enabled for ground truth generation
- Collision passthrough enabled for unobstructed navigation

### Step 4: Download UE4 Environments

Download environment assets from [Fab](https://www.fab.com/) (formerly UE Marketplace). The paper uses 10 environments:

| Environment | Description | Key Features |
|-------------|-------------|--------------|
| City Park | Mixed urban park | Grass, gravel, pavement, trees, paths |
| Downtown West | Dense city center | Streets, curbs, vehicles, traffic lights |
| Suburban | Residential area | Sidewalks, driveways, fences |
| Urban District | Pedestrian zone | Crosswalks, plazas, storefronts |
| Campus | University setting | Walkways, plazas, open spaces |
| Residential | Quiet neighborhood | Sidewalks, front yards |
| Industrial | Commercial area | Loading docks, warehouses |
| Town Square | Public gathering space | Varied paving, fountains |
| Waterfront | Coastal setting | Boardwalks, promenades |
| Rural | Country setting | Informal paths, rural roads |

### Step 5: Install Custom Truncated Dome Assets

We create custom ADA-compliant truncated dome modules:

1. **Geometry**: Truncated domes modeled to ADA [2] and California Title 24 [3] specifications
   - Dome diameter: 23 mm (0.9 in)
   - Dome height: 6.4 mm (0.25 in)
   - Center-to-center spacing: 51–61 mm (2.0–2.4 in)
   - Panel dimensions: 610 mm × 610 mm (24 in × 24 in) minimum

2. **Materials**: High-resolution textures from third-party Tactile Blocks packs
   - Standard yellow
   - Red variants
   - Gray/concrete variants
   - White variants

3. **Placement**: Import dome modules into each UE4 scene at realistic locations
   - Curb ramps and crossings
   - Platform edges
   - Drop-off warnings
   - Transit stops

## Camera Trajectories

### Circular Orbit

The drone orbits around each truncated dome installation, capturing images from multiple angles:

```bash
python data_generation/orbit_navigator.py \
    --environment "CityPark" \
    --radius 2 \
    --altitude 0.3 \
    --speed 3 \
    --iterations 3 \
    --snapshots 30
```

This produces views from low to overhead angles, reflecting the variety of possible robot-mounted camera perspectives.

### Top-Down Sweep

A camera moves from an elevated position downward over the scene, simulating bottom-facing cameras on ground robots:

- Sweeps at multiple heights (0.3m to 2.0m)
- Varying pitch angles
- Grid-based coverage of each paving installation

### Camera Parameters

Along each path, we vary:
- **Height**: 0.3m – 2.0m above ground
- **Orientation**: Multiple pitch/yaw combinations
- **Speed**: 0.5 – 3.0 m/s for motion blur realism

## Ground Truth Generation

### Automatic Annotation Process

1. **Semantic Label Assignment**: Each object in UE4 is assigned a unique semantic label ID
2. **AirSim Rendering**: For each camera frame, AirSim simultaneously produces:
   - RGB image (Scene capture)
   - Segmentation mask (color-coded by object ID)
   - Depth map (planar depth in meters)
3. **Bounding Box Extraction**: 2D bounding boxes computed from segmentation masks by color matching
4. **Label Formatting**: Color-coded masks converted to standardized annotation formats

### Segmentation Color Mapping

Each scene object is assigned a unique RGB color in the segmentation image. The `color_to_object_map` in `orbit_navigator.py` maps these colors to semantic labels:

```python
color_to_object_map = {
    (153, 108, 6): "tactile block",
    (112, 105, 191): "pedestrian sign",
    (89, 121, 72): "tactile block",
}
```

### Environmental Variation

For each environment, we randomize:
- **Weather**: Sunny, overcast, rainy, snowy
- **Time of day**: Daytime, sunset, night
- **Sun position**: Random azimuth and elevation
- **Brightness**: Random intensity scaling
- **Fog/rain effects**: Variable density and intensity
- **Surface materials**: Texture and color randomization on paving surfaces

## Post-Processing Pipeline

After data collection, the raw AirSim outputs undergo:

1. **Format Conversion**: Segmentation masks → YOLO polygon / COCO JSON / SAM2 RLE formats
   ```bash
   python data_utils/format_converters.py --from airsim --to yolo --input raw_labels --output yolo_labels
   ```

2. **Dataset Splitting**: 88% train / 6% val / 6% test
   ```bash
   python data_utils/dataset_generator.py --image-dir images --label-dir labels --output-dir dataset
   ```

3. **Binary Mask Generation** (for models requiring binary masks):
   ```bash
   python data_utils/mask_generator.py --input json_dir --output mask_dir
   ```

## Output Directory Structure

```
<environment>_<id>/
└── Run_<n>/
    ├── rgb/
    │   ├── Env_00_1_0_FV_rgb.png      # Front-view RGB
    │   └── Env_00_1_0_TD_rgb.png      # Top-down RGB
    ├── seg/
    │   ├── Env_00_1_0_FV_segmentation.png
    │   └── Env_00_1_0_TD_segmentation.png
    ├── depth/
    │   ├── Env_00_1_0_FV_depth.png
    │   └── Env_00_1_0_TD_depth.png
    ├── rgb_bb/
    │   ├── Env_00_1_0_FV_rgb_bb.png    # RGB with bounding box overlays
    │   └── Env_00_1_0_TD_rgb_bb.png
    └── label/
        ├── Env_00_1_0_FV.json          # JSON bounding box annotations
        └── Env_00_1_0_TD.json
```

## Scaling Tips

- **Parallel collection**: Run multiple UE4 instances (one per environment) simultaneously
- **Batch rendering**: Use UE4's Movie Render Queue for higher-quality offline rendering
- **Cloud rendering**: Deploy on cloud GPU instances for large-scale generation
- **Variation scripts**: Automate environment randomization between collection runs

## Troubleshooting

| Issue | Solution |
|-------|----------|
| AirSim connection refused | Ensure UE4 editor is running with AirSim plugin loaded |
| Black segmentation masks | Check that `SegmentationSettings.InitMethod` is `"None"` and `OverrideExisting` is `true` |
| Drone position drift | Wait for stabilization (script handles this automatically) |
| Low-quality depth | Ensure `DepthPlanar` image type is used (not `DepthPerspective`) |
| Missing dome colors | Update `color_to_object_map` to match your scene's segmentation IDs |
| Out of memory | Reduce camera resolution in `airsim_settings.json` or use fewer snapshots per orbit |

## References

- [AirSim Documentation](https://microsoft.github.io/AirSim/)
- [Unreal Engine 4 Documentation](https://docs.unrealengine.com/4.27/)
- [ADA Standards for Accessible Design](https://www.ada.gov/)
