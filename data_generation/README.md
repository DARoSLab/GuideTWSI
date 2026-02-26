# Synthetic Data Generation Pipeline

This directory contains the tools for generating synthetic tactile walking surface indicator (TWSI) data using Unreal Engine 4 (UE4) and Microsoft AirSim.

## Prerequisites

- **Unreal Engine 4** (4.27 recommended)
- **AirSim Plugin** ([Microsoft AirSim](https://github.com/microsoft/AirSim)) compiled for UE4
- **Python 3.10+** with the `airsim` package (`pip install airsim`)
- **UE4 Environment Assets** from [Fab](https://www.fab.com/) (formerly UE Marketplace)

## Setup

1. **Install AirSim**: Follow the [AirSim build guide](https://microsoft.github.io/AirSim/build_linux/) for your platform.

2. **Configure AirSim settings**: Copy `airsim_settings.json` to your AirSim settings directory:
   ```bash
   # Linux
   cp airsim_settings.json ~/Documents/AirSim/settings.json
   # Windows
   copy airsim_settings.json %USERPROFILE%\Documents\AirSim\settings.json
   ```

3. **Prepare UE4 environments**: Download and set up the UE4 environment assets. Place custom truncated dome modules into each scene according to the paper's specifications.

4. **Install Python dependencies**:
   ```bash
   pip install airsim numpy opencv-python
   ```

## UE4 Environments

The paper uses 10 diverse UE4 environments from Fab to simulate real-world sidewalk scenarios:

| # | Environment | Description |
|---|-------------|-------------|
| 1 | City Park | Mixed terrain with grass, gravel, pavement, trees, and paths |
| 2 | Downtown West | Streets, curbs, vehicles, traffic lights, and benches |
| 3 | Suburban | Residential sidewalks with driveways and fences |
| 4 | Urban District | Dense urban area with crosswalks and pedestrian zones |
| 5 | Campus | University-style walkways and plazas |
| 6 | Residential | Quiet neighborhood streets and sidewalks |
| 7 | Industrial | Loading docks, warehouses, and service roads |
| 8 | Town Square | Public spaces with varied paving patterns |
| 9 | Waterfront | Boardwalks and coastal paths |
| 10 | Rural | Country roads and informal paths |

Each scene is rendered under multiple lighting/weather conditions (sunny, overcast, rainy, snowy, daytime, sunset, night) using UE4's Directional Lights.

## Usage

Run the orbit navigator to collect data from a UE4 environment:

```bash
python orbit_navigator.py \
    --environment "CityPark" \
    --run 1 \
    --radius 2 \
    --altitude 0.3 \
    --speed 3 \
    --iterations 3 \
    --snapshots 30 \
    --center "0,0"
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--environment` | str | "Test" | Name of the UE4 environment |
| `--run` | int | 1 | Run number for output directory naming |
| `--radius` | float | 2 | Orbit radius in meters |
| `--altitude` | float | 0.3 | Orbit altitude in meters above ground |
| `--speed` | float | 3 | Orbit speed in meters/second |
| `--iterations` | int | 3 | Number of 360-degree orbits |
| `--snapshots` | int | 30 | Snapshots to capture per orbit |
| `--center` | str | "0,0" | x,y direction vector to orbit center |

## Output Structure

```
<environment>_<env_number>/
└── Run_<run_number>/
    ├── rgb/          # RGB images (front-view + top-down)
    ├── rgb_bb/       # RGB images with bounding box overlays
    ├── seg/          # Segmentation masks
    ├── depth/        # Depth images (normalized PNG)
    └── label/        # JSON bounding box annotations
```

Each snapshot produces front-view (`_FV_`) and top-down (`_TD_`) variants.

## Files

| File | Description |
|------|-------------|
| `orbit_navigator.py` | Main data collection script with orbital drone control |
| `airsim_utils.py` | Utility functions for image I/O and coordinate conversion |
| `setup_path.py` | AirSim Python module path bootstrapper |
| `airsim_settings.json` | AirSim simulator configuration (camera, vehicle, segmentation) |

## Troubleshooting

- **Connection refused**: Ensure the UE4 editor with AirSim plugin is running before launching the script.
- **Drone drifting**: The script waits for position stabilization on startup. If it takes too long, check that the drone spawn point is on stable ground.
- **Missing segmentation colors**: Update the `color_to_object_map` in `orbit_navigator.py` or pass a custom mapping via the `segmentation_objects` parameter.
- **Black images**: Verify that the camera settings in `airsim_settings.json` match your UE4 project configuration.

For a comprehensive guide, see [`docs/SYNTHETIC_PIPELINE.md`](../docs/SYNTHETIC_PIPELINE.md).
