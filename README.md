# issacgym_assets

A curated collection of simulation-ready assets (URDFs, meshes, and textures) for NVIDIA Isaac Gym, designed for robotics research, reinforcement learning, and sim-to-real experiments.

## Isaac Gym: Assets & Camera Setup Guide

This guide explains how to:
- Add custom 3D assets into Isaac Gym
- Convert models to usable formats
- Spawn them safely in the environment
- Attach and use cameras for perception

## Adding Custom Assets

### 1. Download 3D Models
Download models from:
https://sketchfab.com/3d-models

### 2. Convert `.fbx` → `.obj`

Isaac Gym does **not directly support `.fbx`**, so convert it using any online converter.

### 3. Required File Structure

Your asset folder should look like:

```bash
resources/objects/office_chair/
├── OfficeChair.obj
├── OfficeChair.mtl     # IMPORTANT
├── OfficeChair.urdf
└── textures/
    ├── texture1.png
    └── ...
```

⚠️ Important Notes:


.mtl file is mandatory → links textures to mesh


### 4. Convert .obj → .urdf
Use the [obj2urdf converter](https://github.com/alaflaquiere/obj2urdf) to generate URDF files from OBJ meshes.

```bash
python ~/obj2urdf/obj2urdf.py \
resources/objects/office_chair/OfficeChair.obj
```

### 5. Load Asset in Isaac Gym

```bash
chair_asset_root = "resources/objects/office_chair"
chair_asset_file = "OfficeChair.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.use_mesh_materials = True

self.chair_asset = self.gym.load_asset(
    self.sim,
    chair_asset_root,
    chair_asset_file,
    asset_options
)
```

### 6. Apply Textures

```bash
chair_texture = self.gym.create_texture_from_file(
      self.sim,
      "/resources/objects/office_chair/textures/OfficeChair_OfficeChair_Main_BaseColor.png"
  )

  cx, cy = self.sample_terrain_aware_position(tile, origin_x, origin_y, final_positions, OBJECT_RADII["chair"])
  cz = env_origin[2].item()

  pose = gymapi.Transform()
  pose.p = gymapi.Vec3(cx, cy, cz - 4.0)
  pose.r = gymapi.Quat.from_euler_zyx(1.57, 0, 1.57)

  chair_handle = self.gym.create_actor(
      env_handle, self.chair_asset, pose, "chair", i, 0, 0
  )

  num_chair_bodies = self.gym.get_actor_rigid_body_count(env_handle, chair_handle)

  for b in range(num_chair_bodies):
      self.gym.set_rigid_body_texture(
          env_handle,
          chair_handle,
          b,
          gymapi.MESH_VISUAL,
          chair_texture
      )
```

## Smart Object Placement (Terrain-Aware)

To avoid floating or colliding objects, use terrain-aware sampling:

```bash
def sample_terrain_aware_position(self, tile, origin_x, origin_y, existing_positions, radius):
```

