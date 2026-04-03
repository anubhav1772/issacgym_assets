# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import math

import numpy as np
from isaacgym import terrain_utils
from numpy.random import choice

from aliengo_gym.envs.base.legged_robot_config import BaseCfg as Cfg


class Terrain:
    def __init__(self, cfg: Cfg.terrain, num_robots, eval_cfg=None, num_eval_robots=0) -> None:

        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.train_rows, self.train_cols, self.eval_rows, self.eval_cols = self.load_cfgs()
        self.tot_rows = len(self.train_rows) + len(self.eval_rows)
        self.tot_cols = max(len(self.train_cols), len(self.eval_cols))
        self.cfg.env_length = cfg.terrain_length
        self.cfg.env_width = cfg.terrain_width

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.initialize_terrains()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

    def load_cfgs(self):
        self._load_cfg(self.cfg)
        self.cfg.row_indices = np.arange(0, self.cfg.tot_rows)
        self.cfg.col_indices = np.arange(0, self.cfg.tot_cols)
        self.cfg.x_offset = 0
        self.cfg.rows_offset = 0
        if self.eval_cfg is None:
            return self.cfg.row_indices, self.cfg.col_indices, [], []
        else:
            self._load_cfg(self.eval_cfg)
            self.eval_cfg.row_indices = np.arange(self.cfg.tot_rows, self.cfg.tot_rows + self.eval_cfg.tot_rows)
            self.eval_cfg.col_indices = np.arange(0, self.eval_cfg.tot_cols)
            self.eval_cfg.x_offset = self.cfg.tot_rows
            self.eval_cfg.rows_offset = self.cfg.num_rows
            return self.cfg.row_indices, self.cfg.col_indices, self.eval_cfg.row_indices, self.eval_cfg.col_indices

    def _load_cfg(self, cfg):
        cfg.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        cfg.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        cfg.width_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        cfg.length_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)

        cfg.border = int(cfg.border_size / cfg.horizontal_scale)
        cfg.tot_cols = int(cfg.num_cols * cfg.width_per_env_pixels) + 2 * cfg.border
        cfg.tot_rows = int(cfg.num_rows * cfg.length_per_env_pixels) + 2 * cfg.border

    def initialize_terrains(self):
        self._initialize_terrain(self.cfg)
        if self.eval_cfg is not None:
            self._initialize_terrain(self.eval_cfg)

    def _initialize_terrain(self, cfg):
        if cfg.curriculum:
            self.curriculum(cfg)
        elif cfg.selected:
            self.selected_terrain(cfg)
        else:
            self.randomized_terrain(cfg)

    def randomized_terrain(self, cfg):
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
            # self.add_terrain_to_map(cfg, terrain, i, j)
            # self.add_scene1_to_map(cfg, terrain, i, j)
            self.add_scene3_to_map(cfg, terrain, i, j)

    def curriculum(self, cfg):
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                difficulty = i / cfg.num_rows * cfg.difficulty_scale
                choice = j / cfg.num_cols + 0.001

                terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
                # self.add_terrain_to_map(cfg, terrain, i, j)
                # self.add_scene1_to_map(cfg, terrain, i, j)
                self.add_scene3_to_map(cfg, terrain, i, j)

    def selected_terrain(self, cfg):
        terrain_type = cfg.terrain_kwargs.pop('type')
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.width_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)

            eval(terrain_type)(terrain, **cfg.terrain_kwargs.terrain_kwargs)
            # self.add_terrain_to_map(cfg, terrain, i, j)
            # self.add_scene1_to_map(cfg, terrain, i, j)
            self.add_scene3_to_map(cfg, terrain, i, j)

    def make_terrain(self, cfg, choice, difficulty, proportions):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=cfg.width_per_env_pixels,
                                           length=cfg.width_per_env_pixels,
                                           vertical_scale=cfg.vertical_scale,
                                           horizontal_scale=cfg.horizontal_scale)
        terrain.height_field_raw[:, :] = 0

        # slope = difficulty * 0.4
        # step_height = 0.05 + 0.18 * difficulty
        # discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)
        # stepping_stones_size = 1.5 * (1.05 - difficulty)
        # stone_distance = 0.05 if difficulty == 0 else 0.1
        # if choice < proportions[0]:
        #     if choice < proportions[0] / 2:
        #         slope *= -1
        #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # elif choice < proportions[1]:
        #     terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        #     terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
        #                                          step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
        # elif choice < proportions[3]:
        #     if choice < proportions[2]:
        #         step_height *= -1
        #     terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        # elif choice < proportions[4]:
        #     num_rectangles = 20
        #     rectangle_min_size = 1.
        #     rectangle_max_size = 2.
        #     terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
        #                                              rectangle_max_size, num_rectangles, platform_size=3.)
        # elif choice < proportions[5]:
        #     terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
        #                                           stone_distance=stone_distance, max_height=0., platform_size=4.)
        # elif choice < proportions[6]:
        #     pass
        # elif choice < proportions[7]:
        #     pass
        # elif choice < proportions[8]:
        #     terrain_utils.random_uniform_terrain(terrain, min_height=-cfg.terrain_noise_magnitude,
        #                                          max_height=cfg.terrain_noise_magnitude, step=0.005,
        #                                          downsampled_scale=0.2)
        # elif choice < proportions[9]:
        #     terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
        #                                          step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
        #     terrain.height_field_raw[0:terrain.length // 2, :] = 0

        return terrain

    def add_terrain_to_map(self, cfg, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
        end_x = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.width_per_env_pixels
        end_y = cfg.border + (j + 1) * cfg.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_width
        x1 = int((cfg.terrain_length / 2. - 1) / terrain.horizontal_scale) + cfg.x_offset
        x2 = int((cfg.terrain_length / 2. + 1) / terrain.horizontal_scale) + cfg.x_offset
        y1 = int((cfg.terrain_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((cfg.terrain_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(self.height_field_raw[start_x: end_x, start_y:end_y]) * terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def add_scene1_to_map(self, cfg, terrain, row, col):
        i = row
        j = col

        # MAP COORDS
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
        end_x   = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.width_per_env_pixels
        end_y   = cfg.border + (j + 1) * cfg.width_per_env_pixels

        # BASE TILE
        tile = terrain.height_field_raw.copy()
        tile[:, :] = 0

        H, W = tile.shape

        # WALL PARAMS
        outer_wall_thickness = 3
        inner_wall_thickness = 2
        wall_height = 1.5
        h = wall_height / terrain.vertical_scale

        # OUTER WALLS
        tile[:, :outer_wall_thickness] = h
        tile[:, -outer_wall_thickness:] = h
        tile[:outer_wall_thickness, :] = h
        tile[-outer_wall_thickness:, :] = h

        # FACTORY WALLS (SPACED)
        aisle_spacing = 35
        door_size = 8

        horizontal_walls = []
        vertical_walls = []

        # Horizontal
        for x in range(aisle_spacing, H - aisle_spacing, aisle_spacing):
            tile[x:x+inner_wall_thickness, :] = h
            horizontal_walls.append(x)

        # Vertical
        for y in range(aisle_spacing, W - aisle_spacing, aisle_spacing):
            tile[:, y:y+inner_wall_thickness] = h
            vertical_walls.append(y)

        # GUARANTEED DOORS (AFTER WALLS)
        for x in horizontal_walls:
            door_y = np.random.randint(10, W - 10)
            tile[x:x+inner_wall_thickness,
                 max(0, door_y-door_size):min(W, door_y+door_size)] = 0

        for y in vertical_walls:
            door_x = np.random.randint(10, H - 10)
            tile[max(0, door_x-door_size):min(H, door_x+door_size),
                 y:y+inner_wall_thickness] = 0

        # EXTRA DOORS (ONLY ON WALLS)
        for _ in range(10):
            x = np.random.randint(10, H - 10)
            y = np.random.randint(10, W - 10)

            if tile[x, y] > 0:
                tile[x-4:x+4, y-4:y+4] = 0

        # WALL FEATURES
        # Protrusions
        for _ in range(150):
            x = np.random.randint(5, H - 5)
            y = np.random.randint(5, W - 5)

            if tile[x, y] > 0:
                size = np.random.randint(1, 3)
                tile[x:x+size, y:y+size] = h * np.random.uniform(0.7, 1.3)

        # Grooves (structured features)
        groove_spacing = 6
        for x in range(0, H, groove_spacing):
            for y in range(0, W):
                if tile[x, y] > 0:
                    tile[x:x+1, y] = (tile[x:x+1, y] * 0.6).astype(tile.dtype)

        # PILLARS
        for _ in range(40):
            x = np.random.randint(10, H - 10)
            y = np.random.randint(10, W - 10)

            # near walls only
            if tile[x, y] == 0 and (
                tile[x+2, y] > 0 or tile[x-2, y] > 0 or
                tile[x, y+2] > 0 or tile[x, y-2] > 0
            ):
                size = np.random.randint(2, 4)
                tile[x:x+size, y:y+size] = h

        # LARGE OBSTACLES
        for _ in range(6):
            x = np.random.randint(10, H - 20)
            y = np.random.randint(10, W - 20)

            if abs(x - H//2) < 15 and abs(y - W//2) < 15:
                continue

            w = np.random.randint(10, 18)
            h_block = np.random.randint(10, 18)

            tile[x:x+w, y:y+h_block] = h

        # SAFE SPAWN ZONE
        cx, cy = H // 2, W // 2
        tile[cx-12:cx+12, cy-12:cy+12] = 0

        self.height_field_raw[start_x:end_x, start_y:end_y] = tile

        # ORIGIN
        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_width
        env_origin_z = np.max(tile) * terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def add_scene2_to_map(self, cfg, terrain, row, col):
        i = row
        j = col

        # MAP COORDS
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
        end_x   = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.width_per_env_pixels
        end_y   = cfg.border + (j + 1) * cfg.width_per_env_pixels

        # BASE TILE
        tile = terrain.height_field_raw.copy()
        tile[:, :] = 0

        H, W = tile.shape

        # WALL PARAMS
        outer_wall_thickness = 4
        wall_height = 1.5
        h = wall_height / terrain.vertical_scale

        # OUTER WALLS (BOUNDARY)
        tile[:, :outer_wall_thickness] = h
        tile[:, -outer_wall_thickness:] = h
        tile[:outer_wall_thickness, :] = h
        tile[-outer_wall_thickness:, :] = h

        # CENTRAL CORRIDOR (CLEAR PATH)
        corridor_width = W // 2   # wide open
        y1 = W//2 - corridor_width//2
        y2 = W//2 + corridor_width//2

        tile[:, y1:y2] = 0  # ensure open corridor

        # DISCONNECTED WALL OBSTACLES (MAIN PART)
        num_obstacles = 5

        for _ in range(num_obstacles):
            x = np.random.randint(10, H - 20)
            y = np.random.randint(y1 + 5, y2 - 5)

            # random orientation
            if np.random.rand() < 0.5:
                # vertical block
                length = np.random.randint(10, 20)
                thickness = np.random.randint(2, 4)

                tile[x:x+length, y:y+thickness] = h

            else:
                # horizontal block
                length = np.random.randint(10, 20)
                thickness = np.random.randint(2, 4)

                tile[x:x+thickness, y:y+length] = h

        # WALL FEATURES
        # Protrusions
        # for _ in range(120):
        #     x = np.random.randint(5, H - 5)
        #     y = np.random.randint(5, W - 5)
        #
        #     if tile[x, y] > 0:
        #         size = np.random.randint(1, 3)
        #         tile[x:x+size, y:y+size] = int(h * np.random.uniform(0.7, 1.3))
        #
        # # Grooves
        # groove_spacing = 6
        # for x in range(0, H, groove_spacing):
        #     for y in range(0, W):
        #         if tile[x, y] > 0:
        #             tile[x, y] = int(tile[x, y] * 0.6)
        #
        # # SMALL PILLARS NEAR WALLS
        # for _ in range(30):
        #     x = np.random.randint(10, H - 10)
        #     y = np.random.randint(10, W - 10)
        #
        #     if tile[x, y] == 0 and (
        #         tile[x+2, y] > 0 or tile[x-2, y] > 0 or
        #         tile[x, y+2] > 0 or tile[x, y-2] > 0
        #     ):
        #         size = np.random.randint(2, 4)
        #         tile[x:x+size, y:y+size] = h

        # STRONG SLAM FEATURES
        # Edge-focused features
        for _ in range(200):
            x = np.random.randint(5, H - 5)
            y = np.random.randint(5, W - 5)

            if tile[x, y] > 0 and (
                tile[x+1, y] == 0 or tile[x-1, y] == 0 or
                tile[x, y+1] == 0 or tile[x, y-1] == 0
            ):
                tile[x, y] = int(h * np.random.uniform(0.5, 1.5))

        # Random grooves (non-periodic)
        for _ in range(150):
            x = np.random.randint(5, H - 5)
            y = np.random.randint(5, W - 5)

            if tile[x, y] > 0:
                tile[x, y] = int(tile[x, y] * np.random.uniform(0.5, 0.9))

        # Corner blobs
        for _ in range(60):
            x = np.random.randint(10, H - 10)
            y = np.random.randint(10, W - 10)

            if tile[x, y] > 0:
                tile[x:x+3, y:y+3] = int(h * 1.4)

        # SAFE SPAWN ZONE (CENTER)
        cx, cy = H // 2, W // 2
        tile[cx-10:cx+10, cy-10:cy+10] = 0

        # FINAL BOUNDARY ENFORCEMENT
        tile[:, :outer_wall_thickness] = h
        tile[:, -outer_wall_thickness:] = h
        # tile[:outer_wall_thickness, :] = h
        # tile[-outer_wall_thickness:, :] = h

        # SMART BOUNDARY WALLS (NO INTERNAL WALLS)
        is_top    = (i == 0)
        is_bottom = (i == cfg.num_rows - 1)
        is_left   = (j == 0)
        is_right  = (j == cfg.num_cols - 1)

        # LEFT WALL
        if is_left:
            tile[:, :outer_wall_thickness] = h

        # RIGHT WALL
        if is_right:
            tile[:, -outer_wall_thickness:] = h

        # TOP WALL
        if is_top:
            tile[:outer_wall_thickness, :] = h

        # BOTTOM WALL
        if is_bottom:
            tile[-outer_wall_thickness:, :] = h

        self.height_field_raw[start_x:end_x, start_y:end_y] = tile

        # ORIGIN
        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_width
        env_origin_z = np.max(tile) * terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def add_scene3_to_map(self, cfg, terrain, row, col):

        start_x = cfg.border
        end_x   = cfg.border + cfg.length_per_env_pixels
        start_y = cfg.border
        end_y   = cfg.border + cfg.width_per_env_pixels

        tile = terrain.height_field_raw.copy()
        tile[:, :] = 0

        H, W = tile.shape

        # wall parameters
        wall_thickness = 4
        wall_height = 1.5
        h = wall_height / terrain.vertical_scale

        # fully sealed outer boundary
        tile[:, :wall_thickness] = h
        tile[:, -wall_thickness:] = h
        tile[:wall_thickness, :] = h
        tile[-wall_thickness:, :] = h

        # strong grooves on outer walls
        for _ in range(500):
            x = np.random.randint(1, H-1)
            y = np.random.randint(1, W-1)

            if (
                x < wall_thickness or x >= H - wall_thickness or
                y < wall_thickness or y >= W - wall_thickness
            ):
                tile[x, y] = int(h * np.random.uniform(0.4, 1.4))

        # corridor
        corridor_width = W // 3
        corridor_center = np.random.randint(W//3, 2*W//3)

        y1 = max(0, corridor_center - corridor_width // 2)
        y2 = min(W, corridor_center + corridor_width // 2)

        tile[:, y1:y2] = 0

        # horizontal connectors
        for _ in range(2):
            x = np.random.randint(H//4, 3*H//4)
            x1 = max(0, x-2)
            x2 = min(H, x+2)
            tile[x1:x2, :] = 0

        # obstacles outside corridor
        num_obstacles = np.random.randint(6, 10)

        for _ in range(num_obstacles):

            x = np.random.randint(10, H - 25)

            if np.random.rand() < 0.5:
                y = np.random.randint(5, max(6, y1 - 5))
            else:
                y = np.random.randint(min(W-25, y2 + 5), W - 10)

            if np.random.rand() < 0.5:
                length = np.random.randint(10, 20)
                thickness = np.random.randint(2, 4)

                x_end = min(x + length, H)
                y_end = min(y + thickness, W)

                tile[x:x_end, y:y_end] = h
                obs_slice = (slice(x, x_end), slice(y, y_end))

            else:
                length = np.random.randint(10, 20)
                thickness = np.random.randint(2, 4)

                x_end = min(x + thickness, H)
                y_end = min(y + length, W)

                tile[x:x_end, y:y_end] = h
                obs_slice = (slice(x, x_end), slice(y, y_end))

            # grooves on obstacle surfaces
            for _ in range(50):
                if obs_slice[0].stop > obs_slice[0].start and obs_slice[1].stop > obs_slice[1].start:
                    gx = np.random.randint(obs_slice[0].start, obs_slice[0].stop)
                    gy = np.random.randint(obs_slice[1].start, obs_slice[1].stop)
                    tile[gx, gy] = int(h * np.random.uniform(0.5, 1.3))

        # landmark blocks
        for _ in range(4):
            x = np.random.randint(20, H-20)
            y = np.random.randint(20, W-20)

            x_end = min(x+6, H)
            y_end = min(y+6, W)

            tile[x:x_end, y:y_end] = int(h * 1.2)

            # grooves on landmarks
            for _ in range(30):
                gx = np.random.randint(x, x_end)
                gy = np.random.randint(y, y_end)
                tile[gx, gy] = int(h * np.random.uniform(0.6, 1.4))

        # global grooves on all walls/obstacles
        for _ in range(400):
            x = np.random.randint(5, H - 5)
            y = np.random.randint(5, W - 5)

            if tile[x, y] > 0:
                tile[x, y] = int(tile[x, y] * np.random.uniform(0.5, 0.9))

        # edge features
        for _ in range(400):
            x = np.random.randint(5, H - 5)
            y = np.random.randint(5, W - 5)

            if tile[x, y] > 0 and (
                tile[x+1, y] == 0 or tile[x-1, y] == 0 or
                tile[x, y+1] == 0 or tile[x, y-1] == 0
            ):
                tile[x, y] = int(h * np.random.uniform(0.6, 1.5))

        # spawn safe zone
        cx, cy = H // 2, W // 2
        tile[cx-15:cx+15, cy-15:cy+15] = 0

        # re-enforce fully sealed outer boundary (FINAL STEP)
        tile[:, :wall_thickness] = h
        tile[:, -wall_thickness:] = h
        tile[:wall_thickness, :] = h
        tile[-wall_thickness:, :] = h

        # re-apply grooves to outer walls (after sealing)
        for _ in range(300):
            x = np.random.randint(1, H-1)
            y = np.random.randint(1, W-1)

            if (
                x < wall_thickness or x >= H - wall_thickness or
                y < wall_thickness or y >= W - wall_thickness
            ):
                tile[x, y] = int(h * np.random.uniform(0.5, 1.4))

        self.height_field_raw[start_x:end_x, start_y:end_y] = tile

        # origin
        # env_origin_x = cfg.terrain_length / 2 + cfg.x_offset * terrain.horizontal_scale
        # env_origin_y = cfg.terrain_width / 2

        margin = 1.0  # VERY important (avoid walls)

        # choose one corner
        corner = "bottom_left"   # bottom_left, bottom_right, top_left, top_right

        if corner == "bottom_left":
            env_origin_x = margin
            env_origin_y = margin

        elif corner == "bottom_right":
            env_origin_x = cfg.terrain_length - margin
            env_origin_y = margin

        elif corner == "top_left":
            env_origin_x = margin
            env_origin_y = cfg.terrain_width - margin

        elif corner == "top_right":
            env_origin_x = cfg.terrain_length - margin
            env_origin_y = cfg.terrain_width - margin

        # env_origin_z = np.max(tile) * terrain.vertical_scale
        env_origin_z = 0

        cfg.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]
