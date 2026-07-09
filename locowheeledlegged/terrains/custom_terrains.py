from __future__ import annotations  # 将来类型注解作为字符串处理, 防止循环导入

import numpy as np
import scipy.interpolate as interpolate
from scipy.ndimage import zoom
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import custom_terrains_cfg


def _perlin_like_noise_2d(
    shape: tuple[int, int],
    frequency: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: int,
    repeat: tuple[int, int] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float64)
    amp = 1.0
    freq_scale = 1.0
    amp_sum = 0.0
    for _ in range(octaves):
        gw = max(2, int(shape[0] * frequency * freq_scale))
        gl = max(2, int(shape[1] * frequency * freq_scale))
        if repeat is not None:
            gw = min(gw, repeat[0])
            gl = min(gl, repeat[1])
        gw = max(2, gw)
        gl = max(2, gl)
        grid = rng.uniform(-1.0, 1.0, (gw + 1, gl + 1))
        if repeat is not None and (repeat[0] > 0 and repeat[1] > 0):
            grid[-1, :] = grid[0, :]
            grid[:, -1] = grid[:, 0]
        zoom_factors = (shape[0] / grid.shape[0], shape[1] / grid.shape[1])
        layer = zoom(grid, zoom_factors, order=1, mode="wrap" if repeat else "nearest")
        out += amp * layer
        amp_sum += amp
        amp *= persistence
        freq_scale *= lacunarity
    if amp_sum > 0:
        out /= amp_sum
    return np.clip(out, -1.0, 1.0)


@height_field_to_mesh
def perlin_noise_terrain(difficulty: float, cfg: custom_terrains_cfg.HfPerlinNoiseTerrainCfg) -> np.ndarray:
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    n = _perlin_like_noise_2d(
        (width_downsampled, length_downsampled),
        frequency=cfg.frequency,
        octaves=cfg.octaves,
        persistence=cfg.persistence,
        lacunarity=cfg.lacunarity,
        seed=cfg.seed,
        repeat=(width_downsampled, length_downsampled),
    )
    height_field_downsampled = height_min + (n + 1.0) * 0.5 * (height_max - height_min)
    # create interpolation function for the sampled heights


    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)



