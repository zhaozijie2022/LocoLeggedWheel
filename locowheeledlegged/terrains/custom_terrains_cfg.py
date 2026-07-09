from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from . import custom_terrains



@configclass
class HfPerlinNoiseTerrainCfg(HfTerrainBaseCfg):

    function = custom_terrains.perlin_noise_terrain

    frequency: float = 0.1                
    octaves: int = 4
    lacunarity: float = 2.0
    persistence: float = 0.5
    seed: int = 42


    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """

