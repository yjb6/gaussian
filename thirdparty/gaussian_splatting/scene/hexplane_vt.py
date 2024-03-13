import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def normalize_time(time,duration):
    return 2*time*duration/(duration-1)-1

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    # print(coords,grid.shape)
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0,
        b: float = 1):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd)) #在range(indim)中得到长度为grid_nd的组合数
    space_grid_coefs = nn.ParameterList()
    time_grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.zeros_(new_grid_coef)
            # nn.init.ones_(new_grid_coef)
            # nn.init.uniform_(new_grid_coef, a=a, b=b)
            time_grid_coefs.append(new_grid_coef)

        else:
            # nn.init.uniform_(new_grid_coef, a=a, b=b)
            nn.init.zeros_(new_grid_coef)
            space_grid_coefs.append(new_grid_coef)

    return space_grid_coefs, time_grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    if pts.shape[-1] ==3:
        #space feature
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]), grid_dimensions)
        )
    elif pts.shape[-1] ==4:
        coo_combs = [(i,pts.shape[-1]-1) for i in range(pts.shape[-1]-1)]
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 0
        # interp_time =0
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # if 3 in coo_comb:
            #     interp_time = interp_time + interp_out_plane
            # else:
            # compute product over planes
            interp_space = interp_space + interp_out_plane
        # print(interp_space)
        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
            # multi_scale_interp.append(interp_time)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField_vt(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires
    ) -> None:
        '''将时间维度和空间维度分割开。insight：空间维度可以标识一个点是否是运动的，时间维度可以标识一个点的运动幅度
        并且分开返回时间维度和空间维度的值
        '''
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.space_grids = nn.ModuleList()
        self.time_grids = nn.ModuleList()

        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            s_gp,t_gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += s_gp[-1].shape[1]
                self.feat_dim += t_gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.space_grids.append(s_gp)
            self.time_grids.append(t_gp)
        # self.feat_dim*=2#时空分割开
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min,duration):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        self.duration =duration

        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor,pts_time, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb)
        pts_time = normalize_aabb(pts_time, self.aabb)
        timestamps = normalize_time(timestamps, self.duration) #[0,(d-1)/d] ->[-1,1]
        # print(timestamps)
        pts_time = torch.cat((pts_time, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        pts_time = pts_time.reshape(-1, pts_time.shape[-1])
        space_features = interpolate_ms_features(
            pts, ms_grids=self.space_grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        time_features = interpolate_ms_features(
            pts_time, ms_grids=self.time_grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        # if len(features) < 1:
        #     features = torch.zeros((0, 1)).to(features.device)


        return space_features,time_features

    def forward(self,
                pts: torch.Tensor,
                pts_time:torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, pts_time,timestamps)

        return features

    @property
    def get_grid_parameters(self):
        return self.grids.parameters()
