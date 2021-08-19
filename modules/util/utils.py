from typing import NamedTuple
from typing import Sequence

import numpy as np


class XyzTuple(NamedTuple):
    x: float
    y: float
    z: float


class IrcTuple(NamedTuple):
    index: int
    row: int
    col: int


def irc2xyz(coord_irc: Sequence[int], origin_xyz: XyzTuple,
            vxSize_xyz: XyzTuple, direction_a: Sequence[int]) -> XyzTuple:
    """
    Flip the coordinates of CT scans to align with XYZ
    Scale the indices with voxel sizes
    :param coord_irc:
    :param origin_xyz:
    :param vxSize_xyz:
    :param direction_a:
    :return: XyzTuple
    """
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)


def xyz2irc(coord_xyz: Sequence[float], origin_xyz: XyzTuple,
            vxSize_xyz: XyzTuple, direction_a: Sequence[float]) -> IrcTuple:
    """
    Convert XYZ coordinates to IRC
    :param coord_xyz:
    :param origin_xyz:
    :param vxSize_xyz:
    :param direction_a:
    :return: IrcTuple
    """
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2], int(cri_a[1]), int(cri_a[0])))
