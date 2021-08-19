import csv
import os
from glob import glob
from functools import lru_cache
from copy import copy

from collections import defaultdict
from typing import NamedTuple, Optional, Sequence

import SimpleITK as sitk
import numpy as np


import torch
from torch.utils.data import Dataset


from ..util.utils import XyzTuple, IrcTuple
from ..util.utils import xyz2irc, irc2xyz


DIRECTORY = '/home/czaku/Documents/Pytorch/data/'


class CandidateInfoTuple(NamedTuple):
    isnodule_bool: bool
    diameter_mm: float
    series_uid: Sequence[float]
    center_xyz: Sequence[float]


@lru_cache(1)
def get_candidate_info_list(requireOnDisk_bool: bool = True) -> list:

    mhd_list = glob(os.path.join(DIRECTORY, 'subset*/*.mhd'))
    present_on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = defaultdict(list)
    with open(os.path.join(DIRECTORY, 'annotations.csv'), 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])

            diameter_dict[series_uid].append(
                (annotation_center_xyz, annotation_diameter_mm))

    candidate_info_list = []
    with open(os.path.join(DIRECTORY, 'candidates.csv'), 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in present_on_disk_set and requireOnDisk_bool:
                continue

            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            candidate_diameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                    if delta_mm > annotation_diameter_mm / 4:
                        break
                    else:
                        candidate_diameter_mm = annotation_diameter_mm
                        break

                candidate_info_list.append(CandidateInfoTuple(
                    is_nodule_bool,
                    candidate_diameter_mm,
                    series_uid,
                    candidate_center_xyz,))

    candidate_info_list.sort(reverse=True)
    return candidate_info_list


class Ct:
    def __init__(self, series_uid: float) -> None:
        mhd_path = glob(os.path.join(DIRECTORY, f'subset*/{series_uid}.mhd'))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        # CT scan voxels are expressed in HU
        # https://en.wikipedia.org/wiki/Hounsfield_scale
        # air is -1000, soft tissue 100-300, bone 1000
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz: Sequence[float],
                          width_irc: Sequence[int]) -> 'ct_chunk, center_irc':
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            rise_assert = center_val >= 0 and center_val < self.hu_a.shape[axis]
            assert rise_assert, repr([self.series_uid, center_xyz, self.origin_xyz,
                                      self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


def get_Ct(series_uid) -> Ct:
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def get_raw_candidate(series_uid: float,
                      center_xyz: list[float],
                      width_irc: tuple[int, int, int]) -> 'ct_chunk center_irc':
    ct = get_Ct(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LoadDataset(Dataset):
    """Torch Dataset implementation"""
    def __init__(self,
                 val_stride: int = 0,
                 is_val_set_bool: Optional[bool] = None,
                 series_uid: Optional[float] =None) -> None:
        self.candidate_info_list = copy(get_candidate_info_list())

        if series_uid:
            self.candidate_info_list = [x for x in self.candidate_info_list
                                        if x.series_uid == series_uid]

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
        elif val_stride > 0:
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list

    def __len__(self):
        return len(self.candidate_info_list)

    def __getitem__(self, index):
        candidate_info_tup = self.candidate_info_list[index]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = get_raw_candidate(
            candidate_info_tup.series_uid,
            candidate_info_tup.center_xyz,
            width_irc)

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
                not candidate_info_tup.isNodule_bool,
                candidate_info_tup.isNodule_bool],
            dtype=torch.int64)

        return (
            candidate_t,
            pos_t,
            candidate_info_tup.series_uid,
            torch.tensor(center_irc))
