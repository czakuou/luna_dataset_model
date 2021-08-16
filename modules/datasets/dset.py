import csv
import os
from glob import glob
from functools import lru_cache

from typing import NamedTuple


DIRECTORY = '/home/czaku/Documents/Pytorch/data/'


class CandidateInfoTuple(NamedTuple):
    isnodule_bool: bool
    diameter_mm: float
    series_uid: tuple[float]
    center_xyz: float


@lru_cache(1)
def get_candidate_info_list(requireOnDisk_bool: bool =True) -> list:

    mhd_list = glob(os.path.join(DIRECTORY, 'subset*/*.mhd'))
    present_on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open(os.path.join(DIRECTORY, 'annotations.csv'), 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
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


if __name__ == '__main__':
    print(get_candidate_info_list()[:5])