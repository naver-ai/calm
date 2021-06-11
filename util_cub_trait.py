"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from os.path import join as ospj
import numpy as np
from PIL import Image

ALL_COORDS = np.array([(x, y) for y in range(224) for x in range(224)])


class CUBTrait:
    def __init__(self, data_dir):
        """
        Every _id starts from 1, not 0
        """
        self.data_dir = data_dir
        self.part_id2name = self._part_id2name()
        self.attribute_id2name = self._attribute_id2name()
        self.class2attribute = self._class2attribute()
        self.image_id2part_locs = self._image_id2part_locs()
        self.image_file2image_id = self._image_file2image_id()
        self.diff2class_pair_dict = self._diff2class_pair_dict()

    def _part_id2name(self):
        with open(ospj(self.data_dir, 'parts', 'parts.txt'), 'r') as f:
            part_id2name = f.readlines()
            part_id2name = {i + 1: ' '.join(name.strip('\n').split(' ')[1:])
                            for i, name in enumerate(part_id2name)}
        return part_id2name

    def _attribute_id2name(self):
        with open(ospj(self.data_dir, 'attributes', 'attributes.txt'),
                  'r') as f:
            attribute_id2name = f.readlines()
            attribute_id2name = {i + 1: name.strip('\n').split(' ')[1]
                                 for i, name in enumerate(attribute_id2name)}
        return attribute_id2name

    def _class2attribute(self):
        with open(ospj(self.data_dir, 'attributes',
                       'class_attribute_labels_continuous.txt'), 'r') as f:
            lines = f.readlines()
        class2attribute = {i + 1: [float(val) for val in vals.split(' ')]
                           for i, vals in enumerate(lines)}
        class2attribute = {key: (np.array(val) >= 50).astype(int) \
                           for key, val in class2attribute.items()}
        return class2attribute

    def _image_id2part_locs(self):
        image_id2part_locs = {}
        with open(ospj(self.data_dir, 'parts', 'part_locs.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            image_id, part_id = int(line[0]), int(line[1])
            x_coord, y_coord = float(line[2]), float(line[3])
            existance = int(line[4])
            if image_id not in image_id2part_locs.keys():
                image_id2part_locs[image_id] = {}
            if existance:
                image_id2part_locs[image_id][part_id] = (x_coord, y_coord)
        return image_id2part_locs

    def _image_file2image_id(self):
        image_file2image_id = {}
        with open(ospj(self.data_dir, 'images.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            image_id, image_file = int(line[0]), line[1]
            image_file2image_id[image_file] = image_id
        return image_file2image_id

    def _get_different_part_ids(self, class_id1, class_id2):
        att1 = self.class2attribute[class_id1]
        att2 = self.class2attribute[class_id2]
        diff_part_ids = self._get_part_id(att1 - att2)
        diff_part_ids = list(set(diff_part_ids))
        return diff_part_ids

    def _diff2class_pair_dict(self):
        diff2class_pair_dict = {}
        for i in range(1, len(self.class2attribute.keys())):
            for j in range(i + 1, len(self.class2attribute.keys()) + 1):
                class_pair = (i, j)
                diff_part_ids = self._get_different_part_ids(i, j)
                diff_size = len(diff_part_ids)
                if diff_size in diff2class_pair_dict.keys():
                    diff2class_pair_dict[diff_size] += [class_pair]
                else:
                    diff2class_pair_dict[diff_size] = [class_pair]
        return diff2class_pair_dict

    def _get_part_id(self, attribute_diff):
        att_diff_part_id = []
        att_diff_names = [self.attribute_id2name[att_id] \
                          for att_id in (np.where(attribute_diff)[0] + 1)]
        for att_name in att_diff_names:
            if any(part in att_name for part in \
                   ['bill', 'crown', 'eye', 'forehead', 'head', 'nape',
                    'throat']):
                att_diff_part_id.append(1)
            if 'back' in att_name:
                att_diff_part_id.append(2)
            if 'belly' in att_name:
                att_diff_part_id.append(3)
            if 'breast' in att_name:
                att_diff_part_id.append(4)
            if 'tail' in att_name:
                att_diff_part_id.append(5)
            if 'leg' in att_name:
                att_diff_part_id.append(6)
            if 'wing' in att_name:
                att_diff_part_id.append(7)
        return att_diff_part_id

    def _union_parts(self, part_coords_resized, part_ids):
        """
        union forehead, beak, crown, eye, throat, nape into head
        union part ids: 1:head, 2:back, 3:belly, 4:breast, 5:tail, 6:leg, 7:wing
        """
        part_head_idxs = [i for i, val in enumerate(part_ids) \
                          if val in [2, 5, 6, 7, 10, 11, 15]]
        part_leg_idxs = [i for i, val in enumerate(part_ids) \
                         if val in [8, 12]]
        part_wing_idxs = [i for i, val in enumerate(part_ids) \
                          if val in [9, 13]]

        part_coords_stack = []
        part_ids_union = []

        if len(part_head_idxs) >= 1:  # head
            part_coords_stack.append(
                np.mean(part_coords_resized[part_head_idxs],
                        axis=0, keepdims=True))
            part_ids_union.append(1)

        # back, belly, breast, tail
        for enum, i in enumerate([1, 3, 4, 14]):
            if i in part_ids:
                part_coords_stack.append(
                    part_coords_resized[[part_ids.index(i)]])
                part_ids_union.append(enum + 2)

        # leg, wing
        for part_id_union, part_name_idxs in \
                [(6, part_leg_idxs), (7, part_wing_idxs)]:
            if len(part_name_idxs) >= 1:
                part_coords_stack.append(part_coords_resized[part_name_idxs])
                part_ids_union += [part_id_union] * len(part_name_idxs)

        part_coords_union = np.concatenate(
            part_coords_stack, axis=0).astype(int)
        return part_coords_union, part_ids_union

    def _get_parts_union(self, image_id, image_origin_size):
        """ get unified part ids and coordiates """
        part_ids = []
        part_coords = []
        for part_id, part_loc in self.image_id2part_locs[image_id].items():
            part_ids.append(part_id)
            part_coords.append(part_loc)
        part_coords = np.array(part_coords)
        part_coords = part_coords / \
                      np.expand_dims(np.array(image_origin_size), axis=0) * 224
        part_coords_union, part_ids_union = \
            self._union_parts(part_coords, part_ids)
        return part_coords_union, part_ids_union

    def _get_gt_difference_coords(self, diff_part_ids,
                                  part_coords_union, part_ids_union):
        """ get GT of difference coordinates of given two classes """
        diff_coords = []
        for diff_id in diff_part_ids:
            if diff_id in part_ids_union:
                diff_coord_id = part_ids_union.index(diff_id)
                diff_coords.append(part_coords_union[diff_coord_id])
        if len(diff_coords) >= 1:
            diff_coords = np.stack(diff_coords, axis=0)
        else:
            diff_coords = None
        return diff_coords

    def _pseudo_segment_mask(self, image_file, part_coords, gt_part_coord_idxs):
        num_all_coords = ALL_COORDS.shape[0]
        num_part_coords = part_coords.shape[0]
        diff_coords_rep = np.concatenate(
            [np.expand_dims(ALL_COORDS, axis=1)] * num_part_coords, axis=1)
        part_coords_rep = np.concatenate(
            [np.expand_dims(part_coords, axis=0)] * num_all_coords, axis=0)
        distance_mtx = np.sum(
            pow((diff_coords_rep - part_coords_rep), 2), axis=2)
        closest_idxs = np.argmin(distance_mtx, axis=1)
        gt_mask_candidate = (closest_idxs[:, None] ==
                             gt_part_coord_idxs).any(axis=1)
        gt_mask_candidate = np.reshape(gt_mask_candidate, (224, 224))

        gt_mask_segment_path = ospj(
            self.data_dir, 'segmentations', image_file[:-4] + '.png')
        gt_mask_segment = Image.open(gt_mask_segment_path).convert('L')
        gt_mask_segment = np.array(gt_mask_segment.resize((224, 224)))
        gt_mask_segment = gt_mask_segment > 0

        gt_mask = np.logical_and(gt_mask_candidate, gt_mask_segment)
        gt_mask = 1 * gt_mask
        return gt_mask

    def get_pseudo_segment_mask(self, image_file, class_pair):
        diff_part_ids = self._get_different_part_ids(*class_pair)
        image_id = self.image_file2image_id[image_file]
        image_path = ospj(self.data_dir, 'images', image_file)
        image_origin_size = Image.open(image_path).size
        part_coords, part_ids = \
            self._get_parts_union(image_id, image_origin_size)
        gt_part_coord_idxs = [i for i, part_id in enumerate(part_ids)
                              if part_id in diff_part_ids]

        diff_coords = self._get_gt_difference_coords(
            diff_part_ids, part_coords, part_ids)
        if diff_coords is None:
            return None

        gt_mask = self._pseudo_segment_mask(
            image_file, part_coords, gt_part_coord_idxs)
        return gt_mask
