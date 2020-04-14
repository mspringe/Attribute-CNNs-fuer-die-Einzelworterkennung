"""
This module provides evaluation utilities for word spotting.

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
import numpy as np
from scipy.spatial.distance import cdist


def ap(bin_relevance, n_occ):
    """
    calculates the average precision (AP) for a given sequence of relevant hits

    :param bin_relevance: binary list of relevant elements in retrieval list
    :return: MAP of sequence
    """
    bin_relevance = np.array(bin_relevance, dtype=bool)
    # floating point representation of relevances
    rels = np.array(bin_relevance, dtype=float)
    # cumlative sum of relevant samples
    cum_sum = np.cumsum(rels)
    # base 1 indices
    idcs = np.arange(len(bin_relevance)) + 1
    # average precision score
    score = np.sum(cum_sum[bin_relevance] / idcs[bin_relevance]) / n_occ
    return score


def map(bin_relevances, occs):
    """calculates mean average precision (MAP) for the relevant hits of retrieval lists"""
    n_samples = len(bin_relevances)
    if not n_samples == len(occs):
        raise ValueError('len(bin_relvances) did not match len(occs)')
    map = 0.
    for idx in range(n_samples):
        map += ap(bin_relevance=bin_relevances[idx], n_occ=occs[idx])
    map /= float(n_samples)
    return map


def ret_list_idcs(attr_vec, phoc_list, metric='cosine'):
    """idcs of sorted retrival list, based on a given metric"""
    dists = cdist([attr_vec], phoc_list, metric=metric)
    idcs = np.argsort(dists)[0]
    return idcs


def relevance(arr_bbox_est, arr_bbox_gt, arr_form_est, arr_form_gt):
    """
    calculates a binary list of relevant elements in a retrieval list

    :param arr_bbox_est: bounding boxes of retrieval list
    :param arr_bbox_gt: bounding boxes of elements with the desired transcription
    :param arr_form_est: page-names/paths of the retrieval list
    :param arr_form_gt: page-names/path of elements with the desired transcription
    :return: binary list indicating relevance of elements in the retrieval list
    """
    # keeping track of evaluated bounding boxes
    N = len(arr_bbox_gt)
    taken = np.zeros(N, dtype=np.uint8)
    # relevant results found
    bin_relevance = []
    # checking for relevant bounding boxes
    for bbox_est, form_est in zip(arr_bbox_est, arr_form_est):
        found = False
        for idx_gt, (bbox_gt, form_gt) in enumerate(zip(arr_bbox_gt, arr_form_gt)):
            # same page
            if form_est == form_gt:
                # same area on page (50% overlap threshold)
                ovlp, bbox = overlap(bbox_gt, bbox_est)
                if ovlp >= 0.5:
                    # marking as relevant bbox_img
                    found = True
                    # appending only if no predecessor has overlapped the gt bbox_img yet (GOOD ESTIMATE)
                    if not taken[idx_gt] == 1:
                        bin_relevance.append(True)
                        taken[idx_gt] = 1
                        break
        # non relevant bbox_img (BAD ESTIMATE)
        if not found:
            bin_relevance.append(False)
        # breaking if all occurrences have been found (all remaining elements would have been False)
        if np.sum(taken) == N:
            break
    found_list = taken.astype(bool).tolist()
    return bin_relevance, found_list


def overlap(bbox1, bbox2):
    """
    Assumption: rectangular bboxes, not areas

    :param bbox1: bbox_img of gt
    :param bbox2: bbox_img of retrieval list
    :return: percentage of area occupied from the gt-bbox_img by the retrieval list-bbox_img
             and the occupied bounding box in that order
    """
    # unpacking bounding boxe
    x00, y00, x01, y01 = bbox1
    x10, y10, x11, y11 = bbox2
    # no overlaps
    if x00 >= x11 or x01 <= x10:
        return 0, (0, 0, 0, 0)
    if y00 >= y11 or y01 <= y10:
        return 0, (0, 0, 0, 0)
    # calculating intersection
    x0 = max(x00, x10)
    y0 = max(y00, y10)
    x1 = min(x01, x11)
    y1 = min(y01, y11)
    # calculating areas
    area_bb1 = abs(x01-x00) * abs(y01-y00)
    area_itsct = abs(x1-x0) * abs(y1-y0)
    # calculating percentage of intersection
    ovlp = area_itsct / area_bb1
    return ovlp, (x0, y0, x1, y1)
