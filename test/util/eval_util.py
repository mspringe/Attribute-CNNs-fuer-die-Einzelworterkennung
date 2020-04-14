from unittest import TestCase
from src.util import eval_util
import numpy as np


class TestModule(TestCase):

    def test_map(self):
        bin_relevances = [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ]
        occs = [3, 1, 1]
        self.assertEqual(eval_util.map(bin_relevances=bin_relevances, occs=occs), 1)
        bin_relevances = [
            [0, 0, 1],
        ]
        occs = [1]
        self.assertAlmostEqual(eval_util.map(bin_relevances=bin_relevances, occs=occs), 1/3)

    def test_ap(self):
        bin_relevance = [True, False, False]
        self.assertEqual(eval_util.ap(bin_relevance=bin_relevance, n_occ=1), 1, msg='1')
        bin_relevance = [True, True, False]
        self.assertEqual(eval_util.ap(bin_relevance=bin_relevance, n_occ=2), 1, msg='2')
        bin_relevance = [True, True, True]
        self.assertEqual(eval_util.ap(bin_relevance=bin_relevance, n_occ=3), 1, msg='3')

    def test_overlap(self):
        b1 = [0, 0, 1, 1]
        b2 = [-1, -1, 0, 0]
        b3 = [-0.5, -0.5, 0.5, 0.5]
        b4 = [-0.5, 0, 0.5, 1]

        self.assertEqual(eval_util.overlap(b1, b1), (1, (0, 0, 1, 1)))
        self.assertEqual(eval_util.overlap(b1, b2), (0, (0, 0, 0, 0)))
        self.assertEqual(eval_util.overlap(b1, b3), (0.25, (0, 0, 0.5, 0.5)))
        self.assertEqual(eval_util.overlap(b1, b4), (0.5, (0, 0, 0.5, 1)))

    def test_relevance(self):
        b1 = [0, 0, 1, 1]
        b2 = [-1, -1, 0, 0]
        b3 = [-0.5, -0.5, 0.5, 0.5]
        b4 = [-0.5, 0, 0.5, 1]
        b5 = [-0.51, -0.01, 0.49, 0.99]
        b6 = [-0.49, 0.01, 0.51, 1.01]

        est_bboxes = [b1, b2, b3, b4, b5, b6]
        est_pages = len(est_bboxes) * ['path/to/page']
        gt_bboxes = [b1] * 3
        gt_pages = ['path/to/page'] * 3

        out = eval_util.relevance(est_bboxes, gt_bboxes, est_pages, gt_pages)
        out_correct = ([True, False, False, True, False, True], [True, True, True])
        self.assertEqual(out, out_correct)

    def test_ret_list_idcs(self):
        vecs = np.array([[0, 1],
                         [1, 0]])
        idx = np.random.randint(2)
        self.assertEqual(eval_util.ret_list_idcs(vecs[idx], vecs).tolist(),
                         [idx, 1-idx])
