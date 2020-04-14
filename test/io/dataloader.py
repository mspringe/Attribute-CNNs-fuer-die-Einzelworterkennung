"""
unit tests for all custom dataloaders and datasets
.. codeauthor:: Maximilian Springenberg <maximilian.springenberg@tu-dortmund.de>
"""
# unit-test relevant imports
from unittest import TestCase
# libraries
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
# own code
from src.io import dataloader


class TestIAMDataset(TestCase):
    """tests for the IAM Dataset adapter"""

    def setUp(self):
        self.iamdb = dataloader.IAMDataset(csvs_path='/Users/mspringe/dsets/iam/ascii/',
                                           imgs_path='/Users/mspringe/dsets/iam/png/')

    def test_train_test_sets(self):
        train, test = self.iamdb.train_test_official()
        train.display(0)
        test.display(0)
        print(len(train))
        print(len(test))

    def test_len(self):
        print(len(self.iamdb))

    def test_display(self):
        self.iamdb.display(np.random.randint(0, len(self.iamdb)-1))

    def test_retrieval(self):
        dset_iam = self.iamdb
        d_loader = DataLoader(dset_iam)
        # trying to retrieve all possible images
        for itm in d_loader:
            bbox = itm['bbox']
            img = itm['img']
            transcript = itm['transcript']
            phoc = itm['phoc']
            word_id = itm['word_id']

    def test_words(self):
        critical = []
        for idx, w in enumerate(self.iamdb.word_list):
            if '?' in w and len(w) > 1:
                critical.append(idx)
        for idx in critical:
            self.iamdb.display(idx)

    def test_image_collage(self):
        idx = 0
        self.iamdb.augment_imgs = False
        for r in range(2):
            for c in range(4):
                plt.subplot(4,4,idx+1)
                plt.axis('off')
                img_idx = np.random.randint(low=0, high=len(self.iamdb))
                img = self.iamdb[img_idx]['img']
                img = self.iamdb.inv_img(img)
                plt.imshow(img, cmap='bone')
                idx += 1
        plt.show()

    def test_max_scale(self):
        max_h, max_w = 0,0
        avg_h = 0
        avg_w = 0
        avg_area = 0
        for i in range(len(self.iamdb)):
            img =  self.iamdb.img(i)
            h, w = img.shape
            avg_h += h
            avg_w += w
            avg_area += (h*w)
            if w > max_w or h > max_h:
                print(img.shape)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        avg_h /= len(self.iamdb)
        avg_w /= len(self.iamdb)
        avg_area /= len(self.iamdb)
        print('results:')
        print(max_h, max_w)
        print(avg_h, avg_w)
        print(avg_area)

    def test_scale(self):
        self.iamdb.scale = (None, 50)
        self.iamdb.display(6)
        self.iamdb.scale = (None, None)
        self.iamdb.display(6)

    def test_len_words(self):
        train, test = self.iamdb.train_test_official()
        print(len(train.words))
        print(len(test.words))


class TestGWDataset(TestCase):
    """tests for the George Washington Dataset adapter"""

    def setUp(self):
        self.gw_dset = dataloader.GWDataSet(csvs_path='/Users/mspringe/dsets/gwdb/almazan/queries/queries.gtp',
                                            imgs_path='/Users/mspringe/dsets/gwdb/almazan/images')

    def test_augment(self):
        dset_aug = self.gw_dset.sub_set([1]).augment(size=5)
        data = dset_aug[0]
        aug, orig = data['img'], data['non_aug_img']
        plt.subplot(211)
        plt.imshow(aug, cmap='bone')
        plt.subplot(212)
        plt.imshow(orig, cmap='bone')
        plt.show()

    def test_train_test_idcs(self):
        for fold in range(1,5):
            train, test = self.gw_dset.fold(k=fold)

    def test_display(self):
        dset_gw = self.gw_dset
        c = 0
        while c < 10:
            idx = np.random.randint(0,len(dset_gw)-1)
            dset_gw.display(idx)
            c +=1

    def test_retrieval(self):
        for k in range(1,5):
            splits = self.gw_dset.fold(k)
            self.assertEqual(len(splits), 2)
            for split in splits:
                d_loader = DataLoader(split, shuffle=True)
                for itm in d_loader:
                    bbox = itm['bbox']
                    img = itm['img']
                    transcript = itm['transcript']
                    phoc = itm['phoc']
                    word_id = itm['word_id']

    def test_augment(self):
        train, _ = self.gw_dset.fold(1)
        size = 10000
        eq = train.augment(size=size, t_quant=dataloader.DSetQuant.EQUAL)
        resp = train.augment(size=size, t_quant=dataloader.DSetQuant.RESPECTIVE)
        rand = train.augment(size=size, t_quant=dataloader.DSetQuant.RANDOM)
        self.assertEqual(size, len(eq))
        self.assertEqual(size, len(resp))
        self.assertEqual(size, len(rand))
        self.assertEqual(eq.words, train.words)
        self.assertEqual(resp.words, train.words)
        self.assertEqual(rand.words, train.words)
        # equal distribution
        counts = defaultdict(int)
        for trans in eq.word_list:
            counts[trans] += 1
        vals = counts.values()
        self.assertTrue(max(vals) - min(vals) <= 1, 'min={}, max={}'.format(min(vals), max(vals)))
        # respective distribution
        orig_counts = defaultdict(int)
        for trans in train.word_list:
            orig_counts[trans] += 1
        orig_pcts = {}
        for trans in train.words:
            orig_pcts[trans] = float(orig_counts[trans]) / len(train)
        resp_counts = defaultdict(int)
        for trans in resp.word_list:
            resp_counts[trans] += 1
        resp_pcts = {}
        for trans in train.words:
            resp_pcts[trans] = float(resp_counts[trans]) / len(resp)
        for w in train.words:
            self.assertAlmostEqual(orig_pcts[w], resp_pcts[w], delta=0.01)
        # visualize random distribution of word classes
        self.visualize_distribution(eq, 'equal')
        self.visualize_distribution(resp, 'respective')
        self.visualize_distribution(rand, 'random')

    def visualize_distribution(self, dset, title):
        words = list(dset.words)
        cls_to_idcs = dset.cls_to_idcs()
        xx = list(range(len(words)))
        yy = [len(cls_to_idcs[w]) for w in words]
        plt.bar(xx, yy)
        plt.title(title)
        plt.show()

    def test_image_collage(self):
        idx = 0
        self.gw_dset.augment_imgs = False
        for r in range(2):
            for c in range(4):
                plt.subplot(4, 4, idx + 1)
                plt.axis('off')
                img_idx = np.random.randint(low=0, high=len(self.gw_dset))
                img = self.gw_dset[img_idx]['img']
                img = self.gw_dset.inv_img(img)
                plt.imshow(img, cmap='bone')
                idx += 1
        plt.show()

    def test_max_scale(self):
        max_h, max_w = 0,0
        for i in range(len(self.gw_dset)):
            img = self.gw_dset.img(i)
            h, w = img.shape
            if w > max_w or h > max_h:
                print(img.shape)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            if i % 100 == 0:
                print(max_h, max_w)
        print(max_h, max_w)

    def test_len_words(self):
        print(len(self.gw_dset.words))


class TestRimes(TestCase):

    def setUp(self):
        self.dset_rimes = dataloader.RimesDataSet(lower_case=False)

    def test_display(self):
        self.dset_rimes.display(np.random.randint(0,len(self.dset_rimes)-1))

    def test_split(self):
        train, test = self.dset_rimes.train_test_official()
        print(len(train), len(test))

    def test_upper_case(self):
        import string
        from src.util import phoc_util
        self.dset_rimes.lower_case = False
        words = self.dset_rimes.word_list
        self.dset_rimes.apply_alphabet(phoc_util.rep_to_alphabet('ludp'))
        for idx, w in enumerate(words):
            if all(c in string.ascii_uppercase for c in w):
                self.dset_rimes.display(idx)
                uppercas_found = True
                break
        self.assertTrue(uppercas_found)

    def test_words(self):
        #print(self.dset_rimes.words)
        critical = []
        chars = set()
        for idx, w in enumerate(self.dset_rimes.word_list):
            for c in w:
                chars.add(c)
        for idx in critical:
            self.dset_rimes.display(idx)
        print(''.join(sorted(chars)))

    def test_image_collage(self):
        idx = 0
        self.dset_rimes.augment_imgs = False
        for r in range(2):
            for c in range(4):
                plt.subplot(4,4,idx+1)
                plt.axis('off')
                idx_img = np.random.randint(low=0, high=len(self.dset_rimes))
                img = self.dset_rimes[idx_img]['img']
                img = self.dset_rimes.inv_img(img)
                plt.imshow(img, cmap='bone')
                idx += 1
        plt.show()

    def test_max_scale(self):
        avg_h, avg_w = 0,0
        max_h, max_w = 0, 0
        for i in range(len(self.dset_rimes)):
            img = self.dset_rimes.img(i)
            h, w = img.shape
            if w > max_w or h > max_h:
                print(img.shape)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            avg_w += w
            avg_h += h
            if i % 100 == 0:
                print(max_h, max_w)
        avg_w /= len(self.dset_rimes)
        avg_h /= len(self.dset_rimes)
        print(max_h, max_w)
        print(avg_h, avg_w)

    def test_len_words(self):
        train, test = self.dset_rimes.train_test_official()
        w_train = set(train.words)
        w_test = set(test.words)
        print(len(w_train))
        print(len(w_test))
        print(len(set(w_test).union(w_train)))


class TestHWSynth(TestCase):

    def setUp(self):
        self.dset_hws = dataloader.HWSynthDataSet(csvs_path='/media/ssd1/data_sets/hws',
                                                  imgs_path='/media/ssd1/data_sets/hws/Images_90K_Normalized')
        test_len = len(self.dset_hws)

    def test_train_test_official(self):
        train, test = self.dset_hws.train_test_official()
        test_lens = len(train), len(test)

    def test_display(self):
        train = self.dset_hws.sub_set(idcs=self.dset_hws.train_idcs[self.dset_hws.train_idcs<len(self.dset_hws)])
        for i in range(10):
            idx = np.random.randint(0,len(train)-1)
            train.display(idx)
            example_phoc = train.phoc(idx)

    def test_image_collage(self):
        idx = 0
        for r in range(2):
            for c in range(4):
                plt.subplot(4,4,idx+1)
                plt.axis('off')
                img = self.dset_hws[idx]['img']
                img = self.dset_hws.inv_img(img)
                plt.imshow(img, cmap='bone')
                idx += 1
        plt.show()

    def test_retrieval(self):
        splits = self.dset_hws.train_test_official()
        self.assertEqual(len(splits), 2)
        for split in splits:
            d_loader = DataLoader(split)
            for itm in d_loader:
                bbox = itm['bbox']
                img = itm['img']
                transcript = itm['transcript']
                phoc = itm['phoc']
                word_id = itm['word_id']
