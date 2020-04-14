"""
This module contains the Dataset-adapters for all relevant datasets

.. moduleauthor:: Maximilian Springenberg <mspringenberg@gmail.com>

|

"""
# extended base-libraries
from copy import deepcopy
from collections import defaultdict
from enum import Enum
from typing import List
import warnings
# libraries
import os
import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
# pytorch relevant imports
from torch.utils.data import Dataset
# own code
from src.util import phoc_util
from src.util.phoc_util import Alphabet
from src.util import augmentation_util


# directory of this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# representation string of DSetQuant instances
REP_STRS = ['eq', 'resp', 'rand']


class DSetQuant(Enum):
    """Enumeration of different quantification approaches, regarding the augmentation of a dataset."""
    RANDOM = 1
    EQUAL = 2
    RESPECTIVE = 3


def quant_to_rep(quant: DSetQuant):
    """parsing an enum instance of :class:`DSetQuant` to a representation string"""
    if quant == DSetQuant.EQUAL:
        return 'eq'
    if quant == DSetQuant.RESPECTIVE:
        return 'resp'
    if quant == DSetQuant.RANDOM:
        return 'rand'
    return None


def rep_to_quant(rep: str):
    """parsing a representation string to an enum instance of :class:`DSetQuant`"""
    if rep == 'eq':
        return DSetQuant.EQUAL
    if rep == 'resp':
        return DSetQuant.RESPECTIVE
    if rep == 'rand':
        return DSetQuant.RANDOM
    return None


class DSetPhoc(Dataset):
    """
    Base-class for a dataset-adapter dealing with PHOC encodings, comes with a set of words and display method.
    Subclasses do not need to write the __getitem__ method **if self.table is a DataFrame with the following collumns**:

        * bbox: x0,y0,x1,y1 -> bounding box of image
        * form_path: path to page / image of origin
        * transcription: the word-imgs transcription
        * word_id: an unique id for the word (usefull for debugging)

    |

    The __getitem__ method then returns a dictionary:

        * bbox: bbox of word in form
        * img: word_img (e.g. augmented, if augment_imgs is True)
        * non_aug_img: none-augmented image (optional, if augment_imgs is True)
        * transcription: transcription of the image
        * phoc: PHOC-encoding of the transcription
        * word_id: an unique id for the word (usefull for debugging)
    """

    def __init__(self, alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION],
                 phoc_levels=3, augment_imgs=True, lower_case=True, scale=(None, None)):
        super().__init__()
        self.scale = scale
        self.table = pd.DataFrame.from_dict({})
        self.alphabet = alphabet
        self.phoc_levels = phoc_levels
        self.form_imgs = {}
        self.lazy_loading = True
        self.trans_idcs = {}
        self.augment_imgs = augment_imgs
        self.lower_case = lower_case

    @property
    def words(self):
        """
        :return: sorted set of word classes in this dataset
        """
        return sorted(set(self.word_list))

    @property
    def word_list(self):
        """
        :return: list of words/ transciptions in this dataset
        """
        return [self.transcript(idx) for idx in range(len(self))]

    @property
    def phoc_list(self):
        """
        :return: list of PHOC in this dataset
        """
        return [phoc_util.phoc(word=word, alphabet=self.alphabet, levels=self.phoc_levels) for word in self.word_list]

    @property
    def bbox_list(self):
        """
        :return: list of bounding boxes in this dataset
        """
        return list(self.table['bbox'])

    @property
    def form_path_list(self):
        """
        :return: list of form-/ image paths in this dataset
        """
        return list(self.table['form_path'])

    @property
    def ids(self):
        """
        :return: list of ids in this dataset
        """
        return [self.id(idx) for idx in range(len(self))]

    def __getitem__(self, idx):
        """
        :param idx: index of item
        :return: dictionary containing:
            * bbox: bbox of word in form
            * img: word_img
            * transcription: transcription of the image
            * phoc: PHOC-encoding of the transcription
            * word_id: an unique id for the word (usefull in debugging / problems with the Model, but no specific use)
        """
        ret_dict = dict()
        ret_dict['bbox'] = self.bbox(idx)
        word_img = self.img(idx)
        if self.augment_imgs:
            ret_dict['non_aug_img'] = deepcopy(word_img)
            word_img = augmentation_util.homography_augm(word_img)
        ret_dict['img'] = word_img
        ret_dict['form'] = self.form(idx)
        ret_dict['transcript'] = self.transcript(idx)
        ret_dict['phoc'] = self.phoc(idx)
        ret_dict['word_id'] = self.id(idx)
        return ret_dict

    def __len__(self):
        return len(self.table)

    def apply_alphabet(self, alphabet: List[Alphabet]):
        """
        makes the dataset fit the alphabet

        :param alphabet: list of alphabet properties
        :return: alternated dataset
        """
        if len(alphabet) == 0:
            raise AttributeError('empty alphabet')
        dset = deepcopy(self)
        # setting lower-case
        dset.lower_case = self.needs_lower(alphabet)
        # no more punctuation
        if Alphabet.ASCII_PUNCTUATION not in alphabet:
            dset = self.exclude_words(phoc_util.alphabet_chars([Alphabet.ASCII_PUNCTUATION]))
        # no more digits
        if Alphabet.ASCII_DIGITS not in alphabet:
            exclude = []
            for w in self.words:
                take = False
                for n in '0123456789':
                    if n in w:
                        take = True
                        break
                if take:
                    exclude.append(w)
            dset = dset.exclude_words(exclude)
        return dset

    def needs_lower(self, alphabet: List[Alphabet]):
        return (Alphabet.ASCII_UPPER not in alphabet)

    def transcript_idcs(self, transcript):
        """
        searches for items with the same transcription in this dataset

        :param transcript: transcript of a word
        :return: indices of occurences in this dataset
        """
        # cached idcs
        if transcript in self.trans_idcs.keys():
            return self.trans_idcs[transcript]
        # calculate
        trans_arr = np.array(self.word_list)
        is_word_arr = (trans_arr == transcript)
        idcs = []
        for idx, is_word in enumerate(is_word_arr):
            if is_word:
                idcs.append(idx)
        # cache idcs
        self.trans_idcs[transcript] = idcs
        return idcs

    def exclude_words(self, words):
        """
        excluding all instances of transcriptions in words from the dataset

        :param words: iterable containing excluded tarnscriptions
        :return: subset without excluded words
        """
        idcs = []
        for i in range(len(self)):
            if not self.transcript(i) in words:
                idcs.append(i)
        subset = self.sub_set(idcs)
        return subset

    def bbox(self, idx):
        """
        bounding box for item at index idx

        :param idx: index of item
        :return: respective bounding box x0,y0, x1,y1
        """
        row = self.table.iloc[idx]
        bbox = row['bbox']
        return bbox

    def img(self, idx):
        """
        image for item at index. NOTE: image is allways unaugmented. Augmentation happends in the __getitem__ method.
        (see also :func:`src.util.augmentation_util.homography_augm` for augmentation)

        :param idx: index of item
        :return: respective image
        """
        # bbox
        bbox = self.bbox(idx)
        form_path = self.form(idx)
        # form
        if form_path not in self.form_imgs.keys():
            img = np.array(Image.open(form_path)).astype(np.uint8)
            self.form_imgs[form_path] = img
        else:
            img = self.form_imgs[form_path]
        # not storing too many images at once
        if len(self.form_imgs) > 100:
            keys = list(self.form_imgs.keys())
            del_k = random.sample(keys, k=1)[0]
            del self.form_imgs[del_k]
        # cutout word-img
        word_img = self.norm_img(self.bbox_img(img, bbox))
        word_img = self.inv_img(word_img)
        word_img = augmentation_util.scale(word_img, *self.scale)
        return word_img

    def form(self, idx):
        """
        the images from

        :param idx: index of item
        :return: path to the images form
        """
        row = self.table.iloc[idx]
        form_path = row['form_path']
        return form_path

    def transcript(self, idx):
        """
        transcription of item

        :param idx: index of item
        :return: respective transcription
        """
        row = self.table.iloc[idx]
        transcript = row['transcription']
        if self.lower_case:
            transcript = transcript.lower()
        return transcript

    def phoc(self, idx):
        """
        Generates the PHOC. The PHOC depends on the global variables self.alphabet, self.phoc_levels.
        (see also :func:`src.util.phoc_util.phoc`)

        :param idx: indesx of item
        :return: respective PHOC
        """
        transcript = self.transcript(idx)
        phoc = phoc_util.phoc(transcript, alphabet=self.alphabet, levels=self.phoc_levels)
        return phoc

    def id(self, idx):
        """
        Unique ID of an item. Sometimes provided by the database, otherwise it should be the index.

        :param idx: index of item
        :return: respective id
        """
        row = self.table.iloc[idx]
        word_id = row['word_id']
        return word_id

    def display(self, index):
        """
        Displaying image at certaint index

        :param index: index of image to be displayed
        """
        img = self.img(index)
        transcription = self.transcript(index)
        plt.imshow(self.norm_img(img), cmap='bone')
        plt.title(transcription, fontdict={'fontsize': 64})
        plt.show()

    def sub_set(self, idcs):
        """
        Generates subset of this set, containing only the items with respective indices in idcs

        :param idcs: idcs of selected items
        :return: SubSet object containing the selected items
        """
        return SubSet(dset=self, idcs=idcs)

    def cls_to_idcs(self):
        # class to idcs of occurrences
        cls_to_idcs = defaultdict(list)
        # indices of new augmented set
        idcs = []
        # fillling classes dict, searching for occurrences
        for i in range(len(self)):
            itm = self[i]
            transcription = itm['transcript']
            cls_to_idcs[transcription].append(i)
        return cls_to_idcs

    def augment(self, size=500000, t_quant=DSetQuant.EQUAL):
        """
        Augmentation of the dataset

        :param size: new size auf augmented dataset
        :param t_quant: type of quantification, regarding the word classes.
                        (see also :class:`src.io.dataloader.DSetQuant`)
        :return: augmented dataset
        """
        N = len(self)
        words = list(self.words)
        # sanity
        if size > N:
            # euqal distribution => all word classes shall have the same number of occurences
            if t_quant == DSetQuant.EQUAL:
                # class to idcs of occurrences
                classes = self.cls_to_idcs()
                # iteration index of respectiv class
                class_iter_idx = {w: 0 for w in words}
                # indices of new augmented set
                idcs = []
                # word classes
                keys = list(classes.keys())
                k_idx = 0
                # collecting samples, iterating word classes
                for i in range(size):
                    k = keys[k_idx]
                    idx = class_iter_idx[k]
                    idcs.append(classes[k][idx])
                    class_iter_idx[k] = (idx+1) % len(classes[k])
                    k_idx = (k_idx+1) % len(keys)
            # respective distribution (to original dataset)
            elif t_quant == DSetQuant.RESPECTIVE:
                idcs = [i % N for i in range(size)]
            # binomial (random) distribution
            elif t_quant == DSetQuant.RANDOM:
                # indices of respective word classes
                cls_idcs = list(range(len(words)))
                # random selection of word classes
                slctd_cls_idcs = [random.sample(cls_idcs, k=1)[0] for _ in range(size)]
                # class to idcs of occurrences
                classes = self.cls_to_idcs()
                # iteration index of respectiv class
                class_iter_idx = {w: 0 for w in words}
                # collecting indices of samples
                idcs = []
                for cls_idx in slctd_cls_idcs:
                    w = words[cls_idx]
                    idx = class_iter_idx[w]
                    idcs.append(classes[w][idx])
                    class_iter_idx[w] = (idx+1) % len(classes[w])
            # sanity
            else:
                raise AttributeError('quantification unknown: {}'.format(t_quant))
        # sanity
        else:
            idcs = list(range(N))
        # generating a DataFrame from collected indices
        table = {k: [] for k in self.table.keys()}
        for idx in idcs:
            row = self.table.iloc[idx]
            for k in row.keys():
                table[k].append(row[k])
        table = pd.DataFrame.from_dict(table)
        # generating the augmented dataset
        augmented_set = deepcopy(self)
        augmented_set.table = table
        augmented_set.idcs = list(range(len(augmented_set)))
        augmented_set.augment_imgs = True
        return augmented_set

    @staticmethod
    def __equal_dist_idcs(transcriptions, n_items):
        # storing idcs by their transcription
        trans_to_idcs = defaultdict(list)
        not_taken = defaultdict(list)
        for i, trans in enumerate(transcriptions):
            trans_to_idcs[trans].append(i)
            not_taken[trans].append(True)
        # equal amount of all Transcription
        idcs = []
        n_items = min(n_items, len(transcriptions)-1)
        keys = list(trans_to_idcs.keys())
        idx_k = 0
        # collecting idcs
        while n_items >= 0:
            # choosing transcript
            key = keys[idx_k]
            # checking if there are idcs left to take
            if any(not_taken[key]):
                # choosing one idx at random
                arr = []
                for trans_idx, n_taken in zip(trans_to_idcs[key], not_taken[key]):
                    if n_taken:
                        arr.append(trans_idx)
                idx = random.sample(arr, k=1)[0]
                # marking index as taken
                taken_idx = trans_to_idcs[key].index(idx)
                not_taken[key][taken_idx] = False
                # storing index
                idcs.append(idx)
                n_items -= 1
            # iteration of keys -> no overflow
            idx_k += 1
            if idx_k == len(keys):
                idx_k = 0
        return idcs

    @staticmethod
    def norm_img(img):
        """
        normalizing image for better visual

        :param img: image of any type
        """
        img_arr = np.array(img).astype(float)
        max_val = np.amax(img_arr)
        if max_val > 0:
            img_arr /= max_val
        return img_arr

    @staticmethod
    def inv_img(img):
        """inverts an image with real values in [0,1]"""
        return np.abs(img - 1.)

    @staticmethod
    def bbox_img(img, bbox):
        """
        extracts infield of image

        :param img: base image (e.g. page)
        :param bbox: bounding box
        :return: infield of bounding box
        """
        if len(bbox) == 4:
            return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        else:
            return img


class SubSet(DSetPhoc):
    """allows easy cutting of a dataset via a list of indices"""

    def __init__(self, dset: DSetPhoc, idcs):
        super().__init__()
        # filtering for indices
        self.table = dset.table.iloc[idcs]
        self.idcs = idcs
        # keeping all other attributes
        self.alphabet = dset.alphabet
        self.phoc_levels = dset.phoc_levels
        self.form_imgs = dset.form_imgs
        self.lazy_loading = dset.lazy_loading
        self.trans_idcs = {}
        self.augment_imgs = dset.augment_imgs
        self.scale = dset.scale


class IAMDataset(DSetPhoc):
    """pytorch Dataset adapter of the IAM dataset"""

    def __init__(self, sep=' ', lazy_loading=True, phoc_levels=3, good_segmentation=False, lower_case=True,
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION], scale=(None, None),
                 csvs_path=os.path.join('/data', 'mspringe', 'iamDB', 'ascii', ''),
                 imgs_path=os.path.join('/data', 'mspringe', 'iamDB', 'forms', 'png', '')):
        """
        :param sep: delimiter of the words.txt file
        :param lazy_loading: Indicates whether all data shall be loaded at once on construct or it shall be loaded
                             'lazy', when it is needed. default: True
        :param phoc_levels: levels of PHOC-Encoding
        :param good_segmentation: Filters for 'ok' segmentation if set to True. default: False
        :param csvs_path: path to the directory of the words.txt file
        :param imgs_path: path to the directory, where all forms are stored
        """
        super().__init__(alphabet=alphabet, phoc_levels=phoc_levels, scale=scale)
        table = defaultdict(list)
        imgs = {}
        # parsing dataset: skipping comments, no headers | gathering: id, path, transcription, bbox_img and segmentation
        self.lower_case = lower_case
        self.csv_path_train = os.path.join(FILE_DIR, 'iam_train.txt')
        self.csv_path_test = os.path.join(FILE_DIR, 'iam_test.txt')
        self.path_stop_words = os.path.join(csvs_path, 'swIAM.txt')
        csv_path_words = os.path.join(csvs_path, 'words.txt')
        with open(csv_path_words, 'r') as csv_f:
            for line in csv_f:
                # skipping comments
                if not line.startswith('#'):
                    # gather relevant information
                    contents = line.split(sep)
                    # segmentation
                    res_seg = contents[1]
                    # skipping segmentation errors of dataset if required
                    if good_segmentation and res_seg != 'ok':
                        continue
                    # bbox
                    bbox = contents[3:7]
                    x0, y0, w, h = [int(str(coord)) for coord in bbox]
                    x1, y1 = x0 + w, y0 + h
                    bbox = x0, y0, x1, y1
                    # one bbox_img has values  -1 -1 -1 -1 ??? => sanity checking validity of bboxes
                    if np.amin(bbox) < 0:
                        continue
                    # id
                    word_id = contents[0]
                    # form file path
                    form_f_name = self.id_to_form_f_name(word_id)
                    form_path = os.path.join(imgs_path, form_f_name)
                    if not os.path.isfile(form_path):
                        continue
                    if not lazy_loading:
                        try:
                            if form_path not in imgs.keys():
                                img = np.array(Image.open(form_path)).astype(np.uint8)
                                imgs[form_path] = img
                        except OSError:
                            continue
                    # transcription
                    transcription = contents[-1].replace('\n', '')
                    # if lower_case:
                    #     transcription = transcription.lower()
                    # storing of data
                    table['word_id'].append(word_id)
                    table['form_path'].append(form_path)
                    table['bbox'].append(bbox)
                    table['res_seg'].append(res_seg)
                    table['transcription'].append(transcription)
        # dataframe containing the meta-data
        self.table = pd.DataFrame.from_dict(table)
        # indicates whether data has been loaded on construct or lazy loading is to be applied
        self.lazy_loading = lazy_loading
        # dictionary of forms
        self.form_imgs = imgs

    def train_test_official(self, stop_words=True):
        """
        generates the official IAM-DB train and test subsets
        (see http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip)

        :return: train and test subsets in that order
        """
        return self.train_set_official(stop_words=stop_words), self.test_set_official(stop_words=stop_words)

    def train_set_official(self, stop_words=True):
        """
        generates the official IAM-DB train subset
        (see http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip)

        :return: official training subset
        """
        subset = self.__sub_set_from_linescsv(csv_path=self.csv_path_train)
        if not stop_words:
            with open(self.path_stop_words, 'r') as f_sw:
                stop_words = []
                for line in f_sw:
                    stop_words += line.split(',')
            subset = subset.exclude_words(stop_words)
        return subset

    def test_set_official(self, stop_words=True):
        """
        generates the official IAM-DB test subset
        (see http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip)

        :return: official testing subset
        """
        subset = self.__sub_set_from_linescsv(csv_path=self.csv_path_test)
        if not stop_words:
            with open(self.path_stop_words, 'r') as f_sw:
                stop_words = []
                for line in f_sw:
                    stop_words += line.split(',')
            subset = subset.exclude_words(stop_words)
        return subset

    def __sub_set_from_linescsv(self, csv_path):
        """extracts line-ids from a csv file and generates the respective subset"""
        # gather line ids
        ids = []
        with open(csv_path, 'r') as f_csv:
            reader = csv.reader(f_csv, delimiter=' ')
            for line_id in reader:
                ids.append(line_id[0])
        # create sub-set based on the indices
        sub_set = self.__line_id_subset(ids=ids)
        return sub_set

    def __line_id_subset(self, ids):
        """generates a subset for a list of line ids"""
        idcs = []
        for idx in range(len(self.table)):
            data = self.table.iloc[idx]
            # extract line_id
            word_id = data['word_id']
            line_id = self.word_id_to_line_id(word_id=word_id)
            # map to indices
            if line_id in ids:
                idcs.append(idx)
        sub_set = self.sub_set(idcs=idcs)
        return sub_set

    def needs_lower(self, alphabet: List[Alphabet]):
        return super().needs_lower(alphabet=alphabet) and (Alphabet.PERFECT_IAM not in alphabet)

    @staticmethod
    def id_to_path(img_id):
        """
        :param img_id: id specified by the IAM-dataset
        :return: path to the image from the words/ directory of the dataset
        """
        img_info = img_id.split('-')
        base_dir = img_info[0]
        sub_dir = '-'.join(img_info[:2])
        file_name = img_id + '.png'
        return os.path.join(base_dir, sub_dir, file_name)

    @staticmethod
    def id_to_form_f_name(img_id):
        """
        :param img_id: id specified by the IAM-dataset
        :return: file name of the corresponding form
        """
        img_info = img_id.split('-')
        form_name = '-'.join(img_info[:2])
        form_f_name = form_name + '.png'
        return form_f_name

    @staticmethod
    def word_id_to_line_id(word_id):
        """
        :param word_id: word id
        :return: line id
        """
        img_info = word_id.split('-')
        line_id = '-'.join(img_info[:2])
        return line_id


class GWDataSet(DSetPhoc):
    """
    pytorch Dataset adapter of the George Washingtion dataset


    .. note::

        We are using the almazan 4 fold cross validation for the George Washington dataset.
        The queries and splits can be obtained from `the links in this repository <https://github.com/almazan/watts/tree/master/datasets>`__.
    """

    def __init__(self, sep=' ', lazy_loading=False, phoc_levels=3, lower_case=True,
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION], scale=(None, None),
                 csvs_path=os.path.join('/data', 'mspringe', 'gwdb', 'almazan', 'queries', 'queries.gtp'),
                 imgs_path=os.path.join('/data', 'mspringe', 'gwdb', 'almazan', 'images', '')):
        """
        :param sep: seperator used in the gt-files
        :param lazy_loading: Indicates whether all data shall be loaded at once on construct or it shall be loaded
                             'lazy', when it is needed. default: False
        :param phoc_levels: levels of PHOC-Encoding
        :param imgs_path: path to pages directory
        :param csvs_path: path to the queries file
        """
        super().__init__(scale=scale)
        table = defaultdict(list)
        imgs = {}
        self.lower_case = lower_case
        # parsing csv
        with open(csvs_path, 'r')as f_queries:
            reader = csv.reader(f_queries, delimiter=sep)
            word_id = 0
            for form_name, x1, y1, x2, y2, transcription in reader:
                table['bbox'].append([int(x1), int(y1), int(x2), int(y2)])
                # if lower_case:
                #     transcription = transcription.lower()
                table['transcription'].append(transcription)
                form_path = os.path.join(imgs_path, form_name)
                table['form_path'].append(form_path)
                table['word_id'].append(word_id)
                word_id += 1
                if not lazy_loading and form_path not in imgs.keys():
                    imgs[form_path] = np.array(Image.open(form_path)).astype(np.uint8)
        # setting DSetPhoc globals
        self.form_imgs = imgs
        self.table = pd.DataFrame.from_dict(table)
        self.lazy_loading = lazy_loading
        self.alphabet = alphabet
        self.phoc_levels = phoc_levels
        self.trans_idcs = {}
        self.augment_imgs = True
        # indices used by Almazan for a 4 fold cross validation
        p_almazan_idcs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'almazan_idcs.npy')
        self.fold_idcs = np.load(p_almazan_idcs)

    def fold(self, k):
        """generates the k-ths fold of the Almazan 4-fold cross validation"""
        if not (0 < k < 5):
            raise ValueError('almazan cross-evaluation proposes a k=4 k-cross evaluation, {} out of range'.format(k))
        test_idcs = [idx for idx, fold in enumerate(self.fold_idcs) if fold == k]
        train_idcs = list(set(range(len(self))).difference(set(test_idcs)))
        return self.sub_set(train_idcs), self.sub_set(test_idcs)


class RimesDataSet(DSetPhoc):
    """
    pytorch Dataset adapter of the RIMES dataset

    .. note::

        * csvs_path should lead to the directory where test and training annotations are stored
        * imgs_path should lead to the directory where all images are stored
    """

    def __init__(self, sep=' ', lazy_loading=True, phoc_levels=3, augment=False, lower_case=True, scale=(None, None),
                 alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION],
                 csvs_path=os.path.join('/data', 'mspringe', 'rimes', 'gt', ''),
                 imgs_path=os.path.join('/data', 'mspringe', 'rimes', 'imgs', '')):
        super().__init__(scale=scale)
        self.sep = sep
        self.lazy_loading = lazy_loading
        self.phoc_levels = phoc_levels
        self.alphabet = alphabet
        self.augment_imgs = augment
        self.lower_case = lower_case
        self.imgs_path = imgs_path
        self.form_imgs = {}
        # searching for training and test GT
        csvs = [os.path.join(csvs_path, fn) for fn in os.listdir(csvs_path)]
        self.p_train = None
        self.p_test = None
        for p_csv in csvs:
            if 'test' in p_csv.lower():
                self.p_test = p_csv
            else:
                self.p_train = p_csv
        if self.p_train is None or self.p_test is None:
            raise FileExistsError('could not find train and test GT-annotations in {}'.format(csvs_path))
        self.idcs_test = []
        self.idcs_train = []
        # parsing csv
        self.table, (self.idcs_train, self.idcs_test) = self.reload_rimes()

    def reload_rimes(self):
        """reloading the dataset"""
        def __load_gt(df_table, df_idx, p_csv):
            """method to parse a gt-annotation"""
            idx_start = df_idx
            with open(p_csv, 'rb') as f_csv:
                for line in f_csv:
                    # decoding french letters and mapping them to UTF-8
                    str_line = line.decode('utf-8', 'replace').strip()
                    # gt annotations may end with empty lines
                    if str_line == '':
                        continue
                    # parsing line
                    img_path, transcript = str_line.split(self.sep)
                    img_path = img_path.encode('ascii', 'ignore').decode('ascii', 'ignore')
                    # parsing french to ASCII
                    transcript = transcript#.lower()
                    transcript = self.to_ascii(transcript)
                    # determine img path
                    img_path = os.path.join(self.imgs_path, img_path)
                    # filling table
                    # if self.lower_case:
                    #     transcript = transcript.lower()
                    if not os.path.isfile(img_path):
                        pass
                        # raise FileNotFoundError('No image exists at {}'.format(img_path))
                        warnings.warn('No image exists at {}'.format(img_path))
                        continue
                    if not self.lazy_loading:
                        self.form_imgs[img_path] = np.array(Image.open(img_path)).astype(np.uint8)
                    # storing relevant data
                    df_table['transcription'].append(transcript)
                    df_table['form_path'].append(img_path)
                    df_table['word_id'].append(df_idx)
                    df_table['bbox'].append([])
                    df_idx += 1
            return df_table, df_idx, list(range(idx_start, df_idx))
        # initializing table and index counter for training/ test splits
        idx = 0
        table = defaultdict(list)
        # loading training data
        table, idx, idcs_train = __load_gt(df_table=table, df_idx=idx, p_csv=self.p_train)
        # loading test data
        table, idx, idcs_test = __load_gt(df_table=table, df_idx=idx, p_csv=self.p_test)
        # DataFrame conversion
        table = pd.DataFrame.from_dict(table)
        return table, (idcs_train, idcs_test)

    def train_test_official(self):
        """the official training and test splits"""
        return self.sub_set(self.idcs_train), self.sub_set(self.idcs_test)

    def needs_lower(self, alphabet: List[Alphabet]):
        return super().needs_lower(alphabet=alphabet) and (Alphabet.PERFECT_RIMES not in alphabet)

    @staticmethod
    def to_ascii(word_str: str):
        """mapping a french word to an ascii encoding"""
        # grammars/ definitions for the mapping of characters
        non_ascii = 'âàêèëéîïôçûùü'
        ascii_mapping = {'âà': 'a',
                         'êèëé': 'e',
                         'îï': 'i',
                         'ô': 'o',
                         'ç': 'c',
                         'ûùü': 'u'}
        non_ascii_upper = non_ascii.upper()
        ascii_mapping_upper = {k.upper(): ascii_mapping[k].upper() for k in ascii_mapping.keys()}
        # building the ascii string
        ret_str = ''
        for char in word_str:
            # lower case french
            if char in non_ascii:
                k = None
                for k_chars in ascii_mapping.keys():
                    if char in k_chars:
                        k = k_chars
                        break
                if k is not None:
                    ret_str += ascii_mapping[k]
            # upper case french
            elif char in non_ascii_upper:
                k = None
                for k_chars in ascii_mapping_upper.keys():
                    if char in k_chars:
                        k = k_chars
                        break
                if k is not None:
                    ret_str += ascii_mapping_upper[k]
            # regular ascii
            else:
                ret_str += char
        # ascii encoding of replaces characters
        ascii_str = ret_str.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        # if '?' in ret_str.encode('ascii', 'replace').decode('ascii', 'ignore'):
        #     #print(word_str, ret_str.encode('ascii', 'replace').decode('ascii', 'ignore'), ascii_str)
        #     print(word_str, ret_str.encode('ascii', 'replace').decode('ascii', 'ignore'), ascii_str)
        #     pass
        #    return ret_str.encode('ascii', 'replace').decode('ascii', 'ignore')
        return ascii_str


class HWSynthDataSet(DSetPhoc):
    """
    pytorch Dataset adapter of the IIIT-HWSynth dataset

    .. note::

        * csvs_path can lead to the directory where the annotations are stored or directly to the annotations
        * imgs_path leads to the directory where the images are stored

    When training on IIIT-HWSynth I would suggest to pass the following arguments:

    ::

        python3 src/training/phocnet_trainer.py \\
        path/to/dir_out \\
        hws \\
        path/to/hws/annotations \\
        path/to/hws/Images_90K_Normalized \\
        --max_iter=1e5 \\
        --model_name=pretrained_PHOCNet \\
        --gpu_idx=cuda:0 \\
        --k_fold=1 \\
        --alphabet=ldp \\
        --s_batch=32 \\
        --augment=none \\
        --shuffle=false
    """

    def __init__(self, phoc_levels=3, alphabet=[Alphabet.ASCII_LOWER, Alphabet.ASCII_DIGITS, Alphabet.ASCII_PUNCTUATION],
                 augment=False, scale=(None, None), lower_case=True,
                 csvs_path=os.path.join('/data', 'mspringe', 'hw_synth', 'groundtruth', ''),
                 imgs_path=os.path.join('/vol', 'corpora', 'document-image-analysis', 'hw-synth',
                                        'Images_90K_Normalized', '')):
        super().__init__(scale=scale, lower_case=lower_case)
        self.phoc_levels = phoc_levels
        self.lazy_loading = False
        self.alphabet = alphabet
        self.augment_imgs = augment
        # searchign GT annotations
        if os.path.isdir(csvs_path):
            dir = csvs_path
        else:
            dir = os.path.dirname(csvs_path)
        files = os.listdir(dir)
        p_train, p_test, p_file_names = 'IIIT-HWS-10K-train-indices.npy', 'IIIT-HWS-10K-val-indices.npy', \
                                        'IIIT-HWS-10K.npy'
        # numpy files already exist
        if all(p in files for p in (p_train, p_test, p_file_names)):
            # loading splits
            self.test_idcs = np.load(os.path.join(dir, p_test))
            self.train_idcs = np.load(os.path.join(dir, p_train))
            file_names = np.load(os.path.join(dir, p_file_names))
        # numpy annotations get created (loads faster)
        else:
            file_names = []
            self.test_idcs = []
            self.train_idcs = []
            df = pd.read_csv(csvs_path, sep=' ', header=None)
            for i in range(len(df)):
                pth, ts, id, is_test = df.iloc[i]
                file_names.append([pth, ts])
                if bool(is_test):
                    self.test_idcs.append(i)
                else:
                    self.train_idcs.append(i)
            self.test_idcs = np.array(self.test_idcs)
            self.train_idcs = np.array(self.train_idcs)
            file_names = np.array(file_names)
            # saving
            for name, arr in [(p_train, self.train_idcs), (p_test, self.test_idcs), (p_file_names, file_names)]:
                np.save(os.path.join(dir, name), arr)
        # NOTE: a million images, hence dictionary approach not feasible
        self.form_imgs = {}
        # filling table
        table = defaultdict(list)
        for i, (p_img, transcript) in enumerate(file_names):
            p_img = str(p_img)
            transcript = str(transcript)
            form_path = os.path.join(imgs_path, p_img)
            bbox = []
            img_id = p_img
            # sanity
            if not os.path.isfile(form_path):
                warnings.warn('{} does not exist'.format(form_path))
                continue
            # storing data
            table['transcription'].append(transcript)
            table['form_path'].append(form_path)
            table['word_id'].append(img_id)
            table['bbox'].append(bbox)
        self.table = pd.DataFrame.from_dict(table)

    def train_test_official(self):
        """the official training and test splits"""
        return self.sub_set(self.train_idcs), self.sub_set(self.test_idcs)

    def needs_lower(self, alphabet: List[Alphabet]):
        return super().needs_lower(alphabet=alphabet) and (Alphabet.PERFECT_RIMES not in alphabet) \
               and (Alphabet.PERFECT_IAM not in alphabet)