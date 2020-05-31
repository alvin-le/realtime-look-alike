#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Indexing classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
import tensorflow as tf

def load_vocab(vocab_file):
    """
    Loads a vocabulary file into dictionary.
    input format: cid gid category
    """
    vocab_cid = collections.OrderedDict()
    vocab_gid = collections.OrderedDict()
    vocab_category = collections.OrderedDict()
    vocab_cid_gid = collections.OrderedDict()
    vocab_cid_category = collections.OrderedDict()
    
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            cols = line.strip().split("\t")
            if 3 > len(cols):
                continue
            cid = cols[0]
            gid = cols[1]
            category = cols[2]
            if cid not in vocab_cid_gid:
                vocab_cid_gid[cid] = gid
            if cid not in vocab_cid_category:
                vocab_cid_category[cid] = category
            if cid not in vocab_cid:
                vocab_cid[cid] = len(vocab_cid.keys())
            if gid not in vocab_gid:
                vocab_gid[gid] = len(vocab_gid.keys())
            if category not in vocab_category:
                vocab_category[category] = len(vocab_category.keys())
    return vocab_cid, vocab_gid, vocab_category, vocab_cid_gid, vocab_cid_category


def convert_by_vocab(vocab, items):
    """Converts a sequence of ids using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


class Indexer(object):
    """Runs end-to-end indexing."""
    
    def __init__(self, vocab_file):
        self.vocab_cid, self.vocab_gid, self.vocab_category, self.vocab_cid_gid, self.vocab_cid_category = load_vocab(vocab_file)
        self.vocab = {
            "cid": self.vocab_cid,
            "gid": self.vocab_gid,
            "category": self.vocab_category,
            "cid_gid": self.vocab_cid_gid,
            "cid_category": self.vocab_cid_category,
        }
        self.cid_cnt = len(self.vocab_cid.keys())
        self.gid_cnt = len(self.vocab_gid.keys())
        self.category_cnt = len(self.vocab_category.keys())
        self.inv_vocab_cid = {v: k for k, v in self.vocab_cid.items()}

    def convert_items_to_ids(self, items):
        cid_ids = convert_by_vocab(self.vocab["cid"], items) 
        gid_ids = convert_by_vocab(self.vocab["gid"],
                                convert_by_vocab(self.vocab["cid_gid"], items)) 
        category_ids = convert_by_vocab(self.vocab["category"],
                                convert_by_vocab(self.vocab["cid_category"], items)) 
        return cid_ids, gid_ids, category_ids
    
    def convert_ids_to_items(self, cids):
        return convert_by_vocab(self.inv_vocab_cid, cids)

