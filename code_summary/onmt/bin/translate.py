#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from code_summary import onmt

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def load_translator():
    parser = ArgumentParser(description='translate.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args()
    ArgumentParser.validate_translate_opts(opt)
    translator = build_translator(opt)
    return translator


def translate(review, translator):
    parser = ArgumentParser(description='translate.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args()
    ArgumentParser.validate_translate_opts(opt)
    # logger = init_logger(opt.log_file)

    # translator = build_translator(opt)
    # src_review = "I could live off of these! I'm not kidding. The regular salted almonds are good too, but I prefer the Smokehouse almonds."
    # src_review = str(opt.review)
    src_review = review
    f_src = open(opt.src,'w',encoding='utf-8')
    f_src.write(src_review)
    f_src.close()
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    shard_pairs = zip(src_shards, tgt_shards)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        # logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug
            )
    f_summary = open(opt.output, 'r', encoding='utf-8')
    summary = f_summary.readlines()[-1].strip('\n')
    f_summary.close()
    return summary
