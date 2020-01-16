#!/usr/bin/env python
from onmt.bin.translate import load_translator
from onmt.bin.translate import translate


if __name__ == "__main__":
    review = "It smells terrible"
    translator = load_translator()
    summary = translate(review, translator)
    print(summary)
