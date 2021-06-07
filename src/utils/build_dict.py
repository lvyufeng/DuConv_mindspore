#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
File: build_dict.py
"""

from __future__ import print_function
import sys
import functools

def cmp(a, b):
    if a[1] > b[1]:
        return -1
    elif a[1] < b[1]:
        return 1
    else:
        return 0

def build_dict(corpus_file, dict_file):
    """
    build words dict
    """
    dict = {}
    max_frequency = 1
    for line in open(corpus_file, 'r'):
        conversation = line.strip().split('\t')
        for i in range(1, len(conversation), 1):
            words = conversation[i].split(' ')
            for word in words:
                if word in dict:
                    dict[word] = dict[word] + 1
                    if dict[word] > max_frequency:
                        max_frequency = dict[word]
                else:
                    dict[word] = 1

    dict["[PAD]"] = max_frequency + 4
    dict["[UNK]"] = max_frequency + 3
    dict["[CLS]"] = max_frequency + 2
    dict["[SEP]"] = max_frequency + 1
    words = sorted(dict.items(), key=functools.cmp_to_key(cmp), reverse=True)
    fout = open(dict_file, 'w')
    for word, frequency in words:
        fout.write(word + '\n')

    fout.close()


def main():
    """
    main
    """
    if len(sys.argv) < 3:
        print("Usage: " + sys.argv[0] + " corpus_file dict_file")
        exit()
    build_dict(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
