#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--doc_root', default='data/Colon', type=str,
                    help='root document of this run.')

parser.add_argument('--transformation', default='rbf', type=str,
                    help='feature transformation.')

parser.add_argument('--actfun', default='sigmoid', type=str,
                    help='activation function.')

args = parser.parse_args()



