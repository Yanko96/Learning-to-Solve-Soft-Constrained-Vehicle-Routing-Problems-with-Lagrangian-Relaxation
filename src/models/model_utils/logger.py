import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd


class Logger(object):
    """
    The class for recording the training process.
    """
    def __init__(self, args):
        self.log_interval = args.log_interval
        self.log_name = os.path.join(args.model_dir, args.log_name)
        self.records = []
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)


    def write_summary(self, summary):
        to_print = ""
        for key, value in summary.items():
            to_print += key
            to_print += ": "
            to_print += str(round(value, 3))
            to_print += " "
        print(to_print)
        self.records.append(summary)
        df = pd.DataFrame(self.records)
        df.to_csv(self.log_name, index=False)

