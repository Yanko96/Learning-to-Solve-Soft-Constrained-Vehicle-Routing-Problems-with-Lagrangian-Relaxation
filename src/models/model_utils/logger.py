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
		self.best_reward = 0
		self.records = []
		if not os.path.exists(args.model_dir):
			os.makedirs(args.model_dir)


	def write_summary(self, summary):
		print("global-step: %(global_step)d, avg-reward: %(avg_reward).3f" % summary)
		self.records.append(summary)
		df = pd.DataFrame(self.records)
		df.to_csv(self.log_name, index=False)
		self.best_reward = max(self.best_reward, summary['avg_reward'])
