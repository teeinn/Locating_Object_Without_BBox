#!/usr/bin/python3
import sys

import argparse

from libs.core import Distancing as CvEngine
from libs.config_engine import ConfigEngine
from ui.web_gui import WebGUI as UI

class DistanceApp():
    def __init__(self, args):
        self.config = ConfigEngine(args.config)
        self.engine = CvEngine(self.config)
        self.ui = UI(self.config, self.engine)
        self.engine.set_ui(self.ui)
        self.ui.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    DistanceApp(args)
