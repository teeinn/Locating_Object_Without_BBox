from flask import Flask, render_template, Response
from distance_inference_one_image_streaming_to_web import distance
from web_gui import WebGUI as UI


class DistanceApp():
    def __init__(self):
        self.engine = distance()
        self.ui = UI(self.engine)
        self.engine.set_ui(self.ui)
        self.ui.start()

if __name__ == '__main__':
    DistanceApp()
