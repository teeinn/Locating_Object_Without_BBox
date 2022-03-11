from flask import Flask, render_template, Response
import threading
import time
import cv2
from distance_inference_one_image_streaming_to_web import distance

class WebGUI:
    def __init__(self, engine):
        self.host='0.0.0.0'
        self.port= '7000'
        self.engine = engine
        self._lock = threading.Lock()
        self.app = self.create_flask_app()

    def create_flask_app(self):
        app = Flask(__name__)
    
        @app.route('/')
        def index():
            return render_template('./index.html')
    
        @app.route('/video_feed')
        def video_feed():
            return Response(self.gen(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        return app

    def gen(self):
        while True:
            with self._lock:
                (flag, encoded_input_img) = cv2.imencode(".jpeg", self.frame_image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_input_img) + b'\r\n')
            
    def run(self):
        self.app.run(
            host=self.host, debug=True, threaded=True, use_reloader=False,
        )

    def update(self, image):
        self.frame_image= image
        
    def start(self):
        """
        Start the thread's activity.
        It must be called at most once. It runes self._run method on a separate thread and starts
        process_video method at engine instance
        """
        threading.Thread(target=self.run).start()
        time.sleep(1)
        self.engine.get_frame()
        
        
