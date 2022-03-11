import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import time
from ..utils.fps_calculator import convert_infr_time_to_fps
import pycuda.autoinit  # Required for initializing CUDA driver


class Detector:
    """
    Perform object detection with the given prebuilt tensorrt engine.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    :param output_layout:
    """

    def _load_plugins(self):
        """ Required as Flattenconcat is not natively supported in TensorRT. """
        ctypes.CDLL("/opt/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        """ Load engine file as a trt Runtime. """
        trt_bin_path = 'libs/detectors/jetson/data/TRT_%s.bin' % self.model
        with open(trt_bin_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        """
        Create some space to store intermediate activation values. 
        Since the engine holds the network definition and trained parameters, additional space is necessary.
        """
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    def __init__(self, config, output_layout=7):
        """ Initialize TensorRT plugins, engine and conetxt. """
        self.config = config
        self.model = self.config.get_section_dict('Detector')['Name']
        self.class_id = int(self.config.get_section_dict('Detector')['ClassID'])
        self.conf_threshold = self.config.get_section_dict('Detector')['MinScore']
        self.output_layout = output_layout
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()
        self.fps = None

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()  # create a CUDA stream to run inference
        self.context = self._create_context()

    def __del__(self):
        """ Free CUDA memories. """
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    @staticmethod
    def _preprocess_trt(img):
        """ Preprocess an image before TRT SSD inferencing. """
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = (2.0 / 255.0) * img - 1.0
        return img

    def _postprocess_trt(self, img, output):
        """ Postprocess TRT SSD output. """
        img_h, img_w, _ = img.shape
        boxes, confs, clss = [], [], []
        for prefix in range(0, len(output), self.output_layout):
            # index = int(output[prefix+0])
            conf = float(output[prefix + 2])
            if conf < float(self.conf_threshold):
                continue
            x1 = (output[prefix + 3])  # * img_w)
            y1 = (output[prefix + 4])  # * img_h)
            x2 = (output[prefix + 5])  # * img_w)
            y2 = (output[prefix + 6])  # * img_h)
            cls = int(output[prefix + 1])
            boxes.append((y1, x1, y2, x2))
            confs.append(conf)
            clss.append(cls)
        return boxes, confs, clss

    def inference(self, img):
        """
        Detect objects in the input image.

        Args:
            img: uint8 numpy array with shape (img_height, img_width, channels)

        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score": s% }, {...}, {...}, ...]
        """
        img_resized = self._preprocess_trt(img)
        # transfer the data to the GPU, run inference and the copy the results back
        np.copyto(self.host_inputs[0], img_resized.ravel())

        # Start inference time
        t_begin = time.perf_counter()
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        inference_time = time.perf_counter() - t_begin  # Seconds

        # Calculate Frames rate (fps)
        self.fps = convert_infr_time_to_fps(inference_time)
        output = self.host_outputs[0]
        boxes, scores, classes = self._postprocess_trt(img, output)
        result = []
        for i in range(len(boxes)):  # number of boxes
            if classes[i] == self.class_id + 1:
                result.append({"id": str(classes[i] - 1) + '-' + str(i), "bbox": boxes[i], "score": scores[i]})

        return result
