# Customized deep learning model inferer. Including model load and model infer.
import json
import threading

from grps_framework.context.context import GrpsContext
from grps_framework.logger.logger import clogger
from grps_framework.monitor.monitor import app_monitor
from grps_framework.model_infer.inferer import inferer_register, ModelInferer
from vllm import EngineArgs, LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


class VllmInferer(ModelInferer):
    class Job:
        def __init__(self, context, prompt='', request=None):
            self._complete_condition = threading.Condition()
            self._context = context
            self._last_len = 0
            self._prompt = prompt
            self._request = request

        def wait(self):
            with self._complete_condition:
                self._complete_condition.wait()
                return

        def done(self):
            with self._complete_condition:
                self._complete_condition.notify()

        @property
        def prompt(self):
            return self._prompt

        @property
        def request(self):
            return self._request

        @property
        def context(self):
            return self._context

        @property
        def last_len(self):
            return self._last_len

        @last_len.setter
        def last_len(self, value):
            self._last_len = value

    def __init__(self):
        super().__init__()
        self._engine_args = None
        self._engine = None
        self._worker_thread = None
        self._job_map = {}
        self._job_cv = threading.Condition()

    def init(self, path, device=None, args=None):
        """
        Initiate model inferer class with model path and device.

        Args:
            path: Model path, it can be a file path or a directory path.
            device: Device to run model.
            args: More args.

        Raises:
            Exception: If init failed, can raise exception. Will be caught by server and show error message to user when
            start service.
        """
        super(VllmInferer, self).init(path, device, args)
        self._engine_args = EngineArgs(**args)

    def load(self):
        """
        Load model from model path.

        Returns:
            True if load model successfully, otherwise False.

        Raises:
            Exception: If load failed, can raise exception and exception will be caught by server and show error message
            to user when start service.
        """
        clogger.info('vllm inferer loading...')
        self._engine = LLMEngine.from_engine_args(self._engine_args)
        self._engine.log_stats = False
        self._worker_thread = threading.Thread(target=self.worker_fn)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        clogger.info('vllm inferer load, engine_args: {}'.format(self._engine_args))
        return True

    def worker_fn(self):
        while True:
            with self._job_cv:
                self._job_cv.wait()

            while self._engine.has_unfinished_requests():
                request_outputs = self._engine.step()
                for request_output in request_outputs:
                    job = self._job_map[request_output.request_id]
                    text = request_output.outputs[0].text

                    if job.context.if_streaming():
                        if job.context.if_disconnected():  # Abort the request if the client disconnects.
                            self._engine.abort_request(request_output.request_id)
                            continue
                        job.context.customized_http_stream_respond(text[job.last_len:])

                    # monitor throughput
                    app_monitor.inc('tp(token/s)',
                                    (len(text) - job.last_len + (len(job.prompt) if job.last_len == 0 else 0)))
                    job.last_len = len(text)

                    if request_output.finished:
                        self._job_map.pop(request_output.request_id)
                        if not job.context.if_streaming():
                            job.context.set_http_response(text)
                        job.done()

    def infer(self, inp, context: GrpsContext):
        """
        The inference function is used to make a prediction call on the given input request.

        Args:
            context: grps context
            inp: Model infer input, which is output of converter preprocess function. When in `no converter mode`, will
            skip converter preprocess and directly use GrpsMessage as input.

        Returns:
            Model infer output, which will be input of converter postprocess function. When in `no converter mode`, it
            will skip converter postprocess and should directly use GrpsMessage as output.

        Raises:
            Exception: If infer failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        request = context.get_http_request()
        req_data = request.get_json()
        # clogger.info('req_data: {}'.format(req_data))
        prompt = req_data.pop('prompt')
        sampling_params = SamplingParams(**req_data)
        request_id = random_uuid()

        if not prompt:
            raise ValueError('prompt is empty.')

        job = self.Job(context, prompt, request)
        with self._job_cv:
            self._job_map[request_id] = job
            self._engine.add_request(request_id, prompt, sampling_params)
            self._job_cv.notify()
        job.wait()


# Register
inferer_register.register('vllm', VllmInferer())
