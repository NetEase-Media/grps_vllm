# Build grps customized project image.

# --------Building stage.--------
FROM registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda11.8_cudnn8.6_vllm0.4.3_py3.10 AS build

# grps archive.
RUN mkdir -p /my_grps
ADD conf /my_grps/conf
ADD data /my_grps/data
ADD src /my_grps/src
ADD requirements.txt /my_grps/requirements.txt
ADD test.py /my_grps/test.py
RUN cd /my_grps && \
    grpst archive . --skip_unittest --output_path server.mar

# --------Release stage.--------
FROM registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda11.8_cudnn8.6_vllm0.4.3_py3.10

# Intall requirements.
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /my_grps
COPY --from=build /my_grps/server.mar /my_grps/server.mar
ENV LANG C.UTF-8

# Can use --timeout (s) to set the timeout time, default is 300s.
CMD ["/bin/sh", "-c", "grpst start server.mar"]