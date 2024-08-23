import random
import sys
import time
import requests
import threading


latency_list = []
tokens_list = []
speed_list = []

fixed_prompt = '华盛顿是谁？ '

random_prompts = [
    '今天天气真好，适合出门散步 ',
    '中国的长城有多长？ ',
    '人工智能对未来社会有什么影响？ ',
    '传统文化在现代社会中的地位如何？ ',
]

def request(server, random_prompt):
    if random_prompt:
        prompt_idx = random.randint(0, len(random_prompts) - 1)
        text_inp = random_prompts[prompt_idx]
    else:
        text_inp = fixed_prompt

    url = f'{server}/generate'
    data = {
        'prompt': text_inp,
        'temperature': 0.1,
        'top_p': 0.5,
        'max_tokens': 4096
    }
    headers = {'Content-Type': 'application/json'}
    start = time.time()
    response = requests.post(url, json=data, headers=headers)
    end = time.time()
    latency = (end - start) * 1000
    text_output = response.content.decode('utf-8')
    print(text_output, flush=True)
    tokens = len(text_output) + len(text_inp)
    speed = tokens / latency * 1000
    print(f'Latency: {latency} ms', flush=True)
    print(f'Tokens: {tokens}', flush=True)
    print(f'Speed: {speed} tokens/s', flush=True)
    latency_list.append(latency)
    tokens_list.append(tokens)
    speed_list.append(speed)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python trt_http_bench.py <concurrency> <random_prompt>")
        exit(1)
    concurrency = sys.argv[1]
    random_prompt = sys.argv[2]


    def run(server, random_prompt):
        for _ in range(10):
            request(server, random_prompt)


    th = []
    for i in range(int(concurrency)):
        if random_prompt == '0':
            t = threading.Thread(target=run,
                                 args=('http://localhost:7080', False))
        elif random_prompt == '1':
            t = threading.Thread(target=run,
                                 args=('http://localhost:7080', True))
        th.append(t)
        t.start()
    for t in th:
        t.join()

    print(f'Average Latency: {sum(latency_list) / len(latency_list)} ms', flush=True)
    print(f'Average Tokens: {sum(tokens_list) / len(tokens_list)}', flush=True)
    print(f'Average Speed: {sum(speed_list) / len(speed_list) * int(concurrency)} tokens/s', flush=True)

