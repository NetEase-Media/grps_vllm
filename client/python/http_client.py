# Http client demo. Complete interface description can be learned from docs/2_Interface.md.

import time
import sys

import requests


def http_request(server, prompt):
    url = f'http://{server}/generate'

    data = {
        'prompt': prompt,
        'temperature': 0.1,
        'top_p': 0.5,
        'max_tokens': 4096
    }

    begin = time.time()
    response = requests.post(url, json=data)
    if response.status_code != 200:
        print(f'Request failed, status code: {response.status_code}')
        return
    end = time.time()
    text = response.content.decode('utf-8')
    token_count = len(text)
    latency = end - begin
    print(text, flush=True)
    print(f'token count: {token_count}, time: {latency}s,'
          f' speed: {token_count / latency} token/s')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 http_client.py <server> <prompt>')
        sys.exit(1)

    while True:
        http_request(sys.argv[1], sys.argv[2])
