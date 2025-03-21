#!/usr/bin/env python3
# First run 'pip install openai' and start the model server
import openai, os, time, requests, base64

# generate text+img->text requests addressed to this URL 
client = openai.OpenAI(
  base_url = os.environ.get('OPENAI_BASE_URL', 'http://0.0.0.0:9000/v1'),
  api_key = 'foo',  # not enforced
)

# ollama and vLLM expect the right model name, others do not
models = [x.id for x in client.models.list()]
config = {
  'model': os.environ.get('MODEL', models[0] if models else 'default'),
  'stream': True,
  'stream_options': {'include_usage': True}
}

if 'MAX_TOKENS' in os.environ:
  config['max_tokens'] = int(os.environ['MAX_TOKENS'])

print(f"Connected to server: {client.base_url}")
print(f"Served models found: {', '.join(models)}")
print(f"Generation config:   {' '.join([str(k)+'='+str(v) for k,v in config.items()])}")

# images can be URL's, or loaded/streamed and base64-encoded 
# (except ollama must use base64 and doesn't support image URLs)
image_base_url = 'https://raw.githubusercontent.com/dusty-nv/jetson-containers/refs/heads/dev/data/images'

prompts = [
  # The image shows a pyramid of colored blocks with numbers on them.
  ('Output the color and number of each box.', 'boxes.jpg'),

  # The image shows a cluster of forget-me-not flowers.
  ('What kind of flower is this?', 'flowers.jpg'),

  # The image is showing a highway exit sign to the Hoover Dam.
  ('What does the sign say?', 'hoover.jpg'),

  # The image shows a husky and golden retriever
  ('What kind of dogs are these?', 'dogs.jpg'),

  # An image of a fruit stand at a farmer's market, with handwritten prices on signs
  ('Where are the peaches at and how much do they cost?', 'fruit.jpg'),

  # Two boys playing soccer, one on the left with a red shirt and one on the right in blue
  ('Describe the players, what sport they are playing, and what is occurring with the ball?', 'soccer.jpg'),
]

def encode_image(url, toBase64=True):
  """ Download + convert to base64, or pass through as image_url """
  if not toBase64:
    return url
  response = requests.get(url)
  return ("data:" + 
       response.headers['Content-Type'] + ";" +
       "base64," + base64.b64encode(response.content).decode())

for (text, img) in prompts:
  time_new = time.perf_counter()
  messages = [{ # single-turn VQA (visual question/answering)
    'role': 'user',
    'content': [
      { 'type': 'text', 'text': text },
      {
        'type': 'image_url',
        'image_url': { 
          'url': encode_image(f'{image_base_url}/{img}')
        },
      },
    ],
  }]

  text_reply = usage = ''
  time_query = time.perf_counter()
  completion = client.chat.completions.create(messages=messages, **config)

  print(f"\n\033[94m{text} <img>{img}</img>\033[00m\n")
  
  for chunk in completion:
    if not chunk.choices: # the last chunk has usage, not reply choices
      usage = chunk.usage
      continue

    delta = chunk.choices[0].delta.content

    if not delta:
      continue

    if not text_reply: # record the Time to First Token latency
      time_first = time.perf_counter()

    text_reply += delta
    print(delta, end='', flush=True)

  time_end = time.perf_counter()
  time_ftl = time_first-time_query
  time_gen = time_end-time_first

  print(f"\n\n\033[32m> {config['model']} | " + " | ".join([
    f"Load Image: {time_query-time_new:.2f}s",
    f"Prefill: {usage.prompt_tokens} tokens @ {usage.prompt_tokens/time_ftl:.2f} t/s" if usage else '',
    f"Time to First Token: {time_ftl:.2f}s",
    f"Generation: {usage.completion_tokens} tokens @ {usage.completion_tokens/time_gen:.2f} t/s" if usage else '',
    f"Total: {time_end-time_query:.2f}s",
  ]) + " \033[00m")
