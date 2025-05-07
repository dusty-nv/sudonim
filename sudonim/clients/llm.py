#!/usr/bin/env python3
# First run 'pip install openai' and start the model server
import openai, os, time

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
  'stream_options': {'include_usage': True},
}

if 'MAX_TOKENS' in os.environ:
  config['max_tokens'] = int(os.environ['MAX_TOKENS'])

print(f"Connected to server: {client.base_url}")
print(f"Found served models: {', '.join(models)}")
print(f"Generation config:   {' '.join([str(k)+'='+str(v) for k,v in config.items()])}")

prompts = [  # generate responses to some default prompts
  "Why did the LLM cross the road?",
  "If a train travels 120 miles in 2 hours, what is its average speed?",
  "Alice’s brother was half her age when she was 6. How old is her brother when she’s 42?",
  "If you were in a race and passed the person in second place, what place would you be in now?",
  "Write a recipe for french onion soup."
]

for prompt in prompts:
  time_beg = time.perf_counter()
  time_tft = None  # Time To First Token latency (TTFT)

  completion = client.chat.completions.create(  # send generation request
    messages=[{
      'role': 'user',
      'content': prompt
    }], **config
  )

  print(f"\n\033[94m{prompt}\033[00m\n")
  
  for chunk in completion:
    if not chunk.choices: # the last chunk has usage stats (not replies)
      usage = chunk.usage
      continue

    delta = chunk.choices[0].delta 
    content = delta.content

    if not content: # check for <thinking> tokens
      reasoning = getattr(delta, 'reasoning_content', None)

      if not reasoning:
        continue

      content = f'\033[2m{reasoning}\033[22m'  # dim CoT

    if time_tft is None: # this was the first output token
      time_tft = time.perf_counter()

    print(content, end='', flush=True)

  # print performance metrics
  time_gen = time.perf_counter() - time_tft
  time_tft = time_tft - time_beg

  print(f"\n\n\033[32m> {config['model']} | " + " | ".join([
    f"Prefill: {usage.prompt_tokens} tokens @ {usage.prompt_tokens/time_tft:.2f} t/s" if usage else '',
    f"Time to First Token: {time_tft:.2f}s",
    f"Generation: {usage.completion_tokens} tokens @ {usage.completion_tokens/time_gen:.2f} t/s" if usage else '',
    f"Total: {time_tft + time_gen:.2f}s",
  ]) + " \033[00m")
