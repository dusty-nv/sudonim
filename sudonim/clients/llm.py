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
  'stream_options': {'include_usage': True}
}

if 'MAX_TOKENS' in os.environ:
  config['max_tokens'] = int(os.environ['MAX_TOKENS'])

print(f"Connected to server: {client.base_url}")
print(f"Found served models: {', '.join(models)}")
print(f"Generation config:   {' '.join([str(k)+'='+str(v) for k,v in config.items()])}")

# generate responses to some default prompts
prompts = [
  "Why did the LLM cross the road?",
  "If a train travels 120 miles in 2 hours, what is its average speed?",
  "Alice’s brother was half her age when she was 6. How old is her brother when she’s 42?",
  "If you were in a race and passed the person in second place, what place would you be in now?",
  "Write a recipe for french onion soup."
]

for prompt in prompts:
  time_new = time.perf_counter()
  messages = [{
    'role': 'user',
    'content': prompt
  }]

  text_reply = usage = ''
  completion = client.chat.completions.create(messages=messages, **config)

  print(f"\n\033[94m{prompt}\033[00m\n")
  
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
    print(delta, end='')

  time_end = time.perf_counter()
  time_ftl = time_first-time_new
  time_gen = time_end-time_first

  print(f"\n\n\033[32m> {config['model']} | " + " | ".join([
    f"Prefill: {usage.prompt_tokens} tokens @ {usage.prompt_tokens/time_ftl:.2f} t/s" if usage else '',
    f"Time to First Token: {time_ftl:.2f}s",
    f"Generation: {usage.completion_tokens} tokens @ {usage.completion_tokens/time_gen:.2f} t/s" if usage else '',
    f"Total: {time_end-time_new:.2f}s",
  ]) + " \033[00m")