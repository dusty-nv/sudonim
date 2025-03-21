#!/usr/bin/env python3
# First run 'pip install openai' and start the model server
import openai, os, requests, base64

client = openai.OpenAI(
  base_url = os.environ.get('OPENAI_BASE_URL', 'http://0.0.0.0:9000/v1'),
  api_key = 'foo',  # not enforced
)

models = [x.id for x in client.models.list()]
print(f"Models from server {client.base_url}  {models}")

url = "https://raw.githubusercontent.com/dusty-nv/jetson-containers/refs/heads/dev/data/images/dogs.jpg"
txt = "What kind of dogs are these?"  # the image shows a husky and golden retriever
img = requests.get(url)  

messages = [{
  'role': 'user',
  'content': [
    { 'type': 'text', 'text': txt },
    {
      'type': 'image_url',
      'image_url': { 
        'url': 'data:' + img.headers['Content-Type'] + ';' +
        'base64,' + base64.b64encode(img.content).decode()
      },
    },
  ],
}]

completion = client.chat.completions.create(
  model=os.environ.get('MODEL', models[0] if models else 'default'), 
  messages=messages,
  stream=True
)

print(f"\n\033[32m{txt} <img>{os.path.basename(url)}</img>\033[00m\n")
  
for chunk in completion:
  print(chunk.choices[0].delta.content, end='', flush=True)
