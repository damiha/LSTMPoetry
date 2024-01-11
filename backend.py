from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from lstm.pytorch_lstm import PytorchLSTMModel

app = Flask(__name__)
CORS(app)

# load the text files
with open("lstm/gedichte_large.txt", "r") as f:
    large_text = f.read()

with open("lstm/medium_model/gedichte_medium.txt", "r") as f:
    medium_text = f.read()

chars_large = list(sorted(set(large_text)))
chars_medium = list(sorted(set(medium_text)))

c_to_i_medium = {c : i for i, c in enumerate(chars_medium)}
i_to_c_medium = {i : c for i, c in enumerate(chars_medium)}

c_to_i_large = {c : i for i, c in enumerate(chars_large)}
i_to_c_large = {i : c for i, c in enumerate(chars_large)}

print("Loading the language models")

block_size = 128

n_chars_medium = len(chars_medium)
n_chars_large = len(chars_large)

german_medium_structured = PytorchLSTMModel(n_chars=n_chars_medium, n_embed= 32, block_size=block_size)
german_medium_structured.load_model("lstm/medium_model/german_poems_medium_size_10.pth")

german_large_unstructured = PytorchLSTMModel(n_chars=n_chars_large, n_embed= 32, block_size=block_size)
german_large_unstructured.load_model("lstm/german_poems_large_1.pth")

def encode(s, c_to_i):

    # make sure it can be translated, filter out all unknown charaters
    s = "".join(list(filter(lambda c: c in c_to_i, s)))

    res = list(map(lambda c: c_to_i[c], s))
    return torch.tensor(res)

def decode(t, i_to_c):
    
    if isinstance(t, torch.Tensor):
        t = t.tolist()
        
    return "".join(list(map(lambda i: i_to_c[i], t)))

def post_process(input_string):

    lines = input_string.split('\n')
    output_lines = []
    consecutive_empty_lines = 0

    for i in range(len(lines)):
        # Check if the current line is empty
        if lines[i].strip() == '':
            consecutive_empty_lines += 1
            # Check if it's an empty line followed by a non-empty line
            if i + 1 < len(lines) and lines[i + 1].strip() != '':
                continue
            # Check if it's an empty line after three consecutive lines
            if consecutive_empty_lines > 3:
                continue
        else:
            consecutive_empty_lines = 0

        output_lines.append(lines[i])

    return '\n'.join(output_lines)

@app.route('/generate', methods=['POST'])
def generate():
    # Extract data from the request
    data = request.get_json()
    model = data.get('model')
    prompt = data.get('prompt')
    n_chars = int(data.get('n_chars'))

    # Print the extracted data to the console
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Number of Characters: {n_chars}")

    if model == "german_medium_structured":
        
        context = encode(s = prompt, c_to_i = c_to_i_medium)
        gen_tokens = german_medium_structured.generate(context=context, n=n_chars, temperature=0.8)

        gen_string = decode(gen_tokens, i_to_c=i_to_c_medium)

    elif model == "german_large_unstructured":
        
        context = encode(s = prompt, c_to_i = c_to_i_large)
        gen_tokens = german_large_unstructured.generate(context=context, n=n_chars, temperature=0.8)

        gen_string = decode(gen_tokens, i_to_c=i_to_c_large)

    gen_string = post_process(gen_string)

    # For demonstration, return a sample response
    response = {
        "generated": gen_string
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
