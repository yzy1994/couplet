from flask import Flask, request
from train import load_vocab
from torch.autograd import Variable
import torch
import json
import sys
PORT = 5000
MODEL_PATH = './model/model3.pkl'

###
USE_CUDA = torch.cuda.is_available()
PADDING = '<pad>'
UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
char2idx = load_vocab()
idx2char = list(char2idx.keys())
model = torch.load(MODEL_PATH)
if USE_CUDA:
    model = model.cuda()
###

app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/couplet')
def couplet():
    output_data = {'result': []}
    if 'inseq' not in request.args.keys():
        output_data['status'] = 'Missing Parameters'
        return json.dumps(output_data)
    inseq = request.args.get('inseq').strip().split(' ')
    input_len = len(inseq)
    if len(inseq)<3 or len(inseq)>15:
        output_data['status'] = 'Invalid Input'
        return json.dumps(output_data)
    x = []
    for char in inseq:
        if char in char2idx.keys():
            x.append(char2idx[char])
        else:
            x.append(char2idx[UNK])
    x.append(char2idx[EOS])
    max_len = len(x)
    x = [x]
    test_x = Variable(torch.LongTensor(x))
    start_decode = Variable(torch.LongTensor([[char2idx[SOS]]] * test_x.size(0)))
    test_x, start_decode = test_x.cuda(), start_decode.cuda()
    preds = model(test_x, start_decode, max_len, None, False)
    preds = torch.max(preds, 1)[1].view(test_x.size(0), max_len)
    result = ''
    for i in range(input_len):
        result += idx2char[preds[0][i].item()]
    result = result.strip()
    output_data['status'] = 'Success'
    output_data['outseq'] = result
    print(result, file=sys.stdout)
    return json.dumps(output_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
