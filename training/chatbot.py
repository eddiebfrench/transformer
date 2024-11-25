from training.transformer import BigramLanguageModel
import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='bpe.model')
encode = lambda s: sp.encode(s, out_type=int)
decode = lambda s: sp.decode(s)

device = 'cuda'

m = BigramLanguageModel(10000)
m.to(device)

m.load_state_dict(torch.load('weights_chat.pth'))


conversation = "A: "

while True:
    conversation += input() + " B: "
    idx = torch.tensor((encode(conversation)), dtype=torch.long, device=device).unsqueeze(0)
    addition = decode(m.generate(idx, max_new_tokens=100, training=False)[0].tolist())
    print(addition[len(conversation):-2])
    conversation += addition



