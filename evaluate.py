import torch
import os
import yaml
import fsspec
import tiktoken
from configs import Snapshot, ModelConfig
from networks import TransformerModel

prompt = "Hello, did you know that "



config_file_path = "/home/jacob/PycharmProjects/CSE453FinalProject/configs/transformer_config.yaml"
model_checkpoint_path = "./final_model.pt"


embedder = tiktoken.get_encoding('p50k_base')
snapshot = fsspec.open(model_checkpoint_path)
with snapshot as f:
    snapshot_data = torch.load(f, map_location="cpu")
config = yaml.load(open(config_file_path, 'r'), yaml.FullLoader)
model_config = ModelConfig(**config['model_config'])
snapshot = Snapshot(**snapshot_data)
model = TransformerModel(model_config)
model.load_state_dict(snapshot.model_state)
model.to('cuda')

sentence = torch.tensor(embedder.encode(prompt)).unsqueeze(0).to('cuda')
print(sentence)
for i in range(60):
    output = model(sentence)
    output = torch.argmax(output, dim=2)
    sentence = torch.cat((sentence, output[:, -1].unsqueeze(0)), dim=1)
print(embedder.decode(sentence[0].tolist()))
