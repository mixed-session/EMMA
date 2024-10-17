import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from safetensors import safe_open


class RetrieverModel(nn.Module):
    def __init__(self):
        super(RetrieverModel, self).__init__()
        self.dialogue_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.memory_encoder = AutoModel.from_pretrained('bert-base-uncased')

    def forward(self, dialogue, positive_memory, negative_memory):
        dialogue_encoding = self.dialogue_encoder(input_ids=dialogue[0], attention_mask=dialogue[1])[0][:, 0, :]
        positive_memory_encoding = self.memory_encoder(input_ids=positive_memory[0], attention_mask=positive_memory[1])[0][:, 0, :]
        negative_memory_encoding = self.memory_encoder(input_ids=negative_memory[0], attention_mask=negative_memory[1])[0][:, 0, :]

        return dialogue_encoding, positive_memory_encoding, negative_memory_encoding


class Retrieval:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = RetrieverModel().to(self.device)
        self.checkpoint = {}
        with safe_open("./retrieval.safetensors", framework="pt", device=0) as f:
            for k in f.keys():
                self.checkpoint[k] = f.get_tensor(k)
        self.model.load_state_dict(self.checkpoint)
        self.model.eval()
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

    def build_memory_embed(self, memory_list):
        memory_embed = torch.zeros((len(memory_list), 768)).to(self.device)
        for i, memory in enumerate(memory_list):
            memory_encoding = self.tokenizer.encode_plus(
                memory.strip(),
                max_length=32,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            output = self.model.memory_encoder(
                input_ids=memory_encoding['input_ids'].to(self.device),
                attention_mask=memory_encoding['attention_mask'].to(self.device)
            )
            output = output.last_hidden_state[:, 0, :]
            memory_embed[i] = output
        return memory_embed

    def predict(self, context, memory_list, top_k=1):
        self.memory_embed = self.build_memory_embed(memory_list)
        context_encoding = self.tokenizer.encode_plus(
            context.strip(),
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        output = self.model.dialogue_encoder(
            input_ids=context_encoding['input_ids'].to(self.device),
            attention_mask=context_encoding['attention_mask'].to(self.device)
        )
        output = output.last_hidden_state[:, 0, :]
        output = output.repeat(len(memory_list), 1)
        distance = self.cosine(output, self.memory_embed)

        score_list = torch.topk(distance, top_k).indices.tolist()
        confidence_list = torch.topk(distance, top_k).values.tolist()

        return memory_list[score_list[0]]
