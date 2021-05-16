import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .layer import AttentionLayer as AL, GlobalAttentionLayer as GoAL, \
    StructAttentionLayer as SAL, ResAttentionLayer as RAL, ContAttentionLayer as CAL
from .dataset import get_lm_path


class TranHGAT(nn.Module):
    def __init__(self, attr_num, device='cpu', finetuning=True, lm='bert', lm_path=None):
        super().__init__()

        # load the model or model checkpoint
        path = get_lm_path(lm, lm_path)
        self.lm = lm
        if lm == 'bert':
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(path)
        elif lm == 'distilbert':
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained(path)
        elif lm == 'roberta':
            from transformers import RobertaModel
            self.bert = RobertaModel.from_pretrained(path)
        elif lm == 'xlnet':
            from transformers import XLNetModel
            self.bert = XLNetModel.from_pretrained(path)

        self.device = device
        self.finetuning = finetuning

        # hard corded for now
        hidden_size = 768
        hidden_dropout_prob = 0.1

        self.inits = nn.ModuleList([
            GoAL(hidden_size, 0.2)
            for _ in range(attr_num)])
        self.alls = nn.ModuleList([
            GoAL(hidden_size, 0.2)
            for _ in range(attr_num)])
        self.oves = nn.ModuleList([
            CAL(hidden_size + hidden_size, 0.2)
            for _ in range(attr_num)])
        self.conts = nn.ModuleList([
            AL(hidden_size + hidden_size, 0.2, device)
            for _ in range(attr_num)])
        self.out = SAL(hidden_size * (attr_num + 1), 0.2)
        self.res = RAL(hidden_size, 0.2, 1/17)

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, xs, zs, y, masks):
        xs = xs.to(self.device)
        zs = zs.to(self.device)
        y = y.to(self.device)
        masks = masks.to(self.device)

        xs = xs.permute(1, 0, 2) #[Attributes, Batch, Tokens]
        masks = masks.permute(0, 2, 1) # [Batch, All Tokens, Attributes]

        attr_outputs = []
        pooled_outputs = []
        attns = []
        if self.training and self.finetuning:
            self.bert.train()
            for x, z, init, all, ove, cont in zip(xs, zs, self.inits, self.alls, self.oves, self.conts):
                attr_embeddings = init(self.bert.get_input_embeddings()(x)) # [Batch, Hidden]
                all_embedding = all(self.bert.get_input_embeddings()(z)) # [1, Hidden]

                attr_embeddings = ove(attr_embeddings, all_embedding)
                attr_outputs.append(attr_embeddings)

                attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings) # [Batch, All Tokens]
                attns.append(attn)

            attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks # [Batch, All Tokens, Attributes]
            attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2) # [Batch, Attributes, Hidden]
            for x in xs:
                if self.lm == 'distilbert':
                    words_emb = self.bert.embeddings(x)
                else:
                    words_emb = self.bert.get_input_embeddings()(x)

                for i in range(words_emb.size()[0]):
                    words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

                output = self.bert(inputs_embeds=words_emb)

                pooled_output = output[0][:, 0, :]
                pooled_output = self.dropout(pooled_output)
                pooled_outputs.append(pooled_output)

            # Struct Attention
            attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
            entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1)
            entity_output = self.res(self.out(attr_outputs, entity_outputs))
        else:
            self.bert.eval()
            with torch.no_grad():
                for x, z, init, all, ove, cont in zip(xs, zs, self.inits, self.alls, self.oves, self.conts):
                    attr_embeddings = init(self.bert.get_input_embeddings()(x))  # [Batch, Hidden]
                    all_embedding = all(self.bert.get_input_embeddings()(z))  # [1, Hidden]

                    attr_embeddings = ove(attr_embeddings, all_embedding)
                    attr_outputs.append(attr_embeddings)

                    attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings)  # [Batch, All Tokens]
                    attns.append(attn)

                attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks
                attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2)
                for x in xs:
                    if self.lm == 'distilbert':
                        words_emb = self.bert.embeddings(x)
                    else:
                        words_emb = self.bert.get_input_embeddings()(x)

                    for i in range(words_emb.size()[0]):
                        words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

                    output = self.bert(inputs_embeds=words_emb)

                    pooled_output = output[0][:, 0, :]
                    pooled_output = self.dropout(pooled_output)
                    pooled_outputs.append(pooled_output)

                attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
                entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1)
                entity_output = self.res(self.out(attr_outputs, entity_outputs))

        logits = self.fc(entity_output)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
