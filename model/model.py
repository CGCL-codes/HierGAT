import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import AttentionLayer as AL, GlobalAttentionLayer as GoAL, StructAttentionLayer as SAL
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

        hidden_size = self.bert.config.hidden_size
        hidden_dropout_prob = 0.1

        self.inits = nn.ModuleList([
            GoAL(hidden_size, 0.2)
            for _ in range(attr_num)])
        self.conts = nn.ModuleList([
            AL(hidden_size + hidden_size, 0.2, device)
            for _ in range(attr_num)])
        self.out = SAL(2 * hidden_size * attr_num, 0.2)

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, xs, left_xs, right_xs, y, token_attr_adjs):
        # Token Sequence
        xs = xs.to(self.device)
        left_xs = left_xs.to(self.device)
        right_xs = right_xs.to(self.device)

        xs = xs.permute(1, 0, 2) #[Attributes, Batch, Tokens]
        left_xs = left_xs.permute(1, 0, 2)
        right_xs = right_xs.permute(1, 0, 2)

        y = y.to(self.device)

        # Token-Attribute Graph Adjacency Matrix
        token_attr_adjs = token_attr_adjs.to(self.device)
        token_attr_adjs = token_attr_adjs.permute(0, 2, 1) # [Batch, All Tokens, Attributes]

        if self.training and self.finetuning:
            self.bert.train()

            # Get Context
            attns, contexts = self.get_context(xs, token_attr_adjs)

            entity_embs = []
            attr_comp_embs = []
            for x, left_x, right_x in zip(xs, left_xs, right_xs):
                # Hierarchical Aggregation
                entity_embs.append(self.hier_aggr(left_x, right_x, attns, contexts))

                # Attribute Comparison
                attr_comp_embs.append(self.hier_attr_comp(x, attns, contexts))

            entity_outputs = torch.stack(entity_embs).permute(1, 0, 2)
            attr_outputs = torch.stack(attr_comp_embs).permute(1, 0, 2)

            # Entity Comparison
            entity_output = self.hier_ent_comp(attr_outputs, entity_outputs)
        else:
            self.bert.eval()
            with torch.no_grad():
                # Get Context
                attns, contexts = self.get_context(xs, token_attr_adjs)

                entity_embs = []
                attr_comp_embs = []
                for x, left_x, right_x in zip(xs, left_xs, right_xs):
                    # Hierarchical Aggregation
                    entity_embs.append(self.hier_aggr(left_x, right_x, attns, contexts))

                    # Attribute Comparison
                    attr_comp_embs.append(self.hier_attr_comp(x, attns, contexts))

                entity_outputs = torch.stack(entity_embs).permute(1, 0, 2)
                attr_outputs = torch.stack(attr_comp_embs).permute(1, 0, 2)

                # Entity Comparison
                entity_output = self.hier_ent_comp(attr_outputs, entity_outputs)

        logits = self.fc(entity_output)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

    # `adjs` is the Token-Attribute graph adjacency matrix
    def get_context(self, xs, adjs):
        attr_outputs = []
        attns = []

        for x, init, cont in zip(xs, self.inits, self.conts):
            # Get Attribute Context Embedding
            attr_embeddings = init(self.bert.get_input_embeddings()(x))  # [Batch, Hidden]
            attr_outputs.append(attr_embeddings)

            # Get Token-Attribute Attention
            attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings)  # [Batch, All Tokens]
            attns.append(attn)

        attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * adjs  # [Batch, All Tokens, Attributes]
        attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2)  # [Batch, Attributes, Hidden]

        return attns, attr_outputs

    def context_embedding(self, x, attns, attr_outputs):
        if self.lm == 'distilbert':
            words_emb = self.bert.embeddings(x)
        else:
            words_emb = self.bert.get_input_embeddings()(x)

        # Add Context Embedding
        for i in range(words_emb.size()[0]):  # i is index of batch
            words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

        return words_emb

    def hier_aggr(self, left_x, right_x, attns, attr_contexts):
        left_attr_emb = self.transform(self.context_embedding(left_x, attns, attr_contexts))
        right_attr_emb = self.transform(self.context_embedding(right_x, attns, attr_contexts))
        entity_emb = torch.cat([left_attr_emb, right_attr_emb])

        return entity_emb

    def hier_attr_comp(self, x, attns, attr_contexts):
        return self.transform(self.context_embedding(x, attns, attr_contexts))

    def hier_ent_comp(self, attr_comp_emb, en_sum_emb):
        # Currently, we only support aligned attributes
        # So the entity is connected to all attributes
        # For simplicity, we omit this particular adjacency matrix
        return self.out(attr_comp_emb, en_sum_emb)

    def transform(self, emb):
        output = self.bert(inputs_embeds=emb)
        pooled_output = output[0][:, 0, :]
        pooled_output = self.dropout(pooled_output)

        return pooled_output
