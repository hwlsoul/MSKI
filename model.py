import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import NCRLoss as ATLoss
# from nclr_loss import NCRLoss as ATLoss

class DocREModel(nn.Module):
    def __init__(self, config, model, rels_input, emb_size=1024, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        self.rels_input = rels_input
        self.mse_criterion = nn.MSELoss()
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.alph = 0.01        
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model = emb_size, nhead = 8, dim_feedforward=2048)
        self.encoder_1 = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model = emb_size, nhead = 8, dim_feedforward=2048)
        self.encoder_2 = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def encode_rel(self):
        outputs = self.model(**self.rels_input)
        rel_emb = outputs.last_hidden_state[:, 0, :]
        rel_emb = rel_emb.unsqueeze(1)
        return rel_emb

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            #entity_es.append(entity_embs)
            #entity_as.append(entity_atts)
            #sequnce_out.to("cuda:1")
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                teacher_logits = None,
                
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        rels_sequence = self.encode_rel()
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        ts_len = len(rels_sequence)
        ress = rels.expand(-1, ts_len, -1)
        # print(ress.shape, ts_len, rs.shape, rels.shape)
        ress = self.encoder_1(ress)
        ress = self.encoder_2(ress)
        rs = rs + ress.mean(dim=0)
        
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
               
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        
                
        logits = self.bilinear(bl)

        if labels is None:
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits)
        
        
        if labels is not None:

            labels = [torch.tensor(label) for label in labels]
            
            labels = torch.cat(labels, dim=0).to(logits)
            
            loss = self.loss_fnt(logits.float(), labels.float())

            
            
            # output = (loss.to(sequence_output),) + output

            output = loss.to(sequence_output)

            if teacher_logits is not None:
                
                teacher_logits = torch.cat(teacher_logits, dim=0).to(logits)
                mse_loss = self.mse_criterion(logits, teacher_logits)
                output = (1-self.alph) output +   self.alph * mse_loss
            

        return output