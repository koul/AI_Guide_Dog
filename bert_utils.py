import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def get_extended_attention_mask(attention_mask):
  # attention_mask [batch_size, seq_length]
  assert attention_mask.dim() == 2
  # [batch_size, 1, 1, seq_length] for multi-head attention
  extended_attention_mask = attention_mask[:, None, None, :]
  extended_attention_mask = extended_attention_mask # fp16 compatibility
  extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
  return extended_attention_mask

class FeatureExtractor(nn.Module):
    def __init__(self, data_type, sensor_dim, sensor_vec_dim):
        super().__init__()
        self.data_type = data_type
        if data_type != 'video':
            self.convert_embedding = nn.Linear(sensor_dim, sensor_vec_dim)
        
    def forward(self, x):
        if self.data_type == 'sensor':
            out = self.convert_embedding(x)
        elif self.data_type == 'multimodal':
            out = self.convert_embedding(x[:, :, -sensor_dim:])
            out = torch.cat((x[:, :, :-sensor_dim], out), 1) 
        else:
            out = x
        return out        

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # this attention is applied after calculating the attention score following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)

        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):

        S = query @ key.transpose(2, 3) / math.sqrt(self.attention_head_size)
        S = attention_mask + S
        S = torch.softmax(S, dim=3)
        V = torch.matmul(S, value)
        V = torch.flatten(V.transpose(1, 2), start_dim=2)

        return V

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self attention
        self.self_attention = BertSelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu

        # layer out
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
        input: the input
        output: the input that requires the sublayer to transform
        dense_layer, dropput: the sublayer
        ln_layer: layer norm that takes input+sublayer(output)
        """
        output = dense_layer(output)
        output = dropout(output)
        normed_output = ln_layer(input + output)
        return normed_output

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf
        each block consists of
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the output of BertSelfAttention and the input of BertSelfAttention
        3. a feed forward layer
        4. a add-norm that takes the output of feed forward layer and the input of feed forward layer
        """
        # multi-head attention w/ self.self_attention
        output = self.self_attention(hidden_states, attention_mask)

        # add-norm layer
        hidden_states = self.add_norm(hidden_states, output, self.attention_dense, self.attention_dropout,
                                      self.attention_layer_norm)

        # feed forward
        output = self.interm_dense(hidden_states)
        output = self.interm_af(output)

        # another add-norm layer
        hidden_states = self.add_norm(hidden_states, output, self.out_dense, self.out_dropout, self.out_layer_norm)

        return hidden_states


class BertModel(nn.Module):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """
    def __init__(self, config, input_vec_size, seq_len, data_type):
        super().__init__()
        self.config = config
        self.seq_len = seq_len

        # embedding
        # self.convert_embedding = nn.Linear(input_vec_size, config.hidden_size)
        self.feature_extractor = FeatureExtractor(data_type, input_vec_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        # self.init_weights()

    def embed(self, input_tensor):

        seq_length = input_tensor.size()[1]
        if seq_length < self.seq_len :
            padded_tensor = torch.zeros((input_shape[0], self.seq_len, input_shape[2]))
            padded_tensor[:, :seq_length, :] = input_tensor
            input_tensor = padded_tensor

        # get word embedding from self.word_embedding
        # inputs_embeds = self.convert_embedding(input_tensor)
        inputs_embeds = self.feature_extractor(input_tensor)
        input_shape = input_tensor.size()
        seq_length = input_shape[1]

        # get position index and position embedding from self.pos_embedding
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)
        
        # add three embeddings together
        embeds = inputs_embeds + pos_embeds

        # layer norm and dropout
        embeds = self.embed_layer_norm(embeds)
        embeds = self.embed_dropout(embeds)

        return embeds

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask = get_extended_attention_mask(attention_mask)

        for i, layer_module in enumerate(self.bert_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_tensor, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_tensor)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return sequence_output, first_tk