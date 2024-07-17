import logging
import math
import sys
import copy

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

sys.path.append('../../mindnlp')
from mindnlp.transformers.models.bert.modeling_bert import BertPreTrainedModel

logger = logging.getLogger(__name__)

BertLayerNorm = nn.LayerNorm 


def gelu(x):
    return x * 0.5 * (1.0 + ops.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * ops.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": ops.relu, "swish": swish}

# done
class BertEmbeddings(nn.Cell): 
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            position_ids = ops.arange(seq_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# done
class BertSelfAttention(nn.Cell):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel construct() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs

# done
class BertSelfOutput(nn.Cell):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# done
class BertAttention(nn.Cell):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def construct(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# done
class BertIntermediate(nn.Cell):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# done
class BertOutput(nn.Cell):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# done
class BertLayer(nn.Cell):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def construct(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

# done
class BertEncoder(nn.Cell):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.CellList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, 
                                         None if head_mask is None else head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

# done
class BertPooler(nn.Cell):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# done
class BertPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# done
class BertLMPredictionHead(nn.Cell):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size,
                                 config.vocab_size,
                                 has_bias=False)

        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size)) # !!not sure

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

# done
class BertOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

#done
class BertOutAttention(nn.Cell):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(ctx_dim, self.all_head_size)
        self.value = nn.Dense(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel construct() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

#done
class BertXAttention(nn.Cell):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def construct(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

# done(not changed)
class LXRTXLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.no_lang_ca = config.no_lang_ca # do not update language embeds

        # Lang self-att and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        if self.no_lang_ca:
            lang_att_output = lang_input
        else:
            lang_att_output, _ = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output, _ = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        if self.no_lang_ca:
            lang_att_output = (lang_input, )
        else:
            lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        if not self.no_lang_ca:
            lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        if self.no_lang_ca:
            lang_output = lang_input
        else:
            lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def construct(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output[0], visn_att_output[0])

        return lang_output, visn_output

# done
class LxmertEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.num_l_layers = config.num_l_layers
        self.num_r_layers = config.num_r_layers
        self.num_h_layers = config.num_h_layers
        self.num_x_layers = config.num_x_layers
        self.update_lang_bert = config.update_lang_bert

        # Using self.layer instead of self.l_layers to support loading BERT weights.
        self.layer = nn.CellList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for param in self.layer.get_parameters():
                param.requires_grad = False

        self.h_layers = nn.CellList(
            [BertLayer(config) for _ in range(self.num_h_layers)]
        ) if self.num_h_layers > 0 else None
        self.r_layers = nn.CellList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        ) if self.num_r_layers > 0 else None
        self.x_layers = nn.CellList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def construct(self, txt_embeds, extended_txt_masks, hist_embeds,
                extended_hist_masks, img_embeds=None, extended_img_masks=None):
        # text encoding
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = ops.stop_gradient(txt_embeds)

        # image encoding
        if img_embeds is not None:
            if self.r_layers is not None:
                for layer_module in self.r_layers:
                    temp_output = layer_module(img_embeds, extended_img_masks)
                    img_embeds = temp_output[0]

        # history encoding
        if self.h_layers is not None:
            for layer_module in self.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]
        hist_max_len = hist_embeds.shape[1]
        
        # cross-modal encoding
        if img_embeds is None:
            hist_img_embeds = hist_embeds
            extended_hist_img_masks = extended_hist_masks
        else:
            hist_img_embeds = ops.cat([hist_embeds, img_embeds], 1)
            extended_hist_img_masks = ops.cat([extended_hist_masks, extended_img_masks], -1)
        
        for layer_module in self.x_layers:
            txt_embeds, hist_img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                hist_img_embeds, extended_hist_img_masks)

        hist_embeds = hist_img_embeds[:, :hist_max_len]
        if img_embeds is not None:
            img_embeds = hist_img_embeds[:, hist_max_len:]
        return txt_embeds, hist_embeds, img_embeds

# done(not edited)
class ImageEmbeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Dense(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)
        self.ang_linear = nn.Dense(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)
        # 0: non-navigable, 1: navigable, 2: stop
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.txt_linear = nn.Dense(512, config.hidden_size)
        self.txt_layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)

        self.args = config.args

    def construct(self, img_feat, ang_feat, type_embeddings, nav_types=None, mode=None, landmark_feat=None):
        if mode == 'text':
            transformed_txt = self.txt_layer_norm(self.txt_linear(landmark_feat))
            embeddings =  transformed_txt + type_embeddings
            if nav_types is not None:
                nav_embeddings = self.nav_type_embedding(nav_types)
                embeddings = embeddings + nav_embeddings
            embeddings = self.layer_norm(embeddings)
            embeddings = self.dropout(embeddings)
        else:
            transformed_im = self.img_layer_norm(self.img_linear(img_feat))
            transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
            embeddings = transformed_im + transformed_ang + type_embeddings
            if nav_types is not None:
                nav_embeddings = self.nav_type_embedding(nav_types)
                embeddings = embeddings + nav_embeddings
            embeddings = self.layer_norm(embeddings)
            embeddings = self.dropout(embeddings)
        return embeddings

# done
class HistoryEmbeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.cls_token = mindspore.Parameter(ops.zeros(1, 1, config.hidden_size))

        self.img_linear = nn.Dense(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)
        self.ang_linear = nn.Dense(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)
        
        self.position_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        # special type embedding for history
        self.type_embedding = nn.Embedding(1, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hist_enc_pano = config.hist_enc_pano
        if config.hist_enc_pano:
            self.pano_img_linear = nn.Dense(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)
            self.pano_ang_linear = nn.Dense(config.angle_feat_size, config.hidden_size)
            self.pano_ang_layer_norm = BertLayerNorm([config.hidden_size], epsilon=1e-12)
            pano_enc_config = copy.copy(config)
            pano_enc_config.num_hidden_layers = config.num_h_pano_layers
            self.pano_encoder = BertEncoder(pano_enc_config)
        else:
            self.pano_encoder = None

    def construct(self, img_feats, ang_feats, pos_ids, 
                pano_img_feats=None, pano_ang_feats=None):
        '''Args:
        - img_feats: (batch_size, dim_feat)
        - pos_ids: (batch_size, )
        - pano_img_feats: (batch_size, pano_len, dim_feat)
        '''

        # device = next(iter(self.parameters())).device   #(580),  ".to(device)" in Line 586 and Line 606 are deleted
        if img_feats is not None:
            batch_size = img_feats.shape[0]
        else:
            batch_size = 1

        type_ids = ops.zeros((batch_size, )).long()
        type_embeddings = self.type_embedding(type_ids)

        if img_feats is None:
            cls_embeddings = self.dropout(self.layer_norm(
                self.cls_token.expand(batch_size, -1, -1)[:, 0] + type_embeddings))
            return cls_embeddings

        # history embedding per step
        embeddings = self.img_layer_norm(self.img_linear(img_feats)) + \
                     self.ang_layer_norm(self.ang_linear(ang_feats)) + \
                     self.position_embeddings(pos_ids) + \
                     type_embeddings

        if self.pano_encoder is not None:
            pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                              self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
            pano_embeddings = self.dropout(pano_embeddings)
            # TODO: mask is always True
            batch_size, pano_len, _ = pano_img_feats.shape
            extended_pano_masks = ops.zeros(batch_size, pano_len).float().unsqueeze(1).unsqueeze(2)
            pano_embeddings = self.pano_encoder(pano_embeddings, extended_pano_masks)[0]
            pano_embeddings = ops.mean(pano_embeddings, 1)

            embeddings = embeddings + pano_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# done(not edited)
class NextActionPrediction(nn.Cell):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.SequentialCell(nn.Dense(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm([hidden_size], epsilon=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Dense(hidden_size, 1))

    def construct(self, x):
        return self.net(x)

# done
class NavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.args = config.args
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)
        
        self.hist_embeddings = HistoryEmbeddings(config)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)

        self.args = config.args

        self.init_weights()

    def construct(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                hist_img_feats=None, hist_ang_feats=None,
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, ob_step_ids=None, hist_masks=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
                ob_masks=None,
                corrected_landmark_feature=None
                ):

        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=mindspore.float32)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            txt_token_type_ids = ops.zeros_like(txt_ids)
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            for layer_module in self.encoder.layer:
                temp_output = layer_module(txt_embeds, extended_txt_masks)
                txt_embeds = temp_output[0]
            if self.config.fix_lang_embedding:
                txt_embeds = ops.stop_gradient(txt_embeds)
            if self.config.no_lang_ca: # run self-attn layers for lang
                all_txt_embeds = [txt_embeds]
                for layer_module in self.encoder.x_layers:
                    lang_att_output = layer_module.lang_self_att(txt_embeds, extended_txt_masks)[0]
                    lang_inter_output = layer_module.lang_inter(lang_att_output)
                    lang_output = layer_module.lang_output(lang_inter_output, lang_att_output)
                    all_txt_embeds.append(lang_output)
                return all_txt_embeds
            return txt_embeds

        # history embedding per step
        if mode == 'history':
            if hist_img_feats is None:
                hist_embeds = self.hist_embeddings(hist_img_feats, hist_ang_feats, ob_step_ids,
                                                   pano_img_feats=hist_pano_img_feats,
                                                   pano_ang_feats=hist_pano_ang_feats)
            else:

                hist_embeds = self.hist_embeddings(hist_img_feats[:, :768], hist_ang_feats, ob_step_ids,
                                                       pano_img_feats=hist_pano_img_feats[:, :, :768],
                                                       pano_ang_feats=hist_pano_ang_feats)

            if self.config.fix_hist_embedding:
                hist_embeds = ops.stop_gradient(hist_embeds)
            return hist_embeds
            
        # cross-modal encoding per step
        elif mode == 'visual':
            ''' LXMERT visual branch'''
            # history embedding
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=mindspore.float32)
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

            ob_token_type_ids = ops.ones((ob_img_feats.shape[0], ob_img_feats.shape[1]), dtype=mindspore.int64
                                           )

            ob_embeds = self.img_embeddings(ob_img_feats[:, :, :768], ob_ang_feats,
                                                self.embeddings.token_type_embeddings(ob_token_type_ids),
                                                nav_types=ob_nav_types)

            # image embedding
            extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
            extended_ob_masks = extended_ob_masks.to(dtype=mindspore.float32)
            extended_ob_masks = (1.0 - extended_ob_masks) * -10000.0

            ob_token_type_ids = ops.ones((ob_img_feats.shape[0], ob_img_feats.shape[1]), dtype=mindspore.int64)

            ob_ori_embeds = ob_embeds

            landmark_embeds = self.img_embeddings(ob_img_feats[:, :, :768], ob_ang_feats,
                                                     self.embeddings.token_type_embeddings(
                                                         ob_token_type_ids),
                                                     nav_types=ob_nav_types,
                                                     mode='text', landmark_feat=corrected_landmark_feature)

            ob_embeds = ob_embeds + landmark_embeds


            if self.encoder.h_layers is not None:
                for layer_module in self.encoder.h_layers:
                    temp_output = layer_module(hist_embeds, extended_hist_masks)
                    hist_embeds = temp_output[0]

            if self.encoder.r_layers is not None:
                for layer_module in self.encoder.r_layers:
                    temp_output = layer_module(ob_embeds, extended_ob_masks)
                    ob_embeds = temp_output[0]
            if self.config.fix_obs_embedding:
                ob_embeds = ops.stop_gradient(ob_embeds)

            # multi-modal encoding
            hist_max_len = hist_embeds.shape[1]
            hist_ob_embeds = ops.cat([hist_embeds, ob_embeds], 1)
            extended_hist_ob_masks = ops.cat([extended_hist_masks, extended_ob_masks], -1)

            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=mindspore.float32)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            if self.config.no_lang_ca:
                all_txt_embeds = txt_embeds
            for l, layer_module in enumerate(self.encoder.x_layers):
                if self.config.no_lang_ca:
                    txt_embeds = all_txt_embeds[l]
                txt_embeds, hist_ob_embeds = layer_module(
                    txt_embeds, extended_txt_masks, 
                    hist_ob_embeds, extended_hist_ob_masks,
                )

            hist_embeds = hist_ob_embeds[:, :hist_max_len]
            ob_embeds = hist_ob_embeds[:, hist_max_len:]

            if self.config.no_lang_ca:
                act_logits = self.next_action(ob_embeds).squeeze(-1)
            else:
                if self.config.act_pred_token == 'ob_txt':
                    act_logits = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob':
                    act_logits = self.next_action(ob_embeds).squeeze(-1)
                elif self.config.act_pred_token == 'ob_hist':
                    act_logits = self.next_action(ob_embeds * hist_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob_txt_hist':
                    act_logits = self.next_action(ob_embeds * (txt_embeds[:, :1] + hist_embeds[:, :1])).squeeze(-1)

            act_logits = ops.masked_fill(act_logits, ob_nav_types==0, -float('inf'))


            return act_logits, txt_embeds, hist_embeds, ob_embeds, ob_ori_embeds, landmark_embeds


