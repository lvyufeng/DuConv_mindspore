from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
# from transformer import TransformerConfig
class TransformerConfig:
    """
    Configuration for `Transformer`.
    """
    def __init__(self,
                 batch_size,
                 emb_size=256,
                 gru_hidden_size=128,
                 num_directions=True,
                 seq_length=256,
                 hidden_size=1024,
                 num_hidden_layers=12,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 hidden_act="relu",
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=256,
                 initializer_range=0.02,
                 label_smoothing=0.1,
                 length_penalty_weight=1.0,
                 dtype=mstype.float16,
                 compute_type=mstype.float16,
                 use_one_hot_embeddings=False):
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.gru_hidden_size = gru_hidden_size
        self.num_directions = num_directions
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing
        self.length_penalty_weight = length_penalty_weight
        self.dtype = dtype
        self.compute_type = compute_type
        self.use_one_hot_embeddings = use_one_hot_embeddings


cfg = edict({
    'transformer_network': 'large',
    'init_loss_scale_value': 1024,
    'scale_factor': 2,
    'scale_window': 2000,
    'optimizer': 'Adam',
    'optimizer_adam_beta2': 0.997,
    'lr_schedule': edict({
        'learning_rate': 2.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    }),
})
'''
two kinds of transformer model version  4096
'''
if cfg.transformer_network == 'large':
    net_cfg = TransformerConfig(
        batch_size=32,
        emb_size=256,
        gru_hidden_size=128,
        seq_length=256,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=256*4,
        hidden_act="gelu",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=256,
        initializer_range=0.02,
        label_smoothing=0.2,
        length_penalty_weight=0.8,
        dtype=mstype.float32,
        compute_type=mstype.float16,
        use_one_hot_embeddings=False)
if cfg.transformer_network == 'base':
    trans_net_cfg = TransformerConfig(
        batch_size=96,
        seq_length=128,
        vocab_size=36560,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="relu",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=256,
        initializer_range=0.02,
        label_smoothing=0.1,
        dtype=mstype.float32,
        compute_type=mstype.float16)


