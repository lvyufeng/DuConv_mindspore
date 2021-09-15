import mindspore
import numpy as np
from mindspore import Tensor, context
from src.model import Retrieval
from src.bert import BertConfig
from src.lr_schedule import Noam
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.nn import WithLossCell, SoftmaxCrossEntropyWithLogits, TrainOneStepCell
from src.dataset import create_dataset

def train(args):
    use_kn = args.use_kn == "true"

    dataset = create_dataset(args.batch_size, data_file_path=args.file_path, use_knowledge=use_kn)
    max_train_steps = args.epoch * dataset.get_dataset_size()
    warmup_steps = int(max_train_steps * args.warmup_proportion)

    config = BertConfig()
    network = Retrieval(config, use_kn)
    loss = SoftmaxCrossEntropyWithLogits(reduction='sum')
    network_with_loss = WithLossCell(network, loss)
    lr_schedule = Noam(config.hidden_size, warmup_steps)
    optimizer = Adam(network_with_loss.trainable_params(), lr_schedule)
    network_one_step = TrainOneStepCell(network_with_loss, optimizer)
    model = Model(network_one_step)

    model.train(args.epoch, dataset)



if __name__ == "__main__":
    # context.set_context(mode=context.PYNATIVE_MODE)
    config = BertConfig()
    net = Retrieval(config, True)
    input_ids = Tensor(np.random.uniform(0, 512, (1, 256)).astype(np.int32))
    segment_ids = Tensor(np.ones((1, 256)).astype(np.int32))
    kn_ids = Tensor(np.random.uniform(0, 512, (1, 256)).astype(np.int32))
    seq_length = Tensor([256], mindspore.int32)

    logits = net(input_ids, segment_ids, kn_ids, seq_length)