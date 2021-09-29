import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.seed import _get_graph_seed
from mindspore.ops import operations as P

class Dropout(nn.Cell):
    r"""
    Dropout layer for the input.

    Randomly set some elements of the input tensor to zero with probability :math:`1 - keep\_prob` during training
    using samples from a Bernoulli distribution.

    The outputs are scaled by a factor of :math:`\frac{1}{keep\_prob}`    during training so
    that the output layer remains at a similar scale. During inference, this
    layer returns the same tensor as the `x`.

    This technique is proposed in paper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ and proved to be effective to reduce
    over-fitting and prevents neurons from co-adaptation. See more details in `Improving neural networks by
    preventing co-adaptation of feature detectors
    <https://arxiv.org/pdf/1207.0580.pdf>`_.

    Note:
        Each channel will be zeroed out independently on every construct call.

    Args:
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. E.g. rate=0.9,
                   dropping out 10% of input units. Default: 0.5.
        dtype (:class:`mindspore.dtype`): Data type of `x`. Default: mindspore.float32.

    Inputs:
        - **x** (Tensor) - The input of Dropout with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, output tensor with the same shape as the `x`.

    Raises:
        TypeError: If `keep_prob` is not a float.
        TypeError: If dtype of `x` is not neither float16 nor float32.
        ValueError: If `keep_prob` is not in range (0, 1].
        ValueError: If length of shape of `x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> net = nn.Dropout(keep_prob=0.8)
        >>> net.set_train()
        Dropout<keep_prob=0.8>
        >>> output = net(x)
        >>> print(output.shape)
        (2, 2, 3)
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        """Initialize Dropout."""
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError("dropout probability should be a number in range (0, 1], but got {}".format(keep_prob))
        self.keep_prob = keep_prob
        seed0, seed1 = _get_graph_seed(0, "dropout")
        self.seed0 = seed0
        self.seed1 = seed1
        self.dropout = P.Dropout(keep_prob, seed0, seed1)

    def construct(self, x):
        if self.training:
            return x

        if self.keep_prob == 1:
            return x

        out, _ = self.dropout(x)
        return out