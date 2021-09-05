import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

class Noam(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps, learning_rate):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.pow = P.Pow()
        self.min = P.Minimum()
        self.cast = P.Cast()

    def construct(self, global_step):
        p = self.cast(self.min(
            self.pow(global_step, -0.5),
            self.pow(self.warmup_steps, -1.5) * global_step),
            mstype.float32)
        return self.learning_rate * self.pow(self.d_model, -0.5) * p
