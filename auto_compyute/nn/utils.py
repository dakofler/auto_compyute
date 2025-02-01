"""Neural network utils"""

from collections.abc import Iterator

from ..autograd import Tensor
from ..backends import Device, cpu
from ..tensors import arange, randperm


class Dataloader:
    def __init__(
        self,
        data: tuple[Tensor, ...],
        batch_size: int = 1,
        device: Device = cpu,
        shuffle_data: bool = True,
        drop_remaining: bool = False,
    ) -> None:
        self.data = data
        self._n = len(self.data[0])
        self.batch_size = min(batch_size, self._n)
        self.device = device
        self.shuffle = shuffle_data
        self._additional_batch = not drop_remaining and self._n % self.batch_size > 0

    def __call__(self) -> Iterator[tuple[Tensor, ...]]:
        idx = randperm(self._n) if self.shuffle else arange(self._n)

        for i in range(len(self)):
            batch_idx = idx[i * self.batch_size : (i + 1) * self.batch_size]
            yield tuple(t[batch_idx].to(self.device) for t in self.data)

    def __len__(self) -> int:
        return max(1, self._n // self.batch_size + self._additional_batch)
