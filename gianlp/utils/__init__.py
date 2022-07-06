from abc import abstractmethod
from typing import Any, Generator


class Sequence:
    """
    Library sequence interface for training
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass

    def __iter__(self) -> Generator[Any, None, None]:
        """
        Create a generator that iterate over the Sequence.
        :return: The generator
        """
        for i in range(len(self)):
            yield self.__getitem__(i)

    def on_epoch_end(self) -> None:
        """
        Method called at the end of every epoch.
        """
        return
