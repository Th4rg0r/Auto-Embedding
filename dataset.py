from collections import deque
from random import shuffle
from tokenizers import Tokenizer
from torchdata.nodes import (
    IterableWrapper,
    ParallelMapper,
    Batcher,
    Loader,
    PinMemory,
    Prefetcher,
)
import random
import torch


class ShuffleBuffer:
    def __init__(self, iterable, buffer_size=1000):
        self.iterable = iterable
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = deque()
        iterator = iter(self.iterable)

        try:
            for _ in range(self.buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass  # fewer than buffer_size items

        while buffer:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            try:
                buffer[idx] = next(iterator)
            except StopIteration:
                buffer.remove(buffer[idx])


class LazyLoader:
    def __init__(self, tokenizer_path, file_path, batch_size):
        self.file_path = file_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.start_id = self.tokenizer.token_to_id("<s>")
        self.end_id = self.tokenizer.token_to_id("</s>")
        self.batch_size = batch_size

    def stream_lines(self, fp):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()

    def transform_input(self, line):
        enc = self.tokenizer.encode(line)
        ids = enc.ids
        src_ids = [self.start_id] + ids + [self.end_id]
        tgt_ids = [self.start_id] + ids + [self.end_id]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )

    def loader(self):
        # 1. Wrap line-stream
        buffered_stream = ShuffleBuffer(self.stream_lines(self.file_path), 2048)
        # node = IterableWrapper(self.stream_lines(self.file_path))
        node = IterableWrapper(buffered_stream)

        # 2. apply paralell processing

        node = ParallelMapper(
            node,
            map_fn=lambda l: self.transform_input(l),
            num_workers=4,
            method="thread",
        )

        # 3. group into batches
        node = Batcher(node, batch_size=self.batch_size, drop_last=False)

        node = PinMemory(node)
        node = Prefetcher(node, prefetch_factor=4)
        loader = Loader(node)
        return loader

    def collate_fn(self, batch):
        # batch is a list of (src_ids, tgt_ids) pairs (as torch Tensors of different lengths)
        src_batch, tgt_batch = zip(*batch)
        src_lens = [len(x) for x in src_batch]
        tgt_lens = [len(x) for x in tgt_batch]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)
        pad_id = self.tokenizer.token_to_id("<pad>")
        # Pad sequences and build masks
        padded_src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
        padded_tgt = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
        src_mask = torch.ones(len(batch), max_src, dtype=torch.bool)
        tgt_mask = torch.ones(len(batch), max_tgt, dtype=torch.bool)
        for i, (src_ids, tgt_ids) in enumerate(batch):
            padded_src[i, : len(src_ids)] = src_ids
            padded_tgt[i, : len(tgt_ids)] = tgt_ids
            src_mask[i, : len(src_ids)] = 0
            tgt_mask[i, : len(tgt_ids)] = 0
            # Transformer expects key_padding_mask where True means **not** allowed
        return padded_src, padded_tgt, src_mask, tgt_mask
