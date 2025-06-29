from alive_progress import alive_bar
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
import os
import subprocess


def get_file_line_cnt(fp):
    # count file lines
    result = subprocess.run(["wc", "-l", fp], capture_output=True, text=True)
    line_count = result.stdout.strip().split()[0]
    return int(line_count)


def split_train_test_set(
    file_path, out_path, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
):
    print("split into three datasets: train, eval, test")
    ratio_sum = train_ratio + validation_ratio + test_ratio
    train_ratio = train_ratio / ratio_sum
    validation_ratio = validation_ratio / ratio_sum
    test_ratio = test_ratio / ratio_sum

    batch_line_cnt = 10000
    line_count = get_file_line_cnt(file_path)

    if line_count < batch_line_cnt:
        batch_line_cnt = line_count

    dataset_dir = os.path.join(out_path, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    train_out_fp = os.path.join(out_path, "datasets", "train.txt")
    eval_out_fp = os.path.join(out_path, "datasets", "eval.txt")
    test_out_fp = os.path.join(out_path, "datasets", "test.txt")

    idxs = list(range(batch_line_cnt))
    random.shuffle(idxs)
    tmp_idx = 0
    batch_iteration = 0
    train_cnt = 0
    valid_cnt = 0
    test_cnt = 0

    with alive_bar(line_count) as bar, open(file_path, "r") as f_input, open(
        train_out_fp, "w"
    ) as f_train, open(eval_out_fp, "w") as f_valid, open(test_out_fp, "w") as f_test:
        valid_threshold = batch_line_cnt * train_ratio
        test_threshold = batch_line_cnt * (validation_ratio + train_ratio)
        for line in f_input:
            cur_idx = idxs[tmp_idx]
            if cur_idx < valid_threshold:
                f_train.write(line + "\n")
                train_cnt += 1
            elif cur_idx < test_threshold:
                f_valid.write(line + "\n")
                valid_cnt += 1
            else:
                f_test.write(line + "\n")
                test_cnt += 1
            tmp_idx += 1
            bar()
            if tmp_idx >= batch_line_cnt:
                tmp_idx = 0
                batch_iteration += 1
                if (batch_iteration + 1) * batch_line_cnt > line_count:
                    batch_line_cnt = line_count - batch_iteration * batch_line_cnt
                    valid_threshold = batch_line_cnt * train_ratio
                    test_threshold = batch_line_cnt * validation_ratio
                    idxs = idxs[:batch_line_cnt]
                random.shuffle(idxs)
    return train_cnt, valid_cnt, test_cnt, train_out_fp, eval_out_fp, test_out_fp


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
    def __init__(self, tokenizer, file_path, batch_size, max_word_per_sentence):
        self.file_path = file_path
        # self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer = tokenizer
        self.start_id = self.tokenizer.token_to_id("<s>")
        self.end_id = self.tokenizer.token_to_id("</s>")
        self.batch_size = batch_size
        self.max_word_per_sentence = max_word_per_sentence

    def stream_lines(self, fp):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if len(line.split()) > self.max_word_per_sentence:
                    continue
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
