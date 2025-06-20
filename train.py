from alive_progress import alive_bar
from dataset import TextDataset, LazyLoader
from network import PositionalEncoding, Seq2SeqTransformer
from tokenizer import sentence_splitter, tokenize
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import argparse
import os
import torch.nn as nn
import torch.optim as optim


def main():
    device = "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Text file with input text. if not already one sentence per line, set --splitSentences to True",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=512,
        help="embedding size, both the input enbedding size and also the generated result embedding size (in this case has to be a multiple of '8'",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Defines the language, and also the subdirectory, where new files will be created",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=20000,
        help="the amount of distinct subwords the model differentiates, greater value make more accurate models, smaller values preserver memory",
    )
    parser.add_argument(
        "--split_sentences", type=bool, default=False, help="Split input into sentences"
    )
    parser.add_argument(
        "--load_tokenizer_from",
        help="If there is already a tokenizer, load it from the given file",
    )
    parser.add_argument(
        "--keep_accents",
        type=bool,
        default=True,
        help="Does not normalize accents (like é to e or á to a) but preservers the original (should be true for most languages)",
    )
    parser.add_argument(
        "--reload_model", type=bool, default=False, help="reloads the trained model"
    )
    parser.add_argument(
        "--model_name", default="model", help="the model name to save the model to"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="the batch size of the training"
    )
    args = parser.parse_args()

    out_dir = args.language

    input_file = args.input
    tokenizer = None

    os.makedirs(out_dir, exist_ok=True)
    if args.split_sentences:
        path, ext = os.path.splitext(args.input)
        input_file = os.path.join(out_dir, "splitted.txt")
        sentence_splitter(args.input, inputFile)

    tokenizer_path = None
    if args.load_tokenizer_from:
        tokenizer_path = args.load_tokenizer_from
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = tokenize(
            input_file, args.vocab_size, keep_accents=args.keep_accents
        )
        tokenizer_path = os.path.join(out_dir, "tokenizer.json")
        tokenizer.save(tokenizer_path)

    # dataset = TextDataset(tokenizer_path=tokenizer_path, file_path = input_file)
    lazy_loader = LazyLoader(
        tokenizer_path=tokenizer_path, file_path=input_file, batch_size=32
    )
    data_loader = lazy_loader.loader()

    vocab_size = tokenizer.get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.embedding_size,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
    )
    if args.reload_model:
        model.load_state_dict(os.path.join(out_dir, args.model_name + ".pt"))

    # training
    pad_id = tokenizer.token_to_id("<pad>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10
    loss_history = []

    model.train()
    print("start training")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with alive_bar(int(26222885 / 32)) as bar:
            for batch in data_loader:
                batch = lazy_loader.collate_fn(batch)
                src_batch, tgt_batch, src_key_mask, tgt_key_mask = batch
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                src_key_mask = src_key_mask.to(device)
                tgt_key_mask = tgt_key_mask.to(device)
                optimizer.zero_grad()
                # Decoder input is all tokens except the last
                decoder_input = tgt_batch[:, :-1]
                # Decoder target is all tokens except the first
                decoder_target = tgt_batch[:, 1:]

                outputs = model(
                    src_batch, decoder_input, src_key_mask, tgt_key_mask[:, :-1]
                )
                # Reshape outputs and target for loss
                outputs = outputs.reshape(-1, vocab_size)
                decoder_target = decoder_target.reshape(-1)
                loss = criterion(outputs, decoder_target)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                bar()

        avg_loss = epoch_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(out_dir, args.model_name + ".pt"))


if __name__ == "__main__":
    main()
