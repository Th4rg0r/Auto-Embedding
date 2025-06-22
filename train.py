from alive_progress import alive_bar
from dataset import LazyLoader, split_train_test_set, get_file_line_cnt
from network import PositionalEncoding, Seq2SeqTransformer
from tokenizer import sentence_splitter, tokenize
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import argparse
import os
import torch.nn as nn
import torch.optim as optim
import torch
import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage.*")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        "--split_sentences", action="store_true", help="Split input into sentences"
    )
    parser.add_argument(
        "--reload_tokenizer", action="store_true",  help="whether to reload the previous tokenizer"
    )
    parser.add_argument(
        "--reload_datasets", action="store_true",  help="whether to reload the splitted datasets from the previous runs")
    parser.add_argument(
        "--normalize_accents",
        action="store_true",
        help="normalize accents (like é to e or á to a) and do NOT preservers the original (should disabled for most languages)",
    )
    parser.add_argument(
        "--reload_latest_model", action="store_true", help="reloads the trained model (latest)"
    )
    parser.add_argument(
        "--reload_best_model", action="store_true", help="reloads the trained model (best)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="the batch size of the training"
    )
    parser.add_argument(
        "--dataset_fraction_percent", type=float, default=100.0, help="the percent of the datasets to be used. Default 100.0")
    args = parser.parse_args()

    out_dir = args.language
    latest_model_fp = os.path.join(out_dir, "latest_model.pt")
    best_model_fp = os.path.join(out_dir, "best_model.pt")
    input_file = args.input
    tokenizer = None

    os.makedirs(out_dir, exist_ok=True)
    if args.split_sentences:
        path, ext = os.path.splitext(args.input)
        input_file = os.path.join(out_dir, "splitted.txt")
        sentence_splitter(args.input, inputFile)

    train_input_fp = None
    eval_input_fp = None
    test_input_fp = None
    train_lcnt = 0
    eval_lcnt = 0
    test_lcnt = 0

    if args.reload_datasets:
        train_input_fp = os.path.join(
            out_dir, "datasets", "train.txt"
        )
        eval_input_fp = os.path.join(
            out_dir, "datasets", "eval.txt"
        )
        test_input_fp = os.path.join(
            out_dir, "datasets", "test.txt"
        )
        print("trainlc")
        train_lcnt = get_file_line_cnt(train_input_fp)
        print("evallc")
        eval_lcnt = get_file_line_cnt(eval_input_fp)
        print("testlc")
        test_lcnt = get_file_line_cnt(test_input_fp)
    else:
        (
            train_lcnt,
            eval_lcnt,
            test_lcnt,
            train_input_fp,
            eval_input_fp,
            test_input_fp,
        ) = split_train_test_set(
            file_path=input_file,
            out_path=out_dir,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
        )

    tokenizer_path = os.path.join(out_dir, "tokenizer.json")
    if args.reload_tokenizer:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = tokenize(
            train_input_fp, args.vocab_size, keep_accents= not args.normalize_accents
        )
        tokenizer.save(tokenizer_path)

    # create loaders for dataset
    lazy_train_loader = LazyLoader(
        tokenizer=tokenizer, file_path=train_input_fp, batch_size=args.batch_size
    )

    train_loader = lazy_train_loader.loader()

    lazy_eval_loader = LazyLoader(
        tokenizer=tokenizer, file_path=eval_input_fp, batch_size=args.batch_size
    )
    eval_loader = lazy_eval_loader.loader()

    lazy_test_loader = LazyLoader(
        tokenizer=tokenizer, file_path=test_input_fp, batch_size=args.batch_size
    )
    test_loader = lazy_train_loader.loader()

    vocab_size = tokenizer.get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.embedding_size,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
    )
    model = model.to(device)
    if args.reload_latest_model:
        model.load_state_dict(latest_model_fp)
    if args.reload_best_model:
        model.load_state_dict(latest_model_fp)

    # training
    pad_id = tokenizer.token_to_id("<pad>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10
    loss_history = []
    eval_loss_history = []
    min_eval_loss = float("inf")

    model.train()
    print("start training")
    max_per_epoch = (args.dataset_fraction_percent * int(train_lcnt/args.batch_size) )/ 100
    for epoch in range(num_epochs):
            
        batch_idx = 0
        epoch_loss = 0.0
        with alive_bar(max_per_epoch) as bar:
            for batch in train_loader:
                if batch_idx > max_per_epoch:
                    break;
                batch_idx += 1
                batch = lazy_train_loader.collate_fn(batch)
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
                bar.text("Loss: " + str(loss.item()))
                bar()

        avg_loss = epoch_loss / batch_idx
        loss_history.append(avg_loss)
        print("evaluation")
        model.eval()
        epoch_val_loss = 0.0
        batch_idx = 0
        max_per_epoch = (args.dataset_fraction_percent * int(eval_lcnt/args.batch_size) )/ 100.0
        with alive_bar(max_per_epoch) as bar, torch.no_grad():
            for batch in eval_loader:
                if batch_idx > max_per_epoch:
                    break;
                batch_idx += 1
                batch = lazy_eval_loader.collate_fn(batch)
                src_batch, tgt_batch, src_key_mask, tgt_key_mask = batch
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                src_key_mask = src_key_mask.to(device)
                tgt_key_mask = tgt_key_mask.to(device)
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
                epoch_val_loss += loss.item()
                bar.text("Loss: " + str(loss.item()))
                bar()

        avg_eval_loss = epoch_val_loss / batch_idx
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {avg_eval_loss}")
        eval_loss_history.append(avg_eval_loss)
        if (avg_eval_loss < min_eval_loss):
            torch.save(model.state_dict(), best_model_fp)
            min_eval_loss = avg_eval_loss;
        torch.save(model.state_dict(), latest_model_fp)

    model.eval()
    epoch_test_loss = 0.0
    batch_idx = 0
    max_per_epoch = (args.dataset_fraction_percent * int(test_lcnt/args.batch_size) )/ 100.0
    with alive_bar(max_per_epoch) as bar, torch.no_grad():
        for batch in test_loader:
            if batch_idx > max_per_epoch:
                break;
            batch_idx += 1
            batch = lazy_test_loader.collate_fn(batch)
            src_batch, tgt_batch, src_key_mask, tgt_key_mask = batch
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            src_key_mask = src_key_mask.to(device)
            tgt_key_mask = tgt_key_mask.to(device)
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
            epoch_test_loss += loss.item()
            bar.text("Loss: " + str(loss.item()))
            bar()
    avg_test_loss = epoch_test_loss / batch_idx
    print(f"Test Loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()
