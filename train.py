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
import gc
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage.*")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Text file with input text. if not already one sentence per line, set --split_sentences for a rudementary split by '.:? etc' ",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Defines the language, and also the subdirectory, where new files will be created. Default: 'en' ",
    )
    parser.add_argument(
        "--model_embedding_size",
        type=int,
        default=512,
        help="embedding size, both the input enbedding size and also the generated result embedding size (in this case has to be a multiple of '8'. Default: 512",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=5000,
        help="the amount of distinct subwords the model differentiates, greater value make more accurate models, smaller values preserver memory. Default: 20000",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="the number of epochs to train. Default: 10"
    )
    parser.add_argument(
        "--split_sentences", action="store_true", help="Split input into sentences, by special characters. Should be avoided for professional result"
    )
    parser.add_argument(
        "--reload_tokenizer", action="store_true",  help="whether to reload the previous tokenizer"
    )
    parser.add_argument(
        "--reload_datasets", action="store_true",  help="whether to reload the splitted datasets from the previous runs")
    parser.add_argument(
        "--normalize_accents",
        action="store_true",
        help="normalize accents (like é to e or á to a) and do NOT preservers the original (should not be set for most languages)",
    )
    parser.add_argument(
        "--reload_latest_model", action="store_true", help="reloads the trained model (latest)"
    )
    parser.add_argument(
        "--reload_best_model", action="store_true", help="reloads the trained model (best)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="the batch size of the training"
    )
    parser.add_argument(
        "--dataset_fraction_percent", type=float, default=100.0, help="the percent of the datasets to be used. Default 100.0")
    args = parser.parse_args()

    out_dir = args.language
    input_file = args.input
    tokenizer = None
        
    models_dir = os.path.join(out_dir, "models")
    embedding_models_dir = os.path.join(out_dir, "embedding_models")
    eb_models_latest_dir = os.path.join(embedding_models_dir, "latest")
    eb_models_best_dir = os.path.join(embedding_models_dir, "best")

    latest_model_fp = os.path.join(models_dir, "latest_model.pt")
    best_model_fp = os.path.join(models_dir, "best_model.pt")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(embedding_models_dir, exist_ok=True)
    os.makedirs(eb_models_latest_dir, exist_ok=True)
    os.makedirs(eb_models_best_dir, exist_ok=True)

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
        train_lcnt = get_file_line_cnt(train_input_fp)
        eval_lcnt = get_file_line_cnt(eval_input_fp)
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
        d_model=args.model_embedding_size,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
    )
    model = model.to(device)
    if args.reload_latest_model:
        state_dict = torch.load(latest_model_fp)
        model.load_state_dict(state_dict)
    if args.reload_best_model:
        state_dict = torch.load(best_model_fp)
        model.load_state_dict(state_dict)

    # training
    pad_id = tokenizer.token_to_id("<pad>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = args.epochs
    loss_history = []
    eval_loss_history = []
    min_eval_loss = float("inf")

    model.train()
    print("start training")
    for epoch in range(num_epochs):
        max_per_epoch = int((args.dataset_fraction_percent * int(train_lcnt/args.batch_size) )/ 100)
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
                bar.text("Loss: " + str(loss.item())+ " Memory: "+str(torch.cuda.memory_allocated() / 1024**2) + "MB" " Free: " + str((torch.cuda.max_memory_allocated()-torch.cuda.memory_allocated()) / 1024**2) + "MB")
                bar()
                #cleanup
                del loss, outputs, src_batch, tgt_batch, batch
                torch.cuda.empty_cache()
                gc.collect()

        avg_loss = epoch_loss / max_per_epoch
        loss_history.append(avg_loss)
        print("evaluation")
        model.eval()
        epoch_val_loss = 0.0
        batch_idx = 0
        max_per_epoch = int((args.dataset_fraction_percent * int(eval_lcnt/args.batch_size) )/ 100.0)
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
                bar.text("Loss: " + str(loss.item())+ " Memory: "+str(torch.cuda.memory_allocated() / 1024**2) + "MB" " Free: " + str((torch.cuda.max_memory_allocated()-torch.cuda.memory_allocated()) / 1024**2) + "MB")
                bar()
                torch.cuda.empty_cache()

        avg_eval_loss = epoch_val_loss / max_per_epoch
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {avg_eval_loss}")
        print(torch.cuda.memory_allocated() / 1024**2, "MB")
        eval_loss_history.append(avg_eval_loss)
        if (avg_eval_loss < min_eval_loss):
            torch.save(model.state_dict(), best_model_fp)
            min_eval_loss = avg_eval_loss;
            print("best validation loss, save model to: "+best_model_fp)
        torch.save(model.state_dict(), latest_model_fp)

    model.eval()
    epoch_test_loss = 0.0
    batch_idx = 0
    max_per_epoch = int((args.dataset_fraction_percent * int(test_lcnt/args.batch_size) )/ 100.0)
    with alive_bar(max_per_epoch) as bar, torch.no_grad():
        for batch in test_loader:
            if batch_idx > max_per_epoch +1:
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
            bar.text("Loss: " + str(loss.item())+ " Memory: "+str(torch.cuda.memory_allocated() / 1024**2) + "MB" " Free: " + str((torch.cuda.max_memory_allocated()-torch.cuda.memory_allocated()) / 1024**2) + "MB")
            bar()
            torch.cuda.empty_cache()
    avg_test_loss = epoch_test_loss / max_per_epoch
    print(f"Test Loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()
