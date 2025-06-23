import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from alive_progress import alive_bar
from dataset import LazyLoader, split_train_test_set, get_file_line_cnt
from network import PositionalEncoding, Seq2SeqTransformer, EncoderOnly
from tokenizer import sentence_splitter, tokenize
from tokenizers import Tokenizer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import torch.nn as nn
import torch.optim as optim
import torch
import warnings
import gc
import json

warnings.filterwarnings(
    "ignore", message="The PyTorch API of nested tensors is in prototype stage.*"
)

def save_model_to_embedding(out_dir, model_type):
    tokenizer_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer_embedding_path = os.path.join(out_dir, "embedding_models", model_type, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    model_cfg = load_model_cfg(out_dir, model_type)
    fp = os.path.join(out_dir, "models", model_type + "_model")
    print(out_dir);
    print(fp)
    enc_fp = os.path.join(out_dir, "embedding_models", model_type, "model")
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )
    state_dict = torch.load(fp+".pt")
    model.load_state_dict(state_dict)

    encoder_model = EncoderOnly(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )
    print(f"model.embedding.shape: {model.embedding.weight.shape} / encoder_model.embedding: {encoder_model.embedding.weight.shape}")
    # copy the weights of full model to encoder model
    encoder_model.embedding.load_state_dict(model.embedding.state_dict())
    encoder_model.pos_encoder.load_state_dict(model.pos_encoder.state_dict())
    encoder_model.encoder.load_state_dict(model.transformer.encoder.state_dict())
    torch.save(encoder_model.state_dict(), enc_fp + ".pt")
    with open(enc_fp + ".cfg", "w") as f:
        json.dump(model_cfg, f, indent=4)
    tokenizer.save(tokenizer_embedding_path)



def save_model(out_dir, model, model_type, model_cfg, tokenizer):
    fp = os.path.join(out_dir, "models", model_type + "_model")
    enc_fp = os.path.join(out_dir, "embedding_models", model_type, "model")
    tokenizer_path = os.path.join(out_dir, "embedding_models", model_type, "tokenizer.json")
    torch.save(model.state_dict(), fp + ".pt")
    with open(fp + ".cfg", "w") as f:
        json.dump(model_cfg, f, indent=4)

    with open(fp + ".loss", "w") as f:
        f.write("Evaluation Loss: "+eval_loss);

    encoder_model = EncoderOnly(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["vocab_size"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_encoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )
    # copy the weights of full model to encoder model
    encoder_model.embedding.load_state_dict(model.embedding.state_dict())
    encoder_model.pos_encoder.load_state_dict(model.pos_encoder.state_dict())
    encoder_model.encoder.load_state_dict(model.transformer.encoder.state_dict())
    torch.save(encoder_model.state_dict(), enc_fp + ".pt")
    with open(enc_fp + ".cfg", "w") as f:
        json.dump(model_cfg, f, indent=4)
    tokenizer.save(tokenizer_path)


def load_model_cfg(out_dir, model_type):
    fp = os.path.join(out_dir, "models", model_type + "_model.cfg")
    model_cfg = None
    with open(fp, "r") as f:
        model_cfg = json.load(f)
    return model_cfg


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
        help="Defines the language, only used for the subdirectory, where new files will be created. Default: 'en' ",
    )
    parser.add_argument(
        "--model_nhead",
        type=int,
        default=8,
        help="the number of heads used in the transformer model. embedding size must be divisible by nhead. More heads give the transformer other viewpoints on the embeddings.",
    )
    parser.add_argument(
        "--model_dim_feedforward",
        type=int,
        default=2048,
        help="the feed forward dimension size.",
    )
    parser.add_argument(
        "--model_dropout",
        type=float,
        default=0.1,
        help="The dropout for the neural network. Helps prevent overfitting",
    )
    parser.add_argument(
        "--model_embedding_size",
        type=int,
        default=512,
        help="embedding size, both the input enbedding size and also the generated result embedding size (in this case has to be a multiple of '8'. Default: 512",
    )
    parser.add_argument(
        "--model_encoder_layer_count",
        type=int,
        default=6,
        help="the encoder layer count the internal transformer. More layers make the model bigger and more komplex, but also increase computation time on inference",
    )
    parser.add_argument(
        "--model_decoder_layer_count",
        type=int,
        default=6,
        help="the decoding layer code of the internal transformer. makes the model bigger and able to handle more komplex tasks. increases training time, but does not effect inference on the embedding model",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=9000,
        help="the amount of distinct subwords the model differentiates, greater value make more accurate models, smaller values preserver memory. Default: 20000",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="the number of epochs to train. Default: 500",
    )
    parser.add_argument(
        "--split_sentences",
        action="store_true",
        help="Split input into sentences, by special characters. Should be avoided for professional result",
    )
    parser.add_argument(
        "--reload_tokenizer",
        action="store_true",
        help="whether to reload the previous tokenizer",
    )
    parser.add_argument(
        "--reload_datasets",
        action="store_true",
        help="whether to reload the splitted datasets from the previous runs",
    )
    parser.add_argument(
        "--normalize_accents",
        action="store_true",
        help="normalize accents (like é to e or á to a) and do NOT preservers the original (should not be set for most languages)",
    )
    parser.add_argument(
        "--reload_latest_model",
        action="store_true",
        help="reloads the trained model (latest)",
    )
    parser.add_argument(
        "--reload_best_model",
        action="store_true",
        help="reloads the trained model (best)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="the batch size of the training"
    )
    parser.add_argument(
        "--dataset_fraction_percent",
        type=float,
        default=100.0,
        help="the percent of the datasets to be used. Default 100.0",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="defines the learning rate. For reduce_lr_on_plateau , this is the start learning rate",
    )
    parser.add_argument(
        "--disable_reduce_lr_on_plateau",
        action="store_true",
        help="disableces reduce_lr_on_plateau. In this case, The model will be trained on a fixed learning rate for fixed epochs",
    )
    parser.add_argument(
        "--disable_early_stopping",
        action="store_true",
        help="disables the early stopping mechanism. In this case the model will be trained further, even if validation loss is persistentley not decreasing.",
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_patience",
        type=int,
        default=5,
        help="defines the patience for lr_on_plateau. after [patience] epochs, where the validation_loss did not improve, learning_rate  will be decreased by factor --reduce_lr_on_plateau_factor",
    )
    parser.add_argument(
        "--max_word_per_sentence",
        type=int,
        default=40,
        help="defines max words a sentence can have. skips lines with more words. Useful if sporadically getting out of gpu memory because of ocassional long sentences",
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_factor",
        type=float,
        default=0.1,
        help="defines at which factor the learning rate will be decreased, when lr_on_plateau is triggered",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="defines the patience for early_stopping. after [patience] epochs, where the validation_loss did not improve, the training stops ",
    )
    args = parser.parse_args()

    model_cfg = {
        "d_model": args.model_embedding_size,
        "nhead": args.model_nhead,
        "num_encoder_layers": args.model_encoder_layer_count,
        "num_decoder_layers": args.model_decoder_layer_count,
        "dim_feedforward": args.model_dim_feedforward,
        "dropout": args.model_dropout,
        "eval_loss": float("inf")
    }

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
        train_input_fp = os.path.join(out_dir, "datasets", "train.txt")
        eval_input_fp = os.path.join(out_dir, "datasets", "eval.txt")
        test_input_fp = os.path.join(out_dir, "datasets", "test.txt")
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
            train_input_fp, args.vocab_size, keep_accents=not args.normalize_accents
        )
        tokenizer.save(tokenizer_path)

    # create loaders for dataset
    lazy_train_loader = LazyLoader(
        tokenizer=tokenizer,
        file_path=train_input_fp,
        batch_size=args.batch_size,
        max_word_per_sentence=args.max_word_per_sentence,
    )
    lazy_eval_loader = LazyLoader(
        tokenizer=tokenizer,
        file_path=eval_input_fp,
        batch_size=args.batch_size,
        max_word_per_sentence=args.max_word_per_sentence,
    )
    lazy_test_loader = LazyLoader(
        tokenizer=tokenizer,
        file_path=test_input_fp,
        batch_size=args.batch_size,
        max_word_per_sentence=args.max_word_per_sentence,
    )

    if args.reload_latest_model:
        model_cfg = load_model_cfg(out_dir, "latest")
    if args.reload_best_model:
        model_cfg = load_model_cfg(out_dir, "best")

    vocab_size = tokenizer.get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size, 
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = None
    if not args.disable_reduce_lr_on_plateau:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.reduce_lr_on_plateau_factor,
            patience=args.reduce_lr_on_plateau_patience,
        )
    num_epochs = args.epochs
    loss_history = []
    eval_loss_history = []
    min_eval_loss = float("inf")

    model.train()
    print("start training")
    stopping_patience = 0
    for epoch in range(num_epochs):
        max_per_epoch = int(
            (args.dataset_fraction_percent * int(train_lcnt / args.batch_size)) / 100
        )
        batch_idx = 0
        epoch_loss = 0.0
        with alive_bar(max_per_epoch) as bar:
            train_loader = lazy_train_loader.loader()
            for batch in train_loader:
                if batch_idx >= max_per_epoch:
                    break
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
                bar.text(
                    "Loss: "
                    + str(loss.item())
                    + " Memory: "
                    + str(torch.cuda.memory_allocated() / 1024**2)
                    + "MB"
                    " Free: "
                    + str(
                        (
                            torch.cuda.max_memory_allocated()
                            - torch.cuda.memory_allocated()
                        )
                        / 1024**2
                    )
                    + "MB"
                    + f"Tensors alive: {len(gc.get_objects())}"
                )
                bar()
                # cleanup
                del loss, outputs, src_batch, tgt_batch, batch
                torch.cuda.empty_cache()
                gc.collect()

        avg_loss = epoch_loss / batch_idx
        loss_history.append(avg_loss)
        print("evaluation")
        model.eval()
        epoch_val_loss = 0.0
        batch_idx = 0
        max_per_epoch = int(
            (args.dataset_fraction_percent * int(eval_lcnt / args.batch_size)) / 100.0
        )
        print(f"Tensors alive: {len(gc.get_objects())}")
        with alive_bar(max_per_epoch) as bar, torch.no_grad():
            eval_loader = lazy_eval_loader.loader()
            for batch in eval_loader:
                if batch_idx >= max_per_epoch:
                    break
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
                bar.text(
                    "Loss: "
                    + str(loss.item())
                    + " Memory: "
                    + str(torch.cuda.memory_allocated() / 1024**2)
                    + "MB"
                    " Free: "
                    + str(
                        (
                            torch.cuda.max_memory_allocated()
                            - torch.cuda.memory_allocated()
                        )
                        / 1024**2
                    )
                    + "MB"
                )
                bar()
                torch.cuda.empty_cache()

        print(f"Tensors alive: {len(gc.get_objects())}")
        avg_eval_loss = epoch_val_loss / batch_idx
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {avg_eval_loss}")
        print(torch.cuda.memory_allocated() / 1024**2, "MB")
        scheduler.step(avg_eval_loss)
        eval_loss_history.append(avg_eval_loss)
    
        model_cfg["eval_loss"] = avg_eval_loss
        save_model(out_dir, model, "latest", model_cfg, tokenizer)
        if avg_eval_loss < min_eval_loss:
            stopping_patience = 0
            #torch.save(model.state_dict(), best_model_fp)
            min_eval_loss = avg_eval_loss
            save_model(out_dir, model, "best", model_cfg, tokenizer)
            print("best validation loss, save model to: " + best_model_fp)
        elif not args.disable_early_stopping:
            stopping_patience += 1
            if stopping_patience >= args.early_stopping_patience:
                print(
                    f"early stopping, because validation loss did not improve for {args.early_stopping_patience} epochs"
                )
                break
        #torch.save(model.state_dict(), latest_model_fp)
        

    model.eval()
    epoch_test_loss = 0.0
    batch_idx = 0
    max_per_epoch = int(
        (args.dataset_fraction_percent * int(test_lcnt / args.batch_size)) / 100.0
    )
    with alive_bar(max_per_epoch) as bar, torch.no_grad():
        test_loader = lazy_train_loader.loader()
        for batch in test_loader:
            if batch_idx >= max_per_epoch:
                break
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
            bar.text(
                "Loss: "
                + str(loss.item())
                + " Memory: "
                + str(torch.cuda.memory_allocated() / 1024**2)
                + "MB"
                " Free: "
                + str(
                    (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated())
                    / 1024**2
                )
                + "MB"
            )
            bar()
            torch.cuda.empty_cache()
    avg_test_loss = epoch_test_loss / batch_idx
    print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    #main()
    save_model_to_embedding("en", "best")
