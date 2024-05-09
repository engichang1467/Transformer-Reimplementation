import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask
from model import build_transformer

from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import lightning as L

import warnings
from pathlib import Path


def greedy_decode(
    lightning_mod, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    """Generate a translation using greedy decoding.

    Args:
        lightning_mod (TransformerModule): THe Lightning Module for the transformer model.
        source (torch.Tensor): The input source sequence.
        source_mask (torch.Tensor): The mask for the source sequence.
        tokenizer_src: The source language tokenizer.
        tokenizer_tgt: The target language tokenizer.
        max_len (int): The maximum length of the output sequence.
        device: The device on which the computation will be performed.

    Returns:
        torch.Tensor: The decoded sequence.
    """
    sos_ids = tokenizer_tgt.token_to_id("[SOS]")
    eos_ids = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = lightning_mod.model.encode(source, source_mask)

    # Initizliae the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_ids).type_as(source)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask)

        # Calculate the output of the decoder
        output = lightning_mod.model.decode(
            encoder_output, source_mask, decoder_input, decoder_mask
        )

        # Get the next token
        prob = lightning_mod.model.project(output[:, -1])
        # Select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()),
            ],
            dim=1,
        )

        if next_word == eos_ids:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    writer,
    num_examples=2,
):
    """Run validation on the provided dataset.

    Args:
        model (Transformer): The trained transformer model.
        validation_ds (Dataset): The dataset for validation.
        tokenizer_src: The source language tokenizer.
        tokenizer_tgt: The target language tokenizer.
        max_len (int): The maximum length of the output sequence.
        device: The device on which the computation will be performed.
        writer: The writer for logging.
        num_examples (int, optional): The number of examples to process. Defaults to 2.
    """
    model.eval()
    count = 0

    source_texts, expected, predicted = [], [], []

    # Size of the control window (just use a default value)
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"]
            encoder_mask = batch["encoder_mask"]

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print to the console
            print("-" * console_width)
            print(f"SOURCE: {source_text}")
            print(f"TARGET: {target_text}")
            print(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break


def get_all_sentences(ds, lang):
    """Generator function to yield all sentences in a dataset for a specific language.

    Args:
        ds: The dataset.
        lang (str): The language code for the target language.

    Yields:
        str: Each sentence in the dataset for the specified language.
    """
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    """Get or build a tokenizer for a specific language.

    Args:
        config (dict): Configuration parameters.
        ds: The dataset.
        lang (str): The language code for the target language.

    Returns:
        Tokenizer: The tokenizer for the specified language.
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    """Get the training and validation dataloaders along with tokenizers.

    Args:
        config (dict): Configuration parameters.

    Returns:
        DataLoader: The training dataloader.
        DataLoader: The validation dataloader.
        Tokenizer: The tokenizer for the source language.
        Tokenizer: The tokenizer for the target language.
    """
    ds_raw = load_dataset(
        "opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train"
    )

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_len_src, max_len_tgt = 0, 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """Builds and returns the Transformer model.

    Args:
        config (dict): Configuration parameters.
        vocab_src_len (int): Length of the source vocabulary.
        vocab_tgt_len (int): Length of the target vocabulary.

    Returns:
        Transformer: The Transformer model.
    """
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )
    return model


def train_model(config):
    """Trains the Transformer model.

    Args:
        config (dict): Configuration parameters.
    """
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    )

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = L.Trainer(
        max_epochs=config["num_epochs"],
        accelerator="auto",  # set to "auto" or "gpu" to use GPUs if available
        devices="auto",  # Uses all available GPUs if applicable
    )

    # Define a LightningModule for the model
    class TransformerModule(L.LightningModule):
        def __init__(self, model, optimizer, loss_fn):
            super().__init__()
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn

        def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
            encoder_output = self.model.encode(encoder_input, encoder_mask)
            decoder_output = self.model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            proj_output = self.model.project(decoder_output)
            return proj_output

        def training_step(self, batch, idx):
            encoder_input = batch["encoder_input"]
            decoder_input = batch["decoder_input"]
            encoder_mask = batch["encoder_mask"]
            decoder_mask = batch["decoder_mask"]

            decoder_output = self(
                encoder_input, decoder_input, encoder_mask, decoder_mask
            )

            label = batch["label"]

            loss = self.loss_fn(
                decoder_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )

            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return self.optimizer

    # Move the model to the appropriate device
    model = TransformerModule(model, optimizer, loss_fn)

    # Train the model using the PyTorch Lightning Trainer
    trainer.fit(model, train_dataloader)

    run_validation(
        model,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt,
        config["seq_len"],
        device,
        writer,
    )

    # Save the trained model
    model_filename = get_weights_file_path(config, "final")
    torch.save(model.state_dict(), model_filename)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
