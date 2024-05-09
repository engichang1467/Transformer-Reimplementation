# Transformer Architecture Reimplementation

This project implements the Transformer Architecture from [Attention is All You Need](https://arxiv.org/abs/1706.03762). The Transformer architecture has significantly impacted sequence-to-sequence learning by introducing an attention-based mechanism, replacing traditional recurrent and convolutional neural network structures. This approach has demonstrated remarkable effectiveness in various tasks including machine translation, language modeling, and text generation.

![Transformer Architecture](img/attention_arch.jpeg)

## Installation

To install the necessary dependencies, run:

```
pip install -r requirements.txt
```

## Training the Model

To train the model, execute the following command:

```
python train.py
```

During training, you will observe predictions from the model after each epoch:

```
Processing epoch 27: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 158/158 [00:51<00:00,  3.10it/s, loss=5.873]
--------------------------------------------------------------------------------
SOURCE: I'm sure I don't want to stay in here any longer!'
TARGET: Eu só sei que não quero mais ficar aqui!'
PREDICTED: Eu só sei que não quero mais ficar aqui!'
```

## Dataset

This implementation uses the [Opus books dataset](https://huggingface.co/datasets/Helsinki-NLP/opus_books), a collection of copyright-free books aligned by Andras Farkas. The default translation direction is from English to Italian, but you can modify this setting in `config.py` to translate between any languages of your choice.

## References

- [GitHub - pytorch-transformer](https://github.com/hkproj/pytorch-transformer)

If you use this code in your research or find it helpful, please consider citing the original paper:

```
@article{vaswani2017attention,
    title   = {Attention is all you need},
    author  = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser,  Lukasz and Polosukhin, Illia},
    journal = {Advances in neural information processing systems},
    volume  = {30},
    year    = {2017}
}
```
