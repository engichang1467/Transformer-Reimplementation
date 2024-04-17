# Transformer Architecture Reimplementation

Implementation of Transformer Architecture from [Attention is All You Need](https://arxiv.org/abs/1706.03762). The Transformer architecture has revolutionized the field of sequence-to-sequence learning by replacing the traditional recurrent and convolutional neural network structures with a novel attention-based mechanism. This approach has proven to be highly effective in tasks such as machine translation, language modeling, and text generation, among others.

![](img/attention_arch.jpeg)

## Install

```
pip install -r requirements.txt
```

## Train the Model

```
python train.py
```

## Reference

```
@article{vaswani2017attention,
    title   = {Attention is all you need},
    author  = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser,  Lukasz and Polosukhin, Illia},
    journal = {Advances in neural information processing systems},
    volume  = {30},
    year    = {2017}
}
```
