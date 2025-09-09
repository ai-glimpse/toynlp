# Bert



## W/O Pretraining on SST2

The SST2 dataset:

- train/validation dataset: [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2)
- test dataset: [SetFit/sst2](https://huggingface.co/datasets/SetFit/sst2)


The model architecture:

```python
class SST2BertModel(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.bert = Bert(config, padding_idx=bert_tokenizer.token_to_id("[PAD]"))
        self.classifier = torch.nn.Linear(config.d_model, 2)  # SST-2 has 2 classes

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        bert_output = self.bert(input_ids, token_type_ids)
        cls_hidden_state = bert_output[:, 0, :]
        logits = self.classifier(cls_hidden_state)
        return logits
```



### Without Pretraining on SST2

The bert config(almost the same as base BERT):

> Note here we use a smaller `max_seq_length=128` while the original BERT uses `max_seq_length=512`.

```python
bert_config = BertConfig(
    max_seq_length=128,
    vocab_size=30522,
    d_model=768,
    attention_d_k=768,
    attention_d_v=768,
    head_num=12,
    d_feed_forward=3072,
    encoder_layers=12,
)
```

The training curve on SST2 dataset with different learning rates:

![](../../docs/images/bert/SST2Bert-Without-Pretrain.png)


Observations:
- With a learning rate of `5e-5`, the model achieves best test accuracy of around 80%
- With learning rates of `1e-5` and `1e-6`, the model got worse performance
- All the learning rates lead to overfitting after several epochs



## The mistakes that I made

### Can not overfit even on a very small dataset

I want to make sure that the full pipeline(data loading, tokenization, model training) is working correctly and that the model can overfit on a small dataset before scaling up. To speed up the overfitting speed, I set a bigger learning rate(0.01/0.001). But I find I can not overfit even on a very small dataset.

I spend lots of time and try many ways to debug this:
- Recheck the data preparation
- Verify the model architecture
- Experiment with different hyperparameters
- Make sure the mask token is being used correctly
- ...

All of these tries did not help me to overfit the model on the small dataset.
You may wonder: can llm/ai helps? The answer is NOOOOOO! They just can't. And they point out MANY meaningless suggestions.
So I try to debug by with smaller dataset and adding more print statements in my code to understand what's going wrong.

I try to train with only ONE data sample and print all the related tensors (input, target, prediction) to see if they make sense.
Finally, I find this output:

```
====================================================================================================
Epoch 146/1000 - Train Loss: 2.3028, Train MLM Loss: 2.3028, Train NSP Loss: 0.0000, Train NSP Accuracy: 1.0000, 
====================================================================================================
Input Tokens: [CLS]|#|#|skyscr|##aper|arch|##ipe|##lag|##o|[SEP]|trying|to|recover|nuclear|weapons|now|[MASK]|[MASK]|bottom|[MASK]|the|ocean|.|.|.|.|[MASK]|were|trying|to|keep|a|nuclear|power|-|plant|from|melting|down|.|hait|.|.|[MASK]|high|[MASK]|ranking|officials|were|still|alive|at|the|submerged|un|compound|.|.|.|.|all|[MASK]|of|[SEP]
Target Tokens: skyscr|at|the|of|they|power|.|some|-|sorts
Predicted Tokens: at|at|at|at|at|at|at|at|at|at
====================================================================================================
Epoch 147/1000 - Train Loss: 2.3028, Train MLM Loss: 2.3028, Train NSP Loss: 0.0000, Train NSP Accuracy: 1.0000, 
====================================================================================================
Input Tokens: [CLS]|#|#|skyscr|##aper|arch|##ipe|##lag|##o|[SEP]|trying|to|recover|nuclear|weapons|now|[MASK]|[MASK]|bottom|[MASK]|the|ocean|.|.|.|.|[MASK]|were|trying|to|keep|a|nuclear|power|-|plant|from|melting|down|.|hait|.|.|[MASK]|high|[MASK]|ranking|officials|were|still|alive|at|the|submerged|un|compound|.|.|.|.|all|[MASK]|of|[SEP]
Target Tokens: skyscr|at|the|of|they|power|.|some|-|sorts
Predicted Tokens: some|some|some|some|some|some|some|some|some|some
====================================================================================================
```

And the loss is keep shake around 2.3(most mlm loss, the nsp loss is near 0.0).
The make me realize that the model maybe stuck in a local minimum, due to the **large learning rate**.
So I try with a smaller learning rate `0.0001` and then the training loss can decrease to 0.0, and the predicted tokens can finally match the target tokens:

```
====================================================================================================
Epoch 29/1000 - Train Loss: 0.0004, Train MLM Loss: 0.0000, Train NSP Loss: 0.0004, Train NSP Accuracy: 1.0000, 
====================================================================================================
Input Tokens: [CLS]|#|#|skyscr|##aper|arch|##ipe|##lag|##o|[SEP]|trying|to|recover|nuclear|weapons|now|[MASK]|[MASK]|bottom|[MASK]|the|ocean|.|.|.|.|[MASK]|were|trying|to|keep|a|nuclear|power|-|plant|from|melting|down|.|reap|.|.|[MASK]|high|[MASK]|ranking|officials|were|still|alive|at|the|submerged|un|compound|.|.|.|.|all|[MASK]|of|[SEP]
Target Tokens: skyscr|at|the|of|they|power|.|some|-|sorts
Predicted Tokens: skyscr|at|the|of|they|power|.|some|-|sorts
====================================================================================================
Epoch 30/1000 - Train Loss: 0.0000, Train MLM Loss: 0.0000, Train NSP Loss: 0.0000, Train NSP Accuracy: 1.0000, 
====================================================================================================
Input Tokens: [CLS]|#|#|skyscr|##aper|arch|##ipe|##lag|##o|[SEP]|trying|to|recover|nuclear|weapons|now|[MASK]|[MASK]|bottom|[MASK]|the|ocean|.|.|.|.|[MASK]|were|trying|to|keep|a|nuclear|power|-|plant|from|melting|down|.|reap|.|.|[MASK]|high|[MASK]|ranking|officials|were|still|alive|at|the|submerged|un|compound|.|.|.|.|all|[MASK]|of|[SEP]
Target Tokens: skyscr|at|the|of|they|power|.|some|-|sorts
Predicted Tokens: skyscr|at|the|of|they|power|.|some|-|sorts
====================================================================================================
```

> To speed up the overfitting speed, I set a bigger learning rate(0.01/0.001)

So that's the story: A bigger learning rate can help the model to converge faster, but it may also lead to local minima.

And, why am I so hurry to want to train the model faster? Because my 4060Ti GPU is so slow and want to finish the training as soon as possible. So, be patient man.


## References
- [google-research/bert](https://github.com/google-research/bert)
- [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
- [dreamgonfly/BERT-pytorch](https://github.com/dreamgonfly/BERT-pytorch)
