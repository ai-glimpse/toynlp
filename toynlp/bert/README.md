# Bert


## The mistakes that I made

### Can not overfit even on very small dataset

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
```

And the loss is keep shake around 2.3(most mlm loss, the nsp loss is near 0.0).
The make me realize that the model maybe stuck in a local minimum, due to the **large learning rate**.
So I try with a smaller learning rate `0.0001` and then the training loss can decrease to 0.0, and the predicted tokens can finally match the target tokens:

```

```


## References
- [google-research/bert](https://github.com/google-research/bert)
- [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
- [dreamgonfly/BERT-pytorch](https://github.com/dreamgonfly/BERT-pytorch)
