# NLLB-200-distilled-600M_dyu-fra

This model is a fine-tuned version of [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) on the [uvci/Koumankan_mt_dyu_fr](https://huggingface.co/datasets/uvci/Koumankan_mt_dyu_fr) dataset.
It achieves the following results on the validation set:
- Loss: xx
- BLEU: xxx

### Training hyperparameters
The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 32
- eval_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results
| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.5055        | 1.0   | 619  | 0.3336          | 0.8868   |
| 0.2832        | 2.0   | 1238 | 0.3039          | 0.9100   |
| 0.1729        | 3.0   | 1857 | 0.3944          | 0.9091   |

### Framework versions
- Transformers 4.42.4
- Pytorch 2.3.1
- Datasets 2.20.0
- Tokenizers 0.19.1
- Kserve 0.11.2