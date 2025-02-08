import os
import sys

sys.path[0] = os.getcwd()

import torch

from dataloader import DataCollator, get_dataset_constructor
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates

from transformers.trainer import Trainer, TrainingArguments
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from peft import PeftModel

@dataclass()
class TrainingArgument:
    epoch: int                 = field()
    dataset: str               = field()

    batchsize: int             = field(default=1)
    gradient_acc: int          = field(default=1)
    learning_rate: float       = field(default=2e-4)
    warmup_ratio: float        = field(default=0.03)

    from_checkpoint: bool      = field(default=False)

@dataclass()
class SavingArgument:
    saving_path: str           = field()
    saving_strategy : str      = field(default='steps')
    saving_step: int           = field(default=60)
    
@dataclass()
class LogArgument:
    project_name: str          = field()
    run_name: str              = field()
    log_step: int              = field()
    
    log_activate: bool         = field(default=True)
    log_model: bool            = field(default=False)
    log_faster: bool           = field(default=True)

parser = HfArgumentParser((TrainingArgument, SavingArgument, LogArgument))
training_args, saving_args, log_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
args = HfArgumentParser((TrainingArguments)).parse_args_into_dataclasses(remaining_args)[0]


tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path='downloaded/llava-med-v1.5-mistral-7b',
    model_base=None,
    model_name='llava-med-v1.5-mistral-7b'
)
    
# dataset = RadDataset('train')
dataset = get_dataset_constructor(training_args.dataset)('train')
conv = conv_templates['mistral_instruct']
collate_fn = DataCollator(
    tokenizer=tokenizer,
    split='train',
    conversation_template=conv,
    pad_token_id=tokenizer.pad_token_id,
    image_processor=image_processor,
    model_config=model.config,
)

model = PeftModel.from_pretrained(
    model, 
    'outputs/vdz_base',
    is_trainable=True
)

model.train()

args.do_train   = True
args.do_eval    = False
args.do_predict = False
args.remove_unused_columns = False
args.num_train_epochs            = training_args.epoch
args.per_device_train_batch_size = training_args.batchsize
args.gradient_accumulation_steps = training_args.gradient_acc
args.learning_rate               = training_args.learning_rate
args.warmup_ratio                = training_args.warmup_ratio

args.save_strategy = saving_args.saving_strategy
args.save_steps    = saving_args.saving_step


#prepare wandb for logging
if log_args.log_activate:
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = log_args.project_name
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "false" if not log_args.log_model else "true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false" if log_args.log_faster else "true"
    
    args.report_to = ['wandb']
    args.run_name = log_args.run_name
    args.logging_steps = log_args.log_step


trainer = Trainer(
    model,
    args,
    data_collator=collate_fn,
    train_dataset=dataset,
)

trainer.train(training_args.from_checkpoint)
trainer.save_model(saving_args.saving_path)