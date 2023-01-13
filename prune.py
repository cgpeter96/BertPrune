"""
@author: cgpeter96
@desc: 根据给定参数裁剪bert层数
@usage:
    python prune.py -mp bert_uncase/ -pmp bert_uncase_prune/ -sl 1,2,3
"""
import os 
import json
import argparse
from typing import List,Union
import copy
import torch
from transformers import BertModel,BertTokenizer,BertConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp","--model_path",type=str,default="")
    parser.add_argument("-pmp","--prune_model_path",type=str,default="./prune_model")
    parser.add_argument("-sl","--select_layers",type=str,default="2",help="可以输入int or 1,2,3,4")
    args = parser.parse_args()
    if hasattr(args,"select_layers"):
        if "," in args.select_layers:
            select_layers = list(map(int,[i.strip() for i in args.select_layers.split(",")]))
        else:
            select_layers=int(args.select_layers)
        setattr(args,"select_layers",select_layers)
    return args

def print_model_parameters(model):
    for name,param in model.named_parameters():
        print(name,param.shape)

def prune_model_parameters(model,maintain_layer_nums:Union[List[int],int]=6):
    """
    裁切bert model权重
    Args:
        model: BertModel
        maintain_layer_nums: 需要保存的参数，可以是list or int
    """
    prune_params={} #裁剪参数
    
    select_layers = []#被选择的层
    if isinstance(maintain_layer_nums,int):
        select_layers = list(range(maintain_layer_nums))
    elif isinstance(maintain_layer_nums,List):
        select_layers = maintain_layer_nums
    else:
        raise NotImplementedError("Not support maintain_layer_nums type:{}".format(type(maintain_layer_nums)))
    max_layer_nums  = []
    for layer_name,layer_param in model.named_parameters():
        if "embeddings" in layer_name:
            prune_params[layer_name]=layer_param 
        elif  "encoder.layer" in layer_name:
            #统计有效层数
            if "attention" in layer_name:
                layer_num = int(layer_name.split(".attention")[0].rsplit(".",1)[1])
                max_layer_nums.append(layer_num)

            for idx,layer_num in enumerate(select_layers):
                name_prefix="encoder.layer.{}.".format(layer_num)
                if layer_name.startswith(name_prefix):
                    prune_params[layer_name]=layer_param
                    continue

        elif "pooler" in layer_name:
            prune_params[layer_name]=layer_param 

    #判断裁剪层数和实际层不一致
    if max(select_layers)>=max(max_layer_nums):
        print("WARN:裁剪最大层数({}),超出模型层数({}),仅保存有效层数".format(max(select_layers),max(max_layer_nums)))
    return prune_params

def prune_model_config(config, maintain_layer_nums:Union[List[int],int]=6):
    """
    修改对应config的参数
    Args:
        config: BertConfig
        maintain_layer_nums: 需要保存的参数，可以是list or int
    """
    prune_config =copy.deepcopy(config)
    prune_config.num_hidden_layers = maintain_layer_nums if isinstance(maintain_layer_nums,int) else len(maintain_layer_nums)
    return prune_config

def check_prune_model(config,model):
    # 保存层数和实际是一致的
    valid_layers = []
    for layer_name,_ in model.named_parameters():
        if "encoder.layer" in layer_name:
            valid_layers.append(layer_name.split("attention")[0])
    valid_layers = list(set(valid_layers))
    if config.num_hidden_layers != len(valid_layers):
        config.num_hidden_layers = len(valid_layers)

def main_prune(args):
    if not os.path.exists(args.model_path):
        print("model_path not exist",file=sys.stderr)
    if not os.path.exists(args.prune_model_path):
        os.makedirs(args.prune_model_path)

    config = BertConfig.from_pretrained(args.model_path)
    model = BertModel.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    print("prune model...")
    prune_model_params = prune_model_parameters(model,args.select_layers)
    prune_config = prune_model_config(config,args.select_layers)
    check_prune_model(prune_config,prune_model_params)

    if args.prune_model_path:
        print(f"save prune model to:{args.prune_model_path}")
        torch.save(prune_model_params,os.path.join(args.prune_model_path,"pytorch_model.bin"))
        prune_config.save_pretrained(args.prune_model_path)
        tokenizer.save_pretrained(args.prune_model_path)

    

if __name__ == "__main__":
    args = parse_args()
    main_prune(args)