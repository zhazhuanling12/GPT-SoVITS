import os
import sys
import torch
from tqdm import tqdm
import yaml
import shutil
import traceback
import json

from config import (
    is_half,
    exp_root,
)

is_half = False

now_dir = os.getcwd()
sys.path.insert(0, now_dir)

tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if os.path.exists(tmp):
    for name in os.listdir(tmp):
        if name == "jieba.cache":
            continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass


def load_and_save(model_path, new_model):
    checkpoint_dict = {}
    step = int(model_path.split("=")[-1].split(".")[0])
    epoch = int(model_path.split("=")[-2].split("e")[0])
    checkpoint_dict["global_step"] = step
    checkpoint_dict["epoch"] = epoch
    checkpoint_dict["state_dict"] = torch.load(
        model_path, map_location="cpu"
    ).state_dict()
    torch.save(checkpoint_dict, new_model)


def prepare_datasets(inp_text, inp_wav_dir, opt_dir):
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")
    from GPT_SoVITS.prepare_datasets.extract_feats import ExtractFeats

    phandler = ExtractFeats(opt_dir)
    for line in tqdm(lines):
        try:
            wav_name, text = line.strip().split("\t")
            if inp_wav_dir != "" and inp_wav_dir != None:
                wav_name = os.path.basename(wav_name)
                if not wav_name.endswith(".wav"):
                    wav_name += ".wav"
                wav_path = "%s/%s" % (inp_wav_dir, wav_name)
            else:
                wav_path = wav_name
                wav_name = os.path.basename(wav_name)
            phandler.pipeline(text, wav_name, wav_path)
        except:
            print(line, traceback.format_exc())


def train_model(
    batch_size=20,
    total_epoch=50,
    exp_name="xxx",
    if_save_latest=True,
    if_save_every_weights=True,
    save_every_epoch=4,
    text_low_lr_rate=0.4,
    pretrained_s2G="GPT_SoVITS/pretrained_models/s2G488k.pth",
    pretrained_s2D="GPT_SoVITS/pretrained_models/s2D488k.pth",
):
    with open("GPT_SoVITS/configs/s2.json") as f:
        data = f.read()
        data = json.loads(data)
    s2_dir = "%s/%s" % (exp_root, exp_name)
    os.makedirs("%s/logs_s2" % (s2_dir), exist_ok=True)
    if is_half == False:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = "1"
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
    SoVITS_weight_root = os.path.join(exp_name, "SoVITS_weights")
    os.makedirs(SoVITS_weight_root, exist_ok=True)
    data["save_weight_dir"] = SoVITS_weight_root
    data["name"] = exp_name.replace('/', '_')
    tmp_config_path = "%s/tmp_s2.json" % tmp
    with open(tmp_config_path, "w") as f:
        f.write(json.dumps(data))

    cmd = 'python GPT_SoVITS/s2_train.py --config "%s"' % (tmp_config_path)
    os.system(cmd)


def main():
    inp_text, inp_wav_dir, opt_dir = sys.argv[1:]
    prepare_datasets(inp_text, inp_wav_dir, opt_dir)
    train_model(exp_name=opt_dir)


if __name__ == "__main__":
    main()
