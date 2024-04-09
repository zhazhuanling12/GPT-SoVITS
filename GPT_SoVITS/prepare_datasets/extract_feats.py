import os
import torch
import traceback
import shutil
import numpy as np
import librosa
from my_utils import load_audio
from time import time as ttime
from scipy.io import wavfile
from tqdm import tqdm
from text.cleaner import clean_text
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from config import pretrained_sovits_path as pretrained_s2G
import utils
from transformers import AutoModelForMaskedLM, AutoTokenizer

maxx = 0.95
alpha = 0.5


def my_save(fea, path, i_part=1):
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s%s.pth" % (ttime(), i_part)
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


def get_semantic_model(s2config_path):
    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    vq_model.load_state_dict(
        torch.load(pretrained_s2G, map_location="cpu")["weight"], strict=False
    )
    return vq_model


def data_to_half(ssl_content, device="cuda", is_half=False):
    if is_half == True:
        ssl_content = ssl_content.half().to(device)
    else:
        ssl_content = ssl_content.to(device)
    return ssl_content


class ExtractFeats:
    def __init__(
        self,
        out_dir,
        s2config_path="GPT_SoVITS/configs/s2.json",
        cnhubert_base_path="GPT_SoVITS/pretrained_models/chinese-hubert-base/",
        bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    ):
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.cnhubert_model = cnhubert.get_model()
        self.vq_model = get_semantic_model(s2config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
        self.nan_fails = []
        self.device = "cuda:0"
        self.is_half = False
        self.name2text = f"{out_dir}/2-name2text.txt"
        self.out_bert_dir = f"{out_dir}/3-bert"
        self.hubert_dir = f"{out_dir}/4-cnhubert"
        self.wav32dir = f"{out_dir}/5-wav32k"
        self.semantic_path = f"{out_dir}/6-name2semantic.tsv"
        self.bert_model = data_to_half(
            self.bert_model, device=self.device, is_half=self.is_half
        )
        self.cnhubert_model = data_to_half(
            self.cnhubert_model, device=self.device, is_half=self.is_half
        )
        self.vq_model = data_to_half(
            self.vq_model, device=self.device, is_half=self.is_half
        )
        os.makedirs(self.out_bert_dir, exist_ok=True)
        os.makedirs(self.hubert_dir, exist_ok=True)
        os.makedirs(self.wav32dir, exist_ok=True)
        if os.path.exists(self.name2text):
            os.remove(self.name2text)
        if os.path.exists(self.semantic_path):
            os.remove(self.semantic_path)

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = data_to_half(
                    inputs[i], device=self.device, is_half=self.is_half
                )

            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def get_text_feat(self, wav_name, text, lang="zh"):
        phones, word2ph, norm_text = clean_text(
            text.replace("%", "-").replace("￥", ","), lang
        )
        bert_feature = self.get_bert_feature(norm_text, word2ph)
        return (
            wav_name,
            [
                " ".join(phones),
                " ".join([str(i) for i in word2ph]),
                norm_text,
            ],
            bert_feature,
        )

    def process_bert_feature(self, bert_out_dir, data, res):
        for name, text, lan in tqdm(data):
            try:
                name = os.path.basename(name)
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lan
                )
                path_bert = f"{bert_out_dir}/{name}.pt"
                if os.path.exists(path_bert) == False and lan == "zh":
                    bert_feature = self.get_bert_feature(norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    my_save(bert_feature, path_bert)
                phones = " ".join(phones)
                res.append([name, phones, word2ph, norm_text])
            except:
                print(name, text, traceback.format_exc())

    def get_hubert_feats(self, wav_name, wav_path):
        tmp_audio = load_audio(wav_path, 32000)
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.2:
            print("%s-filtered,%s" % (wav_name, tmp_max))
            return
        tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha * 32768)) + (
            (1 - alpha) * 32768
        ) * tmp_audio
        tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha * 1145.14)) + (
            (1 - alpha) * 1145.14
        ) * tmp_audio
        tmp_audio = librosa.resample(tmp_audio32b, orig_sr=32000, target_sr=16000)
        # 不是重采样问题
        tensor_wav16 = torch.from_numpy(tmp_audio)
        tensor_wav16 = data_to_half(
            tensor_wav16, device=self.device, is_half=self.is_half
        )
        ssl = (
            self.cnhubert_model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"]
            .transpose(1, 2)
            .cpu()
        )
        if np.isnan(ssl.detach().numpy()).sum() != 0:
            self.nan_fails.append(wav_name)
            print("nan filtered:%s" % wav_name)
            return
        wavfile.write(
            "%s/%s" % (self.wav32dir, wav_name),
            32000,
            tmp_audio32.astype("int16"),
        )
        return ssl

    def get_semantic_feats(self, wav_name, ssl_content=None):
        hubert_path = "%s/%s.pt" % (self.hubert_dir, wav_name)
        if ssl_content is None:
            ssl_content = torch.load(hubert_path, map_location="cpu")
        ssl_content = data_to_half(
            ssl_content, device=self.device, is_half=self.is_half
        )
        codes = self.vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        return semantic

    def pipeline(self, text, wav_name, wav_path):
        wav_id, res, bert_feature = self.get_text_feat(wav_name, text, lang="zh")
        with open(self.name2text, "a", encoding="utf8") as f:
            try:
                f.write("\t".join([wav_id] + res) + "\n")
            except Exception as exc:
                print(f"{exc} err")
        my_save(bert_feature, f"{self.out_bert_dir}/{wav_id}.pt")
        hubert_path = "%s/%s.pt" % (self.hubert_dir, wav_id)
        if not os.path.exists(hubert_path):
            ssl_feat = self.get_hubert_feats(wav_name, wav_path)
            my_save(ssl_feat, hubert_path)
        semantics = self.get_semantic_feats(wav_id)
        with open(self.semantic_path, "a", encoding="utf8") as fw:
            fw.write(wav_id + "\t" + semantics + "\n")

    def pipeline_v2(self, text, wav_name, wav_path):
        wav_id, res, bert_feature = self.get_text_feat(wav_name, text, lang="zh")
        ssl_feat = self.get_hubert_feats(wav_name, wav_path)
        if ssl_feat is None:
            print("aaa")
        semantics = self.get_semantic_feats(wav_name, ssl_feat)
        return wav_id, "\t".join(res), bert_feature, ssl_feat, semantics

    def save_feats(self, data: list, out_path: str, type: str = "dir"):
        if type == "file":
            with open(out_path, "w") as fw:
                fw.write("\n".join(data))
        else:
            for item in data:
                idx, content = item
                out_file = os.path.join(out_path, idx + ".pt")
                my_save(content, out_file)
