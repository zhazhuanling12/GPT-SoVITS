import os
import sys
import shutil
import soundfile as sf

from GPT_SoVITS.inference_webui import (
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)


class Synthesis:
    def __init__(self, GPT_model_path, SoVITS_model_path):
        change_gpt_weights(gpt_path=GPT_model_path)
        change_sovits_weights(sovits_path=SoVITS_model_path)

    def test(
        self,
        ref_text,
        ref_audio_path,
        target_text,
        output_wav_path,
        language_combobox="Chinese",
        language_combobox_02="Chinese",
    ):
        synthesis_result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=language_combobox,
            text=target_text,
            text_language=language_combobox_02,
            ref_free=True,
        )

        result_list = list(synthesis_result)

        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)


def main():
    ref_text = "花婆婆之所以如此感人，是因为它具有饱满的诗情和美学价值，并且它是一部最动人的女性主义绘本。"
    ref_audio_path = "test_16k.wav"
    test_file = sys.argv[1]  #'test.txt'
    # "exp/pengbei/GPT_weight/exp_pengbei-e24.ckpt"
    GPT_model_path = sys.argv[2]
    # "exp/pengbei/SoVITS_weights/exp_pengbei_e48_s960.pth"
    SoVITS_model_path = sys.argv[3]
    output_path = sys.argv[4]
    os.makedirs(output_path, exist_ok=True)
    handler = Synthesis(
        GPT_model_path=GPT_model_path,
        SoVITS_model_path=SoVITS_model_path,
    )
    with open(test_file, "r") as fr:
        target_texts = fr.readlines()
    shutil.copy(ref_audio_path, output_path + "/original.wav")
    for i, target_text in enumerate(target_texts):
        output_wav_path = os.path.join(output_path, f"output_{i}.wav")
        handler.test(ref_text, ref_audio_path, target_text, output_wav_path)


if __name__ == "__main__":
    main()
