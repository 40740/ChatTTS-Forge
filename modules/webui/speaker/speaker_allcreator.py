# -*- coding: utf-8 -*-
from scipy.io.wavfile import write
import gradio as gr
import torch
from modules.speaker import Speaker
from modules.utils.SeedContext import SeedContext
from modules.models import load_chat_tts
from modules.utils.rng import np_rng
from modules.webui import webui_config
from modules.webui.webui_utils import tts_generate
import os
import tempfile
import zipfile

# 创建数据目录
os.makedirs("data", exist_ok=True)
os.makedirs("data/speakers", exist_ok=True)  # 确保speakers文件夹存在

def create_spk_from_seed(seed: int, name: str, gender: str, desc: str):
    chat_tts = load_chat_tts()
    with SeedContext(seed, True):
        emb = chat_tts.sample_random_speaker()
    spk = Speaker(seed=-2, name=name, gender=gender, describe=desc)
    spk.emb = emb

    pt_file_path = os.path.join("data/speakers", f"{seed}.pt")
    torch.save(spk, pt_file_path)
    
    return pt_file_path

def test_spk_voice(seed: int, text: str):
    audio = tts_generate(
        spk=seed,
        text=text,
    )
    audio_file_path = os.path.join("data/speakers", f"{seed}.wav")
    write(audio_file_path, audio[0], audio[1])  # 使用 scipy.io.wavfile.write 保存音频文件

    return audio_file_path

def zip_files():
    zip_path = tempfile.mktemp(".zip")  # 创建一个临时文件路径
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("data/speakers"):
            for file in files:
                zf.write(os.path.join(root, file), arcname=file)
    return zip_path

def batch_generate():
    for i in range(100):
        seed = np_rng()
        text = "大家要用发展的眼光看待自己，千万不能沉迷当下的低落"
        test_spk_voice(seed, text)
        create_spk_from_seed(seed, name=str(seed), gender="*", desc="")
        print(i)

    return "批量生成完成！"

def speaker_allcreator_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Speaker Creator")
        batch_button = gr.Button("批量生成")
        zip_button = gr.Button("打包")
        download_button = gr.File(label="下载ZIP文件")
        
        batch_button.click(
            fn=batch_generate,
            inputs=[],
            outputs=[]
        )
        
        zip_button.click(
            fn=zip_files,
            inputs=[],
            outputs=download_button
        )

    return demo
 
