import os
import tempfile
import zipfile
import torch
import gradio as gr
from scipy.io.wavfile import write  # Added for audio saving

from modules.core.models.tts.ChatTtsModel import ChatTTSModel
from modules.utils.rng import np_rng
from modules.webui import webui_config
from modules.webui.webui_utils import tts_generate

# Create data directories
os.makedirs("data", exist_ok=True)
os.makedirs("data/speakers", exist_ok=True)

@torch.inference_mode()
def create_spk_from_seed(seed: int, name: str, gender: str, desc: str):
    """Create a speaker from seed and save as JSON"""
    spk = ChatTTSModel.create_speaker_from_seed(seed)
    spk.set_name(name=name)
    spk.set_desc(desc=desc)
    spk.set_gender(gender=gender)
    spk.set_author("")  # Default empty author
    spk.set_version("")  # Default empty version
    
    json_path = os.path.join("data/speakers", f"{seed}.spkv1.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(spk.to_json_str())
    
    return json_path

@torch.inference_mode()
def test_spk_voice(seed: int, text: str):
    """Generate test voice for a speaker"""
    spk = ChatTTSModel.create_speaker_from_seed(seed)
    audio = tts_generate(spk=spk, text=text)
    
    # Save the audio file - audio is now expected to be a tuple of (sample_rate, data)
    audio_path = os.path.join("data/speakers", f"{seed}.wav")
    write(audio_path, audio[0], audio[1])  # sample_rate, data
    
    return audio_path

def zip_files():
    """Create a zip file containing all generated speakers and audio"""
    zip_path = tempfile.mktemp(".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("data/speakers"):
            for file in files:
                zf.write(os.path.join(root, file), arcname=file)
    return zip_path

def batch_generate():
    """Batch generate 100 speakers with test voices"""
    default_text = "大家要用发展的眼光看待自己，千万不能沉迷当下的低落"
    
    for i in range(100):
        seed = np_rng()
        # Generate test voice
        test_spk_voice(seed, default_text)
        # Create speaker file
        create_spk_from_seed(
            seed=seed,
            name=str(seed),
            gender="*",
            desc=""
        )
        print(f"Generated speaker {i+1}/100")
    
    return "批量生成完成！"

def speaker_allcreator_ui():
    """Create the Gradio UI for batch speaker creation"""
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