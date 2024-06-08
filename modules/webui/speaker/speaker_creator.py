import gradio as gr
import torch
from modules.speaker import Speaker
from modules.utils.SeedContext import SeedContext
from modules.hf import spaces
from modules.models import load_chat_tts
from modules.utils.rng import np_rng
from modules.webui.webui_utils import get_speakers, tts_generate

import tempfile

names_list = [
    "Alice",
    "Bob",
    "Carol",
    "Carlos",
    "Charlie",
    "Chuck",
    "Chad",
    "Craig",
    "Dan",
    "Dave",
    "David",
    "Erin",
    "Eve",
    "Yves",
    "Faythe",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
    "Mallory",
    "Mallet",
    "Darth",
    "Michael",
    "Mike",
    "Niaj",
    "Olivia",
    "Oscar",
    "Peggy",
    "Pat",
    "Rupert",
    "Sybil",
    "Trent",
    "Ted",
    "Trudy",
    "Victor",
    "Vanna",
    "Walter",
    "Wendy",
]


@torch.inference_mode()
@spaces.GPU
def create_spk_from_seed(
    seed: int,
    name: str,
    gender: str,
    desc: str,
):
    chat_tts = load_chat_tts()
    with SeedContext(seed):
        emb = chat_tts.sample_random_speaker()
    spk = Speaker(seed=-2, name=name, gender=gender, describe=desc)
    spk.emb = emb

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
        torch.save(spk, tmp_file)
        tmp_file_path = tmp_file.name

    return tmp_file_path


@torch.inference_mode()
@spaces.GPU
def test_spk_voice(seed: int, text: str):
    return tts_generate(
        spk=seed,
        text=text,
    )


def random_speaker():
    seed = np_rng()
    name = names_list[seed % len(names_list)]
    return seed, name


creator_ui_desc = """
## Speaker Creator
使用本面板快捷抽卡生成 speaker.pt 文件。

1. **生成说话人**：输入种子、名字、性别和描述。点击 "Generate speaker.pt" 按钮，生成的说话人配置会保存为.pt文件。
2. **测试说话人声音**：输入测试文本。点击 "Test Voice" 按钮，生成的音频会在 "Output Audio" 中播放。
3. **随机生成说话人**：点击 "Random Speaker" 按钮，随机生成一个种子和名字，可以进一步编辑其他信息并测试。
"""


def speaker_creator_ui():
    def on_generate(seed, name, gender, desc):
        file_path = create_spk_from_seed(seed, name, gender, desc)
        return file_path

    def create_test_voice_card(seed_input):
        with gr.Group():
            gr.Markdown("🎤Test voice")
            with gr.Row():
                test_voice_btn = gr.Button("Test Voice", variant="secondary")

                with gr.Column(scale=4):
                    test_text = gr.Textbox(
                        label="Test Text",
                        placeholder="Please input test text",
                        value="说话人测试 123456789 [uv_break] ok, test done [lbreak]",
                    )
                    with gr.Row():
                        current_seed = gr.Label(label="Current Seed", value=-1)
                        with gr.Column(scale=4):
                            output_audio = gr.Audio(label="Output Audio")

        test_voice_btn.click(
            fn=test_spk_voice,
            inputs=[seed_input, test_text],
            outputs=[output_audio],
        )
        test_voice_btn.click(
            fn=lambda x: x,
            inputs=[seed_input],
            outputs=[current_seed],
        )

    gr.Markdown(creator_ui_desc)

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("ℹ️Speaker info")
                seed_input = gr.Number(label="Seed", value=2)
                name_input = gr.Textbox(
                    label="Name", placeholder="Enter speaker name", value="Bob"
                )
                gender_input = gr.Textbox(
                    label="Gender", placeholder="Enter gender", value="*"
                )
                desc_input = gr.Textbox(
                    label="Description",
                    placeholder="Enter description",
                )
                random_button = gr.Button("Random Speaker")
            with gr.Group():
                gr.Markdown("🔊Generate speaker.pt")
                generate_button = gr.Button("Save .pt file")
                output_file = gr.File(label="Save to File")
        with gr.Column(scale=5):
            create_test_voice_card(seed_input=seed_input)
            create_test_voice_card(seed_input=seed_input)
            create_test_voice_card(seed_input=seed_input)
            create_test_voice_card(seed_input=seed_input)

    random_button.click(
        random_speaker,
        outputs=[seed_input, name_input],
    )

    generate_button.click(
        fn=on_generate,
        inputs=[seed_input, name_input, gender_input, desc_input],
        outputs=[output_file],
    )
