import numpy as np
import pytest

from modules.core.handler.datacls.audio_model import AdjustConfig
from modules.core.handler.datacls.tts_model import InferConfig, TTSConfig
from modules.core.pipeline.dcls import TTSPipelineContext
from modules.core.pipeline.factory import PipelineFactory
from modules.core.spk.TTSSpeaker import TTSSpeaker
from tests.pipeline.misc import load_audio, load_audio_wav, save_audio


@pytest.mark.pipe_chat_tts
def test_chat_tts_pipe():
    out_audio_path = "./tests/test_outputs/pipe_chat_tts_out1.wav"

    pipe0 = PipelineFactory.create_chattts_pipeline(
        ctx=TTSPipelineContext(
            text="你好~这里是测试文本~",
            tts_config=TTSConfig(
                mid="chat-tts",
            ),
            infer_config=InferConfig(eos=" ", sync_gen=True),
        ),
    )

    audio_sr, audio_data = pipe0.generate()
    assert audio_data.size != 0
    save_audio(
        #
        file_path=out_audio_path,
        audio_data=audio_data,
        sample_rate=audio_sr,
    )
    # 检查文件不为空
    assert load_audio(out_audio_path)[1].size != 0


@pytest.mark.pipe_chat_tts
def test_chat_tts_voice_clone_pipe():
    voice_target_path = "./tests/test_inputs/chattts_out1.wav"
    out_audio_path = "./tests/test_outputs/pipe_chat_tts_voice_clone_out1.wav"

    voice_target = load_audio_wav(voice_target_path)
    voice_spk = TTSSpeaker.from_ref_wav_bytes(
        ref_wav=voice_target, text="这是一个测试文本。"
    )

    pipe0 = PipelineFactory.create_chattts_pipeline(
        ctx=TTSPipelineContext(
            text="你好，这里是音色克隆测试~",
            tts_config=TTSConfig(
                mid="chat-tts",
            ),
            spk=voice_spk,
            infer_config=InferConfig(eos=" ", sync_gen=True),
        ),
    )

    audio_sr, audio_data = pipe0.generate()
    assert audio_data.dtype == np.float32
    assert audio_data.size != 0
    save_audio(
        #
        file_path=out_audio_path,
        audio_data=audio_data,
        sample_rate=audio_sr,
    )
    # 检查文件不为空
    assert load_audio(out_audio_path)[1].size != 0
