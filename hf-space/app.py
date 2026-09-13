import os
import tempfile
import spaces
import gradio as gr
import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_ID = "THUDM/CogVideoX-5b-I2V"
pipe = None


def load_pipeline():
    global pipe
    if pipe is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        else:
            pipe.to("cpu")
    return pipe


@spaces.GPU(duration=180)
def generate_video(image: Image.Image, prompt: str, seed: int, steps: int, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        raise gr.Error("Загрузите фотографию.")
    if not prompt or not prompt.strip():
        raise gr.Error("Введите описание движения на английском языке.")
    if not torch.cuda.is_available():
        raise gr.Error("GPU ZeroGPU сейчас недоступен. Попробуйте ещё раз через несколько минут.")

    model = load_pipeline()
    image = image.convert("RGB")
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    frames = model(
        prompt=prompt.strip(),
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=int(steps),
        num_frames=49,
        guidance_scale=6,
        generator=generator,
    ).frames[0]

    output_path = os.path.join(tempfile.gettempdir(), "photo-to-motion.mp4")
    export_to_video(frames, output_path, fps=8)
    return output_path


with gr.Blocks(title="Photo → Motion", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Photo → Motion\n### CogVideoX · генерация видео из фотографии")
    gr.Markdown("Загрузите фото, опишите движение на английском и нажмите кнопку. Бесплатный GPU может быть занят — тогда повторите попытку позже.")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Фотография", sources=["upload", "webcam"])
            prompt = gr.Textbox(label="Описание движения", placeholder="Slow cinematic camera push-in, gentle natural movement, realistic lighting...", lines=4)
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="Seed")
                steps = gr.Slider(20, 50, value=35, step=1, label="Качество / шаги")
            generate = gr.Button("Создать видео", variant="primary")
        with gr.Column():
            result = gr.Video(label="Готовое видео", format="mp4")
    generate.click(generate_video, inputs=[image, prompt, seed, steps], outputs=result)
    gr.Markdown("Модель: CogVideoX-5B-I2V. Не загружайте конфиденциальные фотографии.")

if __name__ == "__main__":
    demo.queue().launch()
