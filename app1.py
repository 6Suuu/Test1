import torch
import gradio as gr
from PIL import Image
from accelerate.commands.config.update import description
from safetensors.torch import load_model
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16

    BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
    LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

    USE_4BIT = True if DEVICE == "cuda" else False
    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=TORCH_DTYPE
    )

    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7

    PROMPT_TEMPLATES = {
        "小红书": (
            "你是一个社交媒体专家，请为以下图片生成一条适合小红书的帖子: \n"
            "- 使用2个emoji\n"
            "- 添加3个相关话题标签\n"
            "- 使用口语化的中文\n"
            "- 包含地点或心情描述\n"
            "- 结尾提出一个问题\n"
            "图片内容: {description}"
        ),
        "Instagram": (
            "Generate an engaging Instagram post in English with: \n"
            "- 3 emojis\n"
            "- 4 hashtags\n"
            "- A question to encourage interaction\n"
            "Image content: {description}"
        )
    }

def load_models():
    blip_processor = Blip2Processor.from_pretrained(Config.BLIP2_MODEL)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
       Config.BLIP2_MODEL,
         torch_dtype=Config.TORCH_DTYPE,
         device_map=Config.DEVICE
    )

    llava_processor = AutoProcessor.from_pretrained(Config.LLAVA_MODEL)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        Config.LLAVA_MODEL,
        quantization_config=Config.BNB_CONFIG if Config.USE_4BIT else None,
        torch_dtype=Config.TORCH_DTYPE,
        device_map=Config.DEVICE
    )

    return blip_processor, blip_model, llava_processor, llava_model

def generate_social_post(image_path, platform):
    try:
        image = Image.open(image_path).convert("RGB")

        blip_processor, blip_model, llava_processor, llava_model = load_model()
        description = generate_image_description(blip_processor, blip_model, image)

        prompt_template = Config.PROMPT_TEMPLATES.get(platform, Config.PROMPT_TEMPLATES["小红书"])
        prompt = prompt_template.format(description=description)

        inputs = llava_processor(
            text=prompt,
            image=image,
            return_tensors="pt"
        ).to(llava_model.device)

        outputs = llava_model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE
        )

        return llava_processor.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

    except Exception as e:
        return f"生成失败: {str(e)}"

def create_interface():
    with gr.Blocks(title="智能文案生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 智能社交媒体内容生成器")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="filepath",
                    label="上传图片",
                    height=300
                )
                platform_selector = gr.Dropdown(
                    choices=["小红书", "Instagram"],
                    value="小红书",
                    label="选择平台"
                )
                generate_bnt = gr.Button("生成文案", variant="primary")

            output_text = gr.Textbox(
                label="生成结果",
                placeholder="文案将在这里显示...",
                lines=8,
                interactive=False
            )

        generate_bnt.click(
            fn=generate_social_post,
            inputs=[image_input, platform_selector],
            outputs=output_text
        )

    return demo

if __name__ == "__main__":
    print("正在初始化模型...")
    load_models()

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

