import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any, Generator
import openai
from app.core.config import settings
from app.core.logging import logger

class LLMGenerator:
    def __init__(self):
        self.use_openai = False
        self.model = None
        self.tokenizer = None
        
        if settings.OPENAI_BASE_URL and settings.OPENAI_API_KEY:
            self.use_openai = True
            self.client = openai.OpenAI(
                base_url=settings.OPENAI_BASE_URL,
                api_key=settings.OPENAI_API_KEY
            )
            logger.info(f"Using OpenAI compatible API at {settings.OPENAI_BASE_URL}")
        else:
            self._load_local_model()

    def _load_local_model(self):
        logger.info(f"Loading local SFT model: {settings.SFT_MODEL_ID}")
        try:
            quantization_config = None
            try:
                # 尝试 4-bit 量化配置 (Windows 上 bitsandbytes 可能需要特定版本或 WSL)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
            except ImportError:
                logger.warning("bitsandbytes not installed or compatible, falling back to standard loading.")
            except Exception as e:
                logger.warning(f"bitsandbytes config failed: {e}, falling back to standard loading.")
                quantization_config = None

            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.SFT_MODEL_ID, 
                trust_remote_code=True
            )
            
            # 如果量化配置失败，则不使用 quantization_config
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True
            }
            if quantization_config:
                load_kwargs["quantization_config"] = quantization_config
            else:
                # 如果没有量化，且是 CPU，则默认 float32；如果是 GPU，尝试 float16
                if settings.DEVICE != "cpu":
                    load_kwargs["torch_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                settings.SFT_MODEL_ID,
                **load_kwargs
            )
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise e

    def generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        if self.use_openai:
            return self._generate_openai(prompt, stream)
        else:
            return self._generate_local(prompt, stream)

    def _generate_openai(self, prompt: str, stream: bool):
        try:
            response = self.client.chat.completions.create(
                model="default", # 模型名通常不重要，取决于后端
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.MAX_OUTPUT_TOKENS,
                temperature=0.1,
                stream=stream
            )
            if stream:
                def streamer():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return streamer()
            else:
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Error generating response."

    def _generate_local(self, prompt: str, stream: bool):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 简单的非流式实现，流式需要 TextIteratorStreamer
        if stream:
            # 暂未实现本地流式，回退到非流式
            logger.warning("Local streaming not implemented yet, falling back to non-streaming.")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.MAX_OUTPUT_TOKENS,
                do_sample=False, # 确定性输出
                temperature=0.1
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

llm_generator = LLMGenerator()
