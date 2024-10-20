import os
import json
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import torch
from contextlib import contextmanager
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.ai.generativelanguage_v1beta.types import SafetyRating, HarmCategory

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

@contextmanager
def temporary_env_var(key: str, new_value):
    old_value = os.environ.get(key)
    if new_value is not None:
        os.environ[key] = new_value
    elif key in os.environ:
        del os.environ[key]
    try:
        yield
    finally:
        if old_value is not None:
            os.environ[key] = old_value
        elif key in os.environ:
            del os.environ[key]

class Gemini_Flash:

    def __init__(self, api_key=None, proxy=None):
        config = get_config()
        self.api_key = api_key or config.get("GEMINI_API_KEY")
        self.proxy = proxy or config.get("PROXY")
        if self.api_key is not None:
            self.configure_genai()

    def configure_genai(self):
        genai.configure(api_key=self.api_key, transport='rest')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Analyze the image and make a txt2img detailed prompt. no prefix!", "multiline": True}),
                "vision": ("BOOLEAN", {"default": True}),
                "api_key": ("STRING", {"default": ""}),
                "proxy": ("STRING", {"default": ""}),
                "sexually_block_level": (([
                        "UNSPECIFIED",
                        "HIGH",
                        "MEDIUM",
                        "LOW",
                        "NEGLIGIBLE",
                ]),)
            },
            "optional": {
                "image": ("IMAGE",),  
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("text", "blocked", "sexually_level")
    FUNCTION = "generate_content"

    CATEGORY = "Gemini flash"

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def generate_content(self, prompt, vision, api_key, proxy, sexually_block_level, image=None):
        print(f"Gemini Flash sexually_block_level: {sexually_block_level}")
        config_updated = False
        if api_key and api_key != self.api_key:
            self.api_key = api_key
            config_updated = True
        if proxy != self.proxy:
            self.proxy = proxy
            config_updated = True
        
        if config_updated:
            save_config({"GEMINI_API_KEY": self.api_key, "PROXY": self.proxy})
            self.configure_genai()

        if not self.api_key:
            raise ValueError("API key is required")

        model_name = 'gemini-1.5-flash-002'
        model = genai.GenerativeModel(model_name)
        filteroutput = False
        sexually_level = "UNSPECIFIED"
        
        if sexually_block_level == "UNSPECIFIED":
            sexually_block_level_type = SafetyRating.HarmProbability.HARM_PROBABILITY_UNSPECIFIED
        elif sexually_block_level == "HIGH":
            sexually_block_level_type = SafetyRating.HarmProbability.HIGH
        elif sexually_block_level == "MEDIUM":
            sexually_block_level_type = SafetyRating.HarmProbability.MEDIUM
        elif sexually_block_level == "LOW":
            sexually_block_level_type = SafetyRating.HarmProbability.LOW
        elif sexually_block_level == "NEGLIGIBLE":
            sexually_block_level_type = SafetyRating.HarmProbability.NEGLIGIBLE
        else:
            sexually_block_level_type = SafetyRating.HarmProbability.HARM_PROBABILITY_UNSPECIFIED

        with temporary_env_var('HTTP_PROXY', self.proxy), temporary_env_var('HTTPS_PROXY', self.proxy):
            try:
                if not vision:
                    # Act like a text LLM
                    response = model.generate_content(prompt)
                    textoutput = response.text
                else:
                    # Vision enabled
                    if image is None:
                        raise ValueError(f"{model_name} needs image")
                    else:
                        pil_image = self.tensor_to_image(image)
                        response = model.generate_content(
                            [prompt, pil_image],
                            safety_settings={
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                            }
                            )
                        if response.prompt_feedback.block_reason:
                            textoutput = f"Error: {response.prompt_feedback}"
                            print(textoutput)
                            filteroutput = True
                        elif response.candidates[0].safety_ratings:
                            for rating in response.candidates[0].safety_ratings:
                                if rating.category == HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
                                    print(f"Sexually level: {rating.probability}")
                                    
                                    if rating.probability == SafetyRating.HarmProbability.HARM_PROBABILITY_UNSPECIFIED:
                                        sexually_level = "UNSPECIFIED"
                                    elif rating.probability == SafetyRating.HarmProbability.HIGH:
                                        sexually_level = "HIGH"
                                    elif rating.probability == SafetyRating.HarmProbability.MEDIUM:
                                        sexually_level = "MEDIUM"
                                    elif rating.probability == SafetyRating.HarmProbability.LOW:
                                        sexually_level = "LOW"
                                    elif rating.probability == SafetyRating.HarmProbability.NEGLIGIBLE:
                                        sexually_level = "NEGLIGIBLE"
                                    else:
                                        sexually_level = "UNSPECIFIED"
                                    
                                    if sexually_block_level_type != SafetyRating.HarmProbability.HARM_PROBABILITY_UNSPECIFIED:
                                        if rating.probability >= sexually_block_level_type:
                                            print(f"Blocked!")
                                            filteroutput = True
                            textoutput = response.text
                        else:
                            textoutput = response.text
            except Exception as e:
                textoutput = f"Error: {str(e)}"
                print(textoutput)
                filteroutput = True
        
        return (textoutput, filteroutput, sexually_level)

NODE_CLASS_MAPPINGS = {
    "Gemini_Flash": Gemini_Flash,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini_Flash": "Gemini flash",
}