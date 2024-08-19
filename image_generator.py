import torch
from diffusers import FluxPipeline
import os
import datetime
import json
from dotenv import load_dotenv


BASE_PATH = os.path.abspath(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')


class ImageGenerator:
    config: dict
    pipe: FluxPipeline
    
    def __init__(self, config: dict, token: str):
        self.config = config
        self.pipe = FluxPipeline.from_pretrained(config['model'], torch_dtype=torch.bfloat16, token=token)
    
    def generate_images(self, prompt: str, output_format: str = 'png') -> str:
        print(f'Generating images for: "{prompt}"...')

        image = self.pipe(
            prompt,
            self.config['height'],
            self.config['width'],
            self.config['guidance_scale'],
            self.config['num_inference_steps'],
            self.config['max_sequence_length'],
            generator=torch.Generator('cpu').manual_seed(0)
        )

        datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = os.path.join(OUTPUT_PATH, f'{prompt.lower().replace(" ", "-")}_{datetime_str}')

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        for i, img in enumerate(image.images):
            image_filename = f'{i}.{output_format}'
            img.save(os.path.join(output_dir, image_filename))
        
        print('DONE')


def main() -> None:
    with open(os.path.join(BASE_PATH, 'config.json'), 'r') as f:
        config = json.load(f)
    
    generator = ImageGenerator(config, os.getenv('FLUX_TOKEN'))

    print('FLUX.1-DEV IMAGE GENERATOR')
    print('--------------------------')

    prompt = input(' > Prompt: ')

    generator.generate_images(prompt)


if __name__ == '__main__':
    load_dotenv()

    main()