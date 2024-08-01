from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
import numpy as np
import PIL
import json
import cv2
import torch
import time
import os
import random
from datetime import datetime
from utils.utils import choose_scheduler, resize_and_canny, CannyDetector, concat_image

class T2I():
    def __init__(self, args):
        self.negative_prompt = args.negative_prompt
        self.guidance_scale = args.guidance_scale
        self.num_inference_steps = args.num_inference_steps
        self.width = args.width
        self.height = args.height
        self.controlnet_conditioning_scale = args.controlnet_conditioning_scale
        self.eta = args.eta
        self.clip_skip = args.clip_skip 
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.image_scale = args.image_scale
        self.preprocessor = CannyDetector(args.low_threshold, args.high_threshold)
        self.keep_loc = args.keep_loc
        self.save_concat = args.save_concat
        self.seed = args.seed
        self.args = args
        
        self.img_pool()
        with open(self.args.config, 'r') as f:
            self.config_pool = json.load(f)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if self.save_concat:
            self.concat_path = os.path.realpath(self.save_path) + '_concat'
            if not os.path.exists(self.concat_path):
                os.mkdir(self.concat_path)
        else:
            self.concat_path = None

    def get_configs(self):
        return random.choice(self.config_pool)
    
    def get_color(self):
        color_list = ['light pea green', 'light yellow', 'light pink', 'light sky blue', "cream", 'silver', 'beige']
        color = random.choice(color_list)
        return color 
    
    def img_pool(self):
        img_list = os.listdir(self.args.data_path)
        self.img_list = []
        for item in img_list:
            _, ext = os.path.splitext(item)
            if ext not in ['.png', '.jpg']:
                continue
            self.img_list.append(item)
        self.img_list = list(map(lambda k: os.path.join(self.args.data_path, k), self.img_list))
        self.img_num = len(self.img_list)
    
    def prepare_model(self, args):
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_path, use_safetensors = True)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(args.base_model_path, controlnet = controlnet, safety_checker = None)
        if args.lora_model_path:
            pipe.load_lora_weights(args.lora_model_path)
            pipe.fuse_lora(lora_scale = args.lora_scale)
            print("load lora finished!!")
        pipe.scheduler = choose_scheduler(args.sampler_name).from_config(pipe.scheduler.config)
        pipe.to("cuda")
        return pipe

    @torch.no_grad()
    def inference(self, args):
        self.pipe = self.prepare_model(args)
        print("start inference:")
        for ite in range(args.iterations):
            batch_images = self.get_batch(self.batch_size) 
            current_config = self.config_pool[ite%len(self.config_pool)]
            if current_config["flag"] == 1:
                current_prompt = current_config["prompt"].format(self.get_color())
            else:
                current_prompt = current_config["prompt"]
            print("current prompt: ", current_prompt)
            init_image, mask_image, edge_image, post_masks = resize_and_canny(batch_images, self.preprocessor, current_config["image_scale"], self.width, self.height, self.keep_loc, matting = current_config["matting"])

            start = time.perf_counter()

            if self.seed:
                generators = [torch.Generator().manual_seed(int(self.seed)) for _ in range(len(edge_image))]
            else:
                generators = None
            inpainted_image = self.pipe(prompt = [current_prompt]*len(edge_image),
                                    negative_prompt = [current_config["negative_prompt"]]*len(edge_image),
                                    clip_skip = self.clip_skip,
                                    image = init_image,
                                    mask_image = mask_image,
                                    control_image = edge_image,
                                    guidance_scale = self.guidance_scale,
                                    num_inference_steps = self.num_inference_steps,
                                    width = self.width,
                                    height = self.height,
                                    eta = self.eta,
                                    generator = generators,
                                    controlnet_conditioning_scale = self.controlnet_conditioning_scale).images
            end = time.perf_counter()   
            batch_time = end - start        
            print("batch time:", batch_time)
            self.post_process(init_image, post_masks, inpainted_image, self.save_path, batch_images, current_prompt)
    
    def get_batch(self, num):
        return random.sample(self.img_list, num)
    
    def post_process(self, init_image, mask_image, inpainted_image, img_save_path, img_path, current_prompt):
        current_time = datetime.now()
        format_time = current_time.strftime("%Y%m%d%H%M%S")
        for idx in range(len(inpainted_image)):
            trans_image_path = img_path[idx]
            name, ext = os.path.splitext(os.path.basename(trans_image_path))
            mask_image_arr = np.array(mask_image[idx].convert("L"))
            mask_image_arr = mask_image_arr[:, :, None]
            mask_image_arr = mask_image_arr.astype(np.float32) / 255.0
            unmasked_unchanged_image_arr = (1 - mask_image_arr) * init_image[idx] + mask_image_arr * inpainted_image[idx]
            unmasked_unchanged_image = PIL.Image.fromarray(unmasked_unchanged_image_arr.round().astype("uint8"))
            output_path = os.path.join(img_save_path, '{}_{}{}'.format(name, format_time, ext))
            unmasked_unchanged_image.save(output_path)
            if self.save_concat:
                cat_image = concat_image(trans_image_path, output_path, self.concat_path )
                self.log_output(img_path[idx], output_path, os.path.join(self.concat_path , os.path.basename(output_path)), current_prompt)
            else:
                self.log_output(img_path[idx], output_path, "", current_prompt)
    
    def log_output(self, ori_path, gene_path, cat_path, prompt):
        with open(os.path.join(self.save_path, 'log.txt'), 'a') as f:
            f.write(ori_path + '\t' + gene_path + '\t' + cat_path + '\t' + prompt)
            f.write('\n')
            f.close()

    

   
