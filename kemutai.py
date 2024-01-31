import sys
sys.path.append('/home/curious/sd/ComfyUI/')
import random
import os
import torch
import folder_paths
import comfy.sample
import comfy.utils
import segment_anything
from comfy_extras.chainner_models import model_loading
import comfy.sd
import numpy as np
import ultralytics
from PIL import Image, ImageDraw, ImageFilter

# aha moment
torch.set_grad_enabled(False)

env_var = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', None)
if env_var is None:
    env_var = "backend:cudaMallocAsync"
else:
    env_var += ",backend:cudaMallocAsync"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = env_var

folder_paths.add_model_folder_path('checkpoints', '/home/curious/oogabooga/stable-diffusion-webui/models/Stable-diffusion')
folder_paths.add_model_folder_path('embeddings', '/home/curious/oogabooga/stable-diffusion-webui/embeddings')
folder_paths.add_model_folder_path('vae', '/home/curious/oogabooga/stable-diffusion-webui/models/VAE')
folder_paths.add_model_folder_path('upscale_models', '/home/curious/oogabooga/stable-diffusion-webui/models/RealESRGAN')
folder_paths.add_model_folder_path('loras', '/home/curious/oogabooga/stable-diffusion-webui/models/Lora')
folder_paths.folder_names_and_paths['ultralytics_bbox'] =(['/home/curious/oogabooga/ComfyUI/models/ultralytics/bbox'], folder_paths.supported_pt_extensions)
folder_paths.folder_names_and_paths['sams'] =(['/home/curious/oogabooga/ComfyUI/models/sams'], folder_paths.supported_pt_extensions)

# random
def getSeed():
    seed = random.randrange(4294967294)
    print(f'-- seed requested, chose {seed}')
    return seed

# loading models from disk
def _seekFileFromFolderByPrefix(folder_group, filename):
    candidates = [x for x in folder_paths.get_filename_list(folder_group) if x.startswith(filename)]    
    if len(candidates) == 1:
        filename = candidates[0]
    else:
        raise BaseException(filename)
    return folder_paths.get_full_path(folder_group, filename)

def loadCheckpointSimple(ckpt = 'darkrevpikas_v20.safetensors', load_vae = False):
    ckpt_path = _seekFileFromFolderByPrefix('checkpoints', ckpt)
    model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae = load_vae,
        output_clip = True,
        output_clipvision = False,
        embedding_directory=folder_paths.get_folder_paths('embeddings')
    )
    return (model, clip, vae)

def loadLora(lora_name):
    lora_path = _seekFileFromFolderByPrefix('loras', lora_name)
    lora = comfy.utils.load_torch_file(lora_path, safe_load = True)
    return lora

def loadVAE(ckpt = 'vae-ft-mse-840000-ema-pruned.safetensors'):
    ckpt_path = _seekFileFromFolderByPrefix('vae',ckpt)
    return comfy.sd.VAE(ckpt_path = ckpt_path)

def loadUpscaleModel(ckpt):
    ckpt_path = _seekFileFromFolderByPrefix('upscale_models', ckpt)
    state_dict = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    return model_loading.load_state_dict(state_dict).eval()

def tokenizeAndClipPrompt(prompt, clip):
    if type(prompt) == dict:
        prompt = [f'({k}:{v})' for k, v in prompt.items()]
    if type(prompt) in (list, tuple):
        prompt = ", ".join(prompt)
    tokens = clip.tokenize(prompt)
    # without this, you're going to hit the problem of the new embedding
    # being a leaf variable and colliding with the restriction on inplace
    # operations - https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/2
    conditioning, pooled = clip.encode_from_tokens(tokens, return_pooled = True)
    return [[conditioning, {'pooled_output': pooled}]]

# lora model blending
def blendLora(model, clip, lora, strength_model, strength_clip = None):
    if strength_clip == None:
        strength_clip = strength_model
    if strength_model == 0 and strength_clip == 0:
        return (model, clip)
    modified_model, modified_clip = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
    return (modified_model, modified_clip)

# generic ksampler function
def kSample(model, steps, cfg, con_pos, con_neg, seed, latent, dns = 1.0, sampler = 'euler_ancestral', start_step = None, last_step = None):
    noise = comfy.sample.prepare_noise(latent, seed, None)
    samples = comfy.sample.sample(
        model = model,
        noise = noise,
        steps = steps,
        cfg = cfg,
        sampler_name = sampler,
        scheduler = 'normal',
        positive = con_pos,
        negative = con_neg,
        latent_image = latent,
        denoise = dns,
        start_step = start_step,
        last_step = last_step,
        seed = seed
    )
    return samples

# image and latent utilities
def emptyLatent(width, height):
    return torch.zeros([1,4,height//8,width//8])

def cropToEightMultiple(pixels):
    x_rounded = (pixels.shape[1] // 8) * 8
    y_rounded = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x_rounded or pixels.shape[2] != y_rounded:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x_rounded + x_offset, y_offset:y_rounded + y_offset, :]
    return pixels

def pilToTensorImage(pil_image):
    tensor_image = pil_image.convert('RGB')
    tensor_image = np.array(tensor_image).astype(np.float32)/255.0
    tensor_image = torch.from_numpy(tensor_image)[None,]
    return tensor_image

def tensorImageToPil(tensor_image):
    pil_image = 255.0 * tensor_image.numpy()
    pil_image = np.clip(pil_image, 0, 255).astype(np.uint8)[0]
    pil_image = Image.fromarray(pil_image)
    return pil_image

def savePixels(pixels, fn):
    image_pil = tensorImageToPil(pixels)
    image_pil.save(f'output/{fn}', compress_level = 4)

# image scaling functions
def imageUpscaleWithModel(image, model):
    device = comfy.model_management.get_torch_device()
    model.to(device)
    input_image = image.movedim(-1,-3).to(device)
    tile = 512
    overlap = 32
    out_of_memory = True
    while out_of_memory:
        try:
            scaled = comfy.utils.tiled_scale(
                input_image, 
                lambda a: model(a), 
                tile_x = tile, 
                tile_y = tile, 
                overlap = overlap, 
                upscale_amount = model.scale
            )
            out_of_memory = False
        except comfy.model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e
    model.cpu()
    scaled = torch.clamp(scaled.movedim(-3,-1), min = 0.0, max = 1.0)
    return scaled

def imageScale(pixels, scale):
    samples = pixels.movedim(-1,1)
    width = round(samples.shape[3] * scale)
    height = round(samples.shape[2] * scale)
    samples = comfy.utils.common_upscale(samples, width, height, 'area', 'disabled')
    return samples.movedim(1,-1)

def latentScale(samples, scale):
    nw = round(samples.shape[3] * scale)
    nh = round(samples.shape[2] * scale)
    return comfy.utils.common_upscale(samples, nw, nh, 'area', 'disabled')

 
# ultralytics
def loadUltralyticsBBOX(model_name = 'face_yolov8m.pt'):
    model_path = _seekFileFromFolderByPrefix('ultralytics_bbox', model_name)
    try:
        return ultralytics.YOLO(model_path)
    except ModuleNotFoundError:
        # prime YOLO with the official weights
        # https://github.com/ultralytics/ultralytics/issues/3856
        ultralytics.YOLO('yolov8n.pt')
        return ultralytics.YOLO(model_path)

def loadSAM(model_name = 'sam_vit_b_01ec64.pth'):
    model_path = _seekFileFromFolderByPrefix('sams', model_name)
    if 'vit_h' in model_name:
        model_kind = 'vit_h'
    elif 'vit_l' in model_name:
        model_kind = 'vit_l'
    else:
        model_kind = 'vit_b'
    sam = segment_anything.sam_model_registry[model_kind](checkpoint=model_path)
    device = comfy.model_management.get_torch_device()
    sam.to(device=device)
    return sam

def detectBBOXes(ultra_model, sam_model, pixels, ultra_threshold = 0.5, sam_threshold = 0.93, mask_bloom = None, mask_feather = None, bbox_grow = 0, crop_buffer = 100, drop_size = 5):
    image_pil = tensorImageToPil(pixels)
    ultra_prediction = ultra_model(image_pil, conf=ultra_threshold)
    bboxes = ultra_prediction[0].boxes.xyxy.cpu().numpy()
    sam_predictor = segment_anything.SamPredictor(sam_model)
    sam_predictor.set_image(
        np.clip(255.0 * pixels.cpu().numpy().squeeze(),0,255).astype(np.uint8),
        'RGB'
    )

    masks = []
    cropped_images = []
    cropped_locations = []

    w, h = image_pil.size
    for index, (x0, y0, x1, y1) in enumerate(bboxes):
        if x1 - x0 < drop_size or y1 - y0 < drop_size:
            continue

        bx0 = max(x0 - bbox_grow, 0)
        by0 = max(y0 - bbox_grow, 0)
        bx1 = min(x1 + bbox_grow, w)
        by1 = min(y1 + bbox_grow, h)

        # we are just implementing detection strategy center-1
        dilated_bbox = np.array([[bx0,by0,bx1,by1]])
        center = (bx0+(bx1-bx0)/2, by0+(by1-by0)/2)
        points = np.array([center])
        point_labels = np.array([1])

        sam_masks, sam_scores, _ = sam_predictor.predict(
            point_coords = points,
            point_labels = point_labels,
            box = dilated_bbox
        )

        # get our over-threshold masks, or just the best one if none of them will do
        sam_best_masks = []
        best_index = None
        best_score = -1
        for sam_index, sam_mask in enumerate(sam_masks):
            mask_score = sam_scores[sam_index]
            if mask_score > best_score:
                best_score = mask_score
                best_index = sam_index
            if mask_score > sam_threshold:
                sam_best_masks.append(sam_mask)
        if not sam_best_masks:
            sam_best_masks = [sam_masks[best_index]]
        # still nothing?
        if not sam_best_masks:
            continue
        # merge the masks together
        merged_mask = np.array(sam_best_masks[0])
        for sam_mask in sam_best_masks:
            merged_mask = np.logical_or(merged_mask, np.array(sam_mask))
        mask_pil = (255.0 * merged_mask).astype(np.uint8)
        # greyscale
        mask_pil = Image.fromarray(mask_pil).convert('L')
        if mask_bloom:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius = mask_bloom))
            mask_pil = mask_pil.point(lambda x: 255 if x > 0 else 0)
        if mask_feather:
            mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius = mask_feather))
            # compose the original back on top of it to keep the 'hard centre'
            mask_blurred.paste(mask_pil, (0,0), mask_pil)
            mask_pil = mask_blurred
        
        crop_target_width = (x1 - x0) + crop_buffer
        crop_target_width += (8 - crop_target_width % 8) if (crop_target_width % 8) else 0
        crop_target_width = int(min(w, crop_target_width))

        crop_target_height = (y1 - y0) + crop_buffer
        crop_target_height += (8 - crop_target_height % 8) if (crop_target_height % 8) else 0
        crop_target_height = int(min(h, crop_target_width))
        
        crop_target_x = (x0 + (x1-x0)/2) - crop_target_width / 2
        crop_target_x = int(max(0,crop_target_x))

        crop_target_y = (y0 + (y1-y0)/2) - crop_target_height / 2
        crop_target_y = int(max(0,crop_target_y))

        crop_rectangle = (
            crop_target_x, 
            crop_target_y,
            crop_target_x + crop_target_width, 
            crop_target_y + crop_target_height            
        )
        
        cropped = image_pil.crop(crop_rectangle)
        cropped_torch_image = pilToTensorImage(cropped)
        
        mask_pil = mask_pil.crop(crop_rectangle)
        masks.append(mask_pil)

        cropped_locations.append((crop_target_x, crop_target_y))
        cropped_images.append(cropped_torch_image)

        #cropped.save(f'facecrop-{index}.png')
        #mask_pil.save(f'facemask-{index}.png')

    return (masks, cropped_locations, cropped_images)

def prepareWorkflows(model, clip, clip_skip, vae, lora):
    if model == None or clip == None:
        m_, c_, _ = loadCheckpointSimple()
        if model == None:
            model = m_
        if clip == None:
            clip = c_
    clip.clip_layer(clip_skip)
    if lora:
        for lora_object, lora_strength in lora:
            if type(lora_object) == str:
                lora_object = loadLora(lora_object)
            model, clip = blendLora(model, clip, lora_object, lora_strength)
    if vae == None:
        vae = loadVAE()
    return (model, clip, vae)

def twoStagePixelUpscaleWorkflow(positive, negative, width = 512, height = 512, scale = 1.5, seed = None, dns = 0.7, steps_first = 20, steps_second = 10, lora = [], model = None, clip = None, vae = None, cfg = 7.0, clip_skip = -2, dwell = 0, dwell_step = 0xdeadbeef, latent = None):
    model, clip, vae = prepareWorkflows(model, clip, clip_skip, vae, lora)
    if seed == None:
        seed = getSeed()

    bundled_pos = tokenizeAndClipPrompt(positive, clip)
    bundled_neg = tokenizeAndClipPrompt(negative, clip)

    if latent == None:
        latent = emptyLatent(width, height)

    # first pass sampling
    samples = kSample(model, steps_first, cfg, bundled_pos, bundled_neg, seed, latent)
    pixels = vae.decode(samples)
    # upscale
    upscale_model = loadUpscaleModel('RealESRGAN_x4plus_anime_6B.pth')
    pixels = imageUpscaleWithModel(pixels, upscale_model)
    # bring the image down to our desired scale
    rescale_factor = scale / upscale_model.scale
    pixels = imageScale(pixels, rescale_factor)
    # check image is still divisible by eight
    pixels = cropToEightMultiple(pixels)
    upscaled_latent = vae.encode(pixels)
    returned_latents = []
    returned_pixels = []
    for i in range(dwell + 1):
        samples = kSample(model, steps_second, cfg, bundled_pos, bundled_neg, seed + (i * dwell_step), upscaled_latent, dns = dns )
        pixels = vae.decode(samples)
        returned_latents.append(samples)
        returned_pixels.append(pixels)
    return (returned_pixels, returned_latents, upscaled_latent)

def twoStageLatentUpscaleWorkflow(positive, negative, width = 512, height = 512, scale = 1.5, seed = None, dns = 0.7, steps_first = 20, steps_second = 10, lora = [], model = None, clip = None, vae = None, cfg = 7.0, clip_skip = -2, dwell = 0, dwell_step = 0xdeadbeef, latent = None):
    model, clip, vae = prepareWorkflows(model, clip, clip_skip, vae, lora)
    if seed == None:
        seed = getSeed()

    bundled_pos = tokenizeAndClipPrompt(positive, clip)
    bundled_neg = tokenizeAndClipPrompt(negative, clip)

    if latent == None:
        latent = emptyLatent(width, height)

    # first pass sampling
    samples = kSample(model, steps_first, cfg, bundled_pos, bundled_neg, seed, latent)

    # upscale
    upscaled_latent = latentScale(samples, scale)
    
    returned_latents = []
    returned_pixels = []
    for i in range(dwell + 1):
        samples = kSample(model, steps_second, cfg, bundled_pos, bundled_neg, seed + (i * dwell_step), upscaled_latent, dns = dns )
        pixels = vae.decode(samples)
        returned_latents.append(samples)
        returned_pixels.append(pixels)
    return (returned_pixels, returned_latents, upscaled_latent)

def twoStageUpscaleWithFaces(positive, negative, face_pos = [], face_neg = [], with_latent_upscale = True, steps_face = 20, cfg_face = 7.0, face_dns = 0.3, underlying = twoStageLatentUpscaleWorkflow,ultralytics_model = None, sam_model = None, crop_buffer = 200, mask_bloom = 5, mask_feather = 5, width = 512, height = 512, scale = 1.5, seed = None, dns = 0.7, steps_first = 20, steps_second = 10, lora = [], model = None, clip = None, vae = None, cfg = 7.0, clip_skip = -2, dwell = 0, dwell_step = 0xdeadbeef, latent = None):
    model, clip, vae = prepareWorkflows(model, clip, clip_skip, vae, lora)
    if seed == None:
        seed = getSeed()
    pixels, latents, base_latent = underlying(
        positive,
        negative,
        model = model,
        clip = clip,
        vae = vae,
        width = width, 
        height = height, 
        seed = seed,
        dns = dns, 
        scale = scale,
        steps_first = steps_first, 
        steps_second = steps_second, 
        lora = lora
    )

    if ultralytics_model == None:
        ultralytics_model = loadUltralyticsBBOX()
    if sam_model == None:
        sam_model = loadSAM()
    results = []
    for index, item in enumerate(pixels):
        masks, crop_positions, crop_images = detectBBOXes(ultralytics_model, sam_model, item, crop_buffer = crop_buffer, mask_bloom = mask_bloom, mask_feather = mask_feather)
        if not masks:
            results.append(item)
            continue
        pil_image = tensorImageToPil(item)
        for mask_index, mask_item in enumerate(masks):
            pos_prompt = positive
            if face_pos:
                face_index = mask_index % len(face_pos)
                pos_prompt = face_pos[face_index]
            neg_prompt = negative
            if face_neg:
                face_index = mask_index % len(face_neg)
                neg_prompt = face_neg[face_index]
            bundled_pos = tokenizeAndClipPrompt(pos_prompt, clip)
            bundled_neg = tokenizeAndClipPrompt(neg_prompt, clip)

            face_latent = vae.encode(crop_images[mask_index])
            if with_latent_upscale:
                face_latent = latentScale(face_latent, 2.0)
            face_samples = kSample(model, steps_face, cfg_face, bundled_pos, bundled_neg, seed, face_latent, dns = face_dns)
            face_patch = vae.decode(face_samples)
            if with_latent_upscale:
                face_patch = imageScale(face_patch, 0.5)
            face_patch = tensorImageToPil(face_patch)
            pil_image.paste(face_patch, crop_positions[mask_index], mask_item)
        results.append(pilToTensorImage(pil_image))
    return (pixels, results)

def displayTensorImages(imgs):
    for img in imgs:
        tensorImageToPil(img).show()

# 3:4
# width = 440, 
# height = 592,
# 16:9
# width = 680,
# height = 384, 

if __name__ == '__main__':
    model, clip, _ = loadCheckpointSimple()
    vae = loadVAE()

    argument_package = {
        'positive': '1girl, full body, bubble helmet, (beautiful detailed face, face),  clear helmet, full scene, beautiful, exciting, astronaut, outer space, science fiction, spacesuit, perfect anime picture, perfect lighting, masterpiece, best quality',
        'negative': 'worst quality, low quality, monochrome, bad anatomy, watermark, username, patreon username, patreon logo, text, embedding:easynegative, (embedding:bad-hands-5:0.5)',
        'width': 384,
        'height': 680,
        'scale': 1.5,
        'seed': getSeed(),
        'dns': 0.7,
        'steps_first': 20,
        'steps_second': 10,
        'lora': [],
        'model': model,
        'clip': clip,
        'vae': vae,
        'cfg': 6.5,
        'clip_skip': -2,    
    }
