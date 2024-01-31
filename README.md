# kemutai
An abandoned thin wrapper for ComfyUI nodes - designed to allow improved readability of workflows, allow for the use of Python control structures, and allow mid-process artifacts like latents or loaded CLIP models to be held in memory and reused. It hasn't been maintained as Comfy has continued to change - the code was written around the end of August 2023.

## deprecation warning
If this is something you want to work with, you're probably better off with an option like [ComfyScript](https://github.com/Chaoses-Ib/ComfyScript).

## example workflow (included)
Here's a workflow that performs a single first stage denoise and then multiple 'hires fix' passes on an upscaled latent with different seeds.
```python
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
```
An example workflow that accepts this workflow and repaints the faces is also included.
## example output
![default-ls-3842614795-1](https://github.com/curiousjp/kemutai/assets/48515264/5c2409d8-14ed-4f63-9f6c-8b1732d0aa0e)
