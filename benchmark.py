import wandb
import argparse, sys
from types import SimpleNamespace
import torch
from contextlib import nullcontext
from types import SimpleNamespace

import diffusers
from diffusers import StableDiffusionPipeline


PROJECT = "stable_diffusions"
JOB_TYPE = "benchmark"
GROUP = "pytorch"


defaults = SimpleNamespace(H=512, W=512, steps=20, scale=7.2, temp=1, seed=42)

def grab_setup():
    s = {}
    s["wandb_version"] = wandb.__version__
    s["python_version"] = sys.version.split("|")[0].replace(" ", "")
    s["torch_version"] = torch.__version__
    try:
        import coremltools
        s["coreml_version"] = coremltools.__version__
    except:
        pass
    s["diffusers_version"] = diffusers.__version__
    return s


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The city of Santiago in Chile by Makoto Shinkai")
    parser.add_argument("--H",type=int,default=defaults.H,help="image height, in pixel space")
    parser.add_argument("--W",type=int,default=defaults.W,help="image width, in pixel space")
    parser.add_argument("--scale",type=float,default=defaults.scale,help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--steps",type=int,default=defaults.steps,help="number of ddim sampling steps")
    parser.add_argument("--temp",type=float,default=defaults.temp,help="temperature")
    parser.add_argument("--seed",type=int,default=defaults.seed,help="random seed")
    parser.add_argument("--mp", action="store_true", help="mp")
    parser.add_argument("--log", action="store_true", help="log result to wandb")
    parser.add_argument("--coreml", action="store_true", help="use core ml")
    parser.add_argument("--n", type=int, default=1, help="number of runs")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--tags", type=str, default="", help="tags")
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    args = parser.parse_args()
    return args

def main(args):

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
    if not args.coreml:
        pipe = pipe.to(args.device)
    

    generator = torch.Generator("cuda" if args.device=="cuda" else "cpu").manual_seed(args.seed) 
    ## warm up
    with torch.autocast("cuda") if args.mp else nullcontext():
        if args.coreml:
            from coreml_hack import UNetWrapper
            pipe.safety_checker = lambda images, **kwargs: (images, False)
            pipe.unet = UNetWrapper(pipe.unet)

        results = []
        for _ in range(args.n):
            img = pipe([args.prompt]*args.bs, 
                    num_inference_steps=args.steps, 
                    guidance_scale=args.scale,
                    generator=generator).images[0]

            results.append(img)
    if args.log:
        table = wandb.Table(columns=["prompt", "image"])
        for img in results:
            table.add_data(args.prompt, wandb.Image(img))
        wandb.log({"Inference_results":table})


if __name__ == "__main__":
    args = vars(parse_args())
    args.update(grab_setup())
    args = SimpleNamespace(**args)

    with wandb.init(project=PROJECT, job_type=JOB_TYPE, group=GROUP, config=args):
        main(args)
