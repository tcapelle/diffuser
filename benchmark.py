import wandb
import argparse, random
from PIL import Image
import torch
from contextlib import nullcontext
from types import SimpleNamespace

from diffusers import StableDiffusionPipeline


PROJECT = "stable_diffusions"
JOB_TYPE = "benchmark"
GROUP = "pytorch"


defaults = SimpleNamespace(H=512, W=512, steps=20, scale=7.2, temp=1, seed=42)


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
    parser.add_argument("--n", type=int, default=1, help="number of runs")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    args = parser.parse_args()
    return args

def main(args):


    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
    pipe = pipe.to(args.device)

    generator = torch.Generator("cuda" if args.device=="cuda" else "cpu").manual_seed(args.seed) 
    ## warm up
    with torch.autocast("cuda") if args.mp else nullcontext():
        if args.device=="mps":        
            _ = pipe(args.prompt, num_inference_steps=1).images[0]

        ## actual loop
        results = []
        for _ in range(args.n):
            img = pipe(args.prompt, 
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
    args = parse_args()

    with wandb.init(project=PROJECT, job_type=JOB_TYPE, group=GROUP, config=args):
        main(args)
