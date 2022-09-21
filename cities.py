import wandb
import argparse, random
import torch
from contextlib import nullcontext
from types import SimpleNamespace

from diffusers import StableDiffusionPipeline


PROJECT = "stable_diffusions"
JOB_TYPE = "cities"
GROUP = "pytorch"

defaults = SimpleNamespace(H=512, W=512, steps=20, scale=7.2, temp=1, seed=random.randint(-1e10, 1e10))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H",type=int,default=defaults.H,help="image height, in pixel space")
    parser.add_argument("--W",type=int,default=defaults.W,help="image width, in pixel space")
    parser.add_argument("--scale",type=float,default=defaults.scale,help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--steps",type=int,default=defaults.steps,help="number of ddim sampling steps")
    parser.add_argument("--temp",type=float,default=defaults.temp,help="temperature")
    parser.add_argument("--seed",type=int,default=defaults.seed,help="random seed")
    parser.add_argument("--log", action="store_true", help="log result to wandb")
    parser.add_argument("--n", type=int, default=-1, help="number of artists to do")
    parser.add_argument("--extras",type=str,default=" ",help="to append to prompt")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    args = parser.parse_args()
    return args

def main(args):    
    ## Cities
    
    artists = ("Makoto Shinkai, Joseph Mallord William Turner, Guillermo del Toro,"
               "Salvador Dali, DOOM, Hasui Kawase, Hidetaka Miyazaki, Hans Ruedi Giger,"
               "Diego Rivera, Frida Kahlo, Samy Benmayor, Oswaldo Guayasamín, Roberto Matta, Pedro Lira,"
               "John Martin, Dawid Jurek, Claude Monet, Vincent Van Gogh, Alan Lee, Beksiński").split(",")

    cities = "Santiago in Chile, Valparaiso in Chile".split(",")

    # load model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
    pipe = pipe.to(args.device)

    def run_inference(prompt):
        with torch.autocast("cuda") if args.device == "cuda" else nullcontext():
            img = pipe(prompt, 
                       num_inference_steps=args.steps, 
                       guidance_scale=args.scale,
                       generator=torch.Generator("cuda" if args.device=="cuda" else "cpu").manual_seed(args.seed)).images[0]
        return img
    
    results = []
    for artist in artists[:args.n]:
        for city in cities:
            prompt = f"The city of {city} by {artist} {args.extras}"
            pil_img = run_inference(prompt)
            results.append([prompt, pil_img, artist, city])
            
            
    ## Wandb
    import wandb

    table = wandb.Table(columns=["prompt", "image", "artist", "city"])

    for prompt, pil_img, artist, city in results:
        table.add_data(prompt, wandb.Image(pil_img), artist, city)

    wandb.log({"Results":table})



if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    with wandb.init(project=PROJECT, job_type=JOB_TYPE, group=GROUP, config=args):
        main(args)
