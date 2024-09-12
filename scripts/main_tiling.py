from wsi_mil.tile_wsi.tiler import ImageTiler
from wsi_mil.tile_wsi.pca_partial import main as incremental_pca
from wsi_mil.tile_wsi.arguments import get_arguments
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from pprintpp import pprint
from trackers import timetracker

args, mask_args = get_arguments()
timer = timetracker(name="tracker", verbose=1)
if args.HF_TOKEN:
    os.environ["HF_TOKEN"] = args.HF_TOKEN

outpath = os.path.join(args.path_outputs, args.tiler, f"level_{args.level}")
it = ImageTiler(args=args)

# Tile
if os.path.isfile(args.path_wsi):
    it._load_slide(args.path_wsi, mask_args)
    it.tile_image()
else:
    dirs = []
    extensions = set([".ndpi", ".svs", ".tif"])
    all_files = glob(os.path.join(args.path_wsi, "*.*"))
    files = [f for f in all_files if os.path.splitext(f)[1] in extensions]
    assert len(files) > 0, f"No files with extension {extensions} in {args.path_wsi}"
    iterfile = tqdm(files, desc="Main tiling:", unit="wsi", leave=True)
    timer.tic()
    for f in iterfile:
        name_wsi, _ = os.path.splitext(os.path.basename(f))
        iterfile.set_postfix_str(f"Processing slide with ID = {name_wsi}", refresh=True)
        it._load_slide(f, mask_args)
        it.tile_image()
    print(f"Done!")
    timer.toc()

# PCA
if args.tiler != "simple":
    os.chdir(outpath)
    os.makedirs("pca", exist_ok=True)
    raw_args = ["--path", "."]
    ipca = incremental_pca(raw_args)
    os.makedirs(os.path.join("./mat_pca"), exist_ok=True)
    files = glob("./mat/*.npy")
    iterfile = tqdm(files, desc="PCA:", unit="wsi", leave=True)
    timer.tic()
    for f in iterfile:
        name_wsi, _ = os.path.splitext(os.path.basename(f))
        iterfile.set_postfix_str(f"Processing slide with ID = {name_wsi}", refresh=True)
        m = np.load(f)
        mp = ipca.transform(m)
        np.save(os.path.join("mat_pca", os.path.basename(f)), mp)
    print(f"Done!")
    timer.toc()
