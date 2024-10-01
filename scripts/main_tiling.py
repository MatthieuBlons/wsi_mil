from wsi_mil.tile_wsi.tiler import ImageTiler
from wsi_mil.tile_wsi.pca_partial import main as incremental_pca
from wsi_mil.tile_wsi.arguments import get_arguments
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from trackers import timetracker
import copy
import yaml

args, mask_args = get_arguments()
timer = timetracker(name="tracker", verbose=1)
if args.HF_TOKEN:
    os.environ["HF_TOKEN"] = args.HF_TOKEN
it = ImageTiler(args=args)
# Writes the config.yaml file in output directory
dictio = copy.copy(vars(args))
del dictio["device"]
config_str = yaml.dump(dictio)
os.chdir(it.path_outputs)
with open("config.yaml", "w") as config_file:
    config_file.write(config_str)
# Tile
if os.path.isfile(args.path_wsi):
    it._load_slide(args.path_wsi, mask_args)
    it.tile_image()
else:
    dirs = []
    extensions = set([".ndpi", ".svs", ".tif"])
    all_files = glob(os.path.join(args.path_wsi, "*.*"))
    all_wsi = [f for f in all_files if os.path.splitext(f)[1] in extensions]
    assert len(all_wsi) > 0, f"No files with extension {extensions} in {args.path_wsi}"
    print(f"N={len(all_wsi)} images found")
    processed_already = glob(os.path.join(it.path_outputs, "info", "*.csv*"))
    exclude_tag = [
        os.path.basename(os.path.splitext(p)[0])[:-6] for p in processed_already
    ]
    print(f"N={len(exclude_tag)} ALREADY PROCESSED! WILL BE EXCLUDED...")
    files = [f for f in all_wsi if not any([True for t in exclude_tag if t in f])]
    iterfile = tqdm(files, desc="Main tiling", unit="wsi", leave=True)
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
    os.chdir(it.path_outputs)
    os.makedirs("pca", exist_ok=True)
    raw_args = ["--path", "."]
    # Incremental fit with all embeddings as a single batch
    ipca = incremental_pca(raw_args)
    # Apply dimensionality reduction to each embedding and save results in ./mat_pca:
    os.makedirs(os.path.join("./mat_pca"), exist_ok=True)
    files = glob("./mat/*.npy")
    iterfile = tqdm(files, desc="PCA", unit="wsi", leave=True)
    timer.tic()
    for f in iterfile:
        name_wsi, _ = os.path.splitext(os.path.basename(f))
        iterfile.set_postfix_str(f"Transform slide with ID = {name_wsi}", refresh=True)
        m = np.load(f)
        mp = ipca.transform(m)
        np.save(os.path.join("mat_pca", os.path.basename(f)), mp)
    print(f"Done!")
    timer.toc()
