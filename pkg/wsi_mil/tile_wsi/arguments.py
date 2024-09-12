from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import yaml


def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file. If None, use the command line parser",
        default=None,
    )

    parser.add_argument(
        "--path_wsi",
        type=str,
        default=".",
        help="path of the files to downsample (tiff or svs files)",
    )
    parser.add_argument(
        "--path_mask",
        type=str,
        default="no",
        help="either a path to the xml file, if no, then the whole image is tiled",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="scale to which downsample. I.e a scale of s means dimensions divided by s^2",
    )
    parser.add_argument(
        "--mask_level",
        type=int,
        default=-1,
        help="scale at which has been created the mask. negatives indicate counting levels from the end (slide.level_count)",
    )
    parser.add_argument("--size", type=int, default=256, help="size of patches")
    parser.add_argument(
        "--auto_mask", type=int, default=1, help="if 1, mask is .npy, .xml else"
    )
    parser.add_argument(
        "--tiler",
        type=str,
        default="simple",
        help="type of tiler : imagenet | simple | simclr | moco | uni | conch | gigapath",
    )
    parser.add_argument(
        "--path_outputs", type=str, help="output folder path", default="."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=".",
        help="if using any of the following pretrained embedders :  moco (resnet) | uni | conch |",
    )
    parser.add_argument(
        "--HF_TOKEN",
        type=str,
        default=None,
        help="if using gigapath set the HF_TOKEN environment variable to your Hugging Face API token",
    )
    parser.add_argument("--mask_tolerance", type=float, default=0.75)

    parser.add_argument(
        "--nf",
        action="store_true",
        help="Use this flag when using the nextflow pipeline. Either, dont.",
    )
    parser.add_argument(
        "--max_nb_tiles",
        type=int,
        help="maximum number of tiles to select uniformly. If None, takes all the tiles.",
        default=None,
    )

    args = parser.parse_args()
    # If there is a config file, we populate args with it (still keeping the default arguments)
    if args.config is not None:
        with open(args.config, "r") as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)
    # Set device
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # creat a mask_args object
    mask_args = type(
        "mask_args",
        (object,),
        {
            "auto_mask": args.auto_mask,
            "path_mask": args.path_mask,
            "mask_level": args.mask_level,
            "mask_tolerance": args.mask_tolerance,
        },
    )
    return args, mask_args
