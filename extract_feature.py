import os
import tqdm
import glob
import torch
import argparse
import numpy as np
import diffusion_feature

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()

    # same settings as the diffusion feature package
    parser.add_argument('--layer', type=str, help="which layer's output to be used as features")
    parser.add_argument('--version', type=str, default='xl', choices=('1-5', '2-1', 'xl', 'pgv2', 'pixart-sigma'), help='Stable Diffusion model version')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--offline_lora', type=str, default=None, help='path for pretrained lora weights')  # lora
    parser.add_argument('--offline_lora_filename', type=str, default=None, help='name of lora file')  # lora
    parser.add_argument('--feature_resize', type=int, default=1, help='resize ratio of width and height')
    parser.add_argument('--control', type=str, nargs='+', default=None, help='type of control information to use')  # controlnet
    parser.add_argument('--attention', type=str, nargs='+', default=None, choices=('down_cross', 'mid_cross', 'up_cross', 'down_self', 'mid_self', 'up_self'))
    parser.add_argument('--img_size', type=int, default=1024)  # 1024 for xl and 512 otherwise recommended
    # extraction settings
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--t', type=int, help='Timesteps to compute features')
    parser.add_argument('--denoising_from', type=int, default=None, help='perform multiple denoising from a given t')  # multiple denoising, used along with controlnet
    parser.add_argument('--use_ddim_inversion', action='store_true')
    # io settings
    parser.add_argument('--input_dir', type=str, default=None, help='Where to load images if using raw images inputs. Leave to default to use build-in datasets')
    parser.add_argument('--nested_input_dir', action='store_true')
    parser.add_argument('--prompt_file', type=str, default='prompt.txt')
    parser.add_argument('--output_dir', type=str, default='./output/')
    parser.add_argument('--aggregate_output', action='store_true')
    parser.add_argument('--use_original_filename', action='store_true')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--sample_name_first', action='store_true')
    # debug purpose
    parser.add_argument('--show_all_layers', action='store_true')

    args = parser.parse_args()

    # make run output folder
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Run folder: {args.output_dir}')

    if args.show_all_layers:
        args.layer = None

    df = diffusion_feature.FeatureExtractor(
        args.layer,
        args.version,
        device='cuda',
        dtype=args.dtype,
        offline_lora=args.offline_lora,
        offline_lora_filename=args.offline_lora_filename,
        feature_resize=args.feature_resize,
        control=args.control,
        attention=args.attention,
        img_size=args.img_size,
    )

    # load images
    imgs = sorted(glob.glob(args.input_dir, recursive=True))
    target_dataset = []
    for img in imgs:
        if not args.nested_input_dir:
            save_name = os.path.splitext(os.path.basename(img))[0]
        else:
            save_name = os.path.join(os.path.basename(os.path.split(img)[0]), os.path.splitext(os.path.basename(img))[0])
        target_dataset.append((img, None, save_name))

    # load and encode prompt
    with open(args.prompt_file, 'r') as f:
        prompts = f.read()
        print('prompt:', prompts)
    prompts = df.encode_prompt(prompts)

    # main loop
    i = 0
    num_images = len(target_dataset)
    pbar = tqdm.tqdm(total=num_images)
    with torch.no_grad():
        while i < num_images:
            sublist = [Image.open(target_dataset[j][0]) for j in range(i, min(i+args.batch_size, num_images))]

            features = df.extract(
                prompts,
                len(sublist),
                sublist,
                t=args.t,
                denoising_from=args.denoising_from,
                use_control=args.control is not None,
                use_ddim_inversion=args.use_ddim_inversion,
            )

            # debug mode: show all possible layers
            if args.show_all_layers:
                for k, v in features.items():
                    print(k, v[0].shape)
                exit()

            # save the results
            if args.aggregate_output:
                resize_target = []
                for k, v in features.items():
                    resize_target.append(v.shape[-1])
                resize_target = np.max(resize_target)

                aggregated_feat = []
                for k, v in features.items():
                    v = torch.nn.functional.interpolate(
                        v, resize_target
                    )
                    aggregated_feat.append(v)
                aggregated_feat = torch.cat(aggregated_feat, dim=1).detach().cpu().numpy()
                # print(aggregated_feat.shape)  # batch_size x dim x h x w

                for j in range(len(sublist)):
                    feat = aggregated_feat[j]
                    name = target_dataset[i+j][2] if args.use_original_filename else args.split+str(i+j)
                    os.makedirs(args.output_dir, exist_ok=True)
                    np.save(os.path.join(args.output_dir, name), feat)

            else:
                for j in range(len(sublist)):
                    for k, v in features.items():
                        feat = v.detach().cpu().numpy()[j]
                        name = target_dataset[i+j][2] if args.use_original_filename else args.split+str(i+j)
                        if not args.sample_name_first:
                            out_path = os.path.join(args.output_dir, k, name)
                            os.makedirs(os.path.join(args.output_dir, k), exist_ok=True)
                        else:
                            out_path = os.path.join(args.output_dir, name, k)
                            os.makedirs(os.path.join(args.output_dir, name), exist_ok=True)
                        np.save(out_path, feat)

            pbar.update(len(sublist))
            i += len(sublist)

if __name__ == '__main__':
    main()
