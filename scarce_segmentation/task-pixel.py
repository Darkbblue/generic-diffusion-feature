import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader

import argparse
from segmentation.utils import setup_seed, multi_acc
from segmentation.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, pixel_classifier
from segmentation.datasets import ImageLabelDataset, FeatureDataset, make_transform, shuffle_split
from segmentation.data_util import get_dataset_setting

# from src.feature_extractors import create_feature_extractor, collect_features
# from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
# from guided_diffusion.guided_diffusion.dist_util import dev


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    return torch.device("cpu")


def prepare_data(args, split):
    print(f"Preparing the train set for {args['category']}...")
    dataset = ImageLabelDataset(
        train_dir=args['training_path'],
        test_dir=args['testing_path'],
        targets=args['split_target'][split],
        resolution=args['image_size'],
        num_images=args['training_number'],
    )
    X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)
    y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)

    for (row, label), to_load in zip(enumerate(tqdm(dataset)), args['split_target'][split]):
        all_features = []
        for sub_feature in args['feature_id']:
            features = torch.from_numpy(
                np.load(os.path.join(args['feature_path'], sub_feature, to_load[0]+str(to_load[1])+'.npy'))
            )  # dim x h x w
            features = F.interpolate(
                features.type(torch.FloatTensor).unsqueeze(0),
                (args['dim'][0], args['dim'][1]), mode='bilinear'
            ).squeeze()
            all_features.append(features)
        X[row] = torch.concatenate(all_features, dim=0)
        
        for target in range(args['number_class']):
            if target == args['ignore_label']: continue
            if 0 < (label == target).sum() < 20:
                print(f'Delete small annotation from image {dataset.image_paths[row]} | label {target}')
                label[label == target] = args['ignore_label']
        y[row] = label  # h x w

    # X: n x dim x h x w
    # y: n x h x w
    
    d = X.shape[1]
    print(f'Total dimension {d}')
    X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    y = y.flatten()
    return X[y != args['ignore_label']], y[y != args['ignore_label']]


def evaluation(args, models):
    dataset = ImageLabelDataset(
        train_dir=args['training_path'],
        test_dir=args['testing_path'],
        targets=args['split_target']['test'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
    )

    preds, gts, uncertainty_scores = [], [], []
    for (i, label), to_load in zip(enumerate(tqdm(dataset)), args['split_target']['test']):
        all_features = []
        for sub_feature in args['feature_id']:
            features = torch.from_numpy(
                np.load(os.path.join(args['feature_path'], sub_feature, to_load[0]+str(to_load[1])+'.npy'))
            )  # dim x h x w
            features = F.interpolate(
                features.type(torch.FloatTensor).unsqueeze(0),
                (args['dim'][0], args['dim'][1]), mode='bilinear'
            ).squeeze()
            all_features.append(features)
        features = torch.concatenate(all_features, dim=0)

        # label: image_size x image_size
        # features: dim x image_size x image_size
        # x: #pixel x dim
        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )
        # pred: image_size x image_size
        gts.append(label.numpy())
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item())
    
    save_predictions(args, dataset.image_paths, preds)
    miou = compute_iou(args, preds, gts)
    print(f'Overall mIoU: ', miou)
    print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
    features, labels = prepare_data(args, split='train')
    train_data = FeatureDataset(features, labels)

    print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    print(f" *********************** Current number data {len(features)} ***********************")

    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):

        gc.collect()
        classifier = pixel_classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])
        classifier.init_weights()

        classifier = nn.DataParallel(classifier).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in range(100):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                y_batch = y_batch.type(torch.long)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)
                acc = multi_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                
                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break

        model_path = os.path.join(args['exp_dir'], 
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add_dict_to_argparser(parser, model_and_diffusion_defaults())

    # parser.add_argument('--exp', type=str)
    # model param
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    # io
    parser.add_argument('--log_path', type=str, help='where to store logs and checkpoints')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--feature_path', type=str, help='where are diffusion features stored')
    parser.add_argument('--feature_id', type=str, nargs='+', help='features to be put in the bucket')
    # task info
    parser.add_argument('--category', type=str, help='which dataset to use')
    parser.add_argument('--task_name', type=str, help='logs and checkpoints will be saved at a corresponding folder')
    parser.add_argument('--feature_len', type=int)
    parser.add_argument('--shuffle_dataset', action='store_true')
    parser.add_argument('--load_split', action='store_true')

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = {}
    opts.update(vars(args))
    opts.update(get_dataset_setting(args.category))
    opts['training_path'] = os.path.join(opts['dataset_path'], 'train')
    opts['testing_path'] = os.path.join(opts['dataset_path'], 'test')
    opts['dim'][-1] = opts['feature_len']
    opts['image_size'] = opts['dim'][0]

    # Prepare the experiment folder 
    # if len(opts['steps']) > 0:
    #     suffix = '_'.join([str(step) for step in opts['steps']])
    #     suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
    #     opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = os.path.join(opts['log_path'], opts['task_name'])
    opts['exp_dir'] = path
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))

    if not opts['load_split']:
        opts['split_target'] = shuffle_split(opts['training_path'], opts['testing_path'], opts['shuffle_dataset'])
        with open(os.path.join(opts['log_path'], opts['task_name'], 'split.json'), 'w') as f:
            f.write(json.dumps(opts['split_target']))
    else:
        with open(os.path.join(opts['log_path'], opts['task_name'], 'split.json'), 'r') as f:
            opts['split_target'] = json.load(f)

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train(opts)
    
    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)
