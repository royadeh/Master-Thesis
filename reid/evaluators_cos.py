from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature, extract_cnn_feature_specific, extract_cnn_feature_with_tnorm
from .utils.meters import AverageMeter

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

counter1=1
def visualize_single_camera_features(features, labels, cams_dict, camera_id=5):
    # Convert your features and labels into arrays suitable for t-SNE
    feature_list = [f.cpu().numpy() for f in features.values()]
    label_list = list(labels.values())
    cam_list = list(cams_dict.values())

    # Apply t-SNE transformation
    tsne = TSNE(n_components=2)
    transformed_features = tsne.fit_transform(feature_list)

    # Create a figure
    plt.figure(figsize=(10,10))

    # Add a title to the figure
    plt.title(f'Features for Camera {camera_id}', fontsize=20)

    # Get the unique labels in your label_list
    unique_labels = np.unique(label_list)

    # Create a color for each unique label
    colors = plt.cm.get_cmap('rainbow', len(unique_labels))

    # For each unique label, scatter plot its features in the t-SNE dimensions
    for i, label in enumerate(unique_labels):
        idxs = [idx for idx, (l, c) in enumerate(zip(label_list, cam_list)) if l == label and c == camera_id]  
        label_cam_features = transformed_features[idxs, :]

        # Scatter plot for features of the current label and cam
        plt.scatter(label_cam_features[:, 0], label_cam_features[:, 1], color=colors(i), label=str(label))

    # Add a legend to indicate which points correspond to which label
    plt.legend(title='Label', loc='upper right')

    save_path = f"/home/roya/TSNE_ONECam_SomeIdentities/intra_inter_market_after_training_single_cam{camera_id}.png"

    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()

def updated_visualize_features(features, labels, cams_dict):
    # Convert your features and labels into arrays suitable for t-SNE
    
    feature_list = [f.cpu().numpy() for f in features.values()]
    label_list = list(labels.values())
    cam_list = list(cams_dict.values())

    # Apply t-SNE transformation
    tsne = TSNE(n_components=2)
    transformed_features = tsne.fit_transform(feature_list)

    # Create a figure
    plt.figure(figsize=(10,10))
    
    # Get the unique labels and cams in your data
    unique_labels = np.unique(label_list)
    unique_cams = np.unique(cam_list)

    # Create a marker for each unique cam
    markers = ['o', '^', 's', '*', '+', 'x', 'D', 'h', '1', '2', '3', '4'] # add more markers if you like

    # If there are more unique cams than markers, repeat the marker sequence
    if len(unique_cams) > len(markers):
        markers *= len(unique_cams) // len(markers) + 1

    # Create a color for each unique label
    colors = plt.cm.get_cmap('rainbow', len(unique_labels))

    # For each unique label, scatter plot its features in the t-SNE dimensions
    for i, label in enumerate(unique_labels):
        for j, cam in enumerate(unique_cams):
            idxs = [idx for idx, (l, c) in enumerate(zip(label_list, cam_list)) if l == label and c == cam]  
            label_cam_features = transformed_features[idxs, :]
            
            # Scatter plot for features of the current label and cam
            plt.scatter(label_cam_features[:, 0], label_cam_features[:, 1], color=colors(i), marker=markers[j], label=f'{label}-cam{cam}')

    # Add a legend to indicate which points correspond to which label
    plt.legend(title='Label-Cam', bbox_to_anchor=(1.05, 1), loc='upper left')

    #save_path = f"/home/roya/TSNE_ONECam_SomeIdentities/updated{counter1}.png"

    save_path = f"/home/roya/iidsplots/intra_inter_market_before2_training.png"
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()


def fresh_bn(model, data_loader):
    model.train()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        with torch.no_grad():
            outputs = extract_cnn_feature(model, imgs)
        print('Fresh BN: [{}/{}]\t'.format(i, len(data_loader)))

def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    cams_dict = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, cams) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # print("cam",cams)
        # print("pid",pids)
        # print("fname",fnames)
       
        with torch.no_grad():
            outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid, cam in zip(fnames, outputs, pids, cams):
            features[fname] = output
            labels[fname] = pid
            cams_dict[fname] = cam
            
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    # print("labels",labels)
    # print("****************************")
    # print("cams_dict", cams_dict)
    
    return features, labels, cams_dict




def extract_features_tnorm(model, data_loader, print_freq=1, metric=None, camera_number=1):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, camid) in enumerate(data_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            for i in range(camera_number):
                t = extract_cnn_feature_with_tnorm(model,
                                                   imgs,
                                                   camid,
                                                   i,
                                                   norm=False)
                if i == 0:
                    tmp = t
                else:
                    tmp = tmp + t
            outputs = F.normalize(tmp, p=2, dim=1)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def extract_features_specific(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, camid) in enumerate(data_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            domain_index = camid.cuda()
            outputs = extract_cnn_feature_specific(model, imgs, domain_index)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None, use_cpu=False):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = 1 - torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    if use_cpu:
        dist = 1 - np.matmul(x.cpu().numpy(), y.cpu().numpy().T)
        dist = np.array(dist)
    else:
        dist = 1 - torch.mm(x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), return_mAP=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}'
          .format('market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, 
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    if return_mAP:
        return cmc_scores['market1501'][0], mAP
    else:
        return cmc_scores['market1501'][0]


class Evaluator(object):
    def __init__(self, model, use_cpu=False):
        super(Evaluator, self).__init__()
        self.model = model
        self.use_cpu = use_cpu

    def evaluate(self, data_loader, query, gallery, metric=None, return_mAP=False):
        features, true_labels, cams_dict = extract_features(self.model, data_loader)
        #visualize_single_camera_features(features, true_labels, cams_dict)
        updated_visualize_features(features, true_labels, cams_dict)
        distmat = pairwise_distance(features, query, gallery, metric=metric, use_cpu=self.use_cpu)
        return evaluate_all(distmat, query=query, gallery=gallery, return_mAP=return_mAP)

    def evaluate_specific(self, data_loader, query, gallery, metric=None, return_mAP=False):
        features, _ = extract_features_specific(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric, use_cpu=self.use_cpu)
        return evaluate_all(distmat, query=query, gallery=gallery, return_mAP=return_mAP)

    def evaluate_tnorm(self, data_loader, query, gallery, metric=None, return_mAP=False, camera_number=1):
        features, _ = extract_features_tnorm(self.model, data_loader, camera_number=camera_number)
        distmat = pairwise_distance(features, query, gallery, metric=metric, use_cpu=self.use_cpu)
        return evaluate_all(distmat, query=query, gallery=gallery, return_mAP=return_mAP)

    def fresh_bn(self, data_loader):
        fresh_bn(self.model, data_loader)
