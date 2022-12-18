import torch
import os
from . import utils as solver_utils
from utils.utils import to_cuda, save_config_to_file
import torch.nn.functional as F
from . import clustering
from tqdm import tqdm
from discrepancy.cdd import CDD
from .base_solver import BaseSolver
from torch.utils.tensorboard import SummaryWriter


class AnchorSolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(AnchorSolver, self).__init__(net, dataloader, \
                                                  bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name

        assert ('categorical' in self.train_data)

        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS,
                                                self.opt.CLUSTERING.FEAT_KEY,
                                                self.opt.CLUSTERING.BUDGET)
        self.clustered_target_samples = {}

        if self.opt.TRAIN.LOGGING and (not self.test_only):
            self.tb_writter = SummaryWriter(log_dir=self.opt.SAVE_DIR)
            save_config_to_file(self.opt, self.opt.SAVE_DIR)

        self.build_optimizer()
        self.source_anchors = F.normalize(self.net.module.get_classifier_weights(), dim=1).cpu()

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.cdd = CDD(kernel_num=self.opt.CDD.KERNEL_NUM,
                       kernel_mul=self.opt.CDD.KERNEL_MUL,
                       num_layers=len(self.opt.CDD.ALIGNMENT_FEAT_KEYS),
                       num_classes=self.opt.DATASET.NUM_CLASSES,
                       intra_only=self.opt.CDD.INTRA_ONLY)

    def solve(self):

        while True:
            if self.loop >= self.opt.TRAIN.MAX_LOOP:
                break

            torch.cuda.empty_cache()
            self.net.eval()
            with torch.no_grad():
                print('Clustering based on %s...' % self.target_name)
                self.update_labels()
                self.clustered_target_samples = self.clustering.samples
                target_hypt, filtered_classes = self.filtering()
                self.tb_logging('Clustering/N_filtered_classes', len(filtered_classes))
                self.construct_categorical_dataloader(target_hypt, filtered_classes)
                self.construct_surrogate_feature_sampler(filtered_classes)

            self.update_network(filtered_classes)
            self.loop += 1

        print('Training Done!')

    def update_labels(self):
        init_target_centers = self.net.module.get_classifier_weights()
        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(self.net.module.feature_extractor, target_dataloader)

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples

        chosen_samples = solver_utils.filter_samples(
            target_samples, threshold=threshold
        )
        filtered_classes = solver_utils.filter_class(
            chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES
        )
        print('The number of filtered classes: %d' % len(filtered_classes))
        return chosen_samples, filtered_classes

    def construct_categorical_dataloader(self, samples, filtered_classes):
        target_classwise = solver_utils.split_samples_classwise(
            samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                                   for c in filtered_classes}
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]

        target_sample_labels = samples['Label_target']
        self.selected_classes = [labels[0].item() for labels in target_sample_labels]

        assert (self.selected_classes == [labels[0].item() for labels in samples['Label_target']])
        return target_samples, target_nums, target_sample_labels

    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]

    def calc_loss(self, source_output, target_output, nums_cls):

        feats_toalign_S = self.prepare_feats(source_output)
        feats_toalign_T = self.prepare_feats(target_output)

        cdd_loss = self.cdd.forward(feats_toalign_S, feats_toalign_T,
                                    nums_cls, nums_cls)[self.discrepancy_key]
        cdd_loss *= self.opt.LOSS.CDD_WEIGHT
        self.tb_logging('Train/CDD_Loss', cdd_loss.item())

        return cdd_loss

    def update_network(self, filtered_classes):

        torch.cuda.empty_cache()
        self.net.train()
        self.train_data['categorical']['iterator'] = iter(self.train_data['categorical']['loader'])

        for ii in tqdm(range(self.iters_per_loop), desc=self.opt.EXP_NAME):
            lr = self.update_lr()
            self.optimizer.zero_grad()
            self.tb_logging('Train/LR', lr)

            target_samples_cls, target_nums_cls, target_labels_cls = self.CAS()
            target_cls_concat = torch.cat([to_cuda(samples) for samples in target_samples_cls], dim=0)
            source_features_cls = self.sample_source_features(target_nums_cls,
                                                              target_labels_cls)
            source_features_cls = to_cuda(source_features_cls)
            target_output = self.net(target_cls_concat)
            source_output = self.net.module.classify(source_features_cls)
            loss = self.calc_loss(source_output, target_output, target_nums_cls)

            loss.backward()
            self.optimizer.step()
            self.iters += 1

        with torch.no_grad():
            result = self.test()
        self.tb_logging('Train/TargetAcc(%)', result)
        if result > self.best_result:
            self.best_result = result
            self.save_ckpt()

        print(f'Loop [ {self.loop} | {self.opt.TRAIN.MAX_LOOP} ]')
        print(f' Cur result = {result:.3f} %')
        print(f'Best result = {self.best_result:.3f} %\n')

    def tb_logging(self, tag, value):
        if hasattr(self, 'tb_writter'):
            self.tb_writter.add_scalar(tag, value, global_step=self.iters)

    def save_ckpt(self):
        file_name = os.path.join(self.opt.SAVE_DIR, 'model_best_weight.pth')
        dict2save = {'bn_domain_map': self.bn_domain_map,
                     'state_dict': self.net.module.feature_extractor.state_dict(),
                     'init_model': self.opt.MODEL.PRETRAINED_FEATURE_EXTRACTOR,
                     'result': self.best_result, 'loop': self.loop}
        torch.save(dict2save, file_name)

    def collect_target_feature_mean_std(self, filtered_classes):
        loader = self.train_data['categorical']['loader'].get_classwise_loader()
        feats, plabels = [], []

        for data, plabel in iter(loader):
            feat = self.net(to_cuda(data))[self.opt.CLUSTERING.FEAT_KEY]
            feats += [feat.detach()]
            plabels += [to_cuda(plabel)]

        feats = torch.cat(feats, dim=0)
        plabels = torch.cat(plabels, dim=0)

        assert feats.shape[0] == plabels.shape[0]

        class_mean = torch.zeros(self.opt.DATASET.NUM_CLASSES, feats.shape[-1])
        class_std = torch.zeros(self.opt.DATASET.NUM_CLASSES, feats.shape[-1])
        for c in filtered_classes:
            index = torch.where(plabels == c)[0]
            _std, _mean = torch.std_mean(feats[index], dim=0, unbiased=True)
            class_std[c] = _std
            class_mean[c] = _mean

        del feats
        del plabels
        return class_mean.cpu(), class_std.cpu()

    def construct_surrogate_feature_sampler(self, filtered_classes):
        """
        The time complexity of sampling from a multivariate Gaussian N(mu, cov) is O(n^2).
        Especially when the dimension of feature here is 2048.
        So we only keep those values on diagonal of the covariance matrix,
        in order to make the time complexity of sampling to be near O(n).
        """
        variance_mult = self.opt.TRAIN.VARIANCE_MULT
        print(f'Collecting mean and stddev of target features...variance_mult={variance_mult}')
        target_mean, target_std = self.collect_target_feature_mean_std(filtered_classes)

        normal_sampler = {}

        for i in range(self.opt.DATASET.NUM_CLASSES):
            cur_target_norm = target_mean[i].norm(p=2)
            if cur_target_norm < 1e-3:
                normal_sampler[i] = None
                continue

            estimated_source_mean = self.source_anchors[i] * cur_target_norm
            estimated_source_std = target_std[i]
            # eliminate 0 value in case of ValueError() raised by torch.distribution
            estimated_source_std[estimated_source_std == 0] = 1e-4
            estimated_source_std = estimated_source_std * variance_mult
            normal_sampler[i] = torch.distributions.normal.Normal(
                estimated_source_mean, estimated_source_std)

        self.surrogate_feature_sampler = normal_sampler
        print('surrogate feature samplers constructed.\n')

    def sample_source_features(self, nums_cls, label_cls):
        assert len(nums_cls) == len(label_cls)
        source_features_cls = []

        for num, label in zip(nums_cls, label_cls):
            assert num == label.shape[0]
            cur_label = label[0].item()
            samples = self.surrogate_feature_sampler[cur_label].sample(sample_shape=(num,))
            source_features_cls += [samples]

        return torch.cat(source_features_cls, dim=0)
