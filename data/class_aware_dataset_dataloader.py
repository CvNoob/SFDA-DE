import torch.utils.data
from .categorical_dataset import TargetCategoricalDataset, CategoricalSTDataset
from math import ceil as ceil
from PIL import Image
from .single_dataset import SingleDataset


def collate_fn(data):
    # data is a list: index indicates classes
    data_collate = {}
    num_classes = len(data)
    keys = data[0].keys()
    for key in keys:
        if key.find('Label') != -1:
            data_collate[key] = [torch.tensor(data[i][key]) for i in range(num_classes)]
        if key.find('Img') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]
        if key.find('Path') != -1:
            data_collate[key] = [data[i][key] for i in range(num_classes)]

    return data_collate


class ClassAwareDataLoader(object):
    def name(self):
        return 'ClassAwareDataLoader'

    def __init__(self, source_batch_size, target_batch_size,
                 source_dataset_root="", target_paths=[],
                 transform=None, classnames=[],
                 class_set=[], num_selected_classes=0,
                 seed=None, num_workers=0, drop_last=True,
                 sampler='RandomSampler', **kwargs):
        # dataset type
        self.dataset = CategoricalSTDataset()

        # dataset parameters
        self.source_dataset_root = source_dataset_root
        self.target_paths = target_paths
        self.classnames = classnames
        self.class_set = class_set
        self.source_batch_size = source_batch_size
        self.target_batch_size = target_batch_size
        self.seed = seed
        self.transform = transform

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

    def construct(self):
        self.dataset.initialize(source_root=self.source_dataset_root,
                                target_paths=self.target_paths,
                                classnames=self.classnames, class_set=self.class_set,
                                source_batch_size=self.source_batch_size,
                                target_batch_size=self.target_batch_size,
                                seed=self.seed, transform=self.transform,
                                **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      self.num_selected_classes, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_sampler=batch_sampler,
                                                      collate_fn=collate_fn,
                                                      num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        cid = 0
        for c in self.class_set:
            c_len = max([len(self.dataset.data_paths[d][cid]) // \
                         self.dataset.batch_sizes[d][cid] for d in ['source', 'target']])
            dataset_len += c_len
            cid += 1

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len


class TargetClassAwareDataLoader(object):
    def name(self):
        return 'TargetClassAwareDataLoader'

    def __init__(self, target_batch_size,
                 target_paths=[],
                 transform=None, classnames=[],
                 class_set=[], num_selected_classes=0,
                 seed=None, num_workers=0, drop_last=True,
                 sampler='RandomSampler', **kwargs):
        # dataset type
        self.dataset = TargetCategoricalDataset()

        # dataset parameters
        self.target_paths = target_paths
        self.classnames = classnames
        self.class_set = class_set
        self.target_batch_size = target_batch_size
        self.seed = seed
        self.transform = transform

        # loader parameters
        self.num_selected_classes = min(num_selected_classes, len(class_set))
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs

        self.finish_construction = False

    def construct(self):
        self.dataset.initialize(target_paths=self.target_paths,
                                classnames=self.classnames, class_set=self.class_set,
                                target_batch_size=self.target_batch_size,
                                seed=self.seed, transform=self.transform,
                                **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      self.num_selected_classes,
                                                      drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_sampler=batch_sampler,
                                                      collate_fn=collate_fn,
                                                      num_workers=int(self.num_workers))
        self.finish_construction = True

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        cid = 0
        for c in self.class_set:
            c_len = max([len([cid]) //
                         self.dataset.batch_sizes['target'][cid]])
            dataset_len += c_len
            cid += 1

        dataset_len = ceil(1.0 * dataset_len / self.num_selected_classes)
        return dataset_len

    def get_classwise_loader(self):
        if self.finish_construction:
            dataset = SimpleDataset(self.dataset.data_paths['target'],
                                    self.classnames, self.class_set, self.transform)
            loader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=4)
            return loader
        else:
            raise RuntimeError('Categorical dataset has not been constructed.')


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, classnames, class_set, transform=None, **kwargs):
        super(SimpleDataset, self).__init__()
        self.classnames = classnames
        self.class_set = class_set
        self.data_paths = data_paths
        self.transform = transform
        self.samples, self.labels = self.make_data()
        assert len(self.samples) == len(self.labels)

    def make_data(self):
        samples = []
        labels = []
        for cid, c in enumerate(self.class_set):
            sample_paths = self.data_paths[cid]
            samples += sample_paths
            labels += [self.classnames.index(self.class_set[cid])] * len(sample_paths)
        return samples, labels

    def __getitem__(self, index):
        path = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.samples)