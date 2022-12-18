import os
import data.utils as data_utils
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from data.class_aware_dataset_dataloader import ClassAwareDataLoader, TargetClassAwareDataLoader
from config.config import cfg


def prepare_data_Anchor():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    target = cfg.DATASET.TARGET_NAME
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    # for clustering
    batch_size = cfg.CLUSTERING.TARGET_BATCH_SIZE
    dataset_type = cfg.CLUSTERING.TARGET_DATASET_TYPE
    print('Building clustering_%s dataloader...' % target)
    dataloaders['clustering_' + target] = CustomDatasetDataLoader(
                dataset_root=dataroot_T, dataset_type=dataset_type,
                batch_size=batch_size, transform=train_transform,
                train=False, num_workers=cfg.NUM_WORKERS,
                classnames=classes)

    # initialize the categorical dataloader
    dataset_type = 'TargetCategoricalDataset'
    target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical'] = TargetClassAwareDataLoader(
                dataset_type=dataset_type,
                target_batch_size=target_batch_size,
                transform=train_transform,
                classnames=classes,
                num_workers=cfg.NUM_WORKERS,
                drop_last=True, sampler='RandomSampler')

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    test_domain = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    dataroot_test = os.path.join(cfg.DATASET.DATAROOT, test_domain)
    dataloaders['test'] = CustomDatasetDataLoader(
                    dataset_root=dataroot_test, dataset_type=dataset_type,
                    batch_size=batch_size, transform=test_transform,
                    train=False, num_workers=cfg.NUM_WORKERS,
                    classnames=classes)

    return dataloaders
