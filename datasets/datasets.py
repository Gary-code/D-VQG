import torch.utils.data


def create_dataset(cfg, split='train'):
    dataset = None
    data_loader = None
    if cfg.data.dataset == 'svqa_dataset':
        from datasets.svqa_dataset import SVQADataset, SVQADataLoader

        dataset = SVQADataset(cfg, split)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    elif cfg.data.dataset == 'msvd_qa_dataset':
        from datasets.msvd_dataset import MSVDQADataLoader, MSVDQADataset
        dataset = MSVDQADataset(cfg, split)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=cfg.data.num_workers)
    else:
        raise Exception('Unknown dataset: %s' % cfg.data.dataset)
    
    return dataset, data_loader
