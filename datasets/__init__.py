from .ml_1m import ML1MDataset


DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    # BeautyDataset.code(): BeautyDataset,
    # VideoDataset.code(): VideoDataset,
    # SportsDataset.code(): SportsDataset,
    # SteamDataset.code(): SteamDataset,
    # XLongDataset.code(): XLongDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
