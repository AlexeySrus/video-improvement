import torch
from denoising_pipeline.datasets.separation_dataset_loader import \
    SeriesAndComputingClearDataset
from denoising_pipeline.datasets.pair_dataset_generator import \
    TwoImagesDataset, PairsSeriesDataset


def build_dataset_from_config(config: dict) -> torch.utils.data.Dataset:
    window_size = config['model']['window_size']

    dataset_loader = None
    if config['dataset']['type'] == 'pair':
        dataset_loader = TwoImagesDataset(
            **config['dataset'][config['dataset']['type']],
            window_size=window_size
        )
    elif config['dataset']['type'] == 'series':
        dataset_loader = PairsSeriesDataset(
            folder_path=config[
                'dataset'][config['dataset']['type']
            ]['images_series_folder'],
            window_size=window_size
        )
    elif config['dataset']['type'] == 'sequential':
        dataset_loader = SeriesAndComputingClearDataset(
            **config['dataset'][config['dataset']['type']]['data'],
            window_size=window_size
        )
    elif config['dataset']['type'] == 'separate':
        dataset_loader = SeriesAndComputingClearDataset(
            **config['dataset'][config['dataset']['type']],
            window_size=window_size
        )
    else:
        raise NotImplementedError(
            'Set not supported dataset type: {}'.format(
                config['dataset']['type']
            )
        )

    return dataset_loader
