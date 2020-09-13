import torch
import argparse
from timeit import default_timer as time
from PIL import Image
import yaml
import numpy as np
from denoising_pipeline.model_estimator.model import Model
from denoising_pipeline.architectures.configure_model import \
    build_model_from_config

def parse_args():
    parser = argparse.ArgumentParser(description='Video denoising train script')
    parser.add_argument('--config', required=False, type=str,
                          default='../configuration/example_train_config.yml',
                          help='Path to configuration yml file.'
                        )
    parser.add_argument('--model-weights', required=True, type=str,
                        help='Path to model weights')
    parser.add_argument('--input-image', required=True, type=str,
                        help='Path to image file')
    parser.add_argument('--output-image', required=True, type=str,
                        help='Path to result image file')
    parser.add_argument('--batch-size', required=False, type=int, default=64,
                        help='Inference model batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    window_size = config['model']['window_size']

    series_size = 1 \
        if config['model']['architecture'] != 'sequential' else \
        config['model'][config['model']['architecture']]['series_size']

    denoising_model = build_model_from_config(config)

    model = Model(
        denoising_model,
        device
    )

    print(
        'Count of model trainable parameters: {}'.format(
            model.get_parameters_count()
        )
    )

    model.load(args.model_weights)

    print('Model out shape: {}'.format(
        model.model(*tuple([
            torch.FloatTensor(
                np.zeros((1, 3, window_size, window_size))
            ).to(device)
        ] * series_size)).shape
    ))

    inp_image = np.array(Image.open(args.input_image).convert('RGB'))
    start_time = time()
    out_image = model.predict(
        image=inp_image,
        window_size=window_size,
        batch_size=args.batch_size,
        verbose=True
    )
    finish_time = time()
    Image.fromarray(out_image).save(args.output_image)

    print(
        'Inference time: {:.2f} sec'.format(
            finish_time - start_time
        )
    )

if __name__ == '__main__':
    main()
