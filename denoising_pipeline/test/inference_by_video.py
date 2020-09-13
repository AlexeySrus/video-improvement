import torch
import argparse
import cv2
import tqdm
import yaml
import numpy as np
from denoising_pipeline.model_estimator.model import Model
from denoising_pipeline.architectures.configure_model import \
    build_model_from_config
from denoising_pipeline.datasets.series_dataset_generator import \
    VideoFramesGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Video denoising train script')
    parser.add_argument('--config', required=False, type=str,
                          default='../configuration/example_train_config.yml',
                          help='Path to configuration yml file.'
                        )
    parser.add_argument('--model-weights', required=True, type=str,
                        help='Path to model weights')
    parser.add_argument('--input-video', required=True, type=str,
                        help='Path to video file')
    parser.add_argument('--output-video', required=True, type=str,
                        help='Path to result video file')
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

    video_source = VideoFramesGenerator(args.input_video)

    fps = video_source.get_fps()
    frame_height, frame_width = video_source.get_resolution()

    out_video = cv2.VideoWriter(
        args.output_video,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        fps,
        (frame_width, frame_height)
    )

    for i in tqdm.tqdm(range(len(video_source))):
        frame = video_source.get_next_frame()['frame']
        predicted_frame = model.predict(frame, window_size=window_size)
        predicted_frame = cv2.cvtColor(predicted_frame, cv2.COLOR_RGB2BGR)
        out_video.write(predicted_frame)

    out_video.release()


if __name__ == '__main__':
    main()
