# Реализация подхода шумоподваления

### Инструкции по разворачиванию решения

#### Установка программных зависимостей
Выполните в терминале следующую команду:
```shell script
pip3 install -r requirements.txt
```

#### Загрузка данных
По следующей [ссылке](https://yadi.sk/d/hVr5kLqNfMGILA "Yandex Disk") 
можно скачать наборы изображений.
Далее необходимо распоковать архив в следующую директорию: `data/`

Для скачивания архива и его распоковки потребуется около 30 гигабайт
свободного места на диске.

#### Предобработка данных для обучения моделей
Выполните в терминале следующие команды:
```shell script
python3 video_improvement/scripts/preprocess_dataset.py \
  --dataset-path=data/dataset/after_iphone_denoising/ --verbose
```
```shell script
python3 video_improvement/scripts/preprocess_dataset.py \
  --dataset-path=data/dataset/real_sense_noise/ --verbose
```
```shell script
python3 video_improvement/scripts/preprocess_dataset.py \
  --dataset-path=data/dataset/webcam/ --verbose
```

### Обучение моделей

#### Конфигурация программного комплекса обучения

Скопируйте файл конфигурации `video_improvement/configuration/example_train_config.yml`
в рабочую дирикторию `data/`.

Или же вы можете использовать конфигурацию из следующего примера в сериализации YAML:
```yaml
visualization:
  use_visdom: True
  visdom_port: 9000
  visdom_server: 'http://localhost'

  image:
    every: 10
    scale: 2

model:
  window_size: 224

  architecture: 'simple'  # выберите из 'simple' (мобильная архитектура), 
                          # 'sequential' (работает только с типом 
                          #               загрузчика данных series), 
                          # 'advanced' (архитектура с частотной 
                          #             декомпозицией изображения)

  sequential:
    series_size: 5
    n: 5
    residuals: True
    filters_per_image: 32

  advanced:
    separate_filters_count: 1
    union_filters_count: 1

dataset:
  type: 'pair'  # выберите из 'pair', 'series', 'sequential', 
                # 'separate' (требует большого объема оперативной памяти)

  pair:
    image1_path: 'path to first image from pair'
    image2_path: 'path to second image from pair'

  series:
    images_series_folder: 'path to folder with images folder from one series'

  sequential:
    images_series_folder: 'path to folder with images folders from series'

  separate:
    images_series_folder: 'path to noisy images'
    clear_images_path: 'path to clear images'

train:
  optimizer: 'sgd' # выберите из 'adam', 'nadam', 'radam', 'sgd'
  lr: 0.0000001
  weight_decay: 0

  loss: 'mse' // choose from 'mse', 'l1', 'fourier_loss' (experimental loss)

  epochs: 150
  batch_size: 1
  number_of_processes: 12

  distribution_learning:
    enable: False
    devices: ['cuda:0', 'cuda:1']

  save:
    model: 'path to saved weights and logs'
    every: 10

  load_model: False
  load_optimizer: False

```

Далее вам потребуется настроить параметры конфигурации.

#### Запуск обучения моделей
Для обучения шумоподавляющих моделей используется следующий скрипт: 
`python3 video_improvement/train/train.py`.
До запуска скрипта необходимо настроить окружение, 
для этого нужно выполнить следующую команду:
```shell script
export PYTHONPATH=./
```

Описание скрипта для обучения моделей:
```shell script
usage: train.py [-h] [--config CONFIG] [--no-cuda]

Image denoising train script

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Path to configuration yml file.
  --no-cuda        disables CUDA training
```

Если вы хотите использовать визуализацию процесса обучения, запустите сервис 
**visdom** с помоью следующей команды:
```shell script
visdom -port=9000
```

Замечание: Для удобства запускайте скрипты с помощью утилиты **screen**.

### Запуск скриптов для шумоподавления

До запуска скрипта необходимо настроить окружение, 
для этого нужно выполнить следующую команду:
```shell script
export PYTHONPATH=./
```

#### Запуск скрипта для шумоподавления изображения
Для применения шумоподавляющей сети к изображению используется следующий скрипт: 
`video_improvement/test/inference_by_image.py`


Описание скрипта для шумоподавления изображения:
```shell script
usage: inference_by_image.py [-h] [--config CONFIG] 
  \--model-weights MODEL_WEIGHTS 
  \--input-image INPUT_IMAGE 
  \--output-image OUTPUT_IMAGE [--no-cuda]

Video denoising train script

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration yml file.
  --model-weights MODEL_WEIGHTS
                        Path to model_estimator weights
  --input-image INPUT_IMAGE
                        Path to image file
  --output-image OUTPUT_IMAGE
                        Path to result image file
  --no-cuda             disables CUDA training
```

#### Запуск скрипта для шумоподавления видео
Для применения шумоподавляющей сети к видео файлу используется следующий скрипт: 
`video_improvement/test/inference_by_video.py`

Описание скрипта для шумоподавления видео
```shell script
usage: inference_by_video.py [-h] [--config CONFIG] 
  \--model-weights MODEL_WEIGHTS 
  \--input-video INPUT_VIDEO 
  \--output-video OUTPUT_VIDEO [--no-cuda]

Video denoising train script

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration yml file.
  --model-weights MODEL_WEIGHTS
                        Path to model_estimator weights
  --input-video INPUT_VIDEO
                        Path to video file
  --output-video OUTPUT_VIDEO
                        Path to result video file
  --no-cuda             disables CUDA training
```

#### Pretrained models

Предобученные модели мобильной архитектуры и архитектуры с частотной декомпозицией
изображения можно скачать по следующей  
[ссылке](https://yadi.sk/d/NA6Rg5S2JPpFdw "Yandex Disk").