from typing import List
import logging.config


logging.config.fileConfig('./configuration/logging.conf')
logger = logging.getLogger('Pytorch Image Template')

IMG_EXTENSIONS: List[str] = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
