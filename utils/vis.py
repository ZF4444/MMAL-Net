import numpy as np
import cv2
from config import proposalN

def image_with_boxes(image, coordinates=None, color=None):
    '''
    :param image: image array(CHW) tensor
    :param coordinate: bounding boxs coordinate, coordinates.shape = [proposalN, 4], coordinates[0] = (x0, y0, x1, y1)
    :return:image with bounding box(HWC)
    '''

    if type(image) is not np.ndarray:
        image = image.clone().detach()

        rgbN = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]

        # Anti-normalization
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        image[0] = image[0] * std[0] + mean[0]
        image[1] = image[1] * std[1] + mean[1]
        image[2] = image[2].mul(std[2]) + mean[2]
        image = image.mul(255).byte()

        image = image.data.cpu().numpy()

        image.astype(np.uint8)

        image = np.transpose(image, (1, 2, 0))  # CHW --> HWC
        image = image.copy()

    if coordinates is not None:
        for i, coordinate in enumerate(coordinates):
            if color:
                image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                      (int(coordinate[3]), int(coordinate[2])),
                                      color, 2)
            else:
                if i < proposalN:
                # coordinates(x, y) is reverse in numpy
                    image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])), (int(coordinate[3]), int(coordinate[2])),
                                          rgbN[i], 2)
                else:
                    image = cv2.rectangle(image, (int(coordinate[1]), int(coordinate[0])),
                                          (int(coordinate[3]), int(coordinate[2])),
                                          (255, 255, 255), 2)
    return image
