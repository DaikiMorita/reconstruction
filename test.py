import numpy as np
import LINEUIManager
from PIL import Image
import tqdm


def re_construction(w, b, sigma, h):
    return sigma * np.random.randn(h.shape[0], w.shape[1]) + b + np.dot(h, w)


if __name__ == '__main__':
    w = np.load("W.npy")
    sigma = np.load("sigma.npy")
    c = np.load("C.npy")
    b = np.load("B.npy")
    h = np.random.randint(0, 2, w.shape[0])

    MAX = 3000
    x = re_construction(w, b, sigma, h)
    for _ in tqdm.tqdm(range(MAX)):
        x += re_construction(w, b, sigma, h)

    x = x / MAX

    line_ui_manager = LINEUIManager.LINEUIManager()

    for index, img in enumerate(np.array(x)):
        img = img.reshape(28, 28)
        arr_min = np.min(img)
        img = img + np.abs(arr_min)

        arr_max = np.max(img)

        img = img / arr_max * 255

        Image.fromarray(np.uint8(img)).save("re_%d.jpg" % index)

        line_ui_manager.send_line('re_%d.jpg' % index, "re_%d.jpg" % index)
