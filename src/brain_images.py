import numpy as np
import matplotlib.pyplot as plt
from utils import IsBlank
plt.switch_backend("QT5Agg")

x = np.load("Data/Validation_x.npy")
y = IsBlank(np.load("Data/Validation_y.npy"))

plt.rc("font", size=24)  # controls default text sizes

plt.subplot(121)
plt.imshow(x[1, ...], cmap="Greys")
plt.title("Tumorous")
plt.subplot(122)
plt.imshow(x[2, ...], cmap="Greys")
plt.title("Non-Tumorous")
plt.subplots_adjust(
    top=0.962,
    bottom=0.042,
    left=0.023,
    right=0.992,
    hspace=0.2,
    wspace=0.045
)
plt.show()
