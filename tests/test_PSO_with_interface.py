import sys
sys.path.append("./")

from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from interface.window_face import PSO_interface

import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PSO_interface()
    win.run()
    win.show()
    sys.exit(app.exec_())
