import sys
sys.path.append("./")

from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from interface.UI_1 import Initial_interface
from interface.UI_2 import PSO_interface

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui1 = Initial_interface()
    ui1.open_ui2.connect(ui1.on_optimize_button_clicked)
    ui1.show()

    sys.exit(app.exec_())
