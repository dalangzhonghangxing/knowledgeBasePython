import numpy as np
from algorithm.Optimizer import NoamOpt
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")

opts = [NoamOpt(32, 1, 100, None),
        NoamOpt(32, 1, 200, None),
        NoamOpt(32, 1, 400, None),
        NoamOpt(32, 1, 800, None),
        NoamOpt(32, 1, 1000, None)]
plt.plot(np.arange(1, 2000), [[opt.rate(i) for opt in opts] for i in range(1, 2000)])
plt.legend(["100", "200", "400", "800", "1000"])
plt.show()
