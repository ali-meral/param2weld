import matplotlib.pyplot as plt

# Use LaTeX-rendered text with serif fonts
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Computer Modern Roman"

# Font sizes
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["axes.titlesize"] = 30
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30
plt.rcParams["legend.fontsize"] = 30

# Line/marker styling
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 5

# Legend spacing
plt.rcParams["legend.loc"] = "best"
plt.rcParams["legend.columnspacing"] = 1.0
plt.rcParams["legend.handletextpad"] = 0.4
plt.rcParams["legend.handlelength"] = 1

# Custom color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#e41a1c",  # red
        "#ff7f00",  # orange
        "#984ea3",  # violet
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
)

myblue = "#377eb8"
mygreen = "#4daf4a"
myred = "#e41a1c"
myorange = "#ff7f00"