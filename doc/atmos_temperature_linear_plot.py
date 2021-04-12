import libtide.plot_config as pc
import atmos_consts as ac
import matplotlib.pyplot as plt


fig, ax = pc.setup(ncols=1, nrows=1)
ps = pc.PlotStyles()

ax.plot(ac.temperature, ac.altitude*1e-3, label="Real temp. profile")

R = 287
T0 = ac.temperature[0]
g = 9.81
c_p = 1004
T = T0 - ac.altitude*g/c_p
ax.plot(T, ac.altitude*1e-3, label="Linear temp. profile")

fig.legend()
ax.set_ylabel("Height in $km$")
ax.set_xlabel("Temperature")
ax.set_title("Linear temperature profile")

fig.tight_layout()
filename="output_atmos_temperature_comparison.pdf"

print("Filename: "+filename)
fig.savefig(filename)


