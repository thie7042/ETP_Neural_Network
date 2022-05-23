import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
#
style.use("ggplot")
# data = pd.read_csv("Masterdata.csv")
data = pd.read_csv("testingdata_verylong.csv")
data = data[["STR_A_Bot_02m","TMP_C_Bot_02m"]]
Strain = data.pop('STR_A_Bot_02m')
Temperature = data.pop("TMP_C_Bot_02m")


###
datatwo = pd.read_csv("Masterdata.csv")
datatwo = datatwo[["STR_A_Bot_02m","TMP_C_Bot_02m"]]
Straintwo = datatwo.pop('STR_A_Bot_02m')
Temperaturetwo = datatwo.pop("TMP_C_Bot_02m")

####


# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(Strain, color="red",linewidth=1)
ax.plot(Straintwo, color="red",linewidth=0.01)
# set x-axis label
ax.set_xlabel("Timestamp",fontsize=10)
# set y-axis label
ax.set_ylabel("Strain",color="red",fontsize=10)
plt.ylim(170,230)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(Temperature,color="blue", linewidth=1)
ax2.plot(Temperaturetwo,color="blue", linewidth=0.01)
ax2.set_ylabel("Temperature",color="blue",fontsize=10)
plt.ylim(22,28)

plt.xlim(0,100000)
plt.show()

plt.scatter(Straintwo, Temperaturetwo, s=1)
plt.show()
