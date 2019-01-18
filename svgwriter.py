import numpy as np
import svgwrite
import matplotlib.pyplot as plt

#data_deltas = np.loadtxt('./data_clean/data_2018-03-31 23-43-00.737811/handwritingfull1504.csv', delimiter=',')
#data_deltas = np.loadtxt('./test.csv', delimiter=',')
data_deltas = np.loadtxt('./matchwithascii.csv', delimiter=',')
data = np.zeros_like(data_deltas)

for i in range(len(data_deltas)):
  data[i, 2] = data_deltas[i, 2]
  if i == 0:
    data[i, 0:2] = data_deltas[i, 0:2]
  else:
    data[i, 0:2] = data[i-1, 0:2] + data_deltas[i, 0:2]

data[:, 0:2] = data[:, 0:2]*10.0
data[:, 1] = -1*data[:, 1]

min_x = np.amin(data[:,0])
max_x = np.amax(data[:,0])

min_y = np.amin(data[:,1])
max_y = np.amax(data[:,1])

dims = (max_x - min_x + 50, max_y - min_y + 50)

dwg = svgwrite.Drawing('write_test.svg', size=dims)
dwg.add(dwg.rect(insert=(0,0), size=dims, fill='white'))

p = "M%s,%s " % (min_x, min_y)

command = "M"
lift_pen = 1

for i in range(len(data)):
  if lift_pen == 1:
    command = "M"
  elif command != "L":
    command = "L"
  else:
    command = ""

  x = float(data[i, 0])
  y = float(data[i, 1])
  lift_pen = data[i, 2]
  p += command + str(x) + "," + str(y) + " "
#print(p)

dwg.add(dwg.path(p).stroke("black", 1).fill("none"))

dwg.save()

breaks = np.squeeze(np.where(data[:,2] == 1))
print('breaks:', breaks)

plt.figure()
for i in range(1, len(breaks)):
  #print('x', data[breaks[i-1]:breaks[i], 0], 'y', data[breaks[i-1]:breaks[i], 1])
  plt.plot(data[breaks[i-1]+1:breaks[i], 0], data[breaks[i-1]+1:breaks[i], 1])
plt.show()
