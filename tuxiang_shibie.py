# 开发者 haotian
# 开发时间: 2021/9/28 20:31
import  numpy as np
import  struct
import matplotlib.pyplot as plt
binFile = open('Data/train-images.idx3-ubyte','rb')
buf = binFile.read()

nIndex = 0

magic,nImage,nImgRows,nImgCols = struct.unpack_from('>IIII',buf,nIndex)
nIndex += struct.calcsize('>IIII')

# im = struct.unpack_from('>1568B',buf,nIndex)
im = struct.unpack_from('>1568B',buf,nIndex)

im1 = np.reshape(im[0:784],(28,28))
figure = plt.figure()
plotWindow = figure.add_subplot(111)
plt.imshow(im1,cmap = 'gray')

im2 = np.reshape(im[784:1568],(28,28))
plt.imshow(im2,cmap = 'gray')
plt.show()
