import cv2
import numpy as np
import matplotlib.pyplot as plt

# MODO DEBUG - Activarlo muestra el progreso que se hace con cada parte del programa sobre las imágenes.
DEBUG = False

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
	if new_fig:
		plt.figure()
	if color_img:
		plt.imshow(img)
	else:
		plt.imshow(img, cmap='gray')
	plt.title(title)
	if not ticks:
		plt.xticks([]), plt.yticks([])
	if colorbar:
		plt.colorbar()
	if new_fig:        
		plt.show(block=blocking)

# Para encontrar las monedas, basta con buscar circulos en la imágen
name_img = input("Path de la imagen a procesar: ")
img = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img,5)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=130, minRadius=127, maxRadius=250) # Circulos grandes
circles = np.uint16(np.around(circles))

from math import sqrt
unique = []
for circle in circles[0,:].astype(np.int32):
	close = False
	for i in unique: 
		if sqrt((circle[0]-i[0])**2 + (circle[1]-i[1])**2) < circle[2]:
			close = True
			break
	if not close: unique.append((circle[0],circle[1], circle[2]))
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in unique:
	print(i[0],i[1], i[2])
	_ = cv2.circle(cimg, (i[0],i[1]), i[2], (0,255,0), 2)
# Dibujamos la posición de las monedas
if DEBUG: imshow(cimg, title="Posición de las monedas.")

print("Cantidad de monedas:", len(unique))

# Ahora, pasamos a buscar los dados:
# Esta tarea es más compleja, para lograrlo, debremos encontrar todos los elementos en la imágen,
# y luego, a este conjunto, restarle las monedas.
img = cv2.imread("monedas.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Aplicamos Sobel para encontrar bordes.
ddepth = cv2.CV_16S  # Formato salida
grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=5) 
grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=5) 
abs_grad_x = cv2.convertScaleAbs(grad_x) # abs() + casting uint8
abs_grad_y = cv2.convertScaleAbs(grad_y)
imggrad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

if DEBUG: imshow(imggrad, title="imggrad - Pre-umbralado")

# Sobre el resultado de aplicar Sobel, umbralamos, así obteniendo una imágen binaria
imggrad[imggrad > 240] = 255
imggrad[imggrad <= 240] = 0

if DEBUG: imshow(imggrad, title="imggrad - Umbralada")

# Utilizamos findContours para buscar puntos aislados en la imágen binaria, para así eliminarlos.
imNoiseless = imggrad.copy()
contornos, _ = cv2.findContours(imNoiseless, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
umbral_area = 300
for contorno in contornos:
	area = cv2.contourArea(contorno)
	if area < umbral_area:
		# Rellena el contorno con negro para eliminarlo
		_ = cv2.drawContours(imNoiseless, [contorno], 0, 0, thickness=cv2.FILLED)

# Hacemos apertura para eliminar más.
B = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
imNoiseless = cv2.morphologyEx(imNoiseless, cv2.MORPH_OPEN, B)
if DEBUG: imshow(imNoiseless, title="imNoiseless")

# Hacemos clausura para asegurarnos de que el dado sea reconocido como una única componente conectada
# y no hayan, por imperfecciones en el umbralado + sobel, divisiones en las componentes.
k = 25
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
imClausura = cv2.morphologyEx(imNoiseless, cv2.MORPH_CLOSE, kernel)
if DEBUG: imshow(imClausura, title="imClausura")

connectivity = 8
num_labels, labels, stats, centroids = \
	cv2.connectedComponentsWithStats(imClausura, connectivity, cv2.CV_32S)

labels = np.uint8(255/num_labels*labels)
im_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
# Dibujando las bounding boxes
for st in stats:
	_ = cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=2)
if DEBUG: imshow(img=im_color, color_img=True, title="Componentes conectadas.")

# Finalmente, podemos comenzar a encontrar los dados en la imágen.
dados = []
index = 0
for centroid in centroids.astype(np.int32):
	close = False
	for i in unique: 
		if sqrt((centroid[0]-i[0])**2 + (centroid[1]-i[1])**2) < i[2]:
			close = True
			break
	if not close and stats[index][4] > 4000: dados.append(stats[index])
	index += 1

# Para cada dado obtenemos la subimagen donde se ubican, para luego obtener su valor.
imgs_dados = []
for dado in dados:
	imgs_dados.append(img[dado[1]:(dado[3]+dado[1]), dado[0]:(dado[2]+dado[0])])

# Ontenemos el valor de los dados contando la cantidad de círculos encontrados.
valor_dados = []
for imgdado in imgs_dados:
	#imshow(imgdado)
	imgdado = cv2.medianBlur(imgdado,3)
	circles = cv2.HoughCircles(imgdado,cv2.HOUGH_GRADIENT, 1.2, 20, param1=50, param2=50, minRadius=20, maxRadius=50)
	valor_dados.append(len(circles[0,:]))
	if DEBUG:
		for i in circles[0,:].astype(np.int32):
			_=cv2.circle(imgdado, (i[0],i[1]), i[2], 255, 2)   # draw the outer circle
		imshow(imgdado, title="Dado")
print("El valor de los dados es:", valor_dados)
