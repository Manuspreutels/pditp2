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

name_img = input("Imagen a procesar: ")
img = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
#img = cv2.cvtColor(img_color, cv2.COLOR_RBG2GRAY)

if DEBUG: imshow(img)
# Hacemos sobel sólo horizontalmente, pues resulta más útil para encontrar las letras de la patente.
ddepth = cv2.CV_16S  # Formato salida
sobelx = cv2.Sobel(img, ddepth, 1, 0, ksize=1)
sobelx = cv2.convertScaleAbs(sobelx)

# Umbralamos la imagen
newim = sobelx.copy()
newim[newim > 100] = 255
newim[newim <= 100] = 0
if DEBUG: imshow(newim)

# Aplica una dilatación para agrandar las áreas blancas, de esta forma obtenemos componentes conectadas
kernel_size = 15
kernel = np.ones((kernel_size, kernel_size), np.uint8)
dilated = cv2.dilate(newim, kernel, iterations=1)
if DEBUG: imshow(dilated)

# Con esta apertura, eliminamos elementos no deseados y reducimos el *area negra*, achicando las componentes conectadas
# Luego, se verá cómo esto es útil
B = cv2.getStructuringElement(cv2.MORPH_RECT, (23,23))
abierta = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, B)
if DEBUG:
	imshow(abierta)

num_labels, labels, stats, centroids = \
	cv2.connectedComponentsWithStats(dilated, 8)

# Filtra las componentes utilizando varias métricas que nos permitan predecir que se trata de una patente. 
# Elegimos sólo la componente que cumpla con todos los requisitos que impusimos.
pos_patente = []
for i in range(1, num_labels):
    _, _, ancho, altura, area = stats[i]
    if 2*ancho > 2.5* altura and ancho < 3*altura and altura > kernel_size*1.75 and ancho > kernel_size*5:
        print(stats[i], i)
        if pos_patente == [] or pos_patente[4]/(pos_patente[3]*pos_patente[2]) < area/(ancho*altura):
        	pos_patente = stats[i]

# Obtenemos la subimagen donde está ubicada la patente y la mostramos al usuario.
from math import floor
if pos_patente == []:
	print("Patente no encontrada")
else:
	img_patente = img[(pos_patente[1]+floor(kernel_size/2)):((pos_patente[3]+pos_patente[1])-floor(kernel_size/2)), \
                   (pos_patente[0]+floor(kernel_size/2)):((pos_patente[2]+pos_patente[0])-floor(kernel_size/2))]
	imshow(img_patente, title="Patente")


# Umbralamos la patente
letras = img_patente.copy()
letras[img_patente > 100] = 255
letras[img_patente <= 100] = 0
if DEBUG:
	imshow(letras)

edges = cv2.Sobel(letras, cv2.CV_16S, 0, 1, ksize=1)
edges = cv2.convertScaleAbs(edges)

# Obtenemos las líneas rectas en la imágen. Esto es para poder pintar sobre ellas de negro y que
# no tengamos problemas a la hora de encontrar componentes conectadas porque las letras están pegadas
# al borde de la patente luego de umbralar.
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=45)
for i in range(0, len(lines)):
	if True:
		rho = lines[i][0][0]
		theta = lines[i][0][1]        
		a=np.cos(theta)
		b=np.sin(theta)
		x0=a*rho
		y0=b*rho
		x1=int(x0+100*(-b))
		y1=int(y0+100*(a))
		x2=int(x0-100*(-b))
		y2=int(y0-100*(a))
		_ = cv2.line(letras,(x1,y1),(x2,y2),0,1)


num_labels, labels, stats, centroids = \
	cv2.connectedComponentsWithStats(letras, 4)

pos_letras = []
for i in range(1, num_labels):
	x, y, ancho, altura, area = stats[i]
	if altura > 10 and 4 < ancho < 50 and x !=0:
		pos_letras.append(stats[i])


if len(pos_letras) != 6:
	print("Cantidad incorrecta de caracteres detectada!")
for i in range(len(pos_letras)):
    img_letra = img_patente[pos_letras[i][1]:(pos_letras[i][3]+pos_letras[i][1]), pos_letras[i][0]:(pos_letras[i][2]+pos_letras[i][0])]
    if i == len(pos_letras) - 1: imshow(img_letra, title="Caracter "+str(i+1), blocking=True)
    else: imshow(img_letra, title="Caracter "+str(i+1))

