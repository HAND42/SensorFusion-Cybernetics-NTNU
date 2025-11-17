from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('./WhatsApp Image 2024-09-12 at 20.55.02_7efb7d20.jpg')
model = LatexOCR()
print(model(img))
