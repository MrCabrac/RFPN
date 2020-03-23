from modelo import Modelo
import sys
text = sys.argv[1]

model = Modelo()
prediccion = model.predecir(text)
print("{} --> {}".format(text, prediccion))