from modelo import Modelo

sources = [
    ('data/negativo', 'negativo'),
    ('data/positivo', 'positivo')
]

model = Modelo(sources)
model.entrenar()