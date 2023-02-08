import modelo

def chatbotRespuesta(entradaUsuario):
    bolsadepalabras = modelo.bolsadepalabras
    biblioteca = modelo.biblioteca
    
    vectorentrada = modelo.convVector(entradaUsuario, bolsadepalabras)
    listEtiquetas = modelo.gettag(vectorentrada, LIMITE=0)
    respuesta = modelo.getResponse(listEtiquetas, biblioteca)
    return respuesta

entradaUsuario = ''
while entradaUsuario != 'exit':
    entradaUsuario = input('Usser: ')
    respuesta = chatbotRespuesta(entradaUsuario)

    print(f'Bot: {respuesta}')