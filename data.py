biblioteca = {"intents":
            [
                {"tag": "saludos",
                    "patterns": ['hola', 'buenos dias', 'buenas tardes', 'como estas?'],
                    "responses":['hola soy SIC-BOT como puedo ayudarte?'],
                    "context":[""]
                    },

                {"tag": "despedidas",
                        "patterns": ['chao', 'adios', 'hasta luego', 'nos vemos', 'hasta la proxima'],
                        "responses":['hasta luego que tenga un buen dia'],
                        "context":[],
                        },

                {"tag": "agradecimientos",
                    "patterns": ["gracias",
                                "muchas gracias",
                                "mil gracias",
                                "muy amable",
                                "se lo agradezco",
                                "fue de ayuda",
                                "gracias por la ayuda",
                                "muy agradecido",
                                "gracias por su tiempo"
                                ],
                    "responses":["de nada",
                                "feliz por ayudarlo",
                                "gracias a usted",
                                "estamos para servirle",
                                "fue un placer"
                                ],
                    "context":[""]
                    },

                {"tag": "norespuesta",
                    "patterns": [""],
                    "responses":["no se detecto una respuesta"],
                    "context":[""]
                }
               ]
              }