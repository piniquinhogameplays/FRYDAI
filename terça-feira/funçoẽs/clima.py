def clima(cidade="São Paulo"):
    api_key = ""
    ase_url = f""

    try:
        response = request.get(base_url)
        data = response.json()

        if response.status_code == 200 and 'main' in data:
            main = data["main"]
            temperatura = main["temp"]
            descricao = data["Weather"][0]["description"]
            clima_info = f"O clima atual em {cidade} e de {descricao}"
        else: 
            if data.get("message"):
                clima_info = f"ERRO DA API: {data['mensage']}"
            else:
                clima_info        