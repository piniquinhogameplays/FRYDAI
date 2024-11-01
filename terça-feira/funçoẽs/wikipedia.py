import wikipedia 

def wiki_search(query):
    wikipedia.sel_lang('pt') 
    try:
        page = wikipedia.page(query)
        return page.summary[:500]

    except wikipedia.excepton.DisambiguationError:
        return f"Senhor, ha varias opcoes para '{query}'. Me diga de um jeito mais especifico"
    except wikipedia.exceptions.PageError:
        return "Desculpe, nao encontrei nada na Wikipedia sobre esse tema. O Senhor deseja que eu procure em outro local." 
        
                