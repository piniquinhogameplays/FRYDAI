import time 
import wikipedia
import pywhatkit
import requests 
import psutil 
from pytube import YouTube 
import speech_recognition as sr 
import random 
import socket 
import pyttsx3 
import pyautogui
import math 
import sys 
import cv2 
import os 
import pygetwindow as gw
import datetime 
import threading
from scapy.all import sniff
import yfinance as yf 
import json 
import webbrowser
import psutil
import oandapyV20.endpoints.orders as orders
import pandas as pd 
import subprocess
import flask 
import numpy as np
from scapy.all import * 
from apscheduler.schedulers.blocking import BlockingScheduler
from oandapyV20 import API
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # Adicionando GPT-2
from tqdm import tqdm
from urllib.parse import quote_plus
from oandapyV20.contrib.requests import MarketOrderRequest
from oanda_candles import Pair, Gran, CandleClient
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails

# Inicialização do sistema de voz 
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) # Possivelmente de uma voz masculina

def speak(text):
    engine.say(text)
    engine.runAndWait()
    return speak

def status_sistemas():
    # Obtém a informação da bateria
    battery = psutil.sensors_battery()
    battery_percent = battery.percent

    # Verifica se está carregando ou descarregando
    if battery.power_plugged:
        battery_status = "Carregando"
    else:
        battery_status = "Descarregando"

    # Obtém as janelas abertas
    windows = gw.getWindowsWithTitle("")
    open_windows = [win.title for win in windows]

    # Formata as janelas abertas em uma string
    open_windows_text = ', '.join(open_windows)

    # Retorna o status dos sistemas
    return (f'Todos os sistemas estão funcionando. '
            f'A energia está em {battery_percent}% e o status dela é {battery_status}. '
            f'As janelas abertas são: {open_windows_text}.')

def saudacao():
    return "ola senhor"

# Inicialização de captura de video com resolução definida 
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Identificador de faces 
cascPath = 'haarcascades/haarcascade_frontalface_default.xml'
cascPathOlho = 'haarcascades/haarcascade_eye.xml'

if not os.path.isfile(cascPath) or not os.path.isfile(cascPathOlho):
    print("Arquivos de Haarcascade não encontrados. Verifique o caminho.")
    exit()

# Carregar os classificadores
facePath = cv2.CascadeClassifier(cascPath)
facePathOlho = cv2.CascadeClassifier(cascPathOlho)

if facePath.empty() or facePathOlho.empty():
    print("Não foi possível carregar os classificadores Haarcascade.")
    exit()
'''
def treinamento_faces():
    # Treinamento de faces 
    eigenface = cv2.face.EigenFaceRecognizer_create()
    fisherface = cv2.face.FisherFaceRecognizer_create()
    lbph = cv2.face.LBPHFaceRecognizer_create()

    def getImageWithId():
        
        Percorrer diretório de fotos, ler todas as imagens com o conjunto de faces com seus respectivos ids
        
        pathsImages = [os.path.join('fotos', f) for f in os.listdir('fotos')]
        faces = []
        ids = []

        for pathImage in pathsImages:
            imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
            id = int(os.path.split(pathImage)[-1].split('.')[1])

            ids.append(id)
            faces.append(imageFace)

            cv2.imshow('Face', imageFace)
            cv2.waitKey(10)
        return np.array(ids), faces

    ids, faces = getImageWithId()

    # Gerar classifier do treinamento 
    print('treinamento...')
    eigenface.train(faces, ids)
    eigenface.write('classifier/classificadorEigen.yml')

    # fisherface.train(faces, ids)
    # fisherface.write('classifier/classificadorFisher.yml')

    lbph.train(faces, ids)
    lbph.write('classifier/classificadorLBPH.yml')
    print('Treinamento concluído com sucesso!')

def capturar_faces():
    # Caminho Haarcascade
    cascPath = 'cascade/haarcascade_frontalface_default.xml'
    cascPathOlho = 'cascade/haarcascade-eye.xml'

    # Classificadores baseados nos Haarcascade
    facePath = cv2.CascadeClassifier(cascPath)
    facePathOlho = cv2.CascadeClassifier(cascPathOlho)
    video_capture = cv2.VideoCapture(0)

    increment = 1
    numMostras = 500
    id = input('Digite seu identificador: ')
    width, height = 220, 220
    print('Capturando as faces...')

    # Criar diretório para salvar as imagens
    os.makedirs('fotos', exist_ok=True)

    # Variável para contar o número de amostras capturadas
    amostras_capturadas = 0

    while True:
        conect, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Qualidade da luz sobre a imagem capturada
        print(np.average(gray))

        # Realizar a detecção de faces
        face_detect = facePath.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in face_detect:
            # Desenhar retângulo na face detectada
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Realizar a detecção dos olhos da face
            region = image[y:y + h, x:x + w]
            imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            face_detect_olho = facePathOlho.detectMultiScale(imageOlhoGray)

            for (ox, oy, ow, oh) in face_detect_olho:
                # Desenhar retângulo nos olhos detectados
                cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)

                # Salvando imagem com respectivo ID para treinamento
                if np.average(gray) > 110:
                    face_off = cv2.resize(gray[y:y + h, x:x + w], (width, height))
                    cv2.imwrite(f'fotos/pessoa.{id}.{increment}.jpg', face_off)

                    print('[Foto', increment, 'capturada com sucesso] -', np.average(gray))
                    increment += 1
                    amostras_capturadas += 1

                    # Se atingir o número máximo de amostras, sair do loop
                    if amostras_capturadas >= numMostras:
                        break

        cv2.imshow('Face', image)

        key = cv2.waitKey(1)
        if key & 0xFF == 27 or amostras_capturadas >= numMostras:
            time.sleep(1)
            break

    print('Fotos capturadas com sucesso :)')
    video_capture.release()
    cv2.destroyAllWindows()

def reconhecer_faces():
    # Caminho haarcascade
    detectorFace = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

    # Instanciado Eigen Faces Recognizer
    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.read("classifier/classificadorEigen.yml")

    # Configurações
    height, width = 220, 220
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    camera = cv2.VideoCapture(0)
    detected_count_known = 0
    detected_count_unknown = 0

    # Inicializa a engine de texto para fala
    engine = pyttsx3.init()

    # Configura a voz e a velocidade de fala (opcional)
    voices = engine.getProperty('voices')
    google_voice_id = None
    for voice in voices:
        if "google" in voice.name.lower():
            google_voice_id = voice.id
            break

    # Se não encontrar a voz do Google, usa a primeira voz disponível
    if google_voice_id is None:
        google_voice_id = voices[0].id

    engine.setProperty('voice', google_voice_id)  # Configura a voz para a voz do Google (ou a primeira voz disponível)
    engine.setProperty('rate', 250)  # Configura a velocidade de fala (padrão é 200)

    while True:
        conectado, imagem = camera.read()
        imageGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # Deteccao da face baseado no haarcascade
        faceDetect = detectorFace.detectMultiScale(
            imageGray,
            scaleFactor=1.5,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, h, w) in faceDetect:
            # Desenhando retangulo da face
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)
            image = cv2.resize(imageGray[y:y+h, x:x+w], (width, height))

            # Fazendo comparacao da imagem detectada
            id, confianca = reconhecedor.predict(image)

            if id == 1:
                detected_count_known += 1
                name = 'kauan'
            else:
                detected_count_unknown += 1
                desconhecido = 'Desconhecido'

            # Escrevendo texto no frame
            cv2.putText(imagem, name, (x, y + (h + 24)), font, 1, (0, 255, 0))
            cv2.putText(imagem, str(confianca), (x, y + (h + 43)), font, 1, (0, 0, 255))

            # Se atingir o número máximo de detecções, sair do loop
            if detected_count_known + detected_count_unknown >= 100:
                break

        # Mostrando frame
        cv2.imshow("Face", imagem)
        
        # Verifica se a tecla 'ESC' foi pressionada para fechar a câmera
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

    # Fala a quantidade de vezes que o rosto foi identificado
    engine.say(f"O rosto do {name} foi detectado {detected_count_known} vezes.")
    engine.runAndWait()

    print(f"O rosto do {name} foi detectado {detected_count_known} vezes.")
    print(f"O rosto de um {desconhecido} foi detectado {detected_count_unknown} vezes.")
'''


def reconhecimento_facial_completo():
    # Caminhos e configurações iniciais
    cascPath = 'cascade/haarcascade_frontalface_default.xml'
    cascPathOlho = 'cascade/haarcascade-eye.xml'
    detectorFace = cv2.CascadeClassifier(cascPath)
    detectorFaceOlho = cv2.CascadeClassifier(cascPathOlho)
    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.read("classifier/classificadorEigen.yml")
    
    # Configurações para a captura de fotos
    video_capture = cv2.VideoCapture(0)
    increment = 1
    numMostras = 500
    id = input('Digite seu identificador: ')
    width, height = 220, 220
    print('Capturando as faces...')
    
    # Criar diretório para salvar as imagens
    os.makedirs('fotos', exist_ok=True)
    amostras_capturadas = 0
    
    # Configurações de reconhecimento
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    detected_count_known = 0
    detected_count_unknown = 0
    
    # Inicializa engine de texto para fala
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    google_voice_id = None
    for voice in voices:
        if "google" in voice.name.lower():
            google_voice_id = voice.id
            break
    if google_voice_id is None:
        google_voice_id = voices[0].id
    engine.setProperty('voice', google_voice_id)
    engine.setProperty('rate', 250)

    while True:
        conect, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Qualidade da luz sobre a imagem capturada
        print(np.average(gray))

        # Realizar a detecção de faces
        face_detect = detectorFace.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in face_detect:
            # Desenhar retângulo na face detectada
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Realizar a detecção dos olhos
            region = image[y:y + h, x:x + w]
            imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            face_detect_olho = detectorFaceOlho.detectMultiScale(imageOlhoGray)

            for (ox, oy, ow, oh) in face_detect_olho:
                cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)

                # Salvando imagem se a qualidade de luz for boa
                if np.average(gray) > 110:
                    face_off = cv2.resize(gray[y:y + h, x:x + w], (width, height))
                    cv2.imwrite(f'fotos/pessoa.{id}.{increment}.jpg', face_off)
                    print(f'[Foto {increment} capturada com sucesso] - {np.average(gray)}')
                    increment += 1
                    amostras_capturadas += 1

                    if amostras_capturadas >= numMostras:
                        break

            # Reconhecimento facial usando EigenFaces
            image = cv2.resize(gray[y:y+h, x:x+w], (width, height))
            id_pred, confianca = reconhecedor.predict(image)

            if id_pred == 1:
                detected_count_known += 1
                name = 'kauan'
            else:
                detected_count_unknown += 1
                name = 'Desconhecido'

            # Escrever informações no frame
            cv2.putText(image, name, (x, y + (h + 24)), font, 1, (0, 255, 0))
            cv2.putText(image, str(confianca), (x, y + (h + 43)), font, 1, (0, 0, 255))

        # Exibir a imagem capturada com o retângulo da face
        cv2.imshow('Face', image)

        # Fechar o loop ao pressionar a tecla ESC ou atingir o número máximo de amostras
        key = cv2.waitKey(1)
        if key & 0xFF == 27 or amostras_capturadas >= numMostras:
            break

    # Liberar recursos e destruir janelas abertas
    video_capture.release()
    cv2.destroyAllWindows()

    # Texto para fala - relatar a quantidade de detecções
    engine.say(f"O rosto do {name} foi detectado {detected_count_known} vezes.")
    engine.runAndWait()

    print(f"O rosto do {name} foi detectado {detected_count_known} vezes.")
    print(f"O rosto de um {name} foi detectado {detected_count_unknown} vezes.")

def wiki_search(query):
    """Busca informações na Wikipedia e retorna o resumo."""
    wikipedia.set_lang('pt')  # Configura o idioma para português
    try:
        page = wikipedia.page(query)
        return page.summary[:500]  # Limita o resumo a 500 caracteres
    except wikipedia.exceptions.DisambiguationError:
        return f"Desculpe, há várias opções para '{query}'. Tente ser mais específico."
    except wikipedia.exceptions.PageError:
        return "Desculpe, não encontrei nada na Wikipédia sobre isso."

# PortScan 
def portExplorer(ip):
    open_ports = []
    for port in range(1, 1024):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.setdefaulttimeout(1)
        result = sock.connect_ex((ip, port))
        if result == 0:
            open_ports.append(port)
        sock.close()
    return open_ports        

def localizacao_ip():
    def IPGeolocator(ip):
        resppnse = requests.get(f'')
        data = response.json()
        return data 
    ip_data = IPGeolocator(input('digite o ip:'))
    print(ip_data)    

# Trafico network 

def monitor_traffic(interface):
    # Função que será chamada sempre que um pacote for capturado
    def packet_callback(packet):
        if packet.haslayer('IP'):
            src_ip = packet['IP'].src
            dst_ip = packet['IP'].dst
            proto = packet['IP'].proto

            # Exibe informações básicas sobre o pacote
            print(f"[+] Pacote Capturado: {src_ip} -> {dst_ip} | Protocolo: {proto}")

    print(f"Monitorando tráfego na interface {interface}... Pressione Ctrl+C para parar.")
    # Captura os pacotes na interface especificada
    sniff(iface=interface, prn=packet_callback, store=False)

    # Exemplo de uso da função
    if __name__ == "__main__":
        interface = input("Digite a interface de rede que deseja monitorar (ex: wlan0, eth0): ")
        monitor_traffic(interface)  monitor_traffic(interface)

def scan_website(url):
    # Função para testar SQL Injection
    def test_sql_injection(url):
        payload = "' OR '1'='1"
        test_url = f"{url}?id={payload}"
        
        response = requests.get(test_url)
        if "SQL" in response.text or "syntax error" in response.text:
            print(f"[!] Possível vulnerabilidade de SQL Injection encontrada em {url}")
        else:
            print(f"[-] Nenhuma vulnerabilidade de SQL Injection detectada em {url}")

    # Função para testar XSS
    def test_xss(url):
        payload = "<script>alert('XSS')</script>"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if payload in str(soup):
            print(f"[!] Possível vulnerabilidade de XSS encontrada em {url}")
        else:
            print(f"[-] Nenhuma vulnerabilidade de XSS detectada em {url}")

    # Início do scan
    print(f"Analisando {url} para vulnerabilidades...")
    
    # Testar vulnerabilidade de SQL Injection
    test_sql_injection(url)

    # Testar vulnerabilidade de XSS
    test_xss(url)
    
    if __name__ == "__main__":
        site_url = input("Digite a URL do site que deseja testar: ")
        scan_website(site_url)

# Ataque de wifi 
def attack_wifi():
    def deauth_attack(target_mac, gateway_mac, interface):
        pkt = RadioTap()/Dot11(addr1=target_mac, addr2=gateway_mac, addr3=gateway_mac)/Do11Death()
        sendp(pkt, iface=interface, count=100, inter=.1)

    #exemplo de uso 
    #deauth_attack("FF:FF:FF:FF:FF:FF", "AA:BB:CC:DD:EE:FF", "wlan0mon")




# Trader
# dataF = yf.download("EURUSD=X", start="2024-10-7", end="2022-12-5", interval='3m')
# dataF.iloc[:, :]

# def wiki_search(query):
#     """Busca informações na Wikipedia e retorna o resumo."""
#     wikipedia.set_lang('pt')  # Configura o idioma para português
#     try:
#         page = wikipedia.page(query)
#         return page.summary[:500]  # Limita o resumo a 500 caracteres
#     except wikipedia.exceptions.DisambiguationError:
#         return f"Desculpe, há várias opções para '{query}'. Tente ser mais específico."
#     except wikipedia.exceptions.PageError:
#         return "Desculpe, não encontrei nada na Wikipédia sobre isso."

# def signal_generator(df):
#     open = df.Open.iloc[-1]
#     close = df.Close.iloc[-1]
#     previous_open = df.Open.iloc[-2]
#     previous_close = df.Close.iloc[-2]

#     # Bearish Pattern
#     if (open>close and 
#     previous_open<previous_close and 
#     close<previous_open and
#     open>=previous_close):
#         return 1

#     # Bullish Pattern
#     elif (open<close and 
#         previous_open>previous_close and 
#         close>previous_open and
#         open<=previous_close):
#         return 2

#     # No clear pattern
#     else:
#         return 0

# signal = []
# signal.append(0)
# for i in range(1,len(dataF)):
#     df = dataF[i-1:i+1]
#     signal.append(signal_generator(df))

# dataF["signal"] = signal
# dataF.signal.value_counts()

# def get_candles(n):
#     client = CandleClient(access_token, real=False)
#     collector = client.get_collector(Pair.EUR_USD, Gran.M15)
#     candles = collector.grab(n)
#     return candles

# candles = get_candles(3)
# for candle in candles:
#     print(float(str(candle.bid.o)) > 1)

# def trading_job():
#     candles = get_candles(3)
#     dfstream = pd.DataFrame(columns=['Open','Close','High','Low'])

#     i = 0
#     for candle in candles:
#         dfstream.loc[i, ['Open']] = float(str(candle.bid.o))
#         dfstream.loc[i, ['Close']] = float(str(candle.bid.c))
#         dfstream.loc[i, ['High']] = float(str(candle.bid.h))
#         dfstream.loc[i, ['Low']] = float(str(candle.bid.l))
#         i += 1

#     dfstream['Open'] = dfstream['Open'].astype(float)
#     dfstream['Close'] = dfstream['Close'].astype(float)
#     dfstream['High'] = dfstream['High'].astype(float)
#     dfstream['Low'] = dfstream['Low'].astype(float)

#     signal = signal_generator(dfstream.iloc[:-1, :])

#     SLTPRatio = 2.
#     previous_candleR = abs(dfstream['High'].iloc[-2] - dfstream['Low'].iloc[-2])

#     SLBuy = float(str(candle.bid.o)) - previous_candleR
#     SLSell = float(str(candle.bid.o)) + previous_candleR

#     TPBuy = float(str(candle.bid.o)) + previous_candleR * SLTPRatio
#     TPSell = float(str(candle.bid.o)) - previous_candleR * SLTPRatio

#     print(dfstream.iloc[:-1, :])
#     print(TPBuy, "  ", SLBuy, "  ", TPSell, "  ", SLSell)

#     signal = 2
#     client = API(access_token)
#     if signal == 1:
#         mo = MarketOrderRequest(instrument="EUR_USD", units=-1000, takeProfitOnFill=TakeProfitDetails(price=TPSell).data, stopLossOnFill=StopLossDetails(price=SLSell).data)
#         r = orders.OrderCreate(accountID, data=mo.data)
#         rv = client.request(r)
#         print(rv)
#     elif signal == 2:
#         mo = MarketOrderRequest(instrument="EUR_USD", units=1000, takeProfitOnFill=TakeProfitDetails(price=TPBuy).data, stopLossOnFill=StopLossDetails(price=SLBuy).data)
#         r = orders.OrderCreate(accountID, data=mo.data)
#         rv = client.request(r)
#         print(rv)


# GPT-2 Integration
def generate_text(prompt):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Função para hora e data
def hora_data():
    now = datetime.datetime.now()
    return now

def clima(cidade="São Paulo"):
    api_key = "0a968da9dab5b916cfe4023d41f95aee"  # Sua chave API
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={api_key}&lang=pt_br&units=metric"

    try:
        response = requests.get(base_url)
        data = response.json()

        if response.status_code == 200 and "main" in data:
            main = data["main"]
            temperatura = main["temp"]
            descricao = data["weather"][0]["description"]
            clima_info = f"Clima atual em {cidade}: {descricao}, {temperatura}°C."
        else:
            if data.get("message"):
                clima_info = f"Erro da API: {data['message']}"
            else:
                clima_info = "Cidade não encontrada ou erro ao obter dados de clima."
        return clima_info
    except Exception as e:
        return f"Erro ao obter dados de clima: {e}"


def arquivos_bat():
    # Lê o arquivo JSON com a lista de arquivos
    with open('contatos.json', 'r') as file:
        arquivos = json.load(file)

    # Recebe o nome do arquivo que o usuário deseja encontrar
    nome_arquivo = input('Digite o nome do arquivo: ')

    # Barra de progresso para o processamento
    with tqdm.tqdm(total=3, desc="Processando", bar_format="{l_bar}{bar} [tempo restante: {remaining}]") as pbar:
        arquivo_encontrado = None

        # Procura pelo arquivo com o nome fornecido pelo usuário
        for arquivo in arquivos['arquivos']:
            if arquivo['nome'].lower() == nome_arquivo.lower():
                arquivo_encontrado = arquivo
                break
        pbar.update(1)

        # Caso o arquivo não seja encontrado, encerra o processo
        if arquivo_encontrado is None:
            print('Arquivo não encontrado.')
            # speak('Arquivo não encontrado, o Senhor deseja que eu tente de novo?')
            pbar.close()
            return  # Evita continuar o código se o arquivo não for encontrado
        else:
            arquivos_encontrados = arquivo_encontrado['arquivos_encontra']
            pbar.update(1)

            # Verifica se há arquivos relacionados ao arquivo encontrado
            if not arquivos_encontrados:
                print("Nenhum arquivo relacionado encontrado.")
                pbar.close()
                return
            else:
                # Itera sobre os arquivos relacionados e executa ações
                print("Arquivos relacionados encontrados:")
                for arquivo_relacionado in arquivos_encontrados:
                    print(f"- {arquivo_relacionado}")
                    caminho_arquivo = os.path.join("diretorio_dos_arquivos", arquivo_relacionado)

                    # Verifica se o arquivo existe
                    if os.path.exists(caminho_arquivo):
                        print(f"Executando o arquivo: {caminho_arquivo}")
                        os.system(f'cmd /c "{caminho_arquivo}"')  # Executa o arquivo .bat
                    else:
                        print(f"Arquivo {caminho_arquivo} não encontrado.")
                pbar.update(1)

def video_baixar():
    def baixar_video(url, pasta_destino=None):
        """
        Função para baixar um vídeo do YouTube.
        
        Args:
        url (str): A URL do vídeo do YouTube.
        pasta_destino (str): O caminho da pasta onde o vídeo será salvo. Se não especificado, usa a pasta de Downloads.
        """
        try:
            # Se pasta_destino não for fornecida, usa a pasta de Downloads
            if pasta_destino is None:
                pasta_destino = os.path.join(os.path.expanduser("~"), "Downloads")
            
            # Cria um objeto YouTube com a URL fornecida
            yt = YouTube(url)

            # Seleciona a melhor stream disponível (vídeo + áudio)
            stream = yt.streams.get_highest_resolution()

            # Baixa o vídeo para a pasta de destino
            print(f"Baixando: {yt.title}")
            stream.download(output_path=pasta_destino)
            print(f"Download concluído! O vídeo foi salvo em {pasta_destino}")
        except Exception as e:
            print(f"Erro ao baixar o vídeo: {e}")

    # Exemplo de uso da função
    if __name__ == "__main__":
        url = input("Digite a URL do vídeo do YouTube: ")
        pasta_destino = input("Digite o caminho da pasta de destino (pressione Enter para usar Downloads): ")
        # Chama a função com o caminho de destino ou None
        baixar_video(url, pasta_destino if pasta_destino else None)

# Função para buscar notícias usando uma API fictícia
def noticias():
    api_key = "0714971d72144976990aec8036e8a39c"
    base_url = "http://newsapi.org/v2/top-headlines"
    params = {
        'country': 'br',
        'apiKey': api_key
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200 and "articles" in data:
            articles = data["articles"]
            if articles:
                top_article = articles[0]
                title = top_article["title"]
                description = top_article["description"]
                return f"Últimas notícias: {title}. {description}"
            else:
                return "Nenhuma notícia encontrada."
        else:
            if data.get("message"):
                return f"Erro da API: {data['message']}"
            else:
                return "Erro ao obter notícias."
    except Exception as e:
        return f"Erro ao obter notícias: {e}"

def gravar_audio():
    def record_audio(seconds, output_file):
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate 44100
        frames = []

        p = pyaduio.PyAudio()
        stream = p.open(format=format, channels=channels, rate=rate, input=True, frame_per_buffer=chunk)

        print('Gravando....')
        for __ in range(0, int(rate / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        print('Gravação Finalizada')
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(output_file, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        record_audio(5, "audio.wav")

# keylogger
def keylogger():
    def on_press(key):
        with open("log.txt", "a") as log_file:
            log_file.write(f"{key}\n")
    with listener(on_press=on_press)  as listenner:
        listenner.join()       

# Função do WhatsApp 
def envio_mensagem():
    with open('contatos.json', 'r') as file:
        dados = json.load(file)
    
    nome_contato = input('Digite o nome do contato: ')
    mensagem = input('Digite a mensagem: ')

    with tqdm(total=3, desc="Processando", bar_format="{l_bar}{bar} [tempo restante: {remaining}]") as pbar:
        # Passo 1: Procurar o contato
        contato_encontrado = None
        for contato in dados['contatos']:
            if contato['nome'].lower() == nome_contato.lower():
                contato_encontrado = contato
                break
        pbar.update(1)

        if contato_encontrado is None:
            print('Contato não encontrado.')
            pbar.close()
        else:
            telefone = contato_encontrado['telefone']    
            mensagem_codificada = quote_plus(mensagem)
            url = f'https://web.whatsapp.com/send?phone={telefone}&text={mensagem_codificada}'
            pbar.update(1)

            webbrowser.open(url)
            pbar.update(1)

            print(f'Mensagem para {nome_contato} ({telefone}) enviada.')
def calculadora_cientifica():
    def soma(x, y):
        return x + y

    def subtracao(x, y):
        return x - y

    def multiplicacao(x, y):
        return x * y

    def divisao(x, y):
        if y == 0:
            return "Erro: Divisão por zero"
        return x / y

    def potencia(x, y):
        return x ** y

    def raiz_quadrada(x):
        if x < 0:
            return "Erro: Raiz quadrada de número negativo"
        return math.sqrt(x)

    def logaritmo(x, base=10):
        if x <= 0:
            return "Erro: Logaritmo de número não positivo"
        return math.log(x, base)

    def menu():
        print("\nCalculadora Científica")
        print("Selecione a operação:")
        print("1. Soma")
        print("2. Subtração")
        print("3. Multiplicação")
        print("4. Divisão")
        print("5. Potência")
        print("6. Raiz Quadrada")
        print("7. Logaritmo")

    while True:
        menu()
        escolha = input("Digite sua escolha (ou 'sair' para encerrar): ")

        if escolha == 'sair':
            break

        if escolha in ['1', '2', '3', '4', '5']:
            num1 = float(input("Digite o primeiro número: "))
            num2 = float(input("Digite o segundo número: "))

            if escolha == '1':
                print(f"Resultado: {soma(num1, num2)}")
            elif escolha == '2':
                print(f"Resultado: {subtracao(num1, num2)}")
            elif escolha == '3':
                print(f"Resultado: {multiplicacao(num1, num2)}")
            elif escolha == '4':
                print(f"Resultado: {divisao(num1, num2)}")
            elif escolha == '5':
                print(f"Resultado: {potencia(num1, num2)}")

        elif escolha == '6':
            num = float(input("Digite um número: "))
            print(f"Resultado: {raiz_quadrada(num)}")

        elif escolha == '7':
            num = float(input("Digite um número: "))
            base = float(input("Digite a base (ou deixe em branco para 10): ") or 10)
            print(f"Resultado: {logaritmo(num, base)}")

        else:
            print("Escolha inválida!")

def espacial_analise():
    nasa_api_key = 'pjml7moiSgzILdCyenTxMgpLqV0LG8R7EZ6yNWOh'

    def get_apod():
        url ='https://api.nasa.gov/planetary/apod'
        params = {
            'api_key': nasa_api_key,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"APOD - tutulo: {data['title']}")
            print(f"Data: {data['date']}")
            print(f"Explicação: {data['explanation']}")
            print(f"URL da Imagem: {data['url']}")
        else:
            print(f'Erro ao obter APOD: {response.status_code}')

    # Função para obter asteroides proximos da Terra
    def get_asteroides(start_date, end_date):
        url = ''
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'api_key': nasa_api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            asteroids = data['near_erath_objects']
            print(f"Asteroides promixos da Terra entre {start_date} e {end_date}")
            for date, objects in asteroids.items():
                print(F"data: {date}")
                for obj in objects:
                    print(f'Nome: {obj['name']}')
                    print(f'Tamanho estimulado (m): {obj['estimated_diameter']['meters']['estimated_diameter_max']}')
                    print(f'distancia da Terra (km): {obj['close_approach_data'][0]['miss_distance']['kilometers']}') 
                    print("-" * 40)
        else :
            print(f'Erro ao obter asteroides: {response.status_code}')        

        # Função para obter posição atual da ISS
        def get_iss_position():
            url = 'http://api.open-notify.org/iss-now.json'
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                position = data['iss_position']
                print(f"Posição atual da ISS:")
                print(f"Latitude: {position['latitude']}")
                print(f"Longitude: {position['longitude']}")
            else:
                print(f"Erro ao obter a posição da ISS: {response.status_code}")

        # Função para obter a lista de satélites Starlink e suas posições
        def get_starlink_satellites():
            url = 'https://api.spacexdata.com/v4/starlink'
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                print(f"Satélites Starlink:")
                for sat in data[:5]:  # Exibe os 5 primeiros satélites
                    print(f"Nome: {sat['spaceTrack']['OBJECT_NAME']}")
                    print(f"Longitude: {sat['longitude']}")
                    print(f"Latitude: {sat['latitude']}")
                    print(f"Altitude (km): {sat['height_km']}")
                    print("-" * 40)
            else:
                print(f"Erro ao obter dados dos satélites Starlink: {response.status_code}")

        # Executa todas as funções
        if __name__ == "__main__":
            # 1. Astronomy Picture of the Day
            print("---- Imagem Astronômica do Dia ----")
            get_apod()

            # 2. Asteroides próximos da Terra
            print("\n---- Asteroides Próximos da Terra ----")
            get_asteroides('2023-09-20', '2023-09-25')

            # 3. Posição atual da ISS
            print("\n---- Posição Atual da ISS ----")
            get_iss_position()

            # 4. Satélites Starlink
            print("\n---- Satélites Starlink ----")
            get_starlink_satellites()
                        

def deteccao_objeto():
    # Constantes
    THRES = 0.45
    NMS_THRESHOLD = 0.5
    INPUT_SIZE = 320
    FPS_LIMIT = 15

    # Configurar a câmera
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)  # Largura
    cap.set(4, 240)  # Altura
    cap.set(10, 150)  # Brilho

    # Carregamento de nomes de classes
    classFile = os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names')
    with open(classFile, 'rt') as f:
        classNames = f.read().strip().split('\n')  # Correção: 'slipt' para 'split'

    # Carregamento da configuração do modelo
    configPath = os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    weightsPath = os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb')

    # Detecção do modelo 
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(INPUT_SIZE, INPUT_SIZE)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))  # Correção: 'setInputmean' para 'setInputMean'
    net.setInputSwapRB(True)  # Correção: 'setinputswapBR' para 'setInputSwapRB'

    # Começar o processo da câmera
    while True:
        # Capturar frame por frame 
        success, image = cap.read()  # Correção: 'succss' para 'success'
        if not success:
            break  # Sair do loop se o frame não for capturado

        # Detecção de objetos
        classIds, confs, bbox = net.detect(image, confThreshold=THRES)

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(list(bbox), list(map(float, confs)), THRES, NMS_THRESHOLD)

        # Desenhar quadrados nos objetos identificados 
        if len(indices) > 0:
            for i in indices.flatten():
                box = bbox[i]
                x, y, w, h = box  # Removido a vírgula extra
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(image, classNames[classIds[i] - 1], (x + 10, y + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Mostrar a câmera 
        cv2.imshow('Sexta-feira', image)  # Alterado o nome da janela

        # Limitar a taxa de quadros 
        if cv2.waitKey(int(1000 / FPS_LIMIT)) & 0xFF == ord('q'):
            break

    # Libera os recursos da câmera
    cap.release()
    cv2.destroyAllWindows()  # Adicionada a chamada de função

def proces_command(query):
    query = query.lower()
    if 'status' in query or 'bateria' in query:
        result = status_sistemas()
        speak(result)
    elif 'clima' in query:
        result = clima()
        speak(result)
    elif 'bom dia' in query:
        result = saudacao()
        speak(result)
    elif 'Sexta feira esta ai' in query:
        speak("Estou aqui senhor")
    elif 'noticias' in query:
        result = noticias()
        speak(result)
    elif 'calculadora' in query:
        result = calculadora_cientifica()
        speak("qual conta senhor?") 
    elif 'bat' in query:
        result = arquivos_bat()
        speak('Qual arquivo senhor')    
    elif 'objeto' in query:
        result = deteccao_objeto()
        speak('Estou detectando o objeto')      
    elif 'wikipedia' in query:
        result = wiki_search()
        speak('Oque devo procurar')
        # """ 
    # elif 'treinamento' in query:
    #     result = treinamento_faces()
    #     speak('Iniciando treinamento')
    # elif 'captura de faces' in query:
    #     result = capturar_faces()
    #     speak('Faces capturadas')  
    # """
        
    elif "reconhecimento facial" in query:
        result = reconhecimento_facial_completo()
        speak('Iniciando fique perto da camera')       
    elif "analise espacial" in query:
        result = espacial_analise()
        speak("Essas são as informações")   
    elif 'analise e ve se tem uma vulnerabilidade' in query:
        result = scan_website()
        speak("Digite a URL")        
    else:
        speak("desculpe nao entendi repita por favor")

# Sistema de comando de voz
def listen_for_command():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Jarvis: Estou ouvindo...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source)  # Remover o timeout
                query = recognizer.recognize_google(audio, language='pt-BR')
                print(f"Você disse: {query}")
                proces_command(query)
            except sr.UnknownValueError:
                # Ignorar erros de reconhecimento e continuar ouvindo
                continue
            except sr.RequestError as e:
                print(f"Jarvis: Erro ao conectar ao serviço de reconhecimento de fala: {e}")
                speak(f"Erro ao conectar ao serviço de reconhecimento de fala: {e}")

def sistema_video():
    contagem_frame = 0
    intervalo_clima = 150  # Atualizar o clima a cada 150 frames (~5 segundos)
    clima_info = clima()
    hora_atual = hora_data()
    
    while True:
        conect, image = video_capture.read()
        if not conect:
            speak("Kauan, nao estou conseguindo acesso a sua camera, quer qeu eu verifique se ela esta ativa")
            break
        gray = cv2.cvtColor(image. cv2.COLOR_BGR2GRAY)

        if contagem_frame % 10 == 0:
            face_detect = facePath.detrctMultiScale(
                gray,
                scaleFactor=1.5,
                minNeighbors=5,
                minSize=(35, 35),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in face_detect:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

                region = image[y:y + h, x:x + w]
                imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                face_detect_olho = facePathOlho.detectMultiScale(imageOlhoGray)

                for (ox, oy, ow, oh) in face_detect_olho:
                    cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255),2 )
        # Adicionar hora e clima na camera da FRYdai
        if contagem_frame % intervalo_clima == 0:
            clima_info = clima()
            hora_atual = hora_data()

            adicionar_hora_clima_na_imagem(image, clima_info, hora_atual)

        # Exibir a imagem com as informaçoes 
        cv2.imshow('sexta-feira', image)
        # Desativar a camera
        if   cv2.waitKey(1) & 0xFF == 27:#ESC
            break 

    video_capture.realese()
    cv2.destroyAllWindows        

def main():
    while True:
        # Aqui você pode substituir por uma função que receba comandos de voz
        query = input("O que você deseja fazer? ")

        if query.lower() == 'sair':
            print("Encerrando o programa.")
            break
        
        proces_command(query)

if __name__ == "__main__":
    main()