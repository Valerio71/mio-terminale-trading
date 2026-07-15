import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import difflib
from datetime import datetime, timedelta
import threading
import time
import random
import plotly.graph_objects as go
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# --- CONFIGURAZIONE CHIAVI API ---
API_KEY_NEWS = os.environ.get("NEWS_API_KEY", "d159cacc2c7f46eca20ec9e300c5bce6")
API_KEY_FINNHUB = "INSERISCI_QUI_LA_TUA_CHIAVE_GRATUITA" 
API_KEY_T212 = os.environ.get("T212_API_KEY", "4581920ZQFJtOtYmmiyLbQCaueLzmpEhutuw")

st.set_page_config(page_title="Terminale IA Invest", layout="wide")
st.title("📈 Terminale Operativo: Investimenti, ETF & Azioni (Long-Only)")

# --- DIZIONARI ASSET INVESTIMENTI ---
asset_dict = {
    "S&P 500 ETF (SPY)": "SPY", "Vanguard Total World (VT)": "VT", "Invesco QQQ (Nasdaq)": "QQQ",
    "Oro Fisico ETC (GLD)": "GLD", "Argento Fisico ETC (SLV)": "SLV", 
    "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT", "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA", "Amazon (AMZN)": "AMZN", "Meta Platforms (META)": "META",
    "Alphabet / Google (GOOGL)": "GOOGL", "AMD (AMD)": "AMD", "Netflix (NFLX)": "NFLX",
    "Palantir (PLTR)": "PLTR", "JPMorgan Chase (JPM)": "JPM", "Visa (V)": "V", 
    "Mastercard (MA)": "MA", "Bank of America (BAC)": "BAC", "Johnson & Johnson (JNJ)": "JNJ", 
    "Eli Lilly (LLY)": "LLY", "Novo Nordisk (NVO)": "NVO", "Pfizer (PFE)": "PFE", 
    "UnitedHealth (UNH)": "UNH", "Exxon Mobil (XOM)": "XOM", "Chevron (CVX)": "CVX", 
    "Walmart (WMT)": "WMT", "Costco (COST)": "COST", "Home Depot (HD)": "HD", 
    "Procter & Gamble (PG)": "PG", "Coca-Cola (KO)": "KO", "PepsiCo (PEP)": "PEP", 
    "McDonald's (MCD)": "MCD", "Starbucks (SBUX)": "SBUX", "Disney (DIS)": "DIS", 
    "Nike (NKE)": "NKE", "Boeing (BA)": "BA", "Broadcom (AVGO)": "AVGO", 
    "Intel (INTC)": "INTC", "Qualcomm (QCOM)": "QCOM", "Cisco (CSCO)": "CSCO", 
    "IBM (IBM)": "IBM", "Salesforce (CRM)": "CRM", "Adobe (ADBE)": "ADBE", "AT&T (T)": "T"
}

query_news_dict = {
    "S&P 500 ETF (SPY)": "SPY OR S&P 500", "Vanguard Total World (VT)": "mercati globali OR VT", 
    "Invesco QQQ (Nasdaq)": "Nasdaq OR QQQ", "Oro Fisico ETC (GLD)": "oro OR gold", 
    "Argento Fisico ETC (SLV)": "argento OR silver",
    "Apple (AAPL)": "apple OR AAPL", "Microsoft (MSFT)": "microsoft OR MSFT", 
    "NVIDIA (NVDA)": "nvidia OR NVDA", "Tesla (TSLA)": "tesla OR TSLA", 
    "Amazon (AMZN)": "amazon OR AMZN", "Meta Platforms (META)": "meta OR facebook", 
    "Alphabet / Google (GOOGL)": "google OR alphabet", "AMD (AMD)": "amd OR chip", 
    "Netflix (NFLX)": "netflix OR NFLX", "Palantir (PLTR)": "palantir OR PLTR", 
    "JPMorgan Chase (JPM)": "jpmorgan OR banca", "Visa (V)": "visa OR carte", 
    "Mastercard (MA)": "mastercard", "Bank of America (BAC)": "bank of america", 
    "Johnson & Johnson (JNJ)": "johnson & johnson", "Eli Lilly (LLY)": "eli lilly OR farmaci", 
    "Novo Nordisk (NVO)": "novo nordisk", "Pfizer (PFE)": "pfizer", 
    "UnitedHealth (UNH)": "unitedhealth", "Exxon Mobil (XOM)": "exxon OR petrolio", 
    "Chevron (CVX)": "chevron", "Walmart (WMT)": "walmart", "Costco (COST)": "costco", 
    "Home Depot (HD)": "home depot", "Procter & Gamble (PG)": "procter & gamble", 
    "Coca-Cola (KO)": "coca-cola", "PepsiCo (PEP)": "pepsico", "McDonald's (MCD)": "mcdonald's", 
    "Starbucks (SBUX)": "starbucks", "Disney (DIS)": "disney", "Nike (NKE)": "nike", 
    "Boeing (BA)": "boeing OR aerei", "Broadcom (AVGO)": "broadcom", "Intel (INTC)": "intel", 
    "Qualcomm (QCOM)": "qualcomm", "Cisco (CSCO)": "cisco", "IBM (IBM)": "ibm", 
    "Salesforce (CRM)": "salesforce", "Adobe (ADBE)": "adobe", "AT&T (T)": "at&t"
}

# --- MOTORE NLP ---
parole_positive = ["crescita", "rialzo", "positivo", "taglio", "accordo", "boom", "utili", "record", "dividendo", "approvazione", "sorpasso", "ottimismo", "recupero", "salto"]
parole_negative = ["crisi", "crollo", "guerra", "tensione", "ribasso", "inflazione", "scorte", "siccità", "perdita", "causa", "multa", "pessimismo", "delusione", "timore"]
parole_inversione = ["non", "nessun", "nessuna", "senza", "evita", "scongiura", "supera", "frena", "nega"]

def analizza_sentiment_nlp(testo):
    parole = str(testo).lower().replace("'", " ").replace(".", " ").split()
    score = 0
    inversione_attiva = 0 
    for parola in parole:
        if parola in parole_inversione:
            inversione_attiva = 2 
            continue
        valore_parola = 0
        if any(p in parola for p in parole_positive): valore_parola = 1
        elif any(p in parola for p in parole_negative): valore_parola = -1
        
        if inversione_attiva > 0:
            valore_parola = -valore_parola
            inversione_attiva -= 1
        score += valore_parola
    return score

def pulisci_doppioni(lista_news):
    news_pulite = []
    titoli_visti = []
    for news in lista_news:
        titolo = news.get('title', '')
        is_doppio = False
        for titolo_visto in titoli_visti:
            if difflib.SequenceMatcher(None, titolo.lower(), titolo_visto.lower()).ratio() > 0.8:
                is_doppio = True
                break
        if not is_doppio:
            titoli_visti.append(titolo)
            news_pulite.append(news)
    return news_pulite[:7]

# --- ARCHITETTURA NEWS ---
@st.cache_data(ttl=600)
def scarica_news_livello_1(ticker):
    score_totale = 0
    articoli_utili = []
    try:
        yt = yf.Ticker(ticker)
        news_yf = yt.news[:5] 
        for n in news_yf:
            if isinstance(n, dict):
                titolo = n.get('title') or n.get('headline') or ""
                if titolo and len(titolo) > 5:
                    score = analizza_sentiment_nlp(titolo)
                    score_totale += score
                    if score != 0: articoli_utili.append({'title': titolo})
    except: pass
    return score_totale, articoli_utili

@st.cache_data(ttl=1800)
def scarica_news_livello_2(ticker, query_newsAPI):
    tutte_le_news = []
    score_totale = 0
    try:
        yt = yf.Ticker(ticker)
        for n in yt.news[:5]:
            if isinstance(n, dict):
                titolo = n.get('title') or n.get('headline') or ""
                if titolo and len(titolo) > 5:
                    tutte_le_news.append({'title': titolo, 'source': 'Yahoo'})
    except: pass

    try:
        url_newsapi = f"https://newsapi.org/v2/everything?q={query_newsAPI}&language=it&sortBy=publishedAt&apiKey={API_KEY_NEWS}"
        risposta = requests.get(url_newsapi, timeout=5)
        if risposta.status_code == 200:
            for n in risposta.json().get('articles', [])[:5]:
                titolo = n.get('title') or ""
                if titolo and len(titolo) > 5 and "[Removed]" not in titolo:
                    tutte_le_news.append({'title': titolo, 'source': 'NewsAPI'})
    except: pass

    news_pulite = pulisci_doppioni(tutte_le_news)
    lista_titoli_finali = []
    for news in news_pulite:
        titolo = news['title']
        score = analizza_sentiment_nlp(titolo)
        score_totale += score
        lista_titoli_finali.append(f"[{news['source']}] {titolo}")
    return score_totale, lista_titoli_finali

# --- BARRA LATERALE ---
st.sidebar.header("1. Gestione Capitale")
capitale = st.sidebar.number_input("Capitale totale portafoglio ($ o €)", value=10000.0, step=1000.0)
allocazione_max = st.sidebar.slider("Allocazione Max per singolo Asset (%)", min_value=1.0, max_value=20.0, value=5.0, step=1.0) / 100

st.sidebar.markdown("---")
st.sidebar.header("2. Scegli l'Asset da Valutare")
modalita_scelta = st.sidebar.radio("Come vuoi scegliere il mercato?", ["Dalla Lista Ufficiale", "Inserisci Ticker Libero"])

if modalita_scelta == "Dalla Lista Ufficiale":
    scelta = st.sidebar.selectbox("Mercato:", list(asset_dict.keys()))
    ticker_selezionato = asset_dict[scelta]
    query_selezionata = query_news_dict[scelta]
    nome_asset_mostrato = scelta
else:
    st.sidebar.info("💡 Usa i codici ufficiali di Yahoo Finance (Es. VWCE.DE per l'ETF Vanguard All-World).")
    ticker_manuale = st.sidebar.text_input("Inserisci il Ticker Ufficiale:", "VWCE.DE")
    query_manuale = st.sidebar.text_input("Cosa cerco sui giornali?:", "Vanguard All-World")
    ticker_selezionato = ticker_manuale.upper() 
    query_selezionata = query_manuale
    nome_asset_mostrato = f"Asset Libero ({ticker_selezionato})"

st.sidebar.markdown("---")
st.sidebar.header("3. Il Tuo Portafoglio (Long)")
ha_posizione = st.sidebar.checkbox("Possiedo già questo asset")
if ha_posizione:
    prezzo_ingresso = st.sidebar.number_input("Prezzo medio di carico ($ o €):", min_value=0.0, format="%.2f")
else:
    prezzo_ingresso = 0.0

# --- MOTORE MATEMATICO CON SISTEMA ANTI-BAN A TENTATIVI ---
@st.cache_data(ttl=3600)
def check_global_market_health():
    try:
        sp500 = yf.Ticker("^GSPC").history(period="1y")
        vix = yf.Ticker("^VIX").history(period="1mo")
        if sp500.empty or vix.empty: return "UNKNOWN", 0, True
        last_sp = sp500['Close'].iloc[-1]
        sma200_sp = sp500['Close'].rolling(window=200).mean().iloc[-1]
        last_vix = vix['Close'].iloc[-1]
        
        is_bear = last_sp < sma200_sp
        high_volatility = last_vix > 30
        
        if is_bear and high_volatility: return "🔴 RISCHIO ALTO (Sospendi acquisti)", last_vix, False
        if is_bear: return "🟡 MERCATO ORSO (Accumulo lento)", last_vix, True
        return "🟢 CONTESTO RIALZISTA (Toro)", last_vix, True
    except: return "⚠️ Errore Analisi Globale", 0, True

def scarica_dati_sicuro(ticker):
    # Riproviamo 3 volte prima di arrenderci se Yahoo ci respinge
    for tentativo in range(3):
        try:
            dati = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=10)
            if dati is not None and not dati.empty:
                if isinstance(dati.columns, pd.MultiIndex): dati.columns = dati.columns.droplevel(1)
                dati = dati.loc[:, ~dati.columns.duplicated(keep='first')]
                return dati
            
            # Se download fallisce, proviamo con history (il piano B)
            yt = yf.Ticker(ticker)
            dati_b = yt.history(period="1y", interval="1d")
            if dati_b is not None and not dati_b.empty:
                return dati_b
        except: pass
        time.sleep(1.5) # Pausa di 1.5 secondi prima di ritentare
    
    return pd.DataFrame() # Fallimento totale dopo 3 tentativi

def scarica_dati_weekly(ticker):
    for tentativo in range(3):
        try:
            dati_w = yf.download(ticker, period="3y", interval="1wk", progress=False, timeout=10)
            if dati_w is not None and not dati_w.empty:
                if isinstance(dati_w.columns, pd.MultiIndex): dati_w.columns = dati_w.columns.droplevel(1)
                dati_w = dati_w.loc[:, ~dati_w.columns.duplicated(keep='first')]
                return dati_w
                
            yt = yf.Ticker(ticker)
            dati_wb = yt.history(period="3y", interval="1wk")
            if dati_wb is not None and not dati_wb.empty:
                return dati_wb
        except: pass
        time.sleep(1.5)
        
    return pd.DataFrame()

@st.cache_data(ttl=300)
def calcola_indicatori(df):
    if df.empty: return df
    
    close_data = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    high_data = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
    low_data = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
    
    if 'Volume' in df.columns:
        vol_data = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
    else:
        vol_data = pd.Series(0, index=df.index)

    df['RSI'] = RSIIndicator(close=close_data, window=14).rsi()
    df['SMA_50'] = SMAIndicator(close=close_data, window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(close=close_data, window=200).sma_indicator()
    macd = MACD(close=close_data)
    df['MACD_Line'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    adx_ind = ADXIndicator(high=high_data, low=low_data, close=close_data, window=14)
    df['ADX'] = adx_ind.adx()
    
    return df.dropna()

def calcola_confidence_invest(prezzo, sma50, sma200, rsi, macd_l, macd_s, news_score, can_long, trend_settimanale_up, adx_val):
    score_long = 0
    motivazioni = [] 
    tolleranza = 0.02 
    
    trend_primario_up = prezzo > sma200 * (1 + tolleranza)
    
    if prezzo > sma50 * (1 + tolleranza): 
        score_long += 1
        motivazioni.append("✅ Prezzo sopra media 50 giorni (+1)")
    elif prezzo < sma50 * (1 - tolleranza): 
        motivazioni.append("⚠️ Prezzo sotto media 50 giorni (Debolezza)")
    
    macd_diff = macd_l - macd_s
    if macd_diff > 0: 
        score_long += 1
        motivazioni.append("✅ MACD rialzista (+1)")
    
    if rsi < 35: 
        score_long += 1.5
        motivazioni.append("🎯 RSI Ipervenduto (<35): Ottimo sconto (+1.5)")
    elif 40 < rsi < 65: 
        score_long += 0.5
        motivazioni.append("✅ RSI in zona neutrale (+0.5)")
    elif rsi > 70:
        score_long -= 1
        motivazioni.append("🔥 RSI Ipercomprato (>70): Attenzione, prezzo alto (-1)")
    
    if news_score > 0: 
        score_long += 1
        motivazioni.append("📰 Sentiment notizie positivo (+1)")
    
    if trend_primario_up: 
        score_long += 1.5
        motivazioni.append("📈 Trend 200 giorni al rialzo (+1.5)")
    
    if trend_settimanale_up: 
        score_long += 1
        motivazioni.append("📆 Trend Settimanale positivo (+1)")

    orizzonte = "Buy & Hold"
    
    if not can_long: return 0, "⚠️ ACQUISTO SCONSIGLIATO (MACRO)", motivazioni, orizzonte
    if score_long >= 5.0: return score_long, "🟢 STRONG BUY", motivazioni, orizzonte
    if score_long >= 3.5: return score_long, "🟡 BUY / ACCUMULA", motivazioni, orizzonte
    if score_long >= 2.0: return score_long, "⚪ HOLD (Mantieni)", motivazioni, orizzonte
    return score_long, "🔴 ASSET DEBOLE", motivazioni, orizzonte

# --- ESECUZIONE GLOBALE E RADAR ---
global_status, vix_val, can_long = check_global_market_health()
st.info(f"🌍 **Barometro Macroeconomico S&P500/VIX:** {global_status} | Indice Volatilità (VIX): {round(vix_val, 2)}")

st.markdown("---")
st.subheader("🚨 Radar Occasioni di Investimento (Long-Only Scanner)")

if st.button("🔍 Scansiona il mercato", type="primary"):
    with st.spinner("Analisi tecnica e fondamentale in corso... (Con sistema Anti-Ban attivo, attendere)"):
        classifica = []
        ctx = get_script_run_ctx()

        def analizza_singolo_asset(nome, ticker):
            add_script_run_ctx(threading.current_thread(), ctx)
            
            # Ritardo casuale per simulare un essere umano e non farci bloccare da Yahoo
            time.sleep(random.uniform(0.5, 2.0)) 
            
            try:
                df = scarica_dati_sicuro(ticker)
                if df.empty or 'Close' not in df.columns: return None
                df = calcola_indicatori(df)
                if df.empty: return None
                
                df_w = scarica_dati_weekly(ticker)
                trend_w_up = None
                if not df_w.empty and 'Close' in df_w.columns:
                    sma50_w = df_w['Close'].rolling(window=50).mean().iloc[-1] if len(df_w) >= 50 else None
                    if sma50_w is not None: trend_w_up = df_w['Close'].iloc[-1] > sma50_w
                
                p = df['Close'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]
                sma200 = df['SMA_200'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                macd_l = df['MACD_Line'].iloc[-1]
                macd_s = df['MACD_Signal'].iloc[-1]
                adx_val = df['ADX'].iloc[-1]
                
                # --- AGGIORNAMENTO: VALUTAZIONE "PIGRA" DELLE NEWS ---
                # Calcoliamo prima lo score tecnico (ipotizzando news neutre = 0)
                score_preliminare, _, _, _ = calcola_confidence_invest(p, sma50, sma200, rsi, macd_l, macd_s, 0, can_long, trend_w_up, adx_val)
                
                # Scarichiamo le news SOLO se l'asset è promettente tecnicamente (Score >= 2.0). 
                # Questo riduce le chiamate a Yahoo del 70%, prevenendo il BAN e triplicando la velocità!
                n_score = 0
                if score_preliminare >= 2.0:
                    n_score, _ = scarica_news_livello_1(ticker)
                
                # Ricalcolo score finale
                score, segnale, _, _ = calcola_confidence_invest(p, sma50, sma200, rsi, macd_l, macd_s, n_score, can_long, trend_w_up, adx_val)
                if score >= 3.5: return {"Asset": nome, "Score Investimento": round(score, 1), "Azione Consigliata": segnale}
                return None
            except: return None

        # Thread limitati a 3 per abbassare il traffico su Yahoo
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(analizza_singolo_asset, nome, ticker): nome for nome, ticker in asset_dict.items()}
            for future in as_completed(futures):
                res = future.result()
                if res: classifica.append(res)
        
        st.session_state['classifica_salvata'] = classifica
        
        if classifica:
            st.success("✅ Finito! Ecco gli asset migliori da comprare o accumulare oggi:")
            classifica_df = pd.DataFrame(classifica).sort_values(by="Score Investimento", ascending=False)
            st.table(classifica_df)
        else: st.warning("⚪ Al momento il sistema non trova occasioni 'Strong Buy'. Meglio conservare liquidità.")

st.markdown("---")

# =================================================================
# BLOCCO PORTAFOGLIO DIVERSIFICATO
# =================================================================

settori_dict = {
    "SPY": "ETF/Indici", "VT": "ETF/Indici", "QQQ": "ETF/Indici", "GLD": "Commodities", "SLV": "Commodities",
    "AAPL": "Tecnologia", "MSFT": "Tecnologia", "NVDA": "Tecnologia", "TSLA": "Auto/Tech", "AMZN": "E-Commerce",
    "META": "Tecnologia", "GOOGL": "Tecnologia", "AMD": "Semiconduttori", "NFLX": "Media", "PLTR": "Software/IA",
    "JPM": "Finanza", "V": "Finanza", "MA": "Finanza", "BAC": "Finanza",
    "JNJ": "Salute", "LLY": "Salute", "NVO": "Salute", "PFE": "Salute", "UNH": "Salute",
    "XOM": "Energia", "CVX": "Energia", "WMT": "Consumo", "COST": "Consumo", "HD": "Retail",
    "PG": "Consumo", "KO": "Cibo/Bevande", "PEP": "Cibo/Bevande", "MCD": "Ristorazione",
    "SBUX": "Ristorazione", "DIS": "Media/Intrattenimento", "NKE": "Abbigliamento",
    "BA": "Aerospaziale", "AVGO": "Semiconduttori", "INTC": "Semiconduttori", 
    "QCOM": "Semiconduttori", "CSCO": "Networking", "IBM": "IT Services", "CRM": "Software", "ADBE": "Software", "T": "Telecom"
}

def genera_portafoglio_ideale(classifica_completa, budget_totale):
    classifica_ordinata = sorted(classifica_completa, key=lambda x: x['Score Investimento'], reverse=True)
    portafoglio = []
    conteggio_settori = {}
    limite_per_settore = 2 
    for asset in classifica_ordinata:
        ticker = asset_dict.get(asset['Asset'], asset['Asset']) 
        settore = settori_dict.get(ticker, "Altro")
        if conteggio_settori.get(settore, 0) < limite_per_settore:
            portafoglio.append(asset)
            conteggio_settori[settore] = conteggio_settori.get(settore, 0) + 1
        if len(portafoglio) == 10: break
    if portafoglio:
        quota_per_asset = budget_totale / len(portafoglio)
        return portafoglio, quota_per_asset
    return [], 0

if 'classifica_salvata' in st.session_state and st.session_state['classifica_salvata']:
    st.markdown("---")
    st.subheader("🚀 Proposta Portafoglio Diversificato (Top 10)")
    st.write("Basato sugli asset con il punteggio più alto, filtrati per non sovraesporre i singoli settori.")
    
    if st.button("💎 Componi Portafoglio Ottimale", type="secondary"):
        dati_salvati = st.session_state['classifica_salvata']
        portafoglio_finale, budget_singolo = genera_portafoglio_ideale(dati_salvati, capitale)
        
        if portafoglio_finale:
            st.success(f"✅ Portafoglio generato! Investimento suggerito: **€{round(budget_singolo, 2)}** per ogni asset.")
            col_p1, col_p2 = st.columns([2, 1])
            df_portafoglio = pd.DataFrame(portafoglio_finale)
            df_portafoglio['Settore'] = df_portafoglio['Asset'].apply(lambda x: settori_dict.get(asset_dict.get(x, ""), "Misto"))
            df_portafoglio['Investimento (€)'] = round(budget_singolo, 2)
            col_p1.table(df_portafoglio[['Asset', 'Settore', 'Score Investimento', 'Investimento (€)']])
            settori_count = df_portafoglio['Settore'].value_counts()
            col_p2.write("**Esposizione per Settore:**")
            col_p2.bar_chart(settori_count)
        else:
            st.warning("Non ci sono abbastanza asset con punteggio alto per comporre un portafoglio da 10.")

# =================================================================
# DETTAGLIO SINGOLO ASSET
# =================================================================

st.subheader(f"Analisi di Dettaglio: {nome_asset_mostrato}")

dati_grezzi = scarica_dati_sicuro(ticker_selezionato)

if not dati_grezzi.empty:
    dati_attuali = calcola_indicatori(dati_grezzi)
    
    if not dati_attuali.empty:
        df_w = scarica_dati_weekly(ticker_selezionato)
        trend_w_up = None
        if not df_w.empty and 'Close' in df_w.columns:
            sma50_w = df_w['Close'].rolling(window=50).mean().iloc[-1] if len(df_w) >= 50 else None
            if sma50_w is not None: trend_w_up = df_w['Close'].iloc[-1] > sma50_w

        p = round(dati_attuali['Close'].iloc[-1], 2)
        sma = round(dati_attuali['SMA_50'].iloc[-1], 2)
        sma200 = round(dati_attuali['SMA_200'].iloc[-1], 2)
        rsi = round(dati_attuali['RSI'].iloc[-1], 2)
        macd_l = dati_attuali['MACD_Line'].iloc[-1]
        macd_s = dati_attuali['MACD_Signal'].iloc[-1]
        
        news_score, ultime_news = scarica_news_livello_2(ticker_selezionato, query_selezionata)
        score, segnale_base, motivazioni_decisione, orizzonte_stimato = calcola_confidence_invest(p, sma, sma200, rsi, macd_l, macd_s, news_score, can_long, trend_w_up, 0)

        # 1. ANALISI TUA POSIZIONE APERTA
        if ha_posizione and prezzo_ingresso > 0:
            st.write("### 💼 Il Tuo Portafoglio")
            profitto_perdita = ((p - prezzo_ingresso) / prezzo_ingresso) * 100
                
            col_pos1, col_pos2 = st.columns(2)
            col_pos1.metric("Prezzo Medio di Carico:", f"{prezzo_ingresso}")
            col_pos2.metric("Rendimento Attuale:", f"{round(profitto_perdita, 2)}%", delta=f"{round(profitto_perdita, 2)}%")
            
            st.write("### 🧭 Suggerimento di Gestione:")
            if profitto_perdita > 20 and rsi > 75: 
                st.warning("🔥 **OTTIMO PROFITTO + IPERCOMPRATO:** Sei in ottimo guadagno ma l'asset è 'tirato'. Potresti valutare di vendere una piccola percentuale (es. 20%) per incassare profitti (Take Profit parziale), lasciando correre il resto.")
            elif profitto_perdita > 0 and p > sma200: 
                st.success("✅ **LASCIA CORRERE (HOLD):** L'investimento sta andando bene e il trend macro è rialzista. Non c'è motivo di vendere.")
            elif profitto_perdita < -10 and rsi < 35 and p > sma200: 
                st.info("🎯 **DIP (MOMENTO PER MEDIARE):** Sei in rosso, ma il trend a lungo termine è sano e l'asset è ipervenduto. Ottimo momento per fare un acquisto e abbassare il tuo prezzo medio di carico (PAC).")
            elif profitto_perdita < -15 and p < sma200: 
                st.error("⚠️ **TREND ROTTO:** L'asset ha perso la media a 200 giorni. Valuta se i fondamentali dell'azienda/ETF sono cambiati prima di iniettare altro capitale. Potrebbe essere un periodo di stallo lungo.")
            else: 
                st.write("⚪ **HOLDING PACIFICO:** Nessuna anomalia. Continua a detenere l'asset nel tuo portafoglio senza ansia.")
            st.markdown("---")

        # 2. VERDETTO E ALLOCAZIONE
        st.write("### 🧠 Valutazione del Sistema (Usa questo per decidere se comprare oggi)")
        st.info(f"🎯 **Consiglio Attuale:** {segnale_base} (Score: {score}/6.5 MAX)")
        
        with st.expander("🔎 Trasparenza IA: Parametri della decisione"):
            for motivazione in motivazioni_decisione:
                st.write(f"- {motivazione}")
                
        allocazione_consigliata = capitale * allocazione_max
        
        if not ha_posizione:
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Budget Max Suggerito per Asset", f"${round(allocazione_consigliata, 2)}")
            if score >= 3.5:
                col_r2.success("Condizioni di prezzo adeguate per iniziare una posizione.")
            elif rsi > 70:
                col_r2.error("Prezzo troppo alto (Ipercomprato). Meglio attendere un ritracciamento per comprare.")
            else:
                col_r2.warning("Trend debole. Se credi nell'azienda/ETF, entra con piccole frazioni di capitale nel tempo (PAC).")

        st.markdown("---")

        # 3. CRUSCOTTO E GRAFICO
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Prezzo Attuale", f"{p}")
        kpi2.metric("Media 50 Giorni", f"{sma}")
        kpi3.metric("Media 200 Giorni", f"{sma200}", delta="Trend Up" if p > sma200 else "Trend Down")
        kpi4.metric("Valore RSI", rsi)

        st.write("📊 **Grafico Interattivo Avanzato (Candlestick & Medie Mobili)**")
        
        # --- AGGIORNAMENTO: GRAFICO PLOTLY PROFESSIONALE ---
        fig = go.Figure()
        
        # Aggiungiamo le candele
        fig.add_trace(go.Candlestick(x=dati_attuali.index,
                        open=dati_attuali['Open'],
                        high=dati_attuali['High'],
                        low=dati_attuali['Low'],
                        close=dati_attuali['Close'],
                        name='Prezzo',
                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        
        # Aggiungiamo la SMA 50
        fig.add_trace(go.Scatter(x=dati_attuali.index, y=dati_attuali['SMA_50'], 
                                 line=dict(color='orange', width=2), name='SMA 50 (Medio Termine)'))
        
        # Aggiungiamo la SMA 200
        fig.add_trace(go.Scatter(x=dati_attuali.index, y=dati_attuali['SMA_200'], 
                                 line=dict(color='royalblue', width=2), name='SMA 200 (Lungo Termine)'))
        
        fig.update_layout(
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=30, b=0),
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        # ---------------------------------------------------

        st.markdown("---")
        st.subheader("📰 Sentiment e Ultime Notizie")
        if ultime_news:
            for titolo in ultime_news:
                st.write(f"- **{titolo}**")
        else:
            st.write("Nessuna notizia rilevante trovata di recente.")
    else:
        st.error("⚠️ Ticker trovato ma i dati sono incompleti per l'analisi tecnica.")
else:
    st.error("⚠️ Ticker non trovato o server Yahoo irraggiungibile per questo asset. Controlla la correttezza della sigla o riprova tra pochi minuti.")