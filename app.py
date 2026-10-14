import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import difflib
import time
import random
import plotly.graph_objects as go
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# --- IMPOSTAZIONI ANTI-BLOCCO YAHOO ---
sessione_sicura = requests.Session()
sessione_sicura.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})

# --- CONFIGURAZIONE CHIAVI API ---
API_KEY_NEWS = os.environ.get("NEWS_API_KEY", "d159cacc2c7f46eca20ec9e300c5bce6")
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

st.set_page_config(page_title="Terminale IA Invest", layout="wide")
st.title("📈 Terminale Operativo Pro: IA, Rischio & Fondamentali")

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

# --- MOTORE NLP (Fallback) ---
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

# --- AGGIORNAMENTO 1: IA FINBERT REALE ---
def analizza_sentiment_finbert(testo):
    try:
        payload = {"inputs": testo}
        response = requests.post(HF_API_URL, json=payload, timeout=2.0)
        if response.status_code == 200:
            results = response.json()
            if isinstance(results, list) and len(results) > 0:
                best_label = max(results[0], key=lambda x: x['score'])['label']
                if best_label == 'positive': return 1
                elif best_label == 'negative': return -1
                else: return 0
    except: pass
    return analizza_sentiment_nlp(testo)

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
    try:
        yt = yf.Ticker(ticker, session=sessione_sicura)
        for n in yt.news[:5]:
            if isinstance(n, dict):
                titolo = n.get('title') or n.get('headline') or ""
                if len(titolo) > 5: score_totale += analizza_sentiment_finbert(titolo)
    except: pass
    return score_totale

@st.cache_data(ttl=1800)
def scarica_news_livello_2(ticker, query_newsAPI):
    tutte_le_news = []
    score_totale = 0
    try:
        yt = yf.Ticker(ticker, session=sessione_sicura)
        for n in yt.news[:5]:
            if isinstance(n, dict):
                titolo = n.get('title') or n.get('headline') or ""
                if len(titolo) > 5: tutte_le_news.append({'title': titolo, 'source': 'Yahoo'})
    except: pass

    try:
        url_newsapi = f"https://newsapi.org/v2/everything?q={query_newsAPI}&language=it&sortBy=publishedAt&apiKey={API_KEY_NEWS}"
        risposta = requests.get(url_newsapi, timeout=5)
        if risposta.status_code == 200:
            for n in risposta.json().get('articles', [])[:5]:
                titolo = n.get('title') or ""
                if len(titolo) > 5 and "[Removed]" not in titolo: tutte_le_news.append({'title': titolo, 'source': 'NewsAPI'})
    except: pass

    news_pulite = pulisci_doppioni(tutte_le_news)
    lista_titoli_finali = []
    for news in news_pulite:
        titolo = news['title']
        score_totale += analizza_sentiment_finbert(titolo)
        lista_titoli_finali.append(f"[{news['source']}] {titolo}")
    return score_totale, lista_titoli_finali

# --- AGGIORNAMENTO 3: DATI FONDAMENTALI ---
@st.cache_data(ttl=86400)
def scarica_dati_fondamentali(ticker):
    try:
        yt = yf.Ticker(ticker, session=sessione_sicura)
        info = yt.info
        is_etf = info.get('quoteType') == 'ETF'
        if is_etf: return {"is_etf": True, "pe": None, "margin": None}
        pe_ratio = info.get('forwardPE', None) or info.get('trailingPE', None)
        margin = info.get('profitMargins', None)
        return {"is_etf": False, "pe": pe_ratio, "margin": margin}
    except: return {"is_etf": False, "pe": None, "margin": None}

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
    ticker_manuale = st.sidebar.text_input("Inserisci il Ticker Ufficiale:", "VWCE.DE")
    query_manuale = st.sidebar.text_input("Cosa cerco sui giornali?:", "Vanguard All-World")
    ticker_selezionato = ticker_manuale.upper() 
    query_selezionata = query_manuale
    nome_asset_mostrato = f"Asset Libero ({ticker_selezionato})"

st.sidebar.markdown("---")
st.sidebar.header("3. Il Tuo Portafoglio (Long)")
ha_posizione = st.sidebar.checkbox("Possiedo già questo asset")
prezzo_ingresso = st.sidebar.number_input("Prezzo medio di carico:", min_value=0.0, format="%.2f") if ha_posizione else 0.0

# --- MOTORE MATEMATICO ---
@st.cache_data(ttl=3600)
def check_global_market_health():
    try:
        sp500 = yf.Ticker("^GSPC", session=sessione_sicura).history(period="1y")
        vix = yf.Ticker("^VIX", session=sessione_sicura).history(period="1mo")
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
    try:
        dati = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=8, session=sessione_sicura)
        if dati.empty:
            yt = yf.Ticker(ticker, session=sessione_sicura)
            dati = yt.history(period="1y", interval="1d")
        if dati.empty: return pd.DataFrame()
        if isinstance(dati.columns, pd.MultiIndex): dati.columns = dati.columns.droplevel(1)
        dati = dati.loc[:, ~dati.columns.duplicated(keep='first')]
        return dati
    except: return pd.DataFrame()

def scarica_dati_weekly(ticker):
    try:
        dati_w = yf.download(ticker, period="3y", interval="1wk", progress=False, timeout=8, session=sessione_sicura)
        if dati_w.empty:
            yt = yf.Ticker(ticker, session=sessione_sicura)
            dati_w = yt.history(period="3y", interval="1wk")
        if dati_w.empty: return pd.DataFrame()
        if isinstance(dati_w.columns, pd.MultiIndex): dati_w.columns = dati_w.columns.droplevel(1)
        dati_w = dati_w.loc[:, ~dati_w.columns.duplicated(keep='first')]
        return dati_w
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def calcola_indicatori(df):
    if df.empty: return df
    close_data = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    high_data = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
    low_data = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
    
    # AGGIORNAMENTO 2: ATR
    atr_ind = AverageTrueRange(high=high_data, low=low_data, close=close_data, window=14)
    df['ATR'] = atr_ind.average_true_range()
    df['RSI'] = RSIIndicator(close=close_data, window=14).rsi()
    df['SMA_50'] = SMAIndicator(close=close_data, window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(close=close_data, window=200).sma_indicator()
    macd = MACD(close=close_data)
    df['MACD_Line'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['ADX'] = ADXIndicator(high=high_data, low=low_data, close=close_data, window=14).adx()
    return df.dropna()

def calcola_confidence_invest(prezzo, sma50, sma200, rsi, macd_l, macd_s, news_score, can_long, trend_settimanale_up, fondamentali):
    score_long = 0
    motivazioni = [] 
    tolleranza = 0.02 
    trend_primario_up = prezzo > sma200 * (1 + tolleranza)
    
    if prezzo > sma50 * (1 + tolleranza): 
        score_long += 1
        motivazioni.append("✅ Prezzo sopra media 50 giorni (+1)")
    elif prezzo < sma50 * (1 - tolleranza): 
        motivazioni.append("⚠️ Prezzo sotto media 50 giorni (Debolezza)")
    
    if (macd_l - macd_s) > 0: 
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
        motivazioni.append("📰 Sentiment IA (FinBERT) Positivo (+1)")
    elif news_score < 0:
        score_long -= 0.5
        motivazioni.append("⚠️ Sentiment IA (FinBERT) Negativo (-0.5)")
    
    if trend_primario_up: 
        score_long += 1.5
        motivazioni.append("📈 Trend 200 giorni al rialzo (+1.5)")
    if trend_settimanale_up: 
        score_long += 1
        motivazioni.append("📆 Trend Settimanale positivo (+1)")

    # Valutazione Fondamentali
    if fondamentali:
        if not fondamentali.get("is_etf"):
            pe = fondamentali.get("pe")
            margin = fondamentali.get("margin")
            if pe is not None:
                if 0 < pe < 25:
                    score_long += 0.5
                    motivazioni.append(f"💼 P/E Ratio attrattivo ({round(pe, 1)}) (+0.5)")
                elif pe > 40:
                    score_long -= 0.5
                    motivazioni.append(f"⚠️ P/E Ratio alto ({round(pe, 1)}), rischio bolla (-0.5)")
            if margin is not None and margin > 0.15:
                score_long += 0.5
                motivazioni.append("💰 Margini di profitto eccellenti (>15%) (+0.5)")
        else:
            motivazioni.append("📊 L'asset è un ETF (Dati di bilancio bypassati)")

    if not can_long: return 0, "⚠️ ACQUISTO SCONSIGLIATO (MACRO)", motivazioni
    if score_long >= 5.5: return score_long, "🟢 STRONG BUY", motivazioni
    if score_long >= 3.5: return score_long, "🟡 BUY / ACCUMULA", motivazioni
    if score_long >= 2.0: return score_long, "⚪ HOLD (Mantieni)", motivazioni
    return score_long, "🔴 ASSET DEBOLE", motivazioni

# --- ESECUZIONE GLOBALE E RADAR ---
global_status, vix_val, can_long = check_global_market_health()
st.info(f"🌍 **Barometro Macroeconomico S&P500/VIX:** {global_status} | Indice Volatilità (VIX): {round(vix_val, 2)}")

st.markdown("---")
st.subheader("🚨 Radar Occasioni di Investimento (Scanner Lazy Evaluation)")

if st.button("🔍 Scansiona il mercato", type="primary"):
    with st.spinner("Analisi in corso... (Ricerca veloce per tagliare gli asset deboli)"):
        classifica = []
        ctx = get_script_run_ctx()

        def analizza_singolo_asset(nome, ticker):
            add_script_run_ctx(threading.current_thread(), ctx)
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
                atr_val = df['ATR'].iloc[-1]
                
                # Calcolo Rischio Dinamico (ATR)
                stop_loss = p - (2 * atr_val)
                take_profit = p + (3 * atr_val)
                
                # Lazy Evaluation: calcolo preliminare senza news/fondamentali
                score_preliminare, _, _ = calcola_confidence_invest(p, sma50, sma200, rsi, macd_l, macd_s, 0, can_long, trend_w_up, None)
                
                n_score = 0
                fonds = None
                # Se l'asset è decente (score >= 2.0), approfondiamo. Altrimenti saltiamo e risparmiamo tempo!
                if score_preliminare >= 2.0:
                    n_score = scarica_news_livello_1(ticker)
                    fonds = scarica_dati_fondamentali(ticker)
                
                score, segnale, _ = calcola_confidence_invest(p, sma50, sma200, rsi, macd_l, macd_s, n_score, can_long, trend_w_up, fonds)
                
                if score >= 3.5: 
                    return {
                        "Asset": nome, 
                        "Score": round(score, 1), 
                        "Segnale": segnale,
                        "Prezzo": f"${round(p, 2)}",
                        "Stop Loss": f"${round(stop_loss, 2)}",
                        "Take Profit": f"${round(take_profit, 2)}"
                    }
                return None
            except: return None

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(analizza_singolo_asset, nome, ticker): nome for nome, ticker in asset_dict.items()}
            for future in as_completed(futures):
                res = future.result()
                if res: classifica.append(res)
        
        st.session_state['classifica_salvata'] = classifica
        
        if classifica:
            st.success("✅ Finito! Ecco le migliori occasioni (con Stop Loss e Take Profit automatici calcolati su ATR):")
            classifica_df = pd.DataFrame(classifica).sort_values(by="Score", ascending=False)
            st.table(classifica_df)
        else: st.warning("⚪ Al momento il sistema non trova occasioni 'Strong Buy'. Meglio conservare liquidità.")

st.markdown("---")

# --- DETTAGLIO SINGOLO ASSET ---
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
        atr_val = dati_attuali['ATR'].iloc[-1]
        
        stop_loss = p - (2 * atr_val)
        take_profit = p + (3 * atr_val)
        
        fonds = scarica_dati_fondamentali(ticker_selezionato)
        news_score, ultime_news = scarica_news_livello_2(ticker_selezionato, query_selezionata)
        
        score, segnale_base, motivazioni_decisione = calcola_confidence_invest(p, sma, sma200, rsi, macd_l, macd_s, news_score, can_long, trend_w_up, fonds)

        # 1. RISK MANAGEMENT
        st.write("### 🛡️ Gestione del Rischio (ATR Dinamico)")
        col_rm1, col_rm2, col_rm3 = st.columns(3)
        col_rm1.metric("Prezzo Attuale", f"${p}")
        col_rm2.metric("Stop Loss Suggerito", f"${round(stop_loss, 2)}", "- 2x ATR", delta_color="inverse")
        col_rm3.metric("Take Profit Suggerito", f"${round(take_profit, 2)}", "+ 3x ATR")
        
        st.markdown("---")

        if ha_posizione and prezzo_ingresso > 0:
            st.write("### 💼 Il Tuo Portafoglio")
            profitto_perdita = ((p - prezzo_ingresso) / prezzo_ingresso) * 100
            col_pos1, col_pos2 = st.columns(2)
            col_pos1.metric("Prezzo Medio di Carico:", f"{prezzo_ingresso}")
            col_pos2.metric("Rendimento Attuale:", f"{round(profitto_perdita, 2)}%", delta=f"{round(profitto_perdita, 2)}%")
            st.markdown("---")

        # 2. VERDETTO
        st.write("### 🧠 Valutazione del Sistema (Usa questo per decidere se comprare oggi)")
        st.info(f"🎯 **Consiglio Attuale:** {segnale_base} (Score: {score}/7.5 MAX)")
        
        with st.expander("🔎 Trasparenza IA: Parametri della decisione"):
            for motivazione in motivazioni_decisione:
                st.write(f"- {motivazione}")
                
        if not ha_posizione:
            allocazione_consigliata = capitale * allocazione_max
            st.metric("Budget Max Suggerito per Asset", f"${round(allocazione_consigliata, 2)}")

        st.markdown("---")

        # 3. CRUSCOTTO E GRAFICO PLOTLY
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Media 50 Giorni", f"{sma}")
        kpi2.metric("Media 200 Giorni", f"{sma200}")
        kpi3.metric("Valore RSI", rsi)
        kpi4.metric("Volatilità (ATR)", round(atr_val, 2))

        st.write("📊 **Grafico Professionale Interattivo (Candlestick)**")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=dati_attuali.index,
                        open=dati_attuali['Open'], high=dati_attuali['High'],
                        low=dati_attuali['Low'], close=dati_attuali['Close'],
                        name='Prezzo', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig.add_trace(go.Scatter(x=dati_attuali.index, y=dati_attuali['SMA_50'], line=dict(color='orange', width=2), name='SMA 50'))
        fig.add_trace(go.Scatter(x=dati_attuali.index, y=dati_attuali['SMA_200'], line=dict(color='royalblue', width=2), name='SMA 200'))
        
        fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0), height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📰 Sentiment IA e Ultime Notizie")
        if ultime_news:
            for titolo in ultime_news: st.write(f"- **{titolo}**")
        else: st.write("Nessuna notizia rilevante trovata.")
else:
    st.error("⚠️ Ticker non trovato o irraggiungibile.")
