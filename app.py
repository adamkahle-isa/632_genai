import contextlib as _cl
import time
from datetime import datetime

import numpy as np
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    import anthropic
    _ANT_KEY  = st.secrets.get("ANTHROPIC_API_KEY", "")
    CHATBOT_ON = bool(_ANT_KEY)
except Exception:
    CHATBOT_ON = False
    _ANT_KEY  = ""

st.set_page_config(page_title="FinResearch Terminal", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""<style>
:root {
  --cp:#1A56DB; --cpos:#057A55; --cneg:#C81E1E; --cwarn:#B45309;
  --cneu:#6B7280; --cbg:#FFFFFF; --csub:#F9FAFB; --cbor:#E5E7EB;
  --ctx:#111827; --ctxs:#4B5563; --ctxm:#9CA3AF; --r:5px;
}
/* hide Streamlit chrome so nothing clips the content */
header[data-testid="stHeader"]{display:none!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
footer{display:none!important}
#MainMenu{visibility:hidden!important}
.stApp{background:var(--cbg)!important}
.main .block-container{padding-top:1.5rem;padding-left:2.5rem;padding-right:2.5rem;max-width:1440px}
h1{font-size:20px!important;font-weight:700;margin-bottom:2px!important;color:var(--ctx)}
h3{font-size:11px!important;font-weight:800;letter-spacing:.8px;text-transform:uppercase;
   color:var(--ctxm);border-bottom:1px solid var(--cbor);padding-bottom:5px;margin:14px 0 8px 0}
hr{border-color:var(--cbor);margin:10px 0}
p{font-size:13px;color:var(--ctxs);line-height:1.5}

/* metric card */
.mc{background:var(--cbg);border:1px solid var(--cbor);border-radius:var(--r);padding:11px 14px;height:100%}
.mc-pos{border-top:2px solid var(--cpos)}.mc-neg{border-top:2px solid var(--cneg)}
.mc-warn{border-top:2px solid var(--cwarn)}.mc-neu{border-top:2px solid var(--cp)}
.mc-lbl{font-size:10px;font-weight:700;letter-spacing:.9px;text-transform:uppercase;color:var(--ctxm);margin-bottom:4px}
.mc-val{font-size:20px;font-weight:700;color:var(--ctx);line-height:1.15}
.mc-delta{font-size:12px;font-weight:600;margin-top:3px}
.mc-ctx{font-size:11px;color:var(--ctxs);margin-top:2px;line-height:1.35}
.dpos{color:var(--cpos)}.dneg{color:var(--cneg)}.dwarn{color:var(--cwarn)}.dneu{color:var(--cneu)}

/* 4-quadrant executive brief */
.eq-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin:8px 0 16px 0}
.eqc{background:var(--csub);border-radius:var(--r);border:1px solid var(--cbor);padding:13px 15px}
.eqc.epos{border-top:3px solid var(--cpos)}.eqc.eneg{border-top:3px solid var(--cneg)}
.eqc.ewarn{border-top:3px solid var(--cwarn)}.eqc.eneu{border-top:3px solid var(--cp)}
.eq-lbl{font-size:9px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;color:var(--ctxm);margin-bottom:5px}
.eq-head{font-size:14px;font-weight:700;color:var(--ctx);margin-bottom:3px;line-height:1.25}
.eq-data{font-size:11px;color:var(--ctxs);line-height:1.5}
.eq-sig{font-size:11px;font-weight:600;margin-top:6px}

/* narrative box */
.nb{border-left:3px solid var(--cp);background:#EEF2FF;padding:9px 14px;
    border-radius:0 var(--r) var(--r) 0;margin:6px 0 10px 0}
.nb.nbpos{border-color:var(--cpos);background:#ECFDF5}
.nb.nbneg{border-color:var(--cneg);background:#FEF2F2}
.nb.nbwarn{border-color:var(--cwarn);background:#FFFBEB}
.nb-h{font-size:13px;font-weight:600;color:var(--ctx)}
.nb-b{font-size:11px;color:var(--ctxs);margin-top:2px;line-height:1.4}

/* semantic block */
.blk{background:var(--csub);border-radius:var(--r);border:1px solid var(--cbor);padding:12px 14px;margin-bottom:8px}
.blk-lbl{font-size:10px;font-weight:800;letter-spacing:1px;text-transform:uppercase;
          color:var(--ctxm);margin-bottom:9px;padding-bottom:5px;border-bottom:1px solid var(--cbor)}

/* badge */
.bdg{display:inline-block;padding:2px 6px;border-radius:9px;font-size:10px;font-weight:700}
.bpos{background:#D1FAE5;color:#065F46}.bneg{background:#FEE2E2;color:#991B1B}
.bwarn{background:#FEF3C7;color:#92400E}.bneu{background:#E0E7FF;color:#3730A3}

/* decision card */
.dc{background:#1A2332;color:white;border-radius:var(--r);padding:14px 18px;margin-bottom:12px}
.dc h4{color:white!important;font-size:11px!important;letter-spacing:1px;text-transform:uppercase;
        border:none!important;margin:0 0 8px 0!important}
.di{font-size:12px;color:#CBD5E1;padding:4px 0;border-bottom:1px solid #2D3E55;line-height:1.45}
.di:last-child{border:none}.di strong{color:white}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:2px solid var(--cbor)}
.stTabs [data-baseweb="tab"]{font-size:12px;font-weight:600;letter-spacing:.3px;
    padding:7px 18px;border-radius:4px 4px 0 0;color:var(--ctxs)}
[aria-selected="true"]{color:var(--cp)!important;border-bottom:2px solid var(--cp)!important}
[data-testid="metric-container"]{display:none!important}
[data-baseweb="tab-panel"]{overflow:visible!important;min-height:0!important}
</style>""", unsafe_allow_html=True)

# ── Chart defaults ────────────────────────────────────────────────────────────
_CM = dict(t=10, b=12, l=8, r=8)
_BASE = dict(paper_bgcolor="white", plot_bgcolor="#FAFAFA",
             font=dict(size=11, color="#4B5563"),
             xaxis=dict(gridcolor="#F3F4F6", linecolor="#E5E7EB", automargin=True),
             yaxis=dict(gridcolor="#F3F4F6", linecolor="#E5E7EB", automargin=True),
             hovermode="x unified", legend=dict(orientation="h", y=1.06, font=dict(size=11)))
SM  = dict(**_BASE, height=200, margin=_CM)
MD  = dict(**_BASE, height=280, margin=_CM)
LG  = dict(**_BASE, height=360, margin=dict(t=10,b=16,l=8,r=8))

C_P = "#1A56DB"; C_POS="#057A55"; C_NEG="#C81E1E"; C_WARN="#B45309"
C_BN= "#9CA3AF"; C_OR="#D97706"
FP  = "rgba(26,86,219,0.09)"; FN="rgba(200,30,30,0.12)"

# ── Constants ─────────────────────────────────────────────────────────────────
FRED_API_KEY  = st.secrets["FRED-API"]
FRED_BASE     = "https://api.stlouisfed.org/fred"

SECTOR_FRED: dict[str,list[str]] = {
    "Financial Services":["FEDFUNDS","DGS10","T10Y2Y"],
    "Financials":        ["FEDFUNDS","DGS10","T10Y2Y"],
    "Real Estate":       ["MORTGAGE30US","CSUSHPISA","HOUST"],
    "Consumer Cyclical": ["CPIAUCSL","RSAFS","UMCSENT"],
    "Consumer Defensive":["CPIAUCSL","RSAFS","UMCSENT"],
    "Consumer Discretionary":["CPIAUCSL","RSAFS","UMCSENT"],
    "Consumer Staples":  ["CPIAUCSL","RSAFS","UMCSENT"],
    "Energy":            ["DCOILWTICO","GASREGW","INDPRO"],
    "Industrials":       ["INDPRO","PAYEMS","DGORDER"],
    "Technology":        ["FEDFUNDS","INDPRO","BAMLH0A0HYM2"],
    "Healthcare":        ["CPIAUCSL","PCE","PAYEMS"],
    "Utilities":         ["FEDFUNDS","CPIAUCSL"],
    "Communication Services":["FEDFUNDS","INDPRO","UMCSENT"],
    "Basic Materials":   ["INDPRO","PAYEMS","DGORDER"],
    "Materials":         ["INDPRO","PAYEMS","DGORDER"],
}
FRED_DEFAULT = ["FEDFUNDS","DGS10","INDPRO","CPIAUCSL"]
FRED_LBL: dict[str,str] = {
    "FEDFUNDS":"Fed Funds Rate","DGS10":"10Y Treasury Yield","T10Y2Y":"10Y–2Y Spread",
    "MORTGAGE30US":"30Y Mortgage Rate","CSUSHPISA":"Case-Shiller HPI","HOUST":"Housing Starts",
    "CPIAUCSL":"CPI","RSAFS":"Retail Sales","UMCSENT":"Consumer Sentiment",
    "DCOILWTICO":"WTI Crude Oil","GASREGW":"Gasoline Price","INDPRO":"Industrial Production",
    "PAYEMS":"Nonfarm Payrolls","DGORDER":"Durable Goods Orders",
    "BAMLH0A0HYM2":"HY OAS","PCE":"PCE",
}
SECTOR_PEERS: dict[str,list[str]] = {
    "Technology":        ["AAPL","MSFT","GOOGL","META","NVDA","AMZN","CRM","ORCL"],
    "Financial Services":["JPM","BAC","WFC","GS","MS","C","BLK","AXP"],
    "Financials":        ["JPM","BAC","WFC","GS","MS","C","BLK","AXP"],
    "Healthcare":        ["JNJ","UNH","PFE","MRK","ABBV","TMO","ABT","CVS"],
    "Consumer Cyclical": ["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW"],
    "Consumer Defensive":["PG","KO","PEP","WMT","COST","PM","CL","GIS"],
    "Consumer Discretionary":["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW"],
    "Consumer Staples":  ["PG","KO","PEP","WMT","COST","PM","CL","GIS"],
    "Energy":            ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO"],
    "Industrials":       ["HON","UPS","BA","CAT","DE","MMM","LMT","RTX"],
    "Real Estate":       ["AMT","PLD","CCI","EQIX","PSA","O","DLR","SPG"],
    "Utilities":         ["NEE","DUK","SO","D","AEP","SRE","EXC","XEL"],
    "Communication Services":["GOOGL","META","NFLX","DIS","CMCSA","T","VZ","TMUS"],
    "Basic Materials":   ["LIN","APD","ECL","NEM","FCX","NUE","VMC","ALB"],
    "Materials":         ["LIN","APD","ECL","NEM","FCX","NUE","VMC","ALB"],
}

# ── Data layer ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600,show_spinner=False)
def get_info(t):
    try: return yf.Ticker(t).info or {}
    except Exception as e: return {"_error":str(e)}

@st.cache_data(ttl=3600,show_spinner=False)
def get_hist(t,period="1y",interval="1d"):
    try: return yf.Ticker(t).history(period=period,interval=interval)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600,show_spinner=False)
def get_fin(t):
    try:
        tk=yf.Ticker(t)
        return {"Income Statement":tk.income_stmt,"Balance Sheet":tk.balance_sheet,"Cash Flow":tk.cash_flow}
    except: return {}

@st.cache_data(ttl=3600,show_spinner=False)
def get_qinc(t):
    try: return yf.Ticker(t).quarterly_income_stmt
    except: return pd.DataFrame()

@st.cache_data(ttl=3600,show_spinner=False)
def get_div(t):
    try: tk=yf.Ticker(t); return {"dividends":tk.dividends,"splits":tk.splits}
    except: return {}

@st.cache_data(ttl=3600,show_spinner=False)
def get_holders(t):
    try:
        tk=yf.Ticker(t)
        return {"major":tk.major_holders,"institutional":tk.institutional_holders,"mutualfund":tk.mutualfund_holders}
    except: return {}

@st.cache_data(ttl=3600,show_spinner=False)
def get_recs(t):
    try:
        df=yf.Ticker(t).recommendations
        return df if df is not None else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600,show_spinner=False)
def get_exps(t):
    try: return yf.Ticker(t).options or ()
    except: return ()

@st.cache_data(ttl=3600,show_spinner=False)
def get_chain(t,exp):
    try: ch=yf.Ticker(t).option_chain(exp); return ch.calls,ch.puts
    except: return pd.DataFrame(),pd.DataFrame()

@st.cache_data(ttl=3600,show_spinner=False)
def get_news(t):
    try: return yf.Ticker(t).news or []
    except: return []

@st.cache_data(ttl=86400,show_spinner=False)
def get_fred(sid,start):
    p={"series_id":sid,"api_key":FRED_API_KEY,"file_type":"json","observation_start":start}
    for attempt in range(3):
        try:
            r=requests.get(f"{FRED_BASE}/series/observations",params=p,timeout=10)
            r.raise_for_status()
            df=pd.DataFrame(r.json().get("observations",[]))[["date","value"]]
            df["date"]=pd.to_datetime(df["date"])
            df["value"]=pd.to_numeric(df["value"],errors="coerce")
            return df.dropna(subset=["value"]).set_index("date").rename(columns={"value":sid})
        except requests.HTTPError:
            if r.status_code==429: time.sleep(2**attempt)
            else: return pd.DataFrame()
        except: return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=3600,show_spinner=False)
def get_peers(tups):
    rows=[]
    for t in tups:
        try:
            i=yf.Ticker(t).info or {}
            rows.append({"Ticker":t,"Name":(i.get("shortName") or t)[:18],
                "Mkt Cap":i.get("marketCap"),"P/E":i.get("trailingPE"),
                "Fwd P/E":i.get("forwardPE"),"EV/EBITDA":i.get("enterpriseToEbitda"),
                "P/B":i.get("priceToBook"),"ROE":(i.get("returnOnEquity") or 0)*100,
                "Gross Mgn":(i.get("grossMargins") or 0)*100,
                "Op Mgn":(i.get("operatingMargins") or 0)*100,
                "Div Yld":(i.get("dividendYield") or 0)*100})
        except: pass
    return pd.DataFrame(rows)

# ── Analytics ─────────────────────────────────────────────────────────────────
def calc_risk(rets):
    if rets is None or rets.empty: return {}
    cum=( 1+rets).cumprod()
    ar=float((1+rets.mean())**252-1); av=float(rets.std()*np.sqrt(252))
    sh=ar/av if av else np.nan
    v95=float(np.percentile(rets,5)); cv95=float(rets[rets<=v95].mean())
    rm=cum.cummax(); dd=(cum-rm)/rm; md=float(dd.min())
    de=dd.idxmin(); post=cum[cum.index>de]; hwm=float(rm.loc[de])
    rec=post[post>=hwm]; rd=int((rec.index[0]-de).days) if not rec.empty else None
    return {"ar":ar,"av":av,"sh":float(sh),"v95":v95,"cv95":cv95,
            "md":md,"rd":rd,"dd":dd,"cum":cum-1}

def roll_beta(rets,bench,w=63):
    al=pd.concat([rets,bench],axis=1,join="inner"); al.columns=["s","b"]
    return (al["s"].rolling(w).cov(al["b"])/al["b"].rolling(w).var()).dropna()

def macro_corr(rets,fred_dict):
    r=rets.copy()
    if r.index.tz is not None:
        r.index=r.index.tz_localize(None)
    mo=None
    for freq in ["ME","M"]:
        try: mo=r.resample(freq).apply(lambda x:(1+x).prod()-1); break
        except: continue
    if mo is None or mo.empty:
        return {}
    mo_p=mo.copy(); mo_p.index=mo_p.index.to_period("M")
    out={}
    for sid,df in fred_dict.items():
        try:
            fm=df.resample("ME").last().pct_change().dropna().squeeze()
            if fm.empty: fm=df.resample("M").last().pct_change().dropna().squeeze()
            fm_p=fm.copy(); fm_p.index=fm_p.index.to_period("M")
            al=pd.concat([mo_p,fm_p],axis=1,join="inner").dropna()
            if len(al)>=6: out[sid]=float(al.corr().iloc[0,1])
        except: pass
    return out

def factor_reg(rets,spy_rets,dgs10=None):
    al=pd.concat([rets,spy_rets],axis=1,join="inner").dropna(); al.columns=["s","b"]
    if len(al)<30: return {}
    X=np.column_stack([np.ones(len(al)),al["b"].values])
    b,*_=np.linalg.lstsq(X,al["s"].values,rcond=None)
    yh=X@b; ssr=np.sum((al["s"].values-yh)**2); sst=np.sum((al["s"].values-al["s"].mean())**2)
    out={"alpha":float(b[0])*252,"beta":float(b[1]),"r2":float(1-ssr/sst) if sst else 0}
    if dgs10 is not None and not dgs10.empty:
        try:
            for freq in ["ME","M"]:
                try: ms=rets.resample(freq).apply(lambda x:(1+x).prod()-1); rc=dgs10.resample(freq).last().diff().dropna().squeeze(); break
                except: continue
            am=pd.concat([ms,rc],axis=1,join="inner").dropna()
            if len(am)>=12:
                Xr=np.column_stack([np.ones(len(am)),am.iloc[:,1].values])
                br,*_=np.linalg.lstsq(Xr,am.iloc[:,0].values,rcond=None)
                out["rate_beta"]=float(br[1])
        except: pass
    return out

def monte_carlo(price,ar,av,days=252,sims=1000,seed=42):
    np.random.seed(seed); dt=1/252
    dr=np.exp((ar-.5*av**2)*dt+av*np.sqrt(dt)*np.random.normal(0,1,(days,sims)))
    paths=np.ones((days+1,sims))*price
    for t in range(1,days+1): paths[t]=paths[t-1]*dr[t-1]
    f=paths[-1]
    return {"paths":paths,"pcts":np.percentile(paths,[5,25,50,75,95],axis=1),"finals":f,
            "ploss":float((f<price).mean()),"pup10":float((f>price*1.10).mean()),
            "eret":float((f/price-1).mean()),"mret":float(np.median(f/price-1)),
            "p5":float(np.percentile(f/price-1,5)),"p95":float(np.percentile(f/price-1,95))}

def sim_portfolios(ret_df,n=2000):
    mr=ret_df.mean()*252; cv=ret_df.cov()*252; na=len(ret_df.columns)
    rows=[]
    for _ in range(n):
        w=np.random.dirichlet(np.ones(na)); r=float(np.dot(w,mr))
        v=float(np.sqrt(w@cv.values@w)); rows.append({"vol":v,"ret":r,"sh":r/v if v else 0,"w":w.tolist()})
    return pd.DataFrame(rows)

def risk_contrib(w,cov):
    pv=np.sqrt(w@cov@w); mrc=cov@w; rc=w*mrc/pv; return rc/rc.sum()

def component_var95(w,daily_cov):
    z=1.645; pv=np.sqrt(w@daily_cov@w)
    return z*w*(daily_cov@w)/pv

def sortino_r(rets):
    ar=float((1+rets.mean())**252-1)
    dd=float(rets[rets<0].std()*np.sqrt(252))
    return ar/dd if dd>0 else np.nan

def calmar_r(rets):
    ar=float((1+rets.mean())**252-1)
    cum=(1+rets).cumprod(); md=float((cum/cum.cummax()-1).min())
    return ar/abs(md) if md<0 else np.nan

def omega_r(rets,thr=0.0):
    ex=rets-thr/252; denom=abs(ex[ex<0].sum())
    return float(ex[ex>0].sum()/denom) if denom>0 else np.nan

def rolling_div_ratio(ret_df,w_eq,window=63):
    rows=[]
    for i in range(window,len(ret_df)+1):
        cv60=ret_df.iloc[i-window:i].cov()*252
        pv=float(np.sqrt(w_eq@cv60.values@w_eq))
        wv=float((np.sqrt(np.diag(cv60.values))*w_eq).sum())
        rows.append({"date":ret_df.index[i-1],"dr":wv/pv if pv>0 else np.nan})
    return pd.DataFrame(rows).set_index("date")["dr"].dropna()

def port_full_stats(pr,w,annual_cov):
    ar=float((1+pr.mean())**252-1); av=float(pr.std()*np.sqrt(252))
    sh=ar/av if av else np.nan
    so=sortino_r(pr); ca=calmar_r(pr); om=omega_r(pr)
    cum=(1+pr).cumprod(); md=float((cum/cum.cummax()-1).min())
    var95=float(np.percentile(pr,5)); cvar95=float(pr[pr<=var95].mean())
    rc=risk_contrib(w,annual_cov)
    dr=float((np.sqrt(np.diag(annual_cov))*w).sum()/np.sqrt(w@annual_cov@w))
    return dict(ar=ar,av=av,sh=sh,so=so,ca=ca,om=om,md=md,var95=var95,cvar95=cvar95,rc=rc,dr=dr)

def atm_iv(ticker,price,exps):
    if not exps or not price: return None
    try:
        calls,_=get_chain(ticker,exps[0])
        if calls is None or calls.empty: return None
        iv=calls.loc[(calls["strike"]-price).abs().idxmin(),"impliedVolatility"]
        return float(iv) if pd.notnull(iv) else None
    except: return None

# ── Formatting ────────────────────────────────────────────────────────────────
def fm(n):
    if n is None: return "N/A"
    try: n=float(n)
    except: return "N/A"
    if np.isnan(n): return "N/A"
    for t,s in [(1e12,"T"),(1e9,"B"),(1e6,"M"),(1e3,"K")]:
        if abs(n)>=t: return f"${n/t:.2f}{s}"
    return f"${n:.2f}"

def fp(v):
    if v is None: return "N/A"
    try: return f"{float(v)*100:.2f}%"
    except: return "N/A"

def fx(v,d=1):
    if v is None: return "N/A"
    try: return f"{float(v):.{d}f}x"
    except: return "N/A"

def fmt_stmt(df):
    out=df.copy()
    for c in out.columns: out[c]=out[c].apply(lambda x: fm(x) if pd.notnull(x) else "N/A")
    return out

def norm100(r1,r2):
    al=pd.concat([r1,r2],axis=1,join="inner"); al.columns=["a","b"]
    return (1+al["a"]).cumprod()*100,(1+al["b"]).cumprod()*100

# ── UI components ─────────────────────────────────────────────────────────────
def mcard(lbl,val,delta=None,ctx=None,lvl="neu"):
    arrow={"pos":"▲","neg":"▼","warn":"→","neu":"→"}.get(lvl,"→")
    dcls ={"pos":"dpos","neg":"dneg","warn":"dwarn","neu":"dneu"}.get(lvl,"dneu")
    bcls =f"mc-{lvl}" if lvl in("pos","neg","warn") else "mc-neu"
    dh   =f'<div class="mc-delta {dcls}">{arrow} {delta}</div>' if delta else ""
    ch   =f'<div class="mc-ctx">{ctx}</div>' if ctx else ""
    return f'<div class="mc {bcls}"><div class="mc-lbl">{lbl}</div><div class="mc-val">{val}</div>{dh}{ch}</div>'

def mcols(cards,cols=None):
    n=cols or len(cards); w=f"repeat({n},1fr)"
    st.markdown(f'<div style="display:grid;grid-template-columns:{w};gap:8px;margin-bottom:8px;">{"".join(cards)}</div>',
                unsafe_allow_html=True)

def blk(lbl,cards,ncols=3):
    w=f"repeat({ncols},1fr)"
    inner="".join(cards)
    st.markdown(f'<div class="blk"><div class="blk-lbl">{lbl}</div>'
                f'<div style="display:grid;grid-template-columns:{w};gap:8px;">{inner}</div></div>',
                unsafe_allow_html=True)

def nb(head,body,lvl=""):
    cls={"pos":"nbpos","neg":"nbneg","warn":"nbwarn"}.get(lvl,"")
    st.markdown(f'<div class="nb {cls}"><div class="nb-h">{head}</div><div class="nb-b">{body}</div></div>',
                unsafe_allow_html=True)

def bdg(txt,lvl="neu"):
    cls={"pos":"bpos","neg":"bneg","warn":"bwarn","neu":"bneu"}.get(lvl,"bneu")
    return f'<span class="bdg {cls}">{txt}</span>'

def decision_card(title,items):
    rows="".join(f'<div class="di">{i}</div>' for i in items)
    st.markdown(f'<div class="dc"><h4>{title}</h4>{rows}</div>',unsafe_allow_html=True)

def render_4q(perf,risk,drivers,valuation):
    def qcard(lbl,head,data,sig,lvl):
        cls={"pos":"epos","neg":"eneg","warn":"ewarn","neu":"eneu"}.get(lvl,"eneu")
        sc ={"pos":C_POS,"neg":C_NEG,"warn":C_WARN,"neu":C_P}.get(lvl,C_P)
        return (f'<div class="eqc {cls}"><div class="eq-lbl">{lbl}</div>'
                f'<div class="eq-head">{head}</div><div class="eq-data">{data}</div>'
                f'<div class="eq-sig" style="color:{sc}">→ {sig}</div></div>')
    html='<div class="eq-grid">'+"".join([qcard(**q) for q in [perf,risk,drivers,valuation]])+'</div>'
    st.markdown(html,unsafe_allow_html=True)

# ── Executive brief builder ───────────────────────────────────────────────────
def build_4q(info,rm,spy_rm,rb,corr,peers_df,ticker,mc):
    # Performance quadrant
    if rm and spy_rm and rm.get("ar") and spy_rm.get("ar"):
        diff=rm["ar"]-spy_rm["ar"]
        plvl="pos" if diff>0 else "neg"
        perf=dict(lbl="Performance",
                  head=f"{'Beats' if diff>0 else 'Trails'} SPY {abs(diff)*100:.1f}pp/yr",
                  data=f"Sharpe {rm['sh']:.2f} vs SPY {spy_rm['sh']:.2f} · Ann. return {rm['ar']*100:.1f}%",
                  sig="Risk-adjusted outperformance confirmed" if diff>0 and rm["sh"]>spy_rm["sh"] else "Underperforming on risk-adjusted basis",
                  lvl=plvl)
    else:
        perf=dict(lbl="Performance",head="Insufficient data",data="—",sig="Load more history",lvl="neu")

    # Risk quadrant
    if rm:
        rlvl="warn" if rm.get("md",0)<-0.25 else "neu"
        rb_str=""
        if rb is not None and not rb.empty:
            bv=float(rb.iloc[-1])
            if not np.isnan(bv): rb_str=f" · β={bv:.2f}"
        risk_q=dict(lbl="Risk",
                    head=f"Max DD {rm.get('md',0)*100:.1f}%",
                    data=f"VaR 95% {abs(rm.get('v95',0))*100:.2f}% · CVaR {abs(rm.get('cv95',0))*100:.2f}%{rb_str}",
                    sig="Elevated drawdown profile — monitor position size" if rm.get("md",0)<-0.25 else "Drawdown within normal range",
                    lvl=rlvl)
    else:
        risk_q=dict(lbl="Risk",head="—",data="—",sig="—",lvl="neu")

    # Drivers quadrant
    valid={k:v for k,v in corr.items() if not np.isnan(v)}
    if valid:
        top=max(valid.items(),key=lambda x:abs(x[1]))
        lbl_=FRED_LBL.get(top[0],top[0]); rho=top[1]
        rate_keys={"DGS10","FEDFUNDS","T10Y2Y","MORTGAGE30US"}
        if top[0] in rate_keys and rho<-0.3: sig_d="Rate cuts = tailwind for this position"
        elif top[0] in rate_keys and rho>0.3: sig_d="Rate cuts = headwind — unusual rate sensitivity"
        else: sig_d=f"Monitor {lbl_} as leading indicator"
        drv=dict(lbl="Macro Driver",head=f"{lbl_} (ρ={rho:.2f})",
                 data=f"{'Negatively' if rho<0 else 'Positively'} correlated · monthly returns",
                 sig=sig_d,lvl="warn" if abs(rho)>0.5 else "neu")
    else:
        drv=dict(lbl="Macro Driver",head="No significant correlation",data="—",sig="Macro-independent",lvl="neu")

    # Valuation quadrant
    pe=info.get("trailingPE")
    if peers_df is not None and not peers_df.empty and pe and not np.isnan(float(pe)):
        peer_pes=peers_df[peers_df["Ticker"]!=ticker]["P/E"].dropna()
        if not peer_pes.empty:
            med=peer_pes.median(); prem=(float(pe)-med)/med
            fcf=info.get("freeCashflow"); mc_=info.get("marketCap")
            fcfy=f" · FCF yield {fcf/mc_*100:.1f}%" if fcf and mc_ else ""
            vlvl="warn" if prem>0.3 else "pos" if prem<-0.15 else "neu"
            val_q=dict(lbl="Valuation",
                       head=f"P/E {float(pe):.1f}x — {abs(prem)*100:.0f}% {'premium' if prem>0 else 'discount'}",
                       data=f"vs peer median {med:.1f}x{fcfy}",
                       sig="Premium valuation requires earnings delivery" if prem>0.3 else "Discount to peers — potential value opportunity" if prem<-0.15 else "In-line with peers",
                       lvl=vlvl)
        else: val_q=dict(lbl="Valuation",head=f"P/E {float(pe):.1f}x",data="—",sig="No peer data",lvl="neu")
    else: val_q=dict(lbl="Valuation",head="—",data="—",sig="—",lvl="neu")

    return perf,risk_q,drv,val_q

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 FinResearch")
    st.markdown("---")
    raw=st.text_input("Tickers (comma-separated)","AAPL, MSFT",help="e.g. AAPL, MSFT, JPM")
    tickers=[t.strip().upper() for t in raw.split(",") if t.strip()]
    st.markdown("### History")
    period  =st.selectbox("Period",["1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"],index=3)
    interval=st.selectbox("Interval",["1d","1wk","1mo"],index=0)
    st.markdown("### Monte Carlo")
    mc_sims=st.slider("Simulations",200,2000,1000,100)
    mc_days=st.slider("Horizon (trading days)",63,504,252,21)
    st.markdown("### FRED")
    fred_start=st.date_input("Data start",value=datetime(2010,1,1))
    st.markdown("---")
    st.caption("Yahoo Finance · FRED (St. Louis Fed)")

if not tickers:
    st.warning("Enter at least one ticker in the sidebar.")
    st.stop()

# ── Portfolio-level data ──────────────────────────────────────────────────────
all_rets:dict[str,pd.Series]={}
for t in tickers:
    h=get_hist(t,period,interval)
    if not h.empty: all_rets[t]=h["Close"].pct_change().dropna()

# ── Outer navigation tabs ─────────────────────────────────────────────────────
if len(tickers)>1:
    _outer_labels=[f"🏢 {t}" for t in tickers]+["📊 Portfolio"]
    _outer_tabs=st.tabs(_outer_labels)
    _ticker_ctxs=_outer_tabs[:-1]
else:
    _outer_tabs=None
    _ticker_ctxs=[_cl.nullcontext()]

for ticker,_tab_ctx in zip(tickers,_ticker_ctxs):
  with _tab_ctx:
    with st.spinner(f"Loading {ticker}…"):
        info=get_info(ticker)

    if info.get("_error") or not info.get("symbol"):
        st.error(f"Could not load **{ticker}**: {info.get('_error','not found')}")
        info={}

    name   =info.get("longName") or info.get("shortName") or ticker
    sector =info.get("sector","")
    industry=info.get("industry","")
    price  =info.get("currentPrice") or info.get("regularMarketPrice")
    prev_c =info.get("regularMarketPreviousClose")
    chg    =f"{(price/prev_c-1)*100:+.2f}%" if price and prev_c and prev_c>0 else ""

    # Header
    hcol1,hcol2=st.columns([4,1])
    with hcol1:
        st.markdown(f"# {name} &nbsp;<span style='font-size:14px;color:#6B7280;font-weight:400'>{ticker}</span>",unsafe_allow_html=True)
        if sector or industry:
            st.caption(f"**{sector}** · {industry}")
    with hcol2:
        if price:
            pc=C_POS if chg.startswith("+") else C_NEG
            st.markdown(f'<div style="text-align:right;padding-top:8px"><span style="font-size:26px;font-weight:700;color:#111827">${price:,.2f}</span><br><span style="font-size:14px;font-weight:600;color:{pc}">{chg}</span></div>',unsafe_allow_html=True)

    # ── Shared computation ────────────────────────────────────────────────────
    hist    =get_hist(ticker,period,interval)
    spy_h   =get_hist("SPY",period,interval)
    rets    =hist["Close"].pct_change().dropna()   if not hist.empty      else pd.Series(dtype=float)
    spy_rets=spy_h["Close"].pct_change().dropna()  if not spy_h.empty    else pd.Series(dtype=float)
    rm      =calc_risk(rets)
    spy_rm  =calc_risk(spy_rets)
    rb      =roll_beta(rets,spy_rets,63) if not rets.empty and not spy_rets.empty and interval=="1d" else None

    peers_list=[p for p in SECTOR_PEERS.get(sector,[]) if p!=ticker][:4]
    all_peers =tuple([ticker]+peers_list)
    with st.spinner("Loading peer data…"):
        peers_df=get_peers(all_peers)

    fred_s=fred_start.strftime("%Y-%m-%d")
    s_ids =SECTOR_FRED.get(sector,FRED_DEFAULT)
    fred_d={sid:df for sid in s_ids if not (df:=get_fred(sid,fred_s)).empty}
    corr  =macro_corr(rets,fred_d) if not rets.empty else {}

    dgs10 =get_fred("DGS10",fred_s)
    freg  =factor_reg(rets,spy_rets,dgs10 if not dgs10.empty else None)

    mc={}
    if price and rm:
        mc=monte_carlo(float(price),rm["ar"],rm["av"],days=mc_days,sims=mc_sims)

    # ── Executive Brief ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Executive Brief")
    pq,rq,dq,vq=build_4q(info,rm,spy_rm,rb,corr,peers_df,ticker,mc)
    render_4q(pq,rq,dq,vq)

    # ── Section tabs ──────────────────────────────────────────────────────────
    T1,T2,T3,T4,T5=st.tabs([
        "① Overview","② Risk & Performance",
        "③ Macro & Factors","④ Valuation","⑤ Simulation"])

    # ════════════════════════════════════════════════════════════════════════
    # ① OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    with T1:
        # Normalized price vs SPY
        if not rets.empty and not spy_rets.empty:
            sn,bn=norm100(rets,spy_rets)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=sn.index,y=sn.values,name=ticker,
                line=dict(color=C_P,width=2.5),hovertemplate="%{y:.1f}"))
            fig.add_trace(go.Scatter(x=bn.index,y=bn.values,name="SPY",
                line=dict(color=C_BN,width=1.5,dash="dash"),hovertemplate="%{y:.1f}"))
            fig.add_hline(y=100,line_dash="dot",line_color="#D1D5DB",line_width=1)
            fig.update_layout(title=f"{ticker} vs SPY — Indexed to 100",**LG)
            st.plotly_chart(fig,use_container_width=True)
        elif not hist.empty:
            fig=go.Figure(go.Candlestick(x=hist.index,open=hist["Open"],high=hist["High"],
                low=hist["Low"],close=hist["Close"],name=ticker))
            fig.update_layout(**LG,xaxis_rangeslider_visible=False)
            st.plotly_chart(fig,use_container_width=True)

        # Key stats + news
        sc1,sc2=st.columns([1,1.3])
        with sc1:
            st.markdown("### Key Statistics")
            pe=info.get("trailingPE"); fpe=info.get("forwardPE")
            ev=info.get("enterpriseToEbitda"); pb=info.get("priceToBook")
            peg=info.get("pegRatio"); fcf=info.get("freeCashflow"); mc_=info.get("marketCap")
            st.dataframe(pd.DataFrame({
                "Metric":["P/E (TTM)","Fwd P/E","EV/EBITDA","P/B","PEG","FCF Yield",
                          "Revenue (TTM)","EBITDA","Total Debt","Total Cash",
                          "Profit Margin","Op. Margin","ROE","ROA",
                          "Exchange","Employees","Country"],
                "Value":[fx(pe),fx(fpe),fx(ev),fx(pb,2),fx(peg,2),
                         fp(fcf/mc_ if fcf and mc_ else None),
                         fm(info.get("totalRevenue")),fm(info.get("ebitda")),
                         fm(info.get("totalDebt")),fm(info.get("totalCash")),
                         fp(info.get("profitMargins")),fp(info.get("operatingMargins")),
                         fp(info.get("returnOnEquity")),fp(info.get("returnOnAssets")),
                         info.get("exchange","N/A"),
                         f"{info.get('fullTimeEmployees',0):,}" if info.get("fullTimeEmployees") else "N/A",
                         info.get("country","N/A")]}),
                use_container_width=True,hide_index=True,height=520)
        with sc2:
            st.markdown("### Business Summary")
            st.write(info.get("longBusinessSummary","No description available."))
            st.markdown("### Latest News")
            news=get_news(ticker)
            for a in news[:8]:
                ct=a.get("content",{})
                title=a.get("title") or ct.get("title","No title")
                url  =a.get("link")  or ct.get("canonicalUrl",{}).get("url","")
                pub  =a.get("publisher") or ct.get("provider",{}).get("displayName","")
                ts   =a.get("providerPublishTime") or ct.get("pubDate","")
                if isinstance(ts,(int,float)):
                    try: ts=datetime.fromtimestamp(ts).strftime("%b %d")
                    except: ts=""
                link_=f"[{title}]({url})" if url else title
                st.markdown(f"**{link_}** — {pub} · {ts}")

    # ════════════════════════════════════════════════════════════════════════
    # ② RISK & PERFORMANCE
    # ════════════════════════════════════════════════════════════════════════
    with T2:
        if rm:
            # Performance block
            spy_sh=spy_rm.get("sh",np.nan) if spy_rm else np.nan
            sh_diff=rm["sh"]-spy_sh if not np.isnan(spy_sh) else None
            sh_ctx=f"SPY: {spy_sh:.2f} · Δ {sh_diff:+.2f}" if sh_diff is not None else ""
            av_spy =spy_rm.get("av") if spy_rm else None
            blk("Performance",[
                mcard("Ann. Return",fp(rm["ar"]),
                      delta=f"{(rm['ar']-spy_rm['ar'])*100:+.1f}pp vs SPY" if spy_rm else None,
                      ctx=f"SPY: {fp(spy_rm['ar'])}" if spy_rm else None,
                      lvl="pos" if rm["ar"]>(spy_rm.get("ar",0) if spy_rm else 0) else "neg"),
                mcard("Sharpe Ratio",f"{rm['sh']:.2f}" if not np.isnan(rm["sh"]) else "N/A",
                      delta=f"{sh_diff:+.2f} vs SPY" if sh_diff else None,
                      ctx=sh_ctx,
                      lvl="pos" if sh_diff and sh_diff>0 else "neg"),
                mcard("Ann. Volatility",fp(rm["av"]),
                      delta=f"{(rm['av']-(av_spy or 0))*100:+.1f}pp vs SPY" if av_spy else None,
                      ctx=f"SPY: {fp(av_spy)}" if av_spy else None,
                      lvl="warn" if rm["av"]>0.30 else "neu"),
            ],ncols=3)

            # Risk block
            blk("Risk",[
                mcard("VaR 95% (1-day)",fp(rm["v95"]),
                      ctx="Worst expected daily loss at 95% confidence",lvl="warn"),
                mcard("CVaR 95% (1-day)",fp(rm["cv95"]),
                      ctx="Avg loss on days VaR is breached (tail severity)",lvl="neg"),
                mcard("Max Drawdown",fp(rm["md"]),
                      delta=f"Recovered in {rm['rd']}d" if rm.get("rd") else "Not yet recovered",
                      ctx="Peak-to-trough decline in this period",
                      lvl="neg" if rm["md"]<-0.30 else "warn"),
            ],ncols=3)

        # Cumulative return | Rolling beta
        cr1,cr2=st.columns(2)
        with cr1:
            st.markdown("### Cumulative Return vs SPY")
            if rm and not rets.empty:
                cum=rm["cum"]
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=cum.index,y=cum.values*100,name=ticker,
                    line=dict(color=C_P,width=2.5),hovertemplate="%{y:.2f}%"))
                if spy_rm and not spy_rets.empty:
                    bc=spy_rm["cum"]; idx=cum.index.intersection(bc.index)
                    fig.add_trace(go.Scatter(x=idx,y=bc.loc[idx].values*100,name="SPY",
                        line=dict(color=C_BN,width=1.5,dash="dash"),hovertemplate="%{y:.2f}%"))
                fig.add_hline(y=0,line_dash="dot",line_color="#D1D5DB",line_width=1)
                fig.update_layout(yaxis_title="Return (%)",**MD)
                st.plotly_chart(fig,use_container_width=True)
        with cr2:
            st.markdown("### Rolling 63-Day Beta vs SPY")
            if rb is not None and not rb.empty:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=rb.index,y=rb.values,
                    line=dict(color=C_OR,width=2),fill="tozeroy",fillcolor=FP,
                    hovertemplate="β=%{y:.2f}",name="Beta"))
                fig.add_hline(y=1,line_dash="dash",line_color="#9CA3AF",
                              annotation_text="Market β=1",annotation_position="right")
                fig.update_layout(yaxis_title="Beta",showlegend=False,**MD)
                st.plotly_chart(fig,use_container_width=True)
            else:
                nb("Rolling beta requires daily interval data.",
                   "Set interval to 1d in the sidebar to enable this chart.","warn")

        # Drawdown | Distribution
        dd1,dd2=st.columns(2)
        with dd1:
            st.markdown("### Underwater Equity Curve")
            if rm and "dd" in rm:
                dd=rm["dd"]
                fig=go.Figure(go.Scatter(x=dd.index,y=dd.values*100,
                    fill="tozeroy",fillcolor=FN,
                    line=dict(color=C_NEG,width=1.5),hovertemplate="%{y:.2f}%"))
                fig.update_layout(yaxis_title="Drawdown (%)",showlegend=False,**MD)
                st.plotly_chart(fig,use_container_width=True)
        with dd2:
            st.markdown("### Daily Return Distribution")
            if rm and not rets.empty:
                fig=px.histogram(rets*100,nbins=60,
                    color_discrete_sequence=[C_P],labels={"value":"Daily Return (%)"})
                if rm.get("v95"):
                    fig.add_vline(x=rm["v95"]*100,line_dash="dash",
                                  line_color=C_NEG,annotation_text="VaR 95%")
                fig.update_layout(showlegend=False,**MD)
                st.plotly_chart(fig,use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # ③ MACRO & FACTORS
    # ════════════════════════════════════════════════════════════════════════
    with T3:
        # Narrative: factor regression first
        st.markdown("### Factor Exposure")
        if freg:
            alpha_ann=freg.get("alpha",0); beta_=freg.get("beta",0); r2_=freg.get("r2",0)
            alpha_lvl="pos" if alpha_ann>0.02 else "neg" if alpha_ann<-0.02 else "neu"
            nb(f"Market Beta: {beta_:.2f} — {alpha_ann*100:+.1f}% annualized alpha",
               f"R² = {r2_:.3f} ({r2_*100:.1f}% of return variation explained by SPY). "
               f"{'Strong market driver — limited idiosyncratic returns.' if r2_>0.7 else 'Moderate market linkage — significant stock-specific factors at play.' if r2_>0.4 else 'Weak market correlation — primarily stock-specific.'}",
               lvl=alpha_lvl)
            if "rate_beta" in freg:
                rb_=freg["rate_beta"]; rb_lvl="warn" if abs(rb_)>0.02 else "neu"
                nb(f"Rate Sensitivity: {rb_:.4f} monthly return per 1pp change in 10Y yield",
                   f"{'Negatively' if rb_<0 else 'Positively'} correlated with rates. "
                   f"{'A 100bp rate rise implies ~{:.1f}% monthly headwind.'.format(abs(rb_)*100) if abs(rb_)>0.005 else 'Rate sensitivity is minimal.'}",
                   lvl=rb_lvl)
        else:
            nb("Insufficient data for factor regression.","Requires ≥30 overlapping observations with SPY.","warn")

        # Factor regression table
        if freg:
            reg_rows=[{"Factor":"Market (SPY)","Loading":f"{freg.get('beta',0):.3f}",
                       "Ann. Alpha":fp(freg.get("alpha")),"R²":f"{freg.get('r2',0):.3f}",
                       "Interpretation":"% market move → implied stock move"}]
            if "rate_beta" in freg:
                reg_rows.append({"Factor":"10Y Treasury (DGS10)","Loading":f"{freg['rate_beta']:.4f}",
                                  "Ann. Alpha":"—","R²":"—",
                                  "Interpretation":"Monthly return per 1pp rate change"})
            st.dataframe(pd.DataFrame(reg_rows),use_container_width=True,hide_index=True)

        st.markdown("### Macro Sensitivity")
        # Narrative for top correlation
        valid_c={k:v for k,v in corr.items() if not np.isnan(v)}
        if valid_c:
            top=max(valid_c.items(),key=lambda x:abs(x[1])); top_lbl=FRED_LBL.get(top[0],top[0])
            direction="falls" if top[1]<0 else "rises"
            nb(f"Key driver: {top_lbl} (ρ={top[1]:.2f})",
               f"When {top_lbl} rises, {ticker} monthly returns tend to {'decline' if top[1]<0 else 'increase'}. "
               f"{'This implies rate cuts are a structural tailwind for this position.' if top[0] in ('DGS10','FEDFUNDS','T10Y2Y') and top[1]<-0.3 else ''}",
               lvl="warn" if abs(top[1])>0.5 else "neu")

        mc_l,mc_r=st.columns([1,1])
        with mc_l:
            st.markdown("### Macro Correlation (Monthly Returns)")
            if corr:
                cd=(pd.DataFrame([{"Series":FRED_LBL.get(k,k),"ρ":v}
                                   for k,v in corr.items() if not np.isnan(v)])
                      .sort_values("ρ"))
                colors=[C_NEG if c<0 else C_POS for c in cd["ρ"]]
                fig=go.Figure(go.Bar(x=cd["ρ"],y=cd["Series"],orientation="h",
                    marker_color=colors,text=[f"{c:.2f}" for c in cd["ρ"]],textposition="outside"))
                fig.add_vline(x=0,line_color="#9CA3AF",line_width=1)
                fig.update_layout(xaxis=dict(range=[-1,1],title="Pearson ρ (monthly)"),
                                  margin=dict(t=10,b=10,l=165,r=30),showlegend=False,
                                  height=max(240,len(cd)*52),**{k:v for k,v in _BASE.items() if k not in ("margin","height","xaxis","yaxis")})
                st.plotly_chart(fig,use_container_width=True)
        with mc_r:
            st.markdown("### FRED Series — Trend & Latest")
            for sid,df_f in list(fred_d.items())[:3]:
                lbl_=FRED_LBL.get(sid,sid)
                latest=float(df_f.iloc[-1,0]); prev=float(df_f.iloc[-2,0]) if len(df_f)>1 else None
                delta_=f"{latest-prev:+.4f}" if prev else None
                lvl_="pos" if (delta_ and float(delta_)>0) else "neg" if delta_ else "neu"
                mcols([mcard(lbl_,str(round(latest,4)),delta=delta_,
                             ctx=df_f.index[-1].strftime("%Y-%m-%d"),lvl=lvl_)],cols=1)
                fig=px.line(df_f,color_discrete_sequence=[C_P],labels={sid:lbl_})
                fig.update_layout(height=120,showlegend=False,
                                  margin=dict(t=2,b=2,l=0,r=0),
                                  paper_bgcolor="white",plot_bgcolor="#FAFAFA",
                                  xaxis=dict(showgrid=False,showticklabels=False),
                                  yaxis=dict(gridcolor="#F3F4F6",showticklabels=True))
                st.plotly_chart(fig,use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # ④ VALUATION
    # ════════════════════════════════════════════════════════════════════════
    with T4:
        # Peer comparison with outlier highlighting
        st.markdown("### Peer Comparison — Valuation & Profitability")
        if not peers_df.empty:
            dp=peers_df.copy()
            dp["★"]=dp["Ticker"].apply(lambda t:"★" if t==ticker else "")
            dp["Mkt Cap"]=dp["Mkt Cap"].apply(fm)
            pe_med=peers_df["P/E"].dropna().median()
            def fmt_pe(v):
                if pd.isna(v) or v<=0: return "N/A"
                prem=(v-pe_med)/pe_med*100
                return f"{v:.1f}x  ({prem:+.0f}%)"
            dp["P/E"]=dp["P/E"].apply(fmt_pe)
            for c in ["Fwd P/E","EV/EBITDA","P/B"]:
                dp[c]=dp[c].apply(lambda x:f"{x:.1f}x" if pd.notnull(x) and x>0 else "N/A")
            for c in ["ROE","Gross Mgn","Op Mgn","Div Yld"]:
                dp[c]=dp[c].apply(lambda x:f"{x:.1f}%" if pd.notnull(x) else "N/A")
            cols_=["★","Ticker","Name","Mkt Cap","P/E","Fwd P/E","EV/EBITDA","P/B","ROE","Gross Mgn","Op Mgn","Div Yld"]
            disp=dp[[c for c in cols_ if c in dp.columns]]

            st.dataframe(disp,use_container_width=True,hide_index=True)
            st.caption(f"P/E shown with % premium/discount vs peer median ({pe_med:.1f}x). ★ = current ticker.")

        # Quarterly charts
        q_inc=get_qinc(ticker)
        qc1,qc2=st.columns(2)
        with qc1:
            st.markdown("### Revenue & EBITDA ($B)")
            if not q_inc.empty:
                rk=next((r for r in ["Total Revenue","Revenue"] if r in q_inc.index),None)
                ek=next((r for r in ["EBITDA","Normalized EBITDA"] if r in q_inc.index),None)
                cq=sorted(q_inc.columns)[-8:]
                if rk or ek:
                    fig=go.Figure()
                    if rk:
                        rv=q_inc.loc[rk,cq].apply(lambda x:float(x)/1e9 if pd.notnull(x) else None)
                        fig.add_trace(go.Bar(x=[str(c)[:10] for c in cq],y=rv.values,name="Revenue",marker_color=C_P))
                    if ek:
                        ev=q_inc.loc[ek,cq].apply(lambda x:float(x)/1e9 if pd.notnull(x) else None)
                        fig.add_trace(go.Bar(x=[str(c)[:10] for c in cq],y=ev.values,name="EBITDA",marker_color=C_POS))
                    fig.update_layout(barmode="group",yaxis_title="$ Billions",**MD)
                    st.plotly_chart(fig,use_container_width=True)
        with qc2:
            st.markdown("### Gross & Op. Margin Trend (%)")
            if not q_inc.empty:
                rk2=next((r for r in ["Total Revenue","Revenue"] if r in q_inc.index),None)
                gk=next((r for r in ["Gross Profit"] if r in q_inc.index),None)
                ok=next((r for r in ["Operating Income","EBIT"] if r in q_inc.index),None)
                cq=sorted(q_inc.columns)[-8:]
                if rk2 and (gk or ok):
                    rv2=q_inc.loc[rk2,cq].apply(pd.to_numeric,errors="coerce")
                    fig=go.Figure()
                    if gk:
                        gv=q_inc.loc[gk,cq].apply(pd.to_numeric,errors="coerce")
                        fig.add_trace(go.Scatter(x=[str(c)[:10] for c in cq],y=(gv/rv2*100).where(rv2>0).values,
                            name="Gross Margin",mode="lines+markers",line=dict(color=C_P,width=2)))
                    if ok:
                        ov=q_inc.loc[ok,cq].apply(pd.to_numeric,errors="coerce")
                        fig.add_trace(go.Scatter(x=[str(c)[:10] for c in cq],y=(ov/rv2*100).where(rv2>0).values,
                            name="Op. Margin",mode="lines+markers",line=dict(color=C_POS,width=2,dash="dash")))
                    fig.update_layout(yaxis_title="Margin (%)",**MD)
                    st.plotly_chart(fig,use_container_width=True)

        # Valuation metric cards
        pe=info.get("trailingPE"); fpe=info.get("forwardPE"); ev_=info.get("enterpriseToEbitda")
        pb=info.get("priceToBook"); fcf=info.get("freeCashflow"); mc_=info.get("marketCap")
        fcfy=fcf/mc_ if fcf and mc_ else None
        mcols([mcard("P/E (TTM)",fx(pe),ctx="Trailing twelve months"),
               mcard("Forward P/E",fx(fpe),ctx="Next twelve months consensus"),
               mcard("EV/EBITDA",fx(ev_),ctx="Enterprise value multiple"),
               mcard("P/B",fx(pb,2),ctx="Price-to-book ratio"),
               mcard("FCF Yield",fp(fcfy),
                     lvl="pos" if fcfy and fcfy>0.04 else "warn" if fcfy and fcfy<0.01 else "neu",
                     ctx="Free cash flow / market cap")],cols=5)

        # Financial statements
        st.markdown("### Annual Financial Statements")
        fin=get_fin(ticker)
        if fin:
            fs_t=st.tabs(list(fin.keys()))
            for (lbl_,df_),tab_ in zip(fin.items(),fs_t):
                with tab_:
                    if df_ is not None and not df_.empty:
                        st.dataframe(fmt_stmt(df_),use_container_width=True)

        # Dividends + holders + recs
        st.markdown("### Dividends, Holders & Recommendations")
        dv1,dv2,dv3=st.columns(3)
        with dv1:
            ds=get_div(ticker); divs=ds.get("dividends")
            if divs is not None and not divs.empty:
                st.markdown("**Dividend History**")
                fig=px.bar(divs,color_discrete_sequence=[C_POS],labels={"value":"Dividend ($)"})
                fig.update_layout(showlegend=False,**SM)
                st.plotly_chart(fig,use_container_width=True)
            else: nb("No dividend data available.","This ticker does not pay a dividend.","warn")
        with dv2:
            h_=get_holders(ticker); ih=h_.get("institutional")
            if ih is not None and not ih.empty:
                st.markdown("**Top Institutional Holders**")
                st.dataframe(ih.head(12),use_container_width=True,hide_index=True,height=280)
        with dv3:
            recs=get_recs(ticker)
            if not recs.empty:
                st.markdown("**Analyst Recommendations**")
                st.dataframe(recs.sort_index(ascending=False).head(15),
                             use_container_width=True,height=280)

    # ════════════════════════════════════════════════════════════════════════
    # ⑤ PORTFOLIO & SIMULATION
    # ════════════════════════════════════════════════════════════════════════
    with T5:
        # Monte Carlo
        st.markdown("### Monte Carlo Price Simulation (GBM)")
        if mc:
            # Decision narrative
            pl=mc["ploss"]; lvl_="neg" if pl>0.45 else "warn" if pl>0.30 else "pos"
            nb(f"1-Year Outlook: {pl*100:.0f}% probability of loss · Median return {mc['mret']*100:+.1f}%",
               f"90% of simulated outcomes fall in [{mc['p5']*100:.1f}%, {mc['p95']*100:.1f}%]. "
               f"{'Skewed downside — consider hedging or reduced position.' if pl>0.45 else 'Balanced risk profile.' if pl>0.25 else 'Favorable asymmetry — upside scenarios dominate.'}",
               lvl=lvl_)

            mcols([mcard("Median Return (1Y)",fp(mc["mret"]),lvl="pos" if mc["mret"]>0 else "neg"),
                   mcard("Expected Return",fp(mc["eret"])),
                   mcard("Probability of Loss",fp(mc["ploss"]),lvl="warn" if mc["ploss"]>0.3 else "pos"),
                   mcard("Prob. Return > +10%",fp(mc["pup10"]))],cols=4)

            pcts=mc["pcts"]; da=np.arange(pcts.shape[1])
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([da,da[::-1]]),
                y=np.concatenate([pcts[4],pcts[0][::-1]]),
                fill="toself",fillcolor="rgba(26,86,219,0.12)",
                line=dict(color="rgba(0,0,0,0)"),name="10–90th pct",showlegend=True))
            fig.add_trace(go.Scatter(x=np.concatenate([da,da[::-1]]),
                y=np.concatenate([pcts[3],pcts[1][::-1]]),
                fill="toself",fillcolor="rgba(26,86,219,0.28)",
                line=dict(color="rgba(0,0,0,0)"),name="25–75th pct",showlegend=True))
            fig.add_trace(go.Scatter(x=da,y=pcts[2],line=dict(color=C_P,width=2.5),name="Median"))
            if price:
                fig.add_hline(y=float(price),line_dash="dash",line_color=C_NEG,
                              annotation_text="Current Price",annotation_position="right")
            _mc_layout={k:v for k,v in LG.items() if k!="hovermode"}
            _mc_layout["hovermode"]="x"
            fig.update_layout(xaxis_title="Trading Days",yaxis_title="Price ($)",**_mc_layout)
            st.plotly_chart(fig,use_container_width=True)
        else:
            nb("Monte Carlo unavailable.","Requires price history and return data.","warn")

        # Options
        st.markdown("### Options — Volatility Surface")
        exps=get_exps(ticker)
        rv_s=rets.rolling(21).std()*np.sqrt(252) if not rets.empty else pd.Series(dtype=float)
        rv_now=float(rv_s.dropna().iloc[-1]) if not rv_s.dropna().empty else None
        iv_now=atm_iv(ticker,price,exps)

        mcols([mcard("21-Day Realized Vol",fp(rv_now) if rv_now else "N/A"),
               mcard("ATM IV (front-month)",fp(iv_now) if iv_now else "N/A",
                     lvl="warn" if (rv_now and iv_now and iv_now-rv_now>0.05) else "neu"),
               mcard("IV–RV Spread",
                     f"{(iv_now-rv_now)*100:+.1f}pp" if rv_now and iv_now else "N/A",
                     ctx="Positive = options priced above recent realized vol",
                     lvl="warn" if (rv_now and iv_now and iv_now-rv_now>0.05) else "pos")],cols=3)

        if exps:
            exp_sel=st.selectbox("Expiration date",exps,key=f"exp_{ticker}")
            calls,puts=get_chain(ticker,exp_sel)
            OC=["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility","inTheMoney"]
            oc1,oc2=st.columns(2)
            with oc1:
                st.markdown("**Calls**")
                if calls is not None and not calls.empty:
                    st.dataframe(calls[[c for c in OC if c in calls.columns]],
                                 use_container_width=True,hide_index=True,height=280)
            with oc2:
                st.markdown("**Puts**")
                if puts is not None and not puts.empty:
                    st.dataframe(puts[[c for c in OC if c in puts.columns]],
                                 use_container_width=True,hide_index=True,height=280)

            if calls is not None and not calls.empty and "impliedVolatility" in calls.columns:
                ic1,ic2=st.columns(2)
                with ic1:
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=calls["strike"],y=calls["impliedVolatility"],
                        mode="lines+markers",name="Calls IV",line=dict(color=C_P,width=2)))
                    if puts is not None and not puts.empty:
                        fig.add_trace(go.Scatter(x=puts["strike"],y=puts["impliedVolatility"],
                            mode="lines+markers",name="Puts IV",line=dict(color=C_OR,width=2,dash="dash")))
                    if price:
                        fig.add_vline(x=float(price),line_dash="dash",line_color=C_BN,
                                      annotation_text="Spot")
                    fig.update_layout(title=f"IV Smile — {exp_sel}",xaxis_title="Strike",
                                      yaxis_title="IV",**MD)
                    st.plotly_chart(fig,use_container_width=True)
                with ic2:
                    if "openInterest" in calls.columns:
                        fig=go.Figure()
                        fig.add_trace(go.Bar(x=calls["strike"],y=calls["openInterest"],
                                             name="Calls OI",marker_color=C_P))
                        if puts is not None and not puts.empty:
                            fig.add_trace(go.Bar(x=puts["strike"],y=puts["openInterest"],
                                                 name="Puts OI",marker_color=C_NEG))
                        if price:
                            fig.add_vline(x=float(price),line_dash="dash",line_color=C_BN,
                                          annotation_text="Spot")
                        fig.update_layout(title="Open Interest by Strike",barmode="group",
                                          xaxis_title="Strike",yaxis_title="OI",**MD)
                        st.plotly_chart(fig,use_container_width=True)

        # Chatbot
        if CHATBOT_ON:
            ctx_dict={
                "ticker":ticker,"sector":sector,"price":price,
                "ann_return":fp(rm.get("ar")),"ann_vol":fp(rm.get("av")),
                "sharpe":f"{rm.get('sh',0):.2f}" if rm else "N/A",
                "var95":fp(rm.get("v95")),"cvar95":fp(rm.get("cv95")),
                "max_dd":fp(rm.get("md")),"market_beta":f"{freg.get('beta',0):.3f}" if freg else "N/A",
                "alpha_ann":fp(freg.get("alpha")),"r_squared":f"{freg.get('r2',0):.3f}" if freg else "N/A",
                "rate_beta":f"{freg.get('rate_beta',0):.4f}" if freg and "rate_beta" in freg else "N/A",
                "macro_correlations":{FRED_LBL.get(k,k):round(v,3) for k,v in corr.items()},
                "pe_ttm":info.get("trailingPE"),"fwd_pe":info.get("forwardPE"),
                "monte_carlo":{"median_return":fp(mc.get("mret")),"prob_loss":fp(mc.get("ploss")),
                               "p5":fp(mc.get("p5")),"p95":fp(mc.get("p95"))} if mc else "unavailable",
            }
            st.markdown("---")
            st.markdown("### 🤖 Research Assistant")
            st.caption("Grounded in computed analytics above. Examples: *'Biggest risk?'* · *'Explain Monte Carlo'* · *'How rate-sensitive is this?'*")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history=[]
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.write(msg["content"])
            if prompt:=st.chat_input(f"Ask about {ticker}…"):
                st.session_state.chat_history.append({"role":"user","content":prompt})
                with st.chat_message("user"): st.write(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing…"):
                        try:
                            client=anthropic.Anthropic(api_key=_ANT_KEY)
                            sys_p=(f"You are a quantitative research assistant. Answer ONLY using the provided analytics context. "
                                   f"Be precise and concise. Context:\n{ctx_dict}")
                            resp=client.messages.create(model="claude-haiku-4-5-20251001",max_tokens=512,
                                system=sys_p,messages=st.session_state.chat_history[-6:])
                            reply=resp.content[0].text
                        except Exception as e: reply=f"Error: {e}"
                    st.write(reply)
                st.session_state.chat_history.append({"role":"assistant","content":reply})

# ════════════════════════════════════════════════════════════════════════════
# 📊 PORTFOLIO TAB
# ════════════════════════════════════════════════════════════════════════════
if len(tickers)>1 and _outer_tabs is not None:
  with _outer_tabs[-1]:
    valid_t=list(all_rets.keys())
    if len(valid_t)>=2:
        ret_df=pd.concat(all_rets,axis=1,join="inner").dropna()
        if ret_df.index.tz is not None:
            ret_df.index=ret_df.index.tz_localize(None)
        ret_df.columns=valid_t
        na=len(valid_t)
        ann_cov=ret_df.cov()*252
        w_eq=np.ones(na)/na

        with st.spinner("Computing portfolio analytics…"):
            sim=sim_portfolios(ret_df,n=2000)

        w_ms=np.array(sim.loc[sim["sh"].idxmax(),"w"])
        w_mv=np.array(sim.loc[sim["vol"].idxmin(),"w"])

        pr_eq=(ret_df@w_eq).dropna()
        pr_ms=(ret_df@w_ms).dropna()
        pr_mv=(ret_df@w_mv).dropna()

        ps_eq=port_full_stats(pr_eq,w_eq,ann_cov.values)
        ps_ms=port_full_stats(pr_ms,w_ms,ann_cov.values)
        ps_mv=port_full_stats(pr_mv,w_mv,ann_cov.values)
        corr_mat=ret_df.corr()

        # ── EXECUTIVE DECISION CARD ───────────────────────────────────────────
        items=[]
        sh_gap=ps_ms["sh"]-ps_eq["sh"]
        if sh_gap>0.10:
            items.append(
                f"<strong>REALLOCATION OPPORTUNITY:</strong> The equal-weight allocation leaves {sh_gap:.2f} Sharpe points on the table. "
                f"Shifting to the Max-Sharpe portfolio ({', '.join([f'{valid_t[i]} {w_ms[i]*100:.0f}%' for i in range(na)])}) "
                f"improves risk-adjusted return from {ps_eq['sh']:.2f} to {ps_ms['sh']:.2f} without increasing volatility materially.")
        high_corr=[(valid_t[i],valid_t[j],corr_mat.iloc[i,j])
                   for i in range(na) for j in range(i+1,na)
                   if corr_mat.iloc[i,j]>0.75]
        if high_corr:
            pairs="; ".join([f"{a}/{b} ρ={c:.2f}" for a,b,c in high_corr[:3]])
            items.append(
                f"<strong>CORRELATED PAIRS — ILLUSORY DIVERSIFICATION:</strong> {pairs}. "
                f"These positions co-move; holding both adds cost with minimal risk reduction. "
                f"True diversification requires assets with fundamentally different return drivers.")
        rc_=ps_eq["rc"]; top_i=int(np.argmax(rc_))
        if rc_[top_i]>0.40:
            items.append(
                f"<strong>VOLATILITY CONCENTRATION:</strong> {valid_t[top_i]} contributes {rc_[top_i]*100:.0f}% of portfolio risk "
                f"despite a {100/na:.0f}% equal weight. Risk-weighted sizing would reduce this position.")
        if ps_eq["md"]<-0.25:
            items.append(
                f"<strong>DRAWDOWN PROFILE:</strong> The equal-weight portfolio experienced a max drawdown of {ps_eq['md']*100:.1f}%. "
                f"On the worst 5% of trading days, daily losses averaged {abs(ps_eq['cvar95'])*100:.2f}% (CVaR). "
                f"{'Min-Variance allocation reduces max drawdown to ' + fp(ps_mv['md']) + '.' if ps_mv['md']>ps_eq['md'] else ''}")
        if not items:
            items.append(f"Portfolio is well-structured. Equal-weight Sharpe: {ps_eq['sh']:.2f} · Sortino: {ps_eq['so']:.2f} · Calmar: {ps_eq['ca']:.2f} · Diversification Ratio: {ps_eq['dr']:.2f}x.")
        decision_card("Portfolio Assessment",items)

        # ── SCORECARD ─────────────────────────────────────────────────────────
        st.markdown("### Equal-Weight Portfolio Scorecard")
        def _slvl(v,hi,lo): return "pos" if v>hi else "neg" if v<lo else "warn"
        mcols([
            mcard("Ann. Return",fp(ps_eq["ar"]),lvl="pos" if ps_eq["ar"]>0 else "neg"),
            mcard("Ann. Volatility",fp(ps_eq["av"]),ctx="Lower = less risk",
                  lvl="warn" if ps_eq["av"]>0.25 else "neu"),
            mcard("Sharpe Ratio",f"{ps_eq['sh']:.2f}" if not np.isnan(ps_eq["sh"]) else "N/A",
                  ctx="Return per unit of total risk  ·  >1.0 is strong",
                  lvl=_slvl(ps_eq["sh"] if not np.isnan(ps_eq["sh"]) else 0,1.0,0.4)),
            mcard("Sortino Ratio",f"{ps_eq['so']:.2f}" if not np.isnan(ps_eq["so"]) else "N/A",
                  ctx="Like Sharpe, but penalizes only downside  ·  >1.5 is strong",
                  lvl=_slvl(ps_eq["so"] if not np.isnan(ps_eq["so"]) else 0,1.5,0.7)),
            mcard("Calmar Ratio",f"{ps_eq['ca']:.2f}" if not np.isnan(ps_eq["ca"]) else "N/A",
                  ctx="Ann. return ÷ max drawdown  ·  >1.0 is strong",
                  lvl=_slvl(ps_eq["ca"] if not np.isnan(ps_eq["ca"]) else 0,1.0,0.4)),
            mcard("Omega Ratio",f"{ps_eq['om']:.2f}" if not np.isnan(ps_eq["om"]) else "N/A",
                  ctx="Gains-to-losses ratio above 0% threshold  ·  >1.5 is strong",
                  lvl=_slvl(ps_eq["om"] if not np.isnan(ps_eq["om"]) else 0,1.5,1.0)),
            mcard("Max Drawdown",fp(ps_eq["md"]),ctx="Peak-to-trough decline",
                  lvl="neg" if ps_eq["md"]<-0.30 else "warn"),
            mcard("Diversification Ratio",f"{ps_eq['dr']:.2f}x",
                  ctx="Weighted avg vol ÷ portfolio vol  ·  >1.3 is meaningful",
                  lvl="pos" if ps_eq["dr"]>1.3 else "warn" if ps_eq["dr"]>1.1 else "neg"),
        ],cols=4)

        # ── STRATEGY COMPARISON ───────────────────────────────────────────────
        st.markdown("### Strategy Comparison")
        st.caption("Equal-weight baseline vs. numerically-optimized Max-Sharpe and Min-Variance portfolios from 2,000 Monte Carlo allocations.")
        comp_rows=[]
        for label,ps,wts in [("Equal Weight",ps_eq,w_eq),("Max Sharpe ★",ps_ms,w_ms),("Min Variance",ps_mv,w_mv)]:
            row={"Strategy":label,"Return":fp(ps["ar"]),"Volatility":fp(ps["av"]),
                 "Sharpe":f"{ps['sh']:.2f}","Sortino":f"{ps['so']:.2f}",
                 "Calmar":f"{ps['ca']:.2f}","Max DD":fp(ps["md"]),"Div Ratio":f"{ps['dr']:.2f}x"}
            for i,t in enumerate(valid_t): row[t]=f"{wts[i]*100:.0f}%"
            comp_rows.append(row)
        st.dataframe(pd.DataFrame(comp_rows),use_container_width=True,hide_index=True)

        sc1,sc2=st.columns(2)
        with sc1:
            st.markdown("### Weights by Strategy")
            w_df=pd.DataFrame({"Ticker":valid_t,
                               "Equal Weight":w_eq*100,"Max Sharpe":w_ms*100,"Min Variance":w_mv*100})
            fig=go.Figure()
            for col,col_ in [("Equal Weight",C_BN),("Max Sharpe",C_P),("Min Variance",C_POS)]:
                fig.add_trace(go.Bar(name=col,x=w_df["Ticker"],y=w_df[col],marker_color=col_))
            fig.update_layout(barmode="group",yaxis_title="Weight (%)",**MD)
            st.plotly_chart(fig,use_container_width=True)
        with sc2:
            st.markdown("### Efficient Frontier (2,000 Simulations)")
            fig=px.scatter(sim,x="vol",y="ret",color="sh",color_continuous_scale="RdYlGn",
                opacity=0.45,labels={"vol":"Ann. Volatility","ret":"Ann. Return","sh":"Sharpe"})
            for (wvol,wret,wname,wclr,wsym) in [
                (ps_eq["av"],ps_eq["ar"],"Equal Weight","navy","star"),
                (ps_ms["av"],ps_ms["ar"],"Max Sharpe",C_POS,"diamond"),
                (ps_mv["av"],ps_mv["ar"],"Min Variance",C_OR,"circle"),
            ]:
                fig.add_trace(go.Scatter(x=[wvol],y=[wret],mode="markers+text",
                    text=[wname],textposition="top center",
                    marker=dict(color=wclr,size=13,symbol=wsym),name=wname))
            fig.update_layout(**MD)
            st.plotly_chart(fig,use_container_width=True)

        # ── RISK DECOMPOSITION ────────────────────────────────────────────────
        st.markdown("### Risk Decomposition (Equal-Weight)")
        rd1,rd2=st.columns(2)
        with rd1:
            st.markdown("**Marginal Risk Contribution**")
            st.caption("Each position's % share of total portfolio volatility")
            colors_rc=[C_NEG if r>0.5 else C_WARN if r>0.35 else C_P for r in rc_]
            fig=go.Figure(go.Bar(x=valid_t,y=rc_*100,marker_color=colors_rc,
                text=[f"{r*100:.1f}%" for r in rc_],textposition="outside"))
            fig.add_hline(y=100/na,line_dash="dash",line_color=C_BN,
                          annotation_text=f"Equal {100/na:.0f}%",annotation_position="right")
            fig.update_layout(yaxis_title="% of Portfolio Volatility",showlegend=False,**MD)
            st.plotly_chart(fig,use_container_width=True)
        with rd2:
            st.markdown("**Component VaR 95% (daily)**")
            st.caption("Dollar risk per position as % of portfolio — sums to total portfolio VaR")
            cvars=component_var95(w_eq,ann_cov.values/252)
            fig=go.Figure(go.Bar(x=valid_t,y=cvars*100,
                marker_color=[C_NEG if v>0.01 else C_WARN if v>0.005 else C_P for v in cvars],
                text=[f"{v*100:.3f}%" for v in cvars],textposition="outside"))
            fig.update_layout(yaxis_title="Component VaR (%)",showlegend=False,**MD)
            st.plotly_chart(fig,use_container_width=True)

        # ── CORRELATION & DIVERSIFICATION ─────────────────────────────────────
        st.markdown("### Correlation & Diversification Stability")
        co1,co2=st.columns(2)
        with co1:
            st.markdown("**Correlation Matrix**")
            fig=go.Figure(go.Heatmap(z=corr_mat.values,
                x=corr_mat.columns.tolist(),y=corr_mat.index.tolist(),
                colorscale="RdBu_r",zmin=-1,zmax=1,
                text=np.round(corr_mat.values,2),texttemplate="%{text}",
                colorbar=dict(thickness=12,len=0.8)))
            fig.update_layout(**{**MD,**{"margin":dict(t=10,b=10,l=0,r=0)}})
            st.plotly_chart(fig,use_container_width=True)
        with co2:
            st.markdown("**Rolling 63-Day Diversification Ratio**")
            st.caption("A falling ratio signals correlation creep — diversification benefit eroding over time")
            dr_s=rolling_div_ratio(ret_df,w_eq,window=63)
            if not dr_s.empty:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=dr_s.index,y=dr_s.values,
                    line=dict(color=C_P,width=2),fill="tozeroy",fillcolor=FP,name="Div. Ratio"))
                fig.add_hline(y=1.0,line_dash="dash",line_color=C_NEG,
                              annotation_text="No diversification benefit",annotation_position="right")
                fig.update_layout(yaxis_title="Diversification Ratio",showlegend=False,**MD)
                st.plotly_chart(fig,use_container_width=True)

        # ── CUMULATIVE PERFORMANCE ────────────────────────────────────────────
        st.markdown("### Cumulative Return: Three Allocations vs SPY")
        nb("Max-Sharpe and Min-Variance weights are derived from in-sample optimization and will reflect hindsight bias.",
           "Use these comparisons directionally, not as a backtest. Out-of-sample performance will differ.","warn")
        spy_h2=get_hist("SPY",period,interval)
        spy_r2=spy_h2["Close"].pct_change().dropna() if not spy_h2.empty else pd.Series(dtype=float)
        if spy_r2.index.tz is not None: spy_r2.index=spy_r2.index.tz_localize(None)
        fig=go.Figure()
        for series,name,color,dash in [
            (pr_eq,"Equal Weight",C_P,None),
            (pr_ms,"Max Sharpe",C_POS,"dash"),
            (pr_mv,"Min Variance",C_OR,"dot"),
        ]:
            s=series.copy()
            if s.index.tz is not None: s.index=s.index.tz_localize(None)
            cum=(1+s).cumprod()-1
            ld=dict(color=color,width=2.5) if not dash else dict(color=color,width=2,dash=dash)
            fig.add_trace(go.Scatter(x=cum.index,y=cum.values*100,name=name,
                line=ld,hovertemplate="%{y:.2f}%"))
        if not spy_r2.empty:
            idx=spy_r2.index.intersection(pr_eq.index if pr_eq.index.tz is None else pr_eq.index.tz_localize(None))
            spy_cum=(1+spy_r2.loc[idx]).cumprod()-1
            fig.add_trace(go.Scatter(x=idx,y=spy_cum.values*100,name="SPY",
                line=dict(color=C_BN,width=1.5,dash="dashdot"),hovertemplate="%{y:.2f}%"))
        fig.add_hline(y=0,line_dash="dot",line_color="#D1D5DB",line_width=1)
        fig.update_layout(yaxis_title="Cumulative Return (%)",**LG)
        st.plotly_chart(fig,use_container_width=True)

    else:
        nb("Add 2+ tickers in the sidebar to unlock portfolio analytics.",
           "Portfolio section includes correlation matrix, efficient frontier, and risk attribution.","neu")

st.markdown("---")
st.caption("Yahoo Finance · FRED (St. Louis Fed). For informational purposes only — not investment advice.")
