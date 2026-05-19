import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import io
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from Bitirme import (tam_hesap, vergi_hesapla, gini_hesapla, ort_vergi_hesapla,
                     MEVCUT_ORANLAR, ESKI_SINIRLAR,
                     gelir_st_haric, gelir_dahil, hh_buyukluk, n_pay, n)
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint, LinearConstraint
import time

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(page_title="Vergi Optimizasyon Aracı", layout="wide")
st.title("🇹🇷 Türkiye Gelir Vergisi Optimizasyon Aracı")
st.caption("Gini katsayısını minimize eden optimal vergi sistemi tasarımı")

with st.expander("ℹ️ Bu araç hakkında — Amaç, Kapsam ve Kullanım Kılavuzu"):
    st.markdown("""
    ### Neden Bu Araç?

    Türkiye'de gelir eşitsizliği önemli bir sorun olmaya devam etmektedir.
    Mevcut gelir vergisi sistemi, artan oranlı yapısına rağmen eşitsizliği azaltmada yetersiz kalmaktadır.

    Bu araç, **İstanbul Teknik Üniversitesi Endüstri Mühendisliği** bitirme projesi kapsamında geliştirilmiştir.
    Temel amacı, politika yapıcılara farklı vergi yapılarının gelir dağılımı üzerindeki etkisini
    anlık olarak görebilecekleri bir **karar destek aracı** sunmaktır.

    ---

    ### Karar Değişkenleri

    | Değişken | Açıklama |
    |----------|----------|
    | ✅ Vergi Oranları | Her dilime uygulanan oran (%) optimize edilir |
    | ✅ Dilim Sınırları | Dilim eşik değerleri (TL) optimize edilir |
    | ✅ Dilim Sayısı | Kaç dilim kullanılacağı optimize edilir |

    Üç değişkeni istediğiniz kombinasyonda seçebilirsiniz. Seçilen değişkenler eş zamanlı optimize edilir.

    ---

    ### Kimler Kullanabilir?

    | Kullanıcı | Amaç |
    |-----------|------|
    | 🏛️ Politika yapıcılar | Farklı vergi reformlarının etkisini test etmek |
    | 📊 Araştırmacılar | Vergi-eşitsizlik ilişkisini analiz etmek |
    | 🎓 Akademisyenler | Optimal vergi teorisini uygulamalı görmek |
    """)

baseline_gini, baseline_ort = tam_hesap(MEVCUT_ORANLAR, ESKI_SINIRLAR)

# ============================================================
# SOL PANEL
# ============================================================
with st.sidebar:
    st.header("⚙️ Parametreler")

    st.subheader("🎯 Karar Değişkenleri")
    st.markdown("Optimize edilecek değişkenleri seçin (en az biri zorunlu):")

    opt_oranlar_sec   = st.checkbox("Vergi Oranları",   value=True)
    opt_sinirlar_sec  = st.checkbox("Dilim Sınırları",  value=False)
    opt_dilim_sec     = st.checkbox("Dilim Sayısı",     value=False)

    if not any([opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec]):
        st.error("⚠️ En az bir değişken seçilmeli!")

    # Kombinasyon etiketi
    sec_listesi = []
    if opt_oranlar_sec:  sec_listesi.append("Oranlar")
    if opt_sinirlar_sec: sec_listesi.append("Sınırlar")
    if opt_dilim_sec:    sec_listesi.append("Dilim Sayısı")
    kombinasyon_label = " + ".join(sec_listesi) if sec_listesi else "—"

    with st.expander("📋 Seçilen Kombinasyon"):
        if not sec_listesi:
            st.warning("Hiçbir değişken seçilmedi.")
        else:
            aciklamalar = {
                "Oranlar": "Her dilime uygulanan vergi oranı (%) serbest bırakılır.",
                "Sınırlar": "Dilim eşik noktaları (TL cinsinden) serbest bırakılır.",
                "Dilim Sayısı": "Kaç adet vergi dilimi kullanılacağı serbest bırakılır; oranlar ve sınırlar bu sayıya göre yeniden optimize edilir.",
            }
            for s in sec_listesi:
                st.info(f"**{s}:** {aciklamalar[s]}")

    st.markdown("---")

    # Dilim sayısı — yalnızca "Dilim Sayısı" kutusu seçiliyse aktif
    st.subheader("Dilim Sayısı" + ("" if opt_dilim_sec else " *(pasif)*"))
    if opt_dilim_sec:
        dilim_aralik_min = st.slider("Min dilim sayısı", 5, 12, 5)
        dilim_aralik_max = st.slider("Max dilim sayısı", dilim_aralik_min, 15, 9)
        dilim_sayisi = None  # optimizasyona bırakılacak
    else:
        dilim_sayisi = st.slider("Dilim sayısı", min_value=5, max_value=15,
                                  value=5, disabled=True)
        dilim_aralik_min = dilim_aralik_max = dilim_sayisi

    # Oran kısıtları
    st.markdown("---")
    st.subheader("Oran Kısıtları" + (" *(pasif)*" if not opt_oranlar_sec else ""))
    min_oran      = st.slider("Min oran (%)", 5, 20, 10, disabled=not opt_oranlar_sec) / 100
    max_oran      = st.slider("Max oran (%)", 30, 60, 45, disabled=not opt_oranlar_sec) / 100
    min_oran_fark = st.slider("Oranlar arası min fark (%)", 1, 10, 3, disabled=not opt_oranlar_sec) / 100
    max_oran_fark = st.slider("Oranlar arası max fark (%)", 5, 25, 13, disabled=not opt_oranlar_sec) / 100

    # Sınır kısıtları
    st.markdown("---")
    st.subheader("Dilim Sınırı Kısıtları" + (" *(pasif)*" if not opt_sinirlar_sec else ""))
    min_sinir      = st.number_input("Min dilim sınırı (TL)", min_value=10000,
                                      max_value=200000, value=50000, step=10000,
                                      disabled=not opt_sinirlar_sec)
    son_dilim_min  = st.number_input("Son dilim minimum sınırı (TL)", min_value=500000,
                                      max_value=10000000, value=2000000, step=500000,
                                      disabled=not opt_sinirlar_sec)
    min_sinir_fark = st.number_input("Sınırlar arası min fark (TL)", min_value=10000,
                                      max_value=500000, value=100000, step=10000,
                                      disabled=not opt_sinirlar_sec)

    st.markdown("---")
    st.subheader("Bütçe Toleransı")
    butce_tolerans = st.slider("Tolerans (±%)", 0.0, 10.0, 2.0, step=0.5) / 100

    st.markdown("---")
    optimize_et = st.button("🚀 Optimize Et", type="primary", use_container_width=True,
                             disabled=not any([opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec]))


# ============================================================
# OPTİMİZASYON ÇEKIRDEĞI
# (Bitirme1.py mantığına dayalı, dinamik kombinasyon desteği)
# ============================================================
def calistir_optimizasyon(opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                           dilim_sayisi, dilim_aralik_min, dilim_aralik_max,
                           min_oran, max_oran, min_oran_fark, max_oran_fark,
                           min_sinir, son_dilim_min, min_sinir_fark, butce_tolerans):
    tol_alt = 1 - butce_tolerans
    tol_ust = 1 + butce_tolerans

    if opt_dilim_sec:
        # Dilim sayısı da optimize ediliyor: her k için çöz, en iyiyi al
        best_gini     = np.inf
        best_oranlar  = None
        best_sinirlar = None
        for k in range(dilim_aralik_min, dilim_aralik_max + 1):
            try:
                o, s = _optimize_fixed_k(
                    k, True, True,
                    min_oran, max_oran, min_oran_fark, max_oran_fark,
                    min_sinir, son_dilim_min, min_sinir_fark,
                    tol_alt, tol_ust)
                g = gini_hesapla(vergi_hesapla(o, s))
                if g < best_gini:
                    best_gini     = g
                    best_oranlar  = o
                    best_sinirlar = s
            except Exception:
                continue
        if best_oranlar is None:
            raise ValueError("Dilim sayısı aralığında geçerli çözüm bulunamadı.")
        return best_oranlar, best_sinirlar
    else:
        k = dilim_sayisi if dilim_sayisi else 5
        return _optimize_fixed_k(
            k, opt_oranlar_sec, opt_sinirlar_sec,
            min_oran, max_oran, min_oran_fark, max_oran_fark,
            min_sinir, son_dilim_min, min_sinir_fark,
            tol_alt, tol_ust)


def _optimize_fixed_k(k, opt_oranlar_sec, opt_sinirlar_sec,
                       min_oran, max_oran, min_oran_fark, max_oran_fark,
                       min_sinir, son_dilim_min, min_sinir_fark,
                       tol_alt, tol_ust):
    """
    Bitirme1.py'deki senaryo fonksiyonlarının genelleştirilmiş hali.
    k: dilim sayısı
    opt_oranlar_sec: oranlar optimize edilsin mi
    opt_sinirlar_sec: sınırlar optimize edilsin mi
    """
    if not opt_oranlar_sec and not opt_sinirlar_sec:
        raise ValueError("En az bir değişken seçilmeli.")

    # ── SADECE ORANLAR (Bitirme1 Senaryo 1 mantığı) ─────────────
    if opt_oranlar_sec and not opt_sinirlar_sec:
        # k=5 ise orijinal ESKI_SINIRLAR, değilse eşit aralıklı
        if k == len(ESKI_SINIRLAR) + 1:
            sinirlar = ESKI_SINIRLAR.copy()
        else:
            sinirlar = np.linspace(min_sinir * 2, son_dilim_min, k - 1)

        def hedef(x): return gini_hesapla(vergi_hesapla(x, sinirlar))
        def ort_fn(x): return ort_vergi_hesapla(vergi_hesapla(x, sinirlar))

        bounds = [(min_oran, max_oran)] * k
        kisit_butce = NonlinearConstraint(ort_fn, baseline_ort * tol_alt, baseline_ort * tol_ust)
        A = np.zeros((k - 1, k))
        for i in range(k - 1): A[i, i] = -1; A[i, i + 1] = 1
        kisit_prog = LinearConstraint(A, min_oran_fark, max_oran_fark)

        sonuc = differential_evolution(hedef, bounds,
            constraints=[kisit_butce, kisit_prog],
            seed=42, maxiter=1000, tol=1e-8, popsize=20,
            mutation=(0.5, 1.5), recombination=0.9, polish=True)

        kisitlar = [
            {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * tol_alt},
            {'type': 'ineq', 'fun': lambda x: baseline_ort * tol_ust - ort_fn(x)},
        ]
        for i in range(k - 1):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i + 1] - x[i] - min_oran_fark})
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: max_oran_fark - (x[i + 1] - x[i])})

        sonuc2 = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                          constraints=kisitlar, options={'maxiter': 1000, 'ftol': 1e-10})
        return sonuc2.x, sinirlar

    # ── SADECE SINIRLAR (Bitirme1 Senaryo 2 mantığı) ────────────
    elif opt_sinirlar_sec and not opt_oranlar_sec:
        # k=5 ise orijinal MEVCUT_ORANLAR, değilse eşit aralıklı
        if k == len(MEVCUT_ORANLAR):
            oranlar = MEVCUT_ORANLAR.copy()
        else:
            oranlar = np.linspace(min_oran, max_oran, k)

        def hedef(x): return gini_hesapla(vergi_hesapla(oranlar, x))
        def ort_fn(x): return ort_vergi_hesapla(vergi_hesapla(oranlar, x))

        # Bitirme1 Senaryo 2'deki bounds yapısı: son dilim için ayrı üst sınır
        if k - 1 == 4:  # orijinal 5 dilim: 4 sınır noktası
            sn_bounds = [(50000, 300000), (200000, 600000),
                         (600000, 1500000), (son_dilim_min, son_dilim_min * 4)]
        else:
            sn_bounds = [(min_sinir, son_dilim_min)] * (k - 2) + [(son_dilim_min, son_dilim_min * 4)]

        kisit_butce = NonlinearConstraint(ort_fn, baseline_ort * tol_alt, baseline_ort * tol_ust)
        A = np.zeros((k - 2, k - 1))
        for i in range(k - 2): A[i, i] = -1; A[i, i + 1] = 1
        kisit_sinir = LinearConstraint(A, min_sinir_fark, np.inf)

        sonuc = differential_evolution(hedef, sn_bounds,
            constraints=[kisit_butce, kisit_sinir],
            seed=42, maxiter=1000, tol=1e-8, popsize=20,
            mutation=(0.5, 1.5), recombination=0.9, polish=True)

        kisitlar = [
            {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * tol_alt},
            {'type': 'ineq', 'fun': lambda x: baseline_ort * tol_ust - ort_fn(x)},
        ]
        for i in range(k - 2):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i + 1] - x[i] - min_sinir_fark})

        sonuc2 = minimize(hedef, sonuc.x, method='SLSQP', bounds=sn_bounds,
                          constraints=kisitlar, options={'maxiter': 1000, 'ftol': 1e-10})
        return oranlar, sonuc2.x

    # ── JOINT: ORANLAR + SINIRLAR (Bitirme1 Senaryo 3/4 mantığı) ─
    else:
        def hedef(x): return gini_hesapla(vergi_hesapla(x[:k], x[k:]))
        def ort_fn(x): return ort_vergi_hesapla(vergi_hesapla(x[:k], x[k:]))

        # Bounds: Bitirme1'deki orijinal bounds yapısını miras al (k=5)
        if k == 5:
            sn_bounds = [(50000, 300000), (200000, 600000),
                         (600000, 2000000), (son_dilim_min, son_dilim_min * 4)]
        else:
            sn_bounds = [(min_sinir, son_dilim_min)] * (k - 2) + [(son_dilim_min, son_dilim_min * 4)]
        bounds = [(min_oran, max_oran)] * k + sn_bounds

        kisit_butce = NonlinearConstraint(ort_fn, baseline_ort * tol_alt, baseline_ort * tol_ust)
        toplam = (k - 1) + (k - 2)
        A = np.zeros((toplam, 2 * k - 1))
        for i in range(k - 1): A[i, i] = -1; A[i, i + 1] = 1
        for i in range(k - 2): A[k - 1 + i, k + i] = -1; A[k - 1 + i, k + i + 1] = 1
        alt = np.concatenate([np.full(k - 1, min_oran_fark), np.full(k - 2, min_sinir_fark)])
        ust = np.concatenate([np.full(k - 1, max_oran_fark), np.full(k - 2, np.inf)])
        kisit_yapi = LinearConstraint(A, alt, ust)

        sonuc = differential_evolution(hedef, bounds,
            constraints=[kisit_butce, kisit_yapi],
            seed=42, maxiter=1500, tol=1e-8, popsize=25,
            mutation=(0.5, 1.5), recombination=0.9, polish=True)

        kisitlar = [
            {'type': 'ineq', 'fun': lambda x: ort_fn(x) - baseline_ort * tol_alt},
            {'type': 'ineq', 'fun': lambda x: baseline_ort * tol_ust - ort_fn(x)},
        ]
        for i in range(k - 1):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[i + 1] - x[i] - min_oran_fark})
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: max_oran_fark - (x[i + 1] - x[i])})
        for i in range(k - 2):
            kisitlar.append({'type': 'ineq', 'fun': lambda x, i=i: x[k + i + 1] - x[k + i] - min_sinir_fark})

        # Bitirme1 Senaryo 4'teki gibi çoklu başlangıç noktası (k>5 için)
        if k > 5:
            x0_listesi = [
                np.concatenate([np.linspace(min_oran, max_oran, k),
                                np.linspace(min_sinir * 2, son_dilim_min * 2, k - 1)]),
                np.concatenate([np.linspace(min_oran, max_oran * 0.9, k),
                                np.linspace(min_sinir * 3, son_dilim_min * 2, k - 1)]),
            ]
            en_iyi_gini = np.inf
            en_iyi_x = sonuc.x  # differential_evolution sonucunu varsayılan al
            for x0 in x0_listesi:
                s = minimize(hedef, x0, method='SLSQP', bounds=bounds,
                             constraints=kisitlar, options={'maxiter': 2000, 'ftol': 1e-10})
                if s.fun < en_iyi_gini:
                    en_iyi_gini = s.fun
                    en_iyi_x = s.x
            # differential_evolution sonucunu da değerlendir
            s = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                         constraints=kisitlar, options={'maxiter': 1000, 'ftol': 1e-10})
            if s.fun < en_iyi_gini:
                en_iyi_x = s.x
        else:
            s = minimize(hedef, sonuc.x, method='SLSQP', bounds=bounds,
                         constraints=kisitlar, options={'maxiter': 1000, 'ftol': 1e-10})
            en_iyi_x = s.x

        return en_iyi_x[:k], en_iyi_x[k:]


# ============================================================
# LORENZ EĞRİSİ
# ============================================================
def lorenz_egri(odenen_vergi):
    net = (gelir_dahil - odenen_vergi) / hh_buyukluk
    y   = np.sort(net)
    cg  = np.cumsum(y * n_pay) / np.sum(y * n_pay)
    cp  = np.cumsum(n_pay)
    return np.concatenate([[0], cp]), np.concatenate([[0], cg])


# ============================================================
# KOMBİNASYON ETİKETİ
# ============================================================
def kombinasyon_acikla(opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec):
    parcalar = []
    if opt_oranlar_sec:  parcalar.append("vergi oranlarının")
    if opt_sinirlar_sec: parcalar.append("dilim sınırlarının")
    if opt_dilim_sec:    parcalar.append("dilim sayısının")
    if len(parcalar) == 1:
        return f"yalnızca {parcalar[0]} optimize edilmesiyle"
    elif len(parcalar) == 2:
        return f"{parcalar[0]} ve {parcalar[1]} eş zamanlı optimizasyonuyla"
    else:
        return "vergi oranları, dilim sınırları ve dilim sayısının birlikte optimizasyonuyla"


# ============================================================
# OTOMATİK YORUM
# ============================================================
def policy_type_tespiti(opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                         min_oran, max_oran, min_oran_fark,
                         butce_tolerans, iyilesme_pct, butce_fark, dilim_sayisi_sonuc):
    agresif = 0
    muhafazakar = 0

    if butce_tolerans >= 0.05:     agresif += 2
    elif butce_tolerans <= 0.01:   muhafazakar += 2

    if opt_oranlar_sec:
        if max_oran >= 0.50:       agresif += 2
        elif max_oran <= 0.40:     muhafazakar += 1
        if min_oran <= 0.05:       agresif += 1
        if min_oran_fark <= 0.02:  muhafazakar += 1

    if opt_dilim_sec and dilim_sayisi_sonuc and dilim_sayisi_sonuc >= 10:
        agresif += 1

    n_degisken = sum([opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec])
    agresif += n_degisken - 1  # çok değişkenli kombinasyon agresif sayılır

    yuksek_fiskal_risk = abs(butce_fark) > 3.0

    if yuksek_fiskal_risk and agresif >= 3:
        return ("🔴", "Agresif Yeniden Dağılım / Yüksek Fiskal Risk",
                "Yüksek yeniden dağılım kapasitesi ancak ciddi bütçe sapma riski tespit edildi.")
    elif agresif >= 3:
        return ("🟠", "Agresif Yeniden Dağılım Modeli",
                "Güçlü bir yeniden dağılım yapısı. Geniş parametre esnekliği sisteme büyük manevra alanı tanıyor.")
    elif muhafazakar >= 3:
        return ("🟢", "Muhafazakâr Reform Modeli",
                "Kısıtlayıcı parametre yapısı. Bütçe riski düşük; yeniden dağılım kapasitesi sınırlı.")
    else:
        return ("🟡", "Dengeli Yeniden Dağılım Modeli",
                "Orta düzeyde reform yapısı. Yeniden dağılım kapasitesi ve bütçe riski dengeli.")


def policy_summary_olustur(opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                            iyilesme_pct, butce_fark, opt_gini, baseline_gini,
                            opt_ov, baseline_ov, opt_oranlar, dilim_sayisi_sonuc):
    azalan = sum(1 for f in (opt_ov - baseline_ov) if f < -10)
    artan  = sum(1 for f in (opt_ov - baseline_ov) if f > 10)

    senaryo_acik = kombinasyon_acikla(opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec)

    if iyilesme_pct >= 5:    etki = "anlamlı ölçüde azaltılmıştır"
    elif iyilesme_pct >= 2:  etki = "ölçülebilir biçimde azaltılmıştır"
    else:                     etki = "sınırlı düzeyde azaltılmıştır"

    if butce_fark > 0.5:
        butce_c = f"vergi gelirleri %{butce_fark:.1f} oranında artarken"
    elif butce_fark < -0.5:
        butce_c = f"vergi gelirleri %{abs(butce_fark):.1f} oranında azalırken"
    else:
        butce_c = "vergi gelirleri büyük ölçüde sabit tutulurken"

    dilim_notu = ""
    if opt_dilim_sec and dilim_sayisi_sonuc:
        dilim_notu = f" Optimal dilim sayısı {dilim_sayisi_sonuc} olarak belirlendi."

    ozet = (
        f"Bu optimizasyon, {senaryo_acik} Gini katsayısı "
        f"%{iyilesme_pct:.2f} oranında {etki} ({baseline_gini:.4f} → {opt_gini:.4f}).{dilim_notu} "
        f"{butce_c.capitalize()} yeniden dağılım etkisi ağırlıklı olarak "
        f"üst {artan} gelir grubunun vergi yükünün artırılması ve "
        f"alt {azalan} gelir grubunun yükünün hafifletilmesi yoluyla sağlanmıştır."
    )
    return ozet


def otomatik_yorum(opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                    opt_oranlar, opt_sinirlar, opt_gini, opt_ort,
                    baseline_gini, baseline_ort, opt_ov, baseline_ov,
                    min_oran, max_oran, min_oran_fark, max_oran_fark,
                    min_sinir, son_dilim_min, min_sinir_fark, butce_tolerans,
                    dilim_sayisi_sonuc, sure):
    yorumlar = []
    iyilesme     = baseline_gini - opt_gini
    iyilesme_pct = (iyilesme / baseline_gini) * 100
    butce_fark   = (opt_ort - baseline_ort) / baseline_ort * 100

    emoji, etiket, aciklama = policy_type_tespiti(
        opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
        min_oran, max_oran, min_oran_fark,
        butce_tolerans, iyilesme_pct, butce_fark, dilim_sayisi_sonuc)

    ozet = policy_summary_olustur(
        opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
        iyilesme_pct, butce_fark, opt_gini, baseline_gini,
        opt_ov, baseline_ov, opt_oranlar, dilim_sayisi_sonuc)

    # Bütçe toleransı yorumu
    if butce_tolerans == 0:
        yorumlar.append(("warning", "Bütçe Toleransı",
            "Sıfır tolerans seçildi; tam gelir tarafsızlığı zorunlu kılındı. "
            "Bu kısıt optimizasyonun yeniden dağılım kapasitesini daraltmaktadır."))
    elif butce_tolerans <= 0.02:
        yorumlar.append(("info", "Bütçe Toleransı",
            f"±%{butce_tolerans*100:.1f} tolerans, dengeli bir seçimdir."))
    else:
        yorumlar.append(("info", "Bütçe Toleransı",
            f"±%{butce_tolerans*100:.1f} tolerans, daha agresif yeniden dağılıma olanak tanır."))

    # Karar değişkeni yorumları
    if opt_oranlar_sec:
        if min_oran <= 0.05:
            yorumlar.append(("info", "Alt Oran Sınırı",
                f"Alt oran %{min_oran*100:.0f}, alt gelir gruplarının yükünü önemli ölçüde hafifletebilir."))
        if max_oran >= 0.50:
            yorumlar.append(("warning", "Üst Oran Sınırı",
                f"Üst oran %{max_oran*100:.0f}, uluslararası karşılaştırmalı açıdan yüksektir. "
                "Davranışsal tepkiler bu modelde dikkate alınmamaktadır."))
        if min_oran_fark <= 0.02:
            yorumlar.append(("warning", "Oran Fark Kısıtı",
                "Küçük minimum fark, dilimler arasındaki progressiviteyi sınırlayabilir."))

    if opt_sinirlar_sec:
        if min_sinir_fark < 50000:
            yorumlar.append(("warning", "Dilim Sınır Farkı",
                f"Sınırlar arası min fark ({min_sinir_fark:,.0f} TL) küçük; "
                "dilimler birbirine yakınlaşabilir."))

    if opt_dilim_sec and dilim_sayisi_sonuc:
        if dilim_sayisi_sonuc >= 12:
            yorumlar.append(("warning", "Dilim Sayısı",
                f"Bulunan optimal {dilim_sayisi_sonuc} dilim, OECD ortalamasının üzerindedir. "
                "Uygulamada karmaşıklık artabilir."))
        elif dilim_sayisi_sonuc <= 5:
            yorumlar.append(("info", "Dilim Sayısı",
                f"Optimal dilim sayısı {dilim_sayisi_sonuc}; mevcut sistemle aynı düzeyde."))
        else:
            yorumlar.append(("success", "Dilim Sayısı",
                f"Optimal dilim sayısı {dilim_sayisi_sonuc} olarak belirlendi. "
                "Mevcut 5 dilime göre daha hassas bir yapı oluşturuldu."))

    # Gini yorumu
    if iyilesme_pct >= 5:
        yorumlar.append(("success", "Gini Katsayısı İyileşmesi",
            f"%{iyilesme_pct:.2f} iyileşme ({baseline_gini:.4f} → {opt_gini:.4f}). "
            "Anlamlı bir eşitsizlik azalması."))
    elif iyilesme_pct >= 2:
        yorumlar.append(("success", "Gini Katsayısı İyileşmesi",
            f"%{iyilesme_pct:.2f} iyileşme ({baseline_gini:.4f} → {opt_gini:.4f}). "
            "Politika açısından anlamlı bir sonuç."))
    elif iyilesme_pct >= 0:
        yorumlar.append(("warning", "Gini Katsayısı İyileşmesi",
            f"%{iyilesme_pct:.2f} marjinal iyileşme. Kısıtlar gevşetilerek artırılabilir."))
    else:
        yorumlar.append(("error", "Gini Katsayısı",
            "Optimizasyon sonucunda Gini kötüleşti. Parametre kombinasyonu gözden geçirilmeli."))

    # Yeniden dağılım
    azalan = sum(1 for f in (opt_ov - baseline_ov) if f < -10)
    artan  = sum(1 for f in (opt_ov - baseline_ov) if f > 10)
    if azalan > 0 and artan > 0:
        yorumlar.append(("success", "Yeniden Dağılım Etkisi",
            f"Alt {azalan} grup lehine yük azaldı, üst {artan} grup lehine arttı."))
    elif azalan > 0:
        yorumlar.append(("warning", "Yeniden Dağılım Etkisi",
            f"Alt {azalan} grupta hafifletme var ancak üst gruplar etkilenmedi."))

    # Bütçe etkisi
    if abs(butce_fark) < 0.5:
        yorumlar.append(("success", "Bütçe Etkisi",
            f"%{butce_fark:+.2f} değişimle bütçe tarafsızlığı sağlandı."))
    elif butce_fark > 0:
        yorumlar.append(("info", "Bütçe Etkisi",
            f"Vergi geliri %{butce_fark:.2f} arttı; gelir artırıcı bir reform."))
    else:
        yorumlar.append(("warning", "Bütçe Etkisi",
            f"Vergi geliri %{abs(butce_fark):.2f} azaldı. Bütçe açığı riski göz önünde bulundurulmalı."))

    # Üst dilim bağı
    if opt_oranlar_sec and opt_oranlar[-1] >= max_oran * 0.99:
        yorumlar.append(("warning", "Üst Dilim Oranı",
            f"En yüksek oran üst sınıra (%{max_oran*100:.0f}) dayandı. "
            "Sınır artırılırsa ek iyileşme sağlanabilir."))

    return emoji, etiket, aciklama, ozet, yorumlar


# ============================================================
# PDF RAPORU
# ============================================================
def pdf_olustur(opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                opt_oranlar, opt_sinirlar, opt_gini, opt_ort,
                baseline_gini, baseline_ort, etki_df, dilim_df,
                cp_base, cg_base, cp_opt, cg_opt, dilim_sayisi_sonuc):

    kombinasyon = kombinasyon_acikla(opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec).capitalize()

    def tr(metin):
        s = str(metin)
        for a, b in [("ı","i"),("İ","I"),("ş","s"),("Ş","S"),
                     ("ğ","g"),("Ğ","G"),("ü","u"),("Ü","U"),
                     ("ö","o"),("Ö","O"),("ç","c"),("Ç","C"),
                     ("–","-"),("→","->")]:
            s = s.replace(a, b)
        return s

    MAVI      = (41,  128, 185)
    ACIK_MAVI = (214, 234, 248)
    YESIL     = (39,  174,  96)
    ACIK_YES  = (212, 239, 223)
    KIRMIZI   = (192,  57,  43)
    ACIK_KIR  = (250, 219, 216)
    GRI       = (127, 140, 141)
    ACIK_GRI  = (244, 246, 247)
    BEYAZ     = (255, 255, 255)
    KOYU      = (44,   62,  80)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # SAYFA 1: ÖZET
    pdf.add_page()
    pdf.set_fill_color(*MAVI)
    pdf.rect(0, 0, 210, 42, "F")
    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "Turkiye Gelir Vergisi Optimizasyon Raporu", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(10, 22)
    pdf.cell(0, 7, tr(f"Optimizasyon: {kombinasyon}"), ln=True)
    if dilim_sayisi_sonuc:
        pdf.set_xy(10, 30)
        pdf.cell(0, 7, tr(f"Optimal Dilim Sayisi: {dilim_sayisi_sonuc}"), ln=True)
    pdf.set_text_color(*KOYU)
    pdf.ln(8)

    metrikler = [
        ("Baseline Gini",      f"{baseline_gini:.4f}",                          ACIK_GRI,  KOYU),
        ("Optimal Gini",       f"{opt_gini:.4f}",                               ACIK_YES,  YESIL),
        ("Gini Iyilesme (%)",  f"%{(baseline_gini-opt_gini)/baseline_gini*100:.2f}", ACIK_YES, YESIL),
        ("Baseline Ort.Vergi", f"{baseline_ort:,.0f} TL",                       ACIK_GRI,  KOYU),
        ("Yeni Ort.Vergi",     f"{opt_ort:,.0f} TL",                            ACIK_MAVI, MAVI),
        ("Butce Farki",        f"{((opt_ort-baseline_ort)/baseline_ort)*100:+.2f}%",
                               ACIK_KIR if opt_ort > baseline_ort else ACIK_YES,
                               KIRMIZI  if opt_ort > baseline_ort else YESIL),
    ]
    kutu_w, kutu_h = 60, 22
    x0, y0 = 15, pdf.get_y()
    for idx, (baslik, deger, bg, fg) in enumerate(metrikler):
        col = idx % 3
        row = idx // 3
        x = x0 + col * (kutu_w + 5)
        y = y0 + row * (kutu_h + 5)
        pdf.set_fill_color(*bg)
        pdf.set_draw_color(*GRI)
        pdf.rect(x, y, kutu_w, kutu_h, "FD")
        pdf.set_text_color(*GRI)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(x+2, y+3)
        pdf.cell(kutu_w-4, 5, baslik, ln=True)
        pdf.set_text_color(*fg)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_xy(x+2, y+9)
        pdf.cell(kutu_w-4, 8, deger)
    pdf.set_text_color(*KOYU)
    pdf.set_y(y0 + 2*(kutu_h+5) + 10)

    # Lorenz grafiği
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Tam Esitlik')
    ax.plot(cp_base, cg_base, color='#e74c3c', linewidth=2,
            label=f'Mevcut (Gini={baseline_gini:.4f})')
    cp_int = np.linspace(0, 1, len(cp_opt))
    cg_base_int = np.interp(cp_int, cp_base, cg_base)
    ax.fill_between(cp_opt, cg_base_int, cg_opt, alpha=0.25,
                    color='#27ae60', label='Iyilesme alani')
    ax.plot(cp_opt, cg_opt, color='#27ae60', linewidth=2,
            label=f'Optimal (Gini={opt_gini:.4f})')
    ax.set_xlabel("Nufus Payi"); ax.set_ylabel("Gelir Payi")
    ax.set_title("Lorenz Egrisi Karsilastirmasi")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    y_lor = pdf.get_y()
    if y_lor > 180:
        pdf.add_page()
    pdf.set_fill_color(*MAVI)
    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica","B",12)
    pdf.set_x(15)
    pdf.cell(180, 8, "  Lorenz Egrisi", fill=True, ln=True)
    pdf.set_text_color(*KOYU)
    pdf.image(buf, x=15, y=pdf.get_y()+2, w=175)
    pdf.set_y(pdf.get_y() + 95)

    # SAYFA 2: VERGİ DİLİM TABLOSU
    pdf.add_page()
    pdf.set_fill_color(*MAVI)
    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica","B",14)
    pdf.set_x(15)
    pdf.cell(180, 10, "  Vergi Dilim Karsilastirmasi", fill=True, ln=True)
    pdf.set_text_color(*KOYU)
    pdf.ln(4)

    # Dinamik kolon yapısı
    if not opt_oranlar_sec:
        cols = [("Dilim",15),("Aralik",75),("Oran (Sabit)",25),("Yeni Aralik",60)]
    elif not opt_sinirlar_sec:
        cols = [("Dilim",15),("Aralik (Sabit)",75),("Eski Oran",25),("Yeni Oran",25),("Fark",25)]
    else:
        cols = [("Dilim",15),("Aralik",65),("Eski Oran",25),("Yeni Oran",25),("Fark",25)]

    pdf.set_fill_color(*MAVI)
    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica","B",9)
    pdf.set_x(15)
    for baslik, w in cols:
        pdf.cell(w, 8, baslik, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica","",9)
    for i, (_, row) in enumerate(dilim_df.iterrows()):
        fark_str = tr(row.get("Fark", ""))
        try:
            fark_val = float(fark_str.replace("%","").replace("+",""))
            fill_color = ACIK_YES if fark_val < 0 else (ACIK_KIR if fark_val > 0 else BEYAZ)
            fark_color = YESIL if fark_val < 0 else (KIRMIZI if fark_val > 0 else KOYU)
        except:
            fill_color = ACIK_GRI if i%2==0 else BEYAZ
            fark_color = KOYU
        pdf.set_fill_color(*fill_color)
        pdf.set_text_color(*KOYU)
        pdf.set_x(15)
        pdf.cell(15, 7, tr(row["Dilim"]), border=1, fill=True, align="C")
        for col_key, (_, w) in zip(list(row.index)[1:], cols[1:]):
            val = tr(row.get(col_key, ""))
            if col_key == "Fark":
                pdf.set_text_color(*fark_color)
            else:
                pdf.set_text_color(*KOYU)
            pdf.cell(w, 7, val, border=1, fill=True, align="C")
        pdf.set_text_color(*KOYU)
        pdf.ln()

    # SAYFA 3: ETKİ ANALİZİ
    pdf.add_page()
    pdf.set_fill_color(*MAVI)
    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica","B",14)
    pdf.set_x(15)
    pdf.cell(180, 10, "  Gelir Grubu Etki Analizi", fill=True, ln=True)
    pdf.set_text_color(*KOYU)
    pdf.ln(2)
    pdf.set_font("Helvetica","I",8)
    pdf.set_text_color(*GRI)
    pdf.set_x(15)
    pdf.cell(0, 6, "Yesil: vergi azaldi  |  Kirmizi: vergi artti", ln=True)
    pdf.set_text_color(*KOYU)
    pdf.ln(2)

    ecols = [("Grup",15),("Hane Geliri",40),("Eski Vergi",38),("Yeni Vergi",38),("Fark",38)]
    pdf.set_fill_color(*MAVI)
    pdf.set_text_color(*BEYAZ)
    pdf.set_font("Helvetica","B",9)
    pdf.set_x(15)
    for baslik, w in ecols:
        pdf.cell(w, 8, baslik, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica","",8)
    for i, (_, row) in enumerate(etki_df.iterrows()):
        fark_str = tr(row["Fark"])
        try:
            fark_val = int(fark_str.replace("+","").replace(" TL","").replace(",","").replace(".",""))
            fill_color = ACIK_YES if fark_val <= 0 else ACIK_KIR
            fark_color = YESIL if fark_val <= 0 else KIRMIZI
        except:
            fill_color = ACIK_GRI if i%2==0 else BEYAZ
            fark_color = KOYU
        pdf.set_fill_color(*fill_color)
        pdf.set_text_color(*KOYU)
        pdf.set_x(15)
        pdf.cell(15, 6, tr(row["Grup"]),        border=1, fill=True, align="C")
        pdf.cell(40, 6, tr(row["Hane Geliri"]), border=1, fill=True, align="R")
        pdf.cell(38, 6, tr(row["Eski Vergi"]),  border=1, fill=True, align="R")
        pdf.cell(38, 6, tr(row["Yeni Vergi"]),  border=1, fill=True, align="R")
        pdf.set_text_color(*fark_color)
        pdf.cell(38, 6, fark_str, border=1, fill=True, align="R")
        pdf.set_text_color(*KOYU)
        pdf.ln()

    return bytes(pdf.output())


# ============================================================
# SEKMELER
# ============================================================
tab1, tab2 = st.tabs(["🔧 Optimizasyon", "📈 Duyarlılık Analizi"])

# ============================================================
# TAB 1: OPTİMİZASYON
# ============================================================
with tab1:
    if not optimize_et:
        st.info("Sol panelden **karar değişkenlerini** ve parametreleri ayarlayın, ardından **Optimize Et** butonuna basın.")

        # Seçili kombinasyonu göster
        if sec_listesi:
            st.markdown(f"**Seçilen kombinasyon:** {kombinasyon_label}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Gini", f"{baseline_gini:.4f}")
        col2.metric("Baseline Ort. Vergi", f"{baseline_ort:,.0f} TL")
        col3.metric("Mevcut Dilim Sayısı", "5")

        st.subheader("Mevcut Vergi Sistemi")
        tam = np.concatenate([[0], ESKI_SINIRLAR, [np.inf]])
        rows = []
        for i in range(5):
            aralik = (f"{tam[i]:,.0f} TL ve üzeri" if np.isinf(tam[i+1])
                      else f"{tam[i]:,.0f} – {tam[i+1]:,.0f} TL")
            rows.append({"Dilim": i+1, "Gelir Aralığı": aralik,
                         "Oran": f"%{MEVCUT_ORANLAR[i]*100:.0f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        t0 = time.time()
        with st.spinner("Optimizasyon çalışıyor, lütfen bekleyin..."):
            opt_oranlar_res, opt_sinirlar_res = calistir_optimizasyon(
                opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                dilim_sayisi, dilim_aralik_min, dilim_aralik_max,
                min_oran, max_oran, min_oran_fark, max_oran_fark,
                min_sinir, son_dilim_min, min_sinir_fark, butce_tolerans)
        sure = time.time() - t0

        # Gerçek dilim sayısını bul
        k_sonuc = len(opt_oranlar_res)
        dilim_sayisi_sonuc = k_sonuc if opt_dilim_sec else None

        baseline_ov = vergi_hesapla(MEVCUT_ORANLAR, ESKI_SINIRLAR)
        opt_ov      = vergi_hesapla(opt_oranlar_res, opt_sinirlar_res)
        opt_gini    = gini_hesapla(opt_ov)
        opt_ort     = ort_vergi_hesapla(opt_ov)
        iyilesme    = baseline_gini - opt_gini

        st.caption(f"⏱️ Optimizasyon {sure:.1f} saniyede tamamlandı. | Kombinasyon: **{kombinasyon_label}**"
                   + (f" | Optimal dilim sayısı: **{k_sonuc}**" if opt_dilim_sec else ""))

        st.subheader("📊 Sonuçlar")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Baseline Gini", f"{baseline_gini:.4f}")
        col2.metric("Optimal Gini", f"{opt_gini:.4f}",
                    delta=f"{-iyilesme:.4f}", delta_color="inverse")
        col3.metric("Gini İyileşme (%)", f"%{(iyilesme/baseline_gini)*100:.2f}")
        col4.metric("Mevcut Ort. Vergi", f"{baseline_ort:,.0f} TL")
        col5.metric("Yeni Ort. Vergi", f"{opt_ort:,.0f} TL",
                    delta=f"{opt_ort - baseline_ort:+,.0f} TL")

        if opt_dilim_sec:
            st.info(f"🔢 Dilim sayısı optimizasyonu: {dilim_aralik_min}–{dilim_aralik_max} aralığında **{k_sonuc} dilim** en iyi sonucu verdi.")

        st.markdown("---")
        col_tablo, col_grafik = st.columns([1, 1])

        with col_tablo:
            st.subheader("Vergi Dilim Karşılaştırması")
            tam_eski = np.concatenate([[0], ESKI_SINIRLAR, [np.inf]])
            tam_yeni = np.concatenate([[0], opt_sinirlar_res, [np.inf]])

            dilim_rows = []
            for i in range(k_sonuc):
                row_data = {"Dilim": i + 1}

                # Eski aralık (varsa)
                if i < len(ESKI_SINIRLAR) + 1:
                    eski_aralik = (f"{tam_eski[i]:,.0f} TL ve üzeri" if np.isinf(tam_eski[i+1])
                                   else f"{tam_eski[i]:,.0f} – {tam_eski[i+1]:,.0f} TL")
                else:
                    eski_aralik = "—"

                # Yeni aralık
                yeni_aralik = (f"{tam_yeni[i]:,.0f} TL ve üzeri" if np.isinf(tam_yeni[i+1])
                               else f"{tam_yeni[i]:,.0f} – {tam_yeni[i+1]:,.0f} TL")

                if not opt_sinirlar_sec:
                    row_data["Aralik (Sabit)"] = eski_aralik
                else:
                    row_data["Eski Aralik"] = eski_aralik
                    row_data["Yeni Aralik"] = yeni_aralik

                if not opt_oranlar_sec:
                    oran = MEVCUT_ORANLAR[i] if i < len(MEVCUT_ORANLAR) else opt_oranlar_res[i]
                    row_data["Oran (Sabit)"] = f"%{oran*100:.0f}"
                else:
                    eski_oran = f"%{MEVCUT_ORANLAR[i]*100:.0f}" if i < len(MEVCUT_ORANLAR) else "—"
                    fark = ((opt_oranlar_res[i] - MEVCUT_ORANLAR[i]) * 100
                            if i < len(MEVCUT_ORANLAR) else 0)
                    row_data["Eski Oran"] = eski_oran
                    row_data["Yeni Oran"] = f"%{opt_oranlar_res[i]*100:.1f}"
                    row_data["Fark"] = f"{fark:+.1f}%"

                dilim_rows.append(row_data)

            dilim_df = pd.DataFrame(dilim_rows)
            st.dataframe(dilim_df, use_container_width=True, hide_index=True)

            st.subheader("Gelir Grubu Etki Analizi")
            etki_rows = []
            for i in range(n):
                etki_rows.append({
                    "Grup": f"%{(i+1)*5}",
                    "Hane Geliri": f"{gelir_st_haric[i]:,.0f} TL",
                    "Eski Vergi": f"{baseline_ov[i]:,.0f} TL",
                    "Yeni Vergi": f"{opt_ov[i]:,.0f} TL",
                    "Fark": f"{opt_ov[i]-baseline_ov[i]:+,.0f} TL"
                })
            etki_df = pd.DataFrame(etki_rows)
            st.dataframe(etki_df, use_container_width=True, hide_index=True)

        with col_grafik:
            st.subheader("Lorenz Eğrisi")
            cp_base, cg_base = lorenz_egri(baseline_ov)
            cp_opt,  cg_opt  = lorenz_egri(opt_ov)

            fig_lorenz = go.Figure()
            fig_lorenz.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                name='Tam Eşitlik', line=dict(dash='dash', color='gray', width=1)))
            fig_lorenz.add_trace(go.Scatter(x=cp_base, y=cg_base, mode='lines',
                name=f'Mevcut (Gini={baseline_gini:.4f})',
                line=dict(color='red', width=2)))
            fig_lorenz.add_trace(go.Scatter(x=cp_opt, y=cg_opt, mode='lines',
                name=f'Optimal (Gini={opt_gini:.4f})',
                line=dict(color='green', width=2),
                fill='tonexty', fillcolor='rgba(0,200,0,0.15)'))
            fig_lorenz.update_layout(
                xaxis_title="Nüfus Payı", yaxis_title="Gelir Payı",
                height=380, margin=dict(t=10),
                legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig_lorenz, use_container_width=True)

            st.subheader("Gelir Grubuna Göre Vergi Farkı")
            farklar = opt_ov - baseline_ov
            renkler = ['#2ecc71' if f <= 0 else '#e74c3c' for f in farklar]
            gruplar = [f"%{(i+1)*5}" for i in range(n)]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=gruplar, y=farklar, marker_color=renkler))
            fig_bar.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            fig_bar.update_layout(
                xaxis_title="Gelir Grubu", yaxis_title="Vergi Farkı (TL)",
                height=350, margin=dict(t=10))
            st.plotly_chart(fig_bar, use_container_width=True)

            # Otomatik Yorum
            st.markdown("---")
            st.subheader("🧠 Politika Zekası Katmanı")

            emoji, etiket, aciklama, ozet, yorumlar = otomatik_yorum(
                opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                opt_oranlar_res, opt_sinirlar_res, opt_gini, opt_ort,
                baseline_gini, baseline_ort, opt_ov, baseline_ov,
                min_oran, max_oran, min_oran_fark, max_oran_fark,
                min_sinir, son_dilim_min, min_sinir_fark, butce_tolerans,
                dilim_sayisi_sonuc, sure)

            st.markdown(f"""
            <div style="border:2px solid #2c3e50; border-radius:10px;
                        padding:16px 20px; margin-bottom:16px;
                        background:#f8f9fa;">
                <div style="font-size:13px; color:#7f8c8d;
                            font-weight:600; letter-spacing:1px;
                            text-transform:uppercase; margin-bottom:4px;">
                    Tespit Edilen Reform Tipi
                </div>
                <div style="font-size:20px; font-weight:700;
                            color:#2c3e50; margin-bottom:6px;">
                    {emoji} {etiket}
                </div>
                <div style="font-size:13px; color:#555;">
                    {aciklama}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="border-left:4px solid #2980b9; padding:12px 16px;
                        background:#eaf4fc; border-radius:0 8px 8px 0;
                        margin-bottom:16px;">
                <div style="font-size:12px; color:#2980b9; font-weight:700;
                            letter-spacing:1px; margin-bottom:6px;">
                    REFORM ÖZETİ
                </div>
                <div style="font-size:13px; color:#2c3e50; line-height:1.6;">
                    {ozet}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Detaylı Değerlendirme**")
            for tip, baslik, metin in yorumlar:
                if tip == "success":  st.success(f"**{baslik}:** {metin}")
                elif tip == "warning": st.warning(f"**{baslik}:** {metin}")
                elif tip == "error":   st.error(f"**{baslik}:** {metin}")
                else:                  st.info(f"**{baslik}:** {metin}")

            st.markdown("---")
            pdf_bytes = pdf_olustur(
                opt_oranlar_sec, opt_sinirlar_sec, opt_dilim_sec,
                opt_oranlar_res, opt_sinirlar_res,
                opt_gini, opt_ort, baseline_gini, baseline_ort,
                etki_df, dilim_df, cp_base, cg_base, cp_opt, cg_opt,
                dilim_sayisi_sonuc)
            st.download_button(
                label="📄 PDF Raporu İndir",
                data=pdf_bytes,
                file_name="vergi_optimizasyon.pdf",
                mime="application/pdf")


# ============================================================
# TAB 2: DUYARLILIK ANALİZİ
# ============================================================
with tab2:
    st.subheader("📈 Duyarlılık Analizi")
    st.markdown("""
    Seçilen parametrenin belirli bir aralıkta değiştirilmesiyle optimal Gini'nin
    nasıl değiştiğini gösterir. Her nokta için ayrı optimizasyon çalışır.
    """)

    col_ayar, col_sonuc = st.columns([1, 2])

    with col_ayar:
        st.markdown("**Analiz Ayarları**")

        st.markdown("**Optimize Edilecek Değişkenler**")
        duy_opt_oranlar  = st.checkbox("Vergi Oranları",  value=True, key="duy_oran")
        duy_opt_sinirlar = st.checkbox("Dilim Sınırları", value=False, key="duy_sinir")
        duy_opt_dilim    = st.checkbox("Dilim Sayısı",    value=False, key="duy_dilim")

        if not any([duy_opt_oranlar, duy_opt_sinirlar, duy_opt_dilim]):
            st.error("En az bir değişken seçilmeli!")

        parametre = st.selectbox("Analiz edilecek parametre", [
            "Bütçe Toleransı (%)",
            "Min Oran (%)",
            "Max Oran (%)",
            "Oranlar Arası Min Fark (%)",
            "Sınırlar Arası Min Fark (TL)",
        ])

        col_a, col_b = st.columns(2)
        with col_a:
            default_min = 0.0 if ("Tolerans" in parametre or ("Fark" in parametre and "TL" not in parametre)) else (50000.0 if "TL" in parametre else 5.0)
            p_min = st.number_input("Başlangıç değeri", value=default_min)
        with col_b:
            default_max = 5.0 if ("Tolerans" in parametre or ("Fark" in parametre and "TL" not in parametre)) else (300000.0 if "TL" in parametre else 45.0)
            p_max = st.number_input("Bitiş değeri", value=default_max)

        n_nokta = st.slider("Nokta sayısı", min_value=3, max_value=10, value=5)
        st.warning(f"⚠️ {n_nokta} optimizasyon çalışacak.")

        duy_calistir = st.button("📊 Analizi Çalıştır", type="primary",
                                  use_container_width=True,
                                  disabled=not any([duy_opt_oranlar, duy_opt_sinirlar, duy_opt_dilim]))

    with col_sonuc:
        if duy_calistir:
            degerler = np.linspace(p_min, p_max, n_nokta)
            gini_sonuclari = []
            progress = st.progress(0, text="Hesaplanıyor...")

            for idx, deger in enumerate(degerler):
                p = {
                    "butce_tolerans": 0.02,
                    "min_oran": 0.10,
                    "max_oran": 0.45,
                    "min_oran_fark": 0.03,
                    "max_oran_fark": 0.13,
                    "min_sinir": 50000,
                    "son_dilim_min": 2000000,
                    "min_sinir_fark": 100000,
                    "dilim_sayisi": 5,
                    "dilim_aralik_min": 5,
                    "dilim_aralik_max": 9,
                }

                if parametre == "Bütçe Toleransı (%)":
                    p["butce_tolerans"] = deger / 100
                elif parametre == "Min Oran (%)":
                    p["min_oran"] = deger / 100
                elif parametre == "Max Oran (%)":
                    p["max_oran"] = deger / 100
                elif parametre == "Oranlar Arası Min Fark (%)":
                    p["min_oran_fark"] = deger / 100
                elif parametre == "Sınırlar Arası Min Fark (TL)":
                    p["min_sinir_fark"] = int(deger)

                try:
                    opt_o, opt_s = calistir_optimizasyon(
                        duy_opt_oranlar, duy_opt_sinirlar, duy_opt_dilim,
                        p["dilim_sayisi"], p["dilim_aralik_min"], p["dilim_aralik_max"],
                        p["min_oran"], p["max_oran"],
                        p["min_oran_fark"], p["max_oran_fark"],
                        p["min_sinir"], p["son_dilim_min"],
                        p["min_sinir_fark"], p["butce_tolerans"])
                    g = gini_hesapla(vergi_hesapla(opt_o, opt_s))
                except Exception:
                    g = np.nan

                gini_sonuclari.append(g)
                progress.progress((idx+1)/n_nokta,
                                  text=f"Hesaplanıyor... {idx+1}/{n_nokta}")

            progress.empty()

            gecerli = [(d, g) for d, g in zip(degerler, gini_sonuclari) if not np.isnan(g)]
            if gecerli:
                x_vals = [v[0] for v in gecerli]
                y_vals = [v[1] for v in gecerli]

                # Aktif kombinasyon etiketi
                duy_sec = []
                if duy_opt_oranlar:  duy_sec.append("Oranlar")
                if duy_opt_sinirlar: duy_sec.append("Sınırlar")
                if duy_opt_dilim:    duy_sec.append("Dilim Sayısı")
                duy_label = " + ".join(duy_sec)

                fig_duy = go.Figure()
                fig_duy.add_hline(y=baseline_gini, line_dash="dash",
                                  line_color="red", annotation_text="Baseline Gini")
                fig_duy.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='lines+markers',
                    line=dict(color='#2980b9', width=2),
                    marker=dict(size=8, color='#2980b9'),
                    name="Optimal Gini"))
                fig_duy.update_layout(
                    xaxis_title=parametre,
                    yaxis_title="Optimal Gini Katsayısı",
                    height=400,
                    title=f"{parametre} → Optimal Gini İlişkisi ({duy_label})")
                st.plotly_chart(fig_duy, use_container_width=True)

                tablo = pd.DataFrame({
                    parametre: [f"{d:.2f}" for d in x_vals],
                    "Optimal Gini": [f"{g:.4f}" for g in y_vals],
                    "İyileşme": [f"{baseline_gini-g:.4f}" for g in y_vals],
                    "İyileşme (%)": [f"%{(baseline_gini-g)/baseline_gini*100:.2f}" for g in y_vals],
                })
                st.dataframe(tablo, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("**📝 Duyarlılık Değerlendirmesi**")

                min_g   = min(y_vals)
                max_g   = max(y_vals)
                aralik  = max_g - min_g
                min_idx = y_vals.index(min_g)
                min_x   = x_vals[min_idx]

                if aralik < 0.001:
                    egri_tipi = "duz"
                elif y_vals[-1] < y_vals[0]:
                    egri_tipi = "azalan"
                elif y_vals[-1] > y_vals[0]:
                    egri_tipi = "artan"
                elif min_idx not in [0, len(y_vals)-1]:
                    egri_tipi = "u_sekli"
                else:
                    egri_tipi = "diger"

                if egri_tipi == "duz":
                    st.info(f"**Sınırlı Etki:** Seçilen parametre aralığında Gini yalnızca {aralik:.4f} değişmiştir.")
                elif egri_tipi == "azalan":
                    st.success(f"**Tutarlı İyileşme:** {max_g:.4f}'ten {min_g:.4f}'e geriledi. En iyi sonuç {min_x:.2f} değerinde.")
                elif egri_tipi == "artan":
                    st.warning(f"**Ters Etki:** Parametre artışıyla Gini yükseliyor. En iyi sonuç {min_x:.2f} değerinde.")
                elif egri_tipi == "u_sekli":
                    st.info(f"**Optimal Nokta:** {min_x:.2f} değerinde minimum Gini ({min_g:.4f}). Bu noktanın ötesinde eşitsizlik artıyor.")
                else:
                    st.info(f"**Değerlendirme:** En düşük Gini ({min_g:.4f}) parametre değeri {min_x:.2f} iken elde edildi.")

            else:
                st.error("Hiçbir nokta için geçerli sonuç bulunamadı. Parametreleri kontrol edin.")
        else:
            st.info("Sol taraftan parametreleri ayarlayın ve **Analizi Çalıştır** butonuna basın.")
