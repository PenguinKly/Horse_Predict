import streamlit as st
import pandas as pd
import xgboost as xgb
import requests
import io
from bs4 import BeautifulSoup
import re
import warnings

warnings.filterwarnings('ignore')

# 網頁基本設定 (設定標題、圖示、以及適應手機螢幕)
st.set_page_config(
    page_title="賽馬 AI 戰術預測系統",
    page_icon="🐎",
    layout="centered"
)

# --- 輔助函式區域 ---
def clean_person_name(val):
    return re.sub(r'\(.*?\)', '', str(val)).strip()

def extract_rating(val):
    nums = re.findall(r'\d+', str(val))
    return float(nums[0]) if nums else 52.0

def extract_horse_no(val):
    nums = re.findall(r'\d+', str(val))
    return int(nums[0]) if nums else 0

@st.cache_resource
def load_ai_model():
    """ 載入 10 維度純實力 AI 模型並加上快取 """
    model = xgb.XGBClassifier()
    model.load_model("horse_racing_ai_model.json")
    return model

@st.cache_data
def load_memory_databases():
    """ 載入各種歷史記憶庫並加上快取 """
    try:
        history_db = pd.read_csv("advanced_features.csv")
        horse_mem = history_db.drop_duplicates(subset=['馬匹編號'], keep='last').set_index('馬匹編號')
    except:
        horse_mem = pd.DataFrame()

    try:
        jockey_db = pd.read_csv("jockey_stats.csv").set_index('騎師')
        trainer_db = pd.read_csv("trainer_stats.csv").set_index('練馬師')
        hj_db = pd.read_csv("horse_jockey_stats.csv")
        hj_dict = hj_db.set_index(['馬匹編號', '騎師'])['勝率'].to_dict()
    except:
        jockey_db = pd.DataFrame(columns=['勝率'])
        trainer_db = pd.DataFrame(columns=['勝率'])
        hj_dict = {}
        
    return horse_mem, jockey_db, trainer_db, hj_dict

# --- 共用排位表抓取函式 ---
def fetch_race_cards(date_str, venue_str):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    entries_url = f"https://racing.hkjc.com/zh-hk/local/information/entries?racedate={date_str}"
    try:
        res_entries = requests.get(entries_url, headers=headers)
        soup = BeautifulSoup(res_entries.text, 'html.parser')
        name_to_id = {}
        for link in soup.find_all('a', href=True):
            if "horse?horseid=" in link['href']:
                h_id = link['href'].split("=")[-1]
                h_name = link.text.strip()
                if h_name:
                    name_to_id[h_name] = h_id
    except:
        name_to_id = {}

    today_races = []
    model = load_ai_model()
    horse_memory, jockey_db, trainer_db, hj_dict = load_memory_databases()

    for race_no in range(1, 12):
        url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/RaceCard.aspx?RaceDate={date_str}&Racecourse={venue_str}&RaceNo={race_no}"
        try:
            res = requests.get(url, headers=headers)
            if "找不到" in res.text or "沒有賽事" in res.text:
                break
            tables = pd.read_html(io.StringIO(res.text))
            target_tbl = next((tbl for tbl in tables if ('馬名' in str(tbl.columns) or '馬匹' in str(tbl.columns)) and '檔位' in str(tbl.columns)), None)
            if target_tbl is not None:
                df_race = target_tbl.copy()
                if isinstance(df_race.columns, pd.MultiIndex):
                    df_race.columns = df_race.columns.get_level_values(-1)
                df_race.columns = [str(col).replace(' ', '').replace('\n', '') for col in df_race.columns]
                df_race['場次'] = f"第 {race_no} 場"
                if '馬名' in df_race.columns: df_race.rename(columns={'馬名': '馬匹'}, inplace=True)
                if '配磅' in df_race.columns: df_race.rename(columns={'配磅': '負磅'}, inplace=True)
                
                horse_no_col = next((c for c in df_race.columns if '馬號' in str(c) or 'NO' in str(c).upper()), df_race.columns[0])
                df_race['馬號'] = df_race[horse_no_col].apply(extract_horse_no)
                df_race['檔位'] = pd.to_numeric(df_race['檔位'], errors='coerce')
                df_race['負磅'] = pd.to_numeric(df_race['負磅'], errors='coerce')
                
                rating_col = [c for c in df_race.columns if '評分' in c]
                df_race['評分'] = df_race[rating_col[0]].apply(extract_rating) if rating_col else 52.0
                
                df_race['騎師_清理'] = df_race['騎師'].apply(clean_person_name) if '騎師' in df_race.columns else '未知'
                df_race['練馬師_清理'] = df_race['練馬師'].apply(clean_person_name) if '練馬師' in df_race.columns else '未知'
                
                df_race = df_race.dropna(subset=['檔位', '負磅'])
                df_race = df_race[df_race['馬號'] > 0]
                today_races.append((race_no, df_race, name_to_id))
        except:
            pass
    return today_races, horse_memory, jockey_db, trainer_db, hj_dict, model


# --- UI 介面設計 ---
st.title("🐎 賽馬 AI 戰術預測系統")
st.markdown("基於 XGBoost 機器學習，嚴格剔除目標洩漏，使用 **純賽前 10 維度特徵** 進行深度預測。")

st.divider()

st.subheader("🗓️ 設定目標賽事")
col1, col2 = st.columns(2)

with col1:
    target_date = st.text_input("賽事日期 (格式: YYYY/MM/DD)", value="2026/10/01")
with col2:
    venue = st.selectbox("賽事場地", options=["ST (沙田)", "HV (跑馬地)"])
    venue_code = venue.split(" ")[0]

# --- 按鈕佈局 ---
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
action = None

with col_btn1:
    if st.button("🚀 普通預測", use_container_width=True):
        action = "predict_normal"
with col_btn2:
    if st.button("🚫 忽視騎師", use_container_width=True):
        action = "predict_ignore"
with col_btn3:
    if st.button("📊 歷史回測", use_container_width=True):
        action = "backtest"
with col_btn4:
    if st.button("🎯 智慧投注", use_container_width=True):
        action = "recommend"


# --- 功能 1 & 2：普通預測與忽視騎師預測 ---
if action in ["predict_normal", "predict_ignore"]:
    ignore_jockey = (action == "predict_ignore")
    mode_str = "(忽視騎師權重模式)" if ignore_jockey else ""
    if ignore_jockey:
        st.warning("🚫 已啟動「忽視騎師權重」模式：騎師與練馬師影響力已被完全遮蔽。")
        
    with st.spinner("AI 正在連線賽馬會，讀取排位表與官方即時評分..."):
        today_races, horse_memory, jockey_db, trainer_db, hj_dict, model = fetch_race_cards(target_date, venue_code)
        
        if not today_races:
            st.error(f"無法抓取 {target_date} 的排位表。可能是日期錯誤或賽事尚未公佈。")
        else:
            st.success(f"成功連線！正在為 {len(today_races)} 場賽事進行純實力 10 維度 AI 運算...")
            predictions_list = []
            
            for race_no, race_df, name_to_id in today_races:
                race_name = f"第 {race_no} 場"
                
                for _, row in race_df.iterrows():
                    horse_name = row['馬匹']
                    horse_id = name_to_id.get(horse_name, "")
                    rating = row['評分']
                    horse_no_val = int(row['馬號'])
                    jockey = row.get('騎師_清理', '未知')
                    trainer = row.get('練馬師_清理', '未知')
                    
                    j_win_rate = jockey_db.loc[jockey, '勝率'] if jockey in jockey_db.index else 0.08
                    t_win_rate = trainer_db.loc[trainer, '勝率'] if trainer in trainer_db.index else 0.08
                    hj_win_rate = hj_dict.get((horse_id, jockey), 0.08)
                    
                    if ignore_jockey:
                        j_win_rate, t_win_rate, hj_win_rate = 0.08, 0.08, 0.08
                        
                    recent_rank, hist_win_rate, rest_days, weight_change = 7.0, 0.08, 30.0, 0.0
                    if horse_id in horse_memory.index:
                        mem = horse_memory.loc[horse_id]
                        recent_rank = mem.get('近三仗平均名次', 7.0)
                        hist_win_rate = mem.get('歷史勝率', 0.08)
                        rest_days = mem.get('休賽天數', 30.0)
                        weight_change = mem.get('體重變化', 0.0)
                    
                    # ✨ 嚴格匹配 10 維度
                    features = {
                        '檔位_數值': row['檔位'], '負磅_數值': row['負磅'], '休賽天數': rest_days,
                        '體重變化': weight_change, '近三仗平均名次': recent_rank, '歷史勝率': hist_win_rate,
                        '評分_數值': rating,
                        '騎師勝率': float(j_win_rate), '練馬師勝率': float(t_win_rate), '人馬合作勝率': float(hj_win_rate)
                    }
                    win_prob = model.predict_proba(pd.DataFrame([features]))[:, 1][0]
                    
                    predictions_list.append({
                        '場次': race_name, '馬號': horse_no_val, '馬匹': horse_name,
                        '騎師': jockey, '檔位': int(row['檔位']), '負磅': float(row['負磅']),
                        '評分': int(rating), 'AI預測勝率(%)': round(win_prob * 100, 2)
                    })

            final_predictions = pd.DataFrame(predictions_list)
            st.divider()
            st.subheader(f"🏆 {target_date} {venue_code} 預測結果 {mode_str}")
            
            # ✨ 核心修正：將分組結果轉為 List，並強制提取裡面的數字進行大小排序
            grouped = final_predictions.groupby('場次')
            sorted_groups = sorted(grouped, key=lambda x: int(x[0].replace('第 ', '').replace(' 場', '')))
            
            for name, group in sorted_groups:
                st.markdown(f"**[ {name} ] AI 戰術推薦前四名**")
                sorted_group = group.sort_values(by='AI預測勝率(%)', ascending=False).head(4).drop(columns=['場次'])
                sorted_group['AI預測勝率(%)'] = sorted_group['AI預測勝率(%)'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(sorted_group, hide_index=True, use_container_width=True)

# --- 功能 3：歷史回測 (精簡乾淨的 Wordle 條列風格) ---
elif action == "backtest":
    st.subheader(f"📊 歷史回測詳細報告 (簡潔風格) - {target_date} ({venue_code})")
    with st.spinner("正在載入歷史賽果與記憶庫並執行回測分析..."):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        horse_memory, jockey_db, trainer_db, hj_dict = load_memory_databases()
        model = load_ai_model()
        
        total_races = 0
        ai_top1_hit = 0
        ai_top2_catch_win = 0
        ai_top4_catch_counts = []
        exact_match_count = 0
        
        def clean_name(val): return re.sub(r'\(.*?\)', '', str(val)).strip()

        for race_no in range(1, 12):
            url = f"https://racing.hkjc.com/racing/information/Chinese/Racing/LocalResults.aspx?RaceDate={target_date}&Racecourse={venue_code}&RaceNo={race_no}"
            try:
                res = requests.get(url, headers=headers)
                if "找不到" in res.text or "沒有賽事" in res.text:
                    break
                soup = BeautifulSoup(res.text, 'html.parser')
                name_to_id = {}
                for link in soup.find_all('a', href=True):
                    href = link['href'].lower()
                    if 'horse' in href and ('id=' in href or 'no=' in href):
                        h_id = link['href'].split('=')[-1].split('&')[0].strip()
                        h_name = clean_name(link.text)
                        if h_name and h_id: name_to_id[h_name] = h_id

                tables = pd.read_html(io.StringIO(res.text))
                target_tbl = next((tbl for tbl in tables if '名次' in str(tbl.columns) and '馬名' in str(tbl.columns)), None)
                if target_tbl is not None:
                    df_race = target_tbl.copy()
                    if isinstance(df_race.columns, pd.MultiIndex):
                        df_race.columns = df_race.columns.get_level_values(-1)
                    df_race = df_race[pd.to_numeric(df_race['名次'], errors='coerce').notnull()]
                    
                    predictions_list = []
                            
                    for _, row in df_race.iterrows():
                        horse_name = clean_name(row['馬名'])
                        horse_id = name_to_id.get(horse_name, "")
                        actual_rank = int(row['名次'])
                        jockey = clean_name(row.get('騎師', '未知'))
                        trainer = clean_name(row.get('練馬師', '未知'))
                        
                        j_win_rate = jockey_db.loc[jockey, '勝率'] if jockey in jockey_db.index else 0.08
                        t_win_rate = trainer_db.loc[trainer, '勝率'] if trainer in trainer_db.index else 0.08
                        hj_win_rate = hj_dict.get((horse_id, jockey), 0.08)
                        draw = pd.to_numeric(row.get('檔位', 7), errors='coerce')
                        weight = pd.to_numeric(row.get('實際負磅', 125), errors='coerce')
                        if pd.isna(draw): draw = 7.0
                        if pd.isna(weight): weight = 125.0
                        
                        rating, recent_rank, hist_win_rate = 52.0, 7.0, 0.08
                        if horse_id in horse_memory.index:
                            mem = horse_memory.loc[horse_id]
                            rating = mem.get('評分_數值', mem.get('評分', 52.0))
                            recent_rank = mem.get('近三仗平均名次', 7.0)
                            hist_win_rate = mem.get('歷史勝率', 0.08)
                            
                        # ✨ 嚴格匹配 10 維度
                        features = {
                            '檔位_數值': float(draw), '負磅_數值': float(weight), '休賽天數': 30.0, '體重變化': 0.0,
                            '近三仗平均名次': recent_rank, '歷史勝率': hist_win_rate, '評分_數值': float(rating),
                            '騎師勝率': float(j_win_rate), '練馬師勝率': float(t_win_rate), '人馬合作勝率': float(hj_win_rate)
                        }
                        win_prob = model.predict_proba(pd.DataFrame([features]))[:, 1][0]
                        predictions_list.append({'馬匹': horse_name, 'AI預測勝率': win_prob, '真實名次': actual_rank})

                    df_pred = pd.DataFrame(predictions_list).sort_values(by='AI預測勝率', ascending=False)
                    ai_top4 = df_pred.head(4)['馬匹'].tolist()
                    ai_top2 = df_pred.head(2)['馬匹'].tolist()
                    actual_top4 = df_pred[df_pred['真實名次'] <= 4]['馬匹'].tolist()
                    actual_winner = df_pred[df_pred['真實名次'] == 1]['馬匹'].tolist()[0] if len(df_pred[df_pred['真實名次'] == 1]) > 0 else "無資料"
                    actual_rank_dict = dict(zip(df_pred['馬匹'], df_pred['真實名次']))
                    
                    total_races += 1
                    if ai_top4 and actual_winner != "無資料":
                        if ai_top4[0] == actual_winner: ai_top1_hit += 1
                        if actual_winner in ai_top2: ai_top2_catch_win += 1
                    match_count = len(set(ai_top4) & set(actual_top4))
                    ai_top4_catch_counts.append(match_count)
                    
                    for i, h in enumerate(ai_top4):
                        if (i + 1) == actual_rank_dict.get(h, 99):
                            exact_match_count += 1
                    
                    st.markdown(f"### 🏁 [ 第 {race_no} 場 ]")
                    st.markdown(f"* **真實冠軍**: `{actual_winner}`")
                    
                    st.markdown("* **AI 推薦前四名**:")
                    for idx, h in enumerate(ai_top4):
                        r_rank = actual_rank_dict.get(h, 99)
                        ai_rank = idx + 1
                        if ai_rank == r_rank:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🟢 **#{ai_rank} {h}** (真實 #{r_rank})")
                        elif r_rank <= 4:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🟡 **#{ai_rank} {h}** (真實 #{r_rank})")
                        else:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;⬛ **#{ai_rank} {h}** (真實 #{r_rank})")
                            
                    st.markdown(f"* **前四名命中數**: `{match_count}/4 匹`")
                    st.divider()
            except:
                pass

        if total_races > 0:
            st.subheader("📈 回測績效總結報告")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("單邊獨贏命中率", f"{ai_top1_hit}/{total_races}", f"{round((ai_top1_hit/total_races)*100,1)}%")
            m2.metric("頭馬涵蓋率 (前2名)", f"{ai_top2_catch_win}/{total_races}", f"{round((ai_top2_catch_win/total_races)*100,1)}%")
            m3.metric("前四名平均命中", f"{round(sum(ai_top4_catch_counts)/total_races, 2)} 匹")
            m4.metric("名次完全吻合", f"{exact_match_count} 匹")
        else:
            st.warning("找不到該日期的歷史賽果或尚未有完賽資料。")

# --- 功能 4：AI 全面彩池智慧投注推薦 (含膽拖策略升級版) ---
elif action == "recommend":
    st.subheader(f"🎯 AI 全面彩池智慧投注與膽拖推薦 - {target_date} ({venue_code})")
    with st.spinner("正在連線賽馬會排位與賠率狀態，計算各彩池量化膽拖組合..."):
        today_races, horse_memory, jockey_db, trainer_db, hj_dict, model = fetch_race_cards(target_date, venue_code)
        
        if not today_races:
            st.error(f"無法取得 {target_date} 的排位資料。")
        else:
            for race_no, race_df, name_to_id in today_races:
                race_name = f"第 {race_no} 場"
                predictions_list = []
                
                for _, row in race_df.iterrows():
                    horse_name = row['馬匹']
                    horse_id = name_to_id.get(horse_name, "")
                    rating = row['評分']
                    horse_no_val = int(row['馬號'])
                    jockey = row.get('騎師_清理', '未知')
                    
                    j_win_rate = jockey_db.loc[jockey, '勝率'] if jockey in jockey_db.index else 0.08
                    t_win_rate = trainer_db.loc[row.get('練馬師_清理', '未知'), '勝率'] if row.get('練馬師_清理', '未知') in trainer_db.index else 0.08
                    hj_win_rate = hj_dict.get((horse_id, jockey), 0.08)
                    
                    recent_rank, hist_win_rate, rest_days, weight_change = 7.0, 0.08, 30.0, 0.0
                    if horse_id in horse_memory.index:
                        mem = horse_memory.loc[horse_id]
                        recent_rank = mem.get('近三仗平均名次', 7.0)
                        hist_win_rate = mem.get('歷史勝率', 0.08)
                        rest_days = mem.get('休賽天數', 30.0)
                        weight_change = mem.get('體重變化', 0.0)
                        
                    # ✨ 嚴格匹配 10 維度
                    features = {
                        '檔位_數值': row['檔位'], '負磅_數值': row['負磅'], '休賽天數': rest_days,
                        '體重變化': weight_change, '近三仗平均名次': recent_rank, '歷史勝率': hist_win_rate,
                        '評分_數值': rating, 
                        '騎師勝率': float(j_win_rate), '練馬師勝率': float(t_win_rate), '人馬合作勝率': float(hj_win_rate)
                    }
                    win_prob = model.predict_proba(pd.DataFrame([features]))[:, 1][0]
                    implied_odds = round(max(1.5, 0.85 / (win_prob + 0.001)), 2)
                    
                    predictions_list.append({
                        '馬號': horse_no_val, '馬匹': horse_name, '勝率': win_prob, '預估賠率': implied_odds
                    })
                    
                df_pred = pd.DataFrame(predictions_list).sort_values(by='勝率', ascending=False).reset_index(drop=True)
                
                if len(df_pred) >= 4:
                    h1, h2, h3, h4 = df_pred.iloc[0], df_pred.iloc[1], df_pred.iloc[2], df_pred.iloc[3]
                    
                    st.markdown(f"### 📌 [ {race_name} ] AI 專業膽拖投注策略")
                    rec_col1, rec_col2 = st.columns(2)
                    
                    with rec_col1:
                        st.markdown(f"""
                        * **獨贏 (Win)**: 
                          * 核心推薦：({h1['馬號']}) {h1['馬匹']} (勝率: {h1['勝率']*100:.1f}%)
                        * **連贏 / 位置Q (Quinella / Q.Place)**: 
                          * 膽拖策略：以 **{h1['馬號']} 號** 做膽，拖 **{h2['馬號']}、{h3['馬號']} 號** (共2注)
                        * **二重彩 (Exacta)**: 
                          * 順序策略：({h1['馬號']} 冠軍) ➔ 拖 ({h2['馬號']}、{h3['馬號']} 亞軍)
                        """)
                    with rec_col2:
                        st.markdown(f"""
                        * **三重彩 / 單T (Tricast / Tierce)**: 
                          * 膽拖策略：以 **{h1['馬號']} 號** 做馬膽，配搭 **{h2['馬號']}、{h3['馬號']}、{h4['馬號']} 號** 為配腳
                        * **四連環 (First 4)**: 
                          * 複式策略：({h1['馬號']}), ({h2['馬號']}), ({h3['馬號']}), ({h4['馬號']}) 四匹互聯複式
                        * **四重彩 (Quartet)**: 
                          * 膽拖策略：以 **{h1['馬號']} 號** 做一馬膽，拖 **{h2['馬號']}、{h3['馬號']}、{h4['馬號']} 號**
                        """)
                    st.divider()