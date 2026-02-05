import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tempfile
import os

# ====== データ読み込み ======
@st.cache_data
def load_soraru_data():
    df = pd.read_csv("soraru_data.csv")
    return df

df = load_soraru_data()

# ====== 音声特徴量抽出 ======
def extract_user_features_from_file(uploaded_file, duration=15):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, duration=duration)
    os.remove(tmp_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

# ====== 距離 → スコア変換 ======
def convert_to_score(dist, min_dist, max_dist):
    if max_dist == min_dist:
        return 50.0
    score = 1 - (dist - min_dist) / (max_dist - min_dist)
    score = score * 100
    return max(5, min(score, 100))  # 最低5%

# ====== 総合そらる率 + 曲ランキング ======
def analyze_all(user_feat, df):
    song_feats = df[[f"mfcc_{i}" for i in range(26)]].values

    all_dists = []
    for i in range(len(song_feats)):
        for j in range(i + 1, len(song_feats)):
            all_dists.append(np.linalg.norm(song_feats[i] - song_feats[j]))

    min_dist = min(all_dists)
    max_dist = max(all_dists)

    soraru_center = song_feats.mean(axis=0)
    dist_total = np.linalg.norm(user_feat - soraru_center)
    soraru_rate = convert_to_score(dist_total, min_dist, max_dist)

    results = []
    for (_, row), song_feat in zip(df.iterrows(), song_feats):
        dist = np.linalg.norm(user_feat - song_feat)
        score = convert_to_score(dist, min_dist, max_dist)
        results.append({
            "song": row["song"],
            "url": row["youtube_url"],
            "score": score
        })

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)
    return soraru_rate, df_res

# ====== コメント生成 ======
def generate_comment(rate: float) -> str:
    # ★ 智康さんが書いた長文コメントをそのまま使用（省略）
    # ここは前回あなたが送ってくれた内容をそのまま貼り付けてOK
    # 文字数制限のためここでは省略するけど、実際のコードには全文入れてね
    return "（ここにあなたの長文コメントが入ります）"


# ====== Streamlit UI ======
st.set_page_config(page_title="【精密解析】そらる・シンクロ率チェッカー", layout="centered")

# ====== カスタムCSS（そらるテーマ） ======
st.markdown("""
<style>
body {
    background-color: #f7fbff;
}
.title-card {
    background: linear-gradient(135deg, #dceeff, #b7d7ff);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 20px;
    border: 1px solid #aac8ff;
}
.result-box {
    background: #e8f2ff;
    padding: 20px;
    border-left: 6px solid #7fbfff;
    border-radius: 10px;
    margin: 15px 0;
}
.song-card {
    background: #ffffff;
    border: 1px solid #cfe2ff;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ====== タイトル ======
st.markdown("""
<div class="title-card">
    <h1 style="margin:0; line-height:1.3;">
        【精密解析】<br>そらる・シンクロ率チェッカー
    </h1>
    <p>あなたの声に最も近い楽曲も判定！</p>
</div>
""", unsafe_allow_html=True)

# SNSアイコン
st.markdown("""
<div style="text-align:center;">
<a href="https://twitter.com/soraruru" target="_blank">
    <img src="https://abs.twimg.com/favicons/twitter.ico" width="32">
</a>
&nbsp;&nbsp;
<a href="https://www.youtube.com/@soraru" target="_blank">
    <img src="https://www.youtube.com/s/desktop/fe2f1f8e/img/favicon_32x32.png" width="32">
</a>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ====== アップロード ======
st.subheader("① 音声ファイルをアップロード")
uploaded_file = st.file_uploader("対応形式：wav / mp3 / m4a", type=["wav", "mp3", "m4a"])

st.markdown("---")

# ====== 精密解析 ======
st.subheader("② 精密解析")
analyze_button = st.button("🔍 精密解析スタート")

if analyze_button:
    if uploaded_file is None:
        st.warning("先に音声ファイルをアップロードしてください。")
    else:
        with st.spinner("解析中…"):
            try:
                user_feat = extract_user_features_from_file(uploaded_file, duration=15)
                soraru_rate, result = analyze_all(user_feat, df)
            except Exception as e:
                st.error(f"解析中にエラーが発生しました：{e}")
                st.stop()

        st.success("解析が完了しました！")

        # ====== 結果 ======
        st.subheader("③ 結果")

        st.markdown(f"""
        <div class="result-box">
            <h2>あなたのそらる・シンクロ率： {soraru_rate:.1f}%</h2>
            <p>{generate_comment(soraru_rate)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ====== ランキング ======
        st.subheader("④ あなたに近い そらる楽曲 TOP5")

        top5 = result.head(5).reset_index(drop=True)

        # 1位
        top1 = top5.iloc[0]
        st.markdown(f"""
        <div class="song-card">
            <h3>🥇 第1位：{top1['song']}（{top1['score']:.1f}%）</h3>
        </div>
        """, unsafe_allow_html=True)
        st.video(top1["url"])
        st.write(f"[YouTubeで開く]({top1['url']})")

        # 2〜5位
        for i in range(1, len(top5)):
            row = top5.iloc[i]
            st.markdown(f"""
            <div class="song-card">
                <h4>🥈 第{i+1}位：{row['song']}（{row['score']:.1f}%）</h4>
                <a href="{row['url']}" target="_blank">YouTubeで開く</a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("⑤ Xでシェア（後で実装）")
else:
    st.info("音声ファイルをアップロードしてから「精密解析スタート」を押してください。")