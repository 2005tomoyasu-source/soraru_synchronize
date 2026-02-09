import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tempfile
import os

# ★ Streamlit のページ設定（必ず最上部） ★
st.set_page_config(
    page_title="【精密解析】そらる・シンクロ率チェッカー",
    layout="centered"
)

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

# ====== コメント生成（あなたの長文コメントをそのまま使用） ======
def generate_comment(rate: float) -> str:
    # ★ 智康さんが作った長文コメントをそのまま貼ってください
    # ここでは省略しますが、前回のメッセージの内容をそのまま入れてOK
    return "（ここにあなたの長文コメント全文を貼ってください）"

# ====== URLパラメータ読み取り ======
params = st.query_params

shared_rate = None
shared_song = None

if "rate" in params:
    try:
        shared_rate = float(params["rate"])
    except:
        shared_rate = None

if "song1" in params:
    shared_song = params["song1"]

# ====== カスタムCSS（そらるテーマ） ======
st.markdown("""
<style>

body {
    background-color: #f4f8ff;
    font-family: 'Hiragino Maru Gothic ProN', 'Yu Gothic', sans-serif;
}

/* タイトルカード */
.title-card {
    background: linear-gradient(135deg, #e9f2ff, #d7e6ff);
    padding: 35px 20px;
    border-radius: 18px;
    border: 2px solid #8fb4ff;
    margin-bottom: 25px;
    box-shadow: 0 4px 12px rgba(120, 150, 220, 0.25);
}

/* タイトル文字 */
.title-text {
    color: #0f1a33;  /* ★濃いネイビーで視認性UP */
    font-weight: 800;
    font-size: 2.3rem;
    line-height: 1.3;
    text-align: center;
    text-shadow: 0px 1px 3px rgba(255,255,255,0.9);
}

/* サブタイトル */
.subtitle-text {
    color: #2a4d8f;
    font-size: 1.1rem;
    text-align: center;
    margin-top: 8px;
}

/* コメントボックス */
.result-box {
    background: #f9fbff;  /* ★ほぼ白に変更 */
    padding: 22px;
    border-left: 6px solid #5fa8ff;
    border-radius: 10px;
    margin: 20px 0;
    border: 1px solid #bcd4ff;  /* ★枠線追加 */
    box-shadow: 0 3px 10px rgba(150, 180, 255, 0.25);
}

.result-box h2, .result-box p {
    color: #0f1a33;  /* ★濃いネイビーで統一 */
}

/* ランキングカード */
.song-card {
    background: #ffffff;
    border: 2px solid #bcd4ff;  /* ★枠線を濃く */
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 14px;
    box-shadow: 0 3px 10px rgba(180, 200, 255, 0.25);
}

.song-card h3, .song-card h4, .song-card a {
    color: #0f1a33;  /* ★文字色を濃くして視認性UP */
}

/* フォント統一 */
h1, h2, h3, h4, p, div {
    font-family: 'Hiragino Maru Gothic ProN', 'Yu Gothic', sans-serif;
}

</style>
""", unsafe_allow_html=True)


# ====== タイトル ======
st.markdown("""
<div class="title-card">
    <div class="title-text">
        【精密解析】<br>そらる・シンクロ率チェッカー
    </div>
    <div class="subtitle-text">
        あなたの声に最も近い楽曲も判定！
    </div>
</div>
""", unsafe_allow_html=True)

# ====== SNSアイコン ======
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
# ====== アップロード ======
st.subheader("① 音声ファイルをアップロード")
st.write("対応形式：wav / mp3")
st.write("推奨：10〜20秒のサビや盛り上がり部分（声が大きいところ）")
st.write("※ 声だけ・アカペラだとより精度が上がります")

uploaded_file = st.file_uploader(
    "ここに音声ファイルをドラッグ＆ドロップしてください",
    type=["wav", "mp3"]
)

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