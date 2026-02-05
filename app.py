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
    return np.concatenate([mfcc_mean, mfcc_std])  # 26次元

# ====== 距離 → スコア変換（絶対スケール） ======
def convert_to_score(dist, min_dist, max_dist):
    if max_dist == min_dist:
        return 50.0
    score = 1 - (dist - min_dist) / (max_dist - min_dist)
    return max(0, min(score * 100, 100))

# ====== 総合そらる率 + 曲ランキング ======
def analyze_all(user_feat, df):
    song_feats = df[[f"mfcc_{i}" for i in range(26)]].values

    # 曲同士の距離分布
    all_dists = []
    for i in range(len(song_feats)):
        for j in range(i + 1, len(song_feats)):
            all_dists.append(np.linalg.norm(song_feats[i] - song_feats[j]))

    min_dist = min(all_dists)
    max_dist = max(all_dists)

    # 総合そらる率
    soraru_center = song_feats.mean(axis=0)
    dist_total = np.linalg.norm(user_feat - soraru_center)
    soraru_rate = convert_to_score(dist_total, min_dist, max_dist)

    # 曲ランキング
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

    if rate >= 95:
        return "【神域の同調】もはや判別不能。声の立ち上がりから消え際の減衰まで、そらるさんの波形をそのままなぞったかのような一致です。"

    elif rate >= 90:
        return "【転生クラス】驚異的なシンクロ率。高音域へ抜ける際の切なさを孕んだ息の混ぜ方は、もはや本人級。"

    elif rate >= 85:
        return "【至高の共鳴】澄んだ響きと繊細なウィスパー成分が黄金比。初見リスナーは本人と聞き間違えるレベル。"

    elif rate >= 80:
        return "【極めて高い親和性】中音域の厚みと吐息の混ざり方が非常に近いです。バラードで特に映えるタイプ。"

    elif rate >= 75:
        return "【ハイレベルな同調】フレーズ終わりの息の抜き方がそらるさんと共鳴。低音の響きを深めるとさらに近づきます。"

    elif rate >= 70:
        return "【確かなそらる成分】鼻に抜ける甘い響きにそらるさんのエッセンスを強く感じます。脱力感を意識するとさらに寄ります。"

    elif rate >= 65:
        return "【潜在的シンクロ】中低音域でハッとするほど似た響きを見せます。個性とそらる成分の絶妙なブレンド。"

    elif rate >= 60:
        return "【ハイブリッド・ボイス】静寂を纏った響きを一部に持っています。シリアスな曲で特に高いシンクロ率を発揮。"

    elif rate >= 55:
        return "【共鳴の予感】声の密度や帯域バランスに共通パーツを確認。ウィスパーを磨けばさらに伸びます。"

    elif rate >= 50:
        return "【唯一無二の響き】個性とそらる成分が半々。無理に寄せず、世界観を乗せることで新しい魅力が生まれます。"

    elif rate >= 40:
        return "【ハイブリッド・ポテンシャル】あなた独自の力強い響きが際立っています。自分流に歌いこなせるタイプ。"

    elif rate >= 30:
        return "【ニュー・ジェネレーション】そらるさんとは異なるベクトルでキャラの立った歌声。芯の強さが魅力。"

    elif rate >= 20:
        return "【アンリミテッド・カラー】そらるさんとは対極の色彩を持つ声。創造に向いた唯一無二の響きです。"

    elif rate >= 10:
        return "【アイデンティティの確立】完全オリジナルの声質。個性を純粋に保てているのは大きな才能です。"

    else:
        return "【究極のオリジナリティ】測定不能！そらる成分をほぼ検知できないほど個性が突き抜けています。"

# ====== Streamlit UI ======
st.set_page_config(page_title="【精密解析】そらる・シンクロ率チェッカー", layout="centered")

# タイトル（折り返し改善＋中央揃え）
st.markdown("<h1 style='text-align:center;'>【精密解析】そらる・シンクロ率チェッカー</h1>", unsafe_allow_html=True)
st.caption("あなたの声に最も近い楽曲も判定！")

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

# ① 音声アップロード
st.subheader("① 音声ファイルをアップロード")
st.write("**対応形式：** wav / mp3 / m4a")
st.write("**推奨：** 10〜20秒のサビや盛り上がり部分（声が大きいところ）")
st.write("※ 声だけ・アカペラだとより精度が上がります")

uploaded_file = st.file_uploader("ここに音声ファイルをドラッグ＆ドロップしてください", type=["wav", "mp3", "m4a"])

st.markdown("---")

# ② 精密解析
st.subheader("② 精密解析")

analyze_button = st.button("🔍 精密解析スタート")

if analyze_button:
    if uploaded_file is None:
        st.warning("先に音声ファイルをアップロードしてください。")
    else:
        with st.spinner("解析中です… 音声ファイルの長さによって少し時間がかかる場合があります。"):
            try:
                user_feat = extract_user_features_from_file(uploaded_file, duration=15)
                soraru_rate, result = analyze_all(user_feat, df)
            except Exception as e:
                st.error(f"解析中にエラーが発生しました：{e}")
                st.stop()

        st.success("解析が完了しました！")
        st.caption("※ あなたの音声ファイルは診断後に破棄されます。他者に利用されることはありません。")

        st.markdown("---")

        # ③ 結果
        st.subheader("③ 結果")
        st.markdown(f"### あなたのそらる・シンクロ率： **{soraru_rate:.1f}%**")
        st.write(generate_comment(soraru_rate))

        st.markdown("---")

        # ④ 楽曲ランキング
        st.subheader("④ あなたに近い そらる楽曲 TOP5")

        top5 = result.head(5).reset_index(drop=True)

        # 1位
        top1 = top5.iloc[0]
        st.markdown(f"#### 🥇 第1位：{top1['song']}  （{top1['score']:.1f}%）")
        st.video(top1["url"])
        st.write(f"[YouTubeで開く]({top1['url']})")

        # 2〜5位
        for i in range(1, len(top5)):
            row = top5.iloc[i]
            st.markdown(f"#### 🥈 第{i+1}位：{row['song']}  （{row['score']:.1f}%）")
            st.write(f"[YouTubeで開く]({row['url']})")

        st.markdown("---")

        # ⑤ Xシェア（後で実装）
        st.subheader("⑤ Xでシェア")
        st.write("※ 診断結果を画像化してXに投稿できる機能は、あとで一緒に作り込みましょう。")

else:
    st.info("音声ファイルをアップロードしてから「精密解析スタート」を押してください。")