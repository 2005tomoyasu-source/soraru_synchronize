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
    score = score * 100
    return max(5, min(score, 100))  # ★ 最低値を5％に固定

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
        return ("【神域の同調】もはや判別不能、本人降臨レベルです。声の立ち上がりから消え際のスーッとした減衰まで、"
                "そらるさんの波形をそのままなぞったかのような一致を見せています。空気に溶けるような透明感と、"
                "耳元で囁かれているような実在感を同時に持っており、聴く人を一瞬で『碧の世界』へ引きずり込む魔力を持っています。")

    elif rate >= 90:
        return ("【転生クラス】驚異的なシンクロ率です。そらるさん特有の『温かみのある無機質さ』を見事に再現しています。"
                "特に高音域へ抜ける際の、切なさを孕んだ息の混ぜ方は天性のものと言えるでしょう。マイクを通した瞬間に完成されるその響きは、"
                "もはや転生したそらるさん本人と言っても過言ではありません。自信を持って『そらるボイス』を名乗ってください。")

    elif rate >= 85:
        return ("【至高の共鳴】『ビー玉の中の宇宙』をそのまま体現したような、澄んだ響きを持っています。中音域の安定感と、"
                "そこに乗る繊細なウィスパー成分の比率が黄金比に近い状態です。ふとした瞬間のニュアンスが驚くほど本人に似ているため、"
                "初見のリスナーは間違いなく耳を疑うはず。ミックスでリバーブを深めに掛ければ、完璧に化けるポテンシャルがあります。")

    elif rate >= 80:
        return ("【極めて高い親和性】かなりの高シンクロ率です。中音域の厚みと、吐息が混ざり合う『エモーショナルな質感』が非常に近く、"
                "そらるさんの楽曲、特にバラードでの表現力が爆発的に高まるタイプです。声の抜け感が非常にスムーズで、"
                "聴き手にストレスを与えない癒やしの成分がたっぷり詰まっています。あと一歩で、神域に手が届く位置にいます。")

    elif rate >= 75:
        return ("【ハイレベルな同調】かなり似ています。特にフレーズ終わりの『息の抜き方』や、言葉の頭に置くエッジボイスの使い方が、"
                "そらるさんの歌唱スタイルと深く共鳴しています。歌い方や声の表情に確かな『そらる味』があり、ファンなら思わずニヤリとしてしまうはず。"
                "意識して低音の響きを深めるだけで、さらにシンクロ率は跳ね上がる可能性を秘めています。")

    elif rate >= 70:
        return ("【確かなそらる成分】声の抜け感や、鼻に抜ける甘い響きにそらるさんのエッセンスを強く感じます。全体的に落ち着いたトーンでありながら、"
                "サビなどで見せる芯の強さがそらるさんの歌唱設計に非常に似ています。今のままでも十分『似ている』と言われるレベルですが、"
                "もう少しだけ『脱力感』を意識して歌うと、より本人に近いアンニュイな魅力が増すでしょう。")

    elif rate >= 65:
        return ("【潜在的シンクロ】ところどころに強い『そらる成分』を検知しました。全ての帯域ではありませんが、"
                "特定の音域（特に中低音）において、ハッとするほど似た響きを見せることがあります。自分の個性をベースにしつつ、"
                "そらるさんのエッセンスを絶妙なスパイスとして持っている状態です。寄せる技術を磨けば、まだまだ上を狙える伸び代を感じます。")

    elif rate >= 60:
        return ("【ハイブリッド・ボイス】あなた自身の個性を主軸にしつつ、そらるさんのような『静寂を纏った響き』を一部に持っています。"
                "全ての曲というよりは、特定の楽曲（例えば『銀の祈誓』のようなシリアスな曲）で特に高いシンクロ率を発揮するタイプです。"
                "自分の声を活かしながら、要所でそらるさんのテクニックを取り入れるのが一番輝くスタイルと言えます。")

    elif rate >= 55:
        return ("【共鳴の予感】声の密度や帯域のバランスに、そらるさんと共通するパーツを確認しました。現在はあなた独自の歌い方が強く出ていますが、"
                "声質そのものには『透明感』の素質が十分にあります。ウィスパーボイスの練習を重ねることで、"
                "あなたの喉の中に眠っている『そらる成分』をもっと引き出すことができるはず。可能性に満ちた数値です。")

    elif rate >= 50:
        return ("【唯一無二の響き】自分自身の個性が半分を占めている、非常に魅力的なブレンド具合です。そらるさんのような落ち着きを持ちつつも、"
                "あなたにしか出せない独自の色彩を纏った歌声です。無理に寄せるよりも、今の響きにそらるさんの楽曲の世界観を乗せることで、"
                "全く新しい『そらるソング』を生み出せる才能を秘めています。その個性を大切にしてください。")

    elif rate >= 40:
        return ("【独創のアーティスト・ボイス】そらるさんの成分をベースにしつつも、あなた独自の力強い響きがはっきりと顔を出しています。"
                "今のままでも『そらるさんの曲を自分流に歌いこなせる』、模倣を超えた自立したバランスの声質です。"
                "シンクロ率という枠に収まらない、歌い手としての強いアイデンティティを感じさせる結果となりました。")

    elif rate >= 30:
        return ("【ニュー・ジェネレーション】そらるさんの声質とは異なるベクトルで、非常にキャラクターの立った歌声です。"
                "シンクロ率は低めですが、それはあなたの声にしっかりとした『芯』があり、誰の影響も受けていない証拠です。"
                "そらるさんの楽曲をカバーしても、原曲の影に隠れない圧倒的な存在感を放つことができる、自立したボイスタイプと言えます。")

    elif rate >= 20:
        return ("【アンリミテッド・カラー】そらるさんの『碧』の世界とは対極にあるような、エネルギッシュまたは独自の色彩を持った声です。"
                "波形解析の結果、そらるさんの成分とは別の帯域で非常に高いエネルギーを検知しました。これは模倣ではなく創造に向いた声であり、"
                "あなたにしか出せない響きを武器に、新しい音楽の道を切り拓くべき喉の持ち主です。")

    elif rate >= 10:
        return ("【アイデンティティの確立】あなたの声が誰にも似ていない『完全オリジナル』であることを示しています。"
                "そらるさんのファンでありながら、自分自身の個性をこれほど純粋に保てているのは一つの才能です。"
                "そらるさんの楽曲をあなたが歌うことで、原曲とは全く違う、あなたにしか救えないリスナーに届く新しい命が吹き込まれるでしょう。")

    else:
        return ("【究極のオリジナリティ】測定不能！そらるさんの成分をほぼ検知できないほど、あなたの個性は突き抜けています。"
                "ある意味、このサイトで最も貴重な『0%に近い希少な響き』の持ち主です。既存の枠組みにはまらないその声を大切に、"
                "自分だけの道を突き進んでください。その響きは、誰にも真似できないあなただけの宝物です。")

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