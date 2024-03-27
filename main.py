# pipelines機能の利用
from transformers import pipeline

# 【文書分類】
text_classification_pipeline = pipeline(
  model="llm-book/bert-base-japanese-v3-marc_ja"
)
# 商品レビューで学習しているモデルを利用している
# text = "世界には言葉が出ないほどひどい音楽がある。"
# text = "段ボールが破損していて商品も破損していました。"
# text = "色が可愛くてまたリピートしたいと思います。"
# text = "色は可愛いですが、コスパが悪いのでリピートはしません。"
text = "高評価だったので購入しましたが、私の肌には合いませんでした。"
# textの極性を予測
print(text_classification_pipeline(text)[0])

