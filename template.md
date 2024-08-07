---
marp: true
theme: default
paginate: true
---

# GPT-4を活用したインタラクティブPDF学習支援システム
## AIによる個別最適化学習の実現

---

# このシステムの作成を始めた背景

- pythonやAI
  - 一方向的な情報提供
  - 個別ニーズへの対応が不十分
- AIによる個別学習支援の可能性
  - 動的なコンテンツ生成
  - リアルタイムフィードバック

---

# システム概要

![システム概要図](https://example.com/system_overview.png)

- PDFアップロード → テキスト抽出 → 問題生成 → 回答評価 → フィードバック

---

# 技術スタック

- 言語：Python
- フレームワーク：Streamlit
- 主要ライブラリ：
  - LangChain: GPT-4との連携
  - pdfplumber: PDFテキスト抽出
- AI：OpenAI GPT-4

---

# 主要機能の詳細

1. PDFテキスト抽出
   - pdfplumberを使用して高精度な抽出を実現
2. 動的問題生成
   - GPT-4を活用し、PDFの内容に基づいた問題を生成
3. 回答評価とフィードバック
   - ユーザーの回答をGPT-4が評価し、詳細なフィードバックを提供

---

# デモンストレーション

![デモスクリーンショット](https://example.com/demo_screenshot.png)

---

# 技術的課題と解決策

課題：
- LangChainの互換性問題

解決策：
- バージョン管理の徹底
- 代替アプローチの検討（直接OpenAI APIの使用）

---

# プロジェクトの成果

- PDFベースの柔軟な学習システムの実現
- AIによる個別化された問題生成と評価
- ユーザーフィードバック：
  - "従来の学習方法より効果的"
  - "即時フィードバックが学習意欲を向上"

---

# 今後の展望

- 機能拡張
  - 複数言語対応
  - 学習履歴の分析と可視化
- 他分野への応用
  - 企業研修
  - 技術文書の理解支援

---

# 学んだこと・感想

- AIとの協働による開発の insights
  - プロンプトエンジニアリングの重要性
  - AIの能力と限界の理解
- 個人的成長
  - 最新技術の実践的応用スキル
  - プロジェクト管理能力の向上

---

# まとめ

- GPT-4を活用した革新的な学習支援システムの開発
- PDFコンテンツの柔軟な活用
- AIによる個別最適化学習の実現
- 技術と教育の融合による新たな可能性の開拓

---

# ご清聴ありがとうございました

質問やコメントをお待ちしています。