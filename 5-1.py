import streamlit as st
# Streamlitモジュールをインポート

from langchain_community.chat_models import ChatOpenAI
# langchain_communityからChatOpenAIをインポート

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
# langchainのスキーマからメッセージクラスをインポート

from langchain_community.callbacks.manager import get_openai_callback
# langchain_communityからget_openai_callbackをインポート

import requests
# HTTPリクエストを行うためのrequestsライブラリをインポート

from bs4 import BeautifulSoup
# HTML解析を行うためのBeautifulSoupライブラリをインポート

from urllib.parse import urlparse
# URL解析を行うためのurlparse関数をインポート

def init_page():
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    # ページ設定を行い、タイトルとアイコンを設定

    st.header("Website Summarizer 🤗")
    # ページのヘッダーを設定

    st.sidebar.title("Options")
    # サイドバーのタイトルを設定

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # サイドバーに「Clear Conversation」ボタンを作成

    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        # メッセージが存在しない場合、またはクリアボタンが押された場合、メッセージを初期化

        st.session_state.costs = []
        # コストを記録するリストを初期化

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4", "GPT-4o"))
    # サイドバーにラジオボタンを作成し、使用するモデルを選択

    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    elif model == "GPT-4o":
        model_name = "gpt-4o"
    else:
        model_name = "gpt-4"
    # 選択されたモデルに応じてモデル名を設定

    return ChatOpenAI(temperature=0, model_name=model_name)
    # 選択されたモデルでChatOpenAIインスタンスを返す

def get_url_input():
    url = st.text_input("URL: ", key="input")
    # テキスト入力フィールドを作成し、URLを取得

    return url

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
        # URLのスキームとネットワークロケーションをチェックし、妥当性を確認
    except ValueError:
        return False

def get_content(url):
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            # URLにHTTPリクエストを送信

            soup = BeautifulSoup(response.text, 'html.parser')
            # レスポンスをBeautifulSoupで解析

            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
            # main、article、bodyタグのテキストを順に取得
    except:
        st.write('something wrong')
        return None

def build_prompt(content, n_chars=300):
    return f"""以下はとあるWebページのコンテンツである。内容を{n_chars}文字程度でわかりやすく要約してください。

========

{content[:1000]}

========

日本語で書いてね！
"""
    # コンテンツの最初の1000文字を使い、要約を依頼するプロンプトを作成

def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
        # チャットモデルにメッセージを送信し、応答を取得

    return answer.content, cb.total_cost
    # 応答の内容とコストを返す

def main():
    init_page()
    # ページを初期化

    llm = select_model()
    # モデルを選択

    init_messages()
    # メッセージを初期化

    container = st.container()
    response_container = st.container()
    # コンテナを作成

    with container:
        url = get_url_input()
        # URL入力フィールドを表示

        is_valid_url = validate_url(url)
        # 入力されたURLの妥当性をチェック

        if not is_valid_url:
            st.write('Please input valid url')
            answer = None
            # URLが無効な場合、エラーメッセージを表示し、answerをNoneに設定
        else:
            content = get_content(url)
            # 有効なURLの場合、コンテンツを取得

            if content:
                prompt = build_prompt(content)
                # コンテンツが取得できた場合、プロンプトを作成

                st.session_state.messages.append(HumanMessage(content=prompt))
                # プロンプトをセッションのメッセージに追加

                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                    # モデルにプロンプトを送信し、応答とコストを取得

                st.session_state.costs.append(cost)
                # コストをセッションのコストリストに追加
            else:
                answer = None

    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            # 応答がある場合、要約を表示

            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)
            # オリジナルのテキストを表示

    costs = st.session_state.get('costs', [])
    # セッションからコストリストを取得

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    total_cost_jpy = sum(costs) * 150
    st.sidebar.markdown(f"**総費用: ¥{total_cost_jpy:.0f}**")
    # サイドバーに総コストを表示

    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")
        # 各リクエストのコストをサイドバーに表示

if __name__ == '__main__':
    main()
    # メイン関数を実行

