import streamlit as st
import pandas as pd
from zhipuai import ZhipuAI
import random
import json
import os
from datetime import datetime


# ==============================================================================
# --- SECTION 1: CORE CONFIG & FUNCTIONS ---
# ==============================================================================

def inject_custom_css():
    """Injects custom CSS to style the Streamlit application."""
    st.markdown("""
        <style>
            html, body, [class*="st-"] { font-family: 'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', 'sans-serif'; }
            [data-testid="stSidebarCollapseButton"] svg, [data-testid="stSidebarCollapseButton"] span { display: none; }
            [data-testid="stSidebarCollapseButton"]::before {
                content: '≡'; font-size: 24px !important; color: #31333F !important;
                margin-left: 8px; margin-top: 5px; display: inline-block;
            }
            .st-emotion-cache-1c7y2kd { background-color: #F0F2F6; border-radius: 20px 20px 20px 5px; padding: 18px 22px; }
            .st-emotion-cache-4oy321 { background-color: #C9E4FF; border-radius: 20px 20px 5px 20px; padding: 18px 22px; }
            .stButton>button {
                border-radius: 30px; border: 1px solid #E0E0E0; background-color: #FFFFFF;
                width: 100%; transition: all 0.2s ease-in-out; min-height: 60px;
                white-space: normal; word-wrap: break-word; line-height: 1.4;
            }
            .stButton>button:hover { border-color: #007BFF; color: #007BFF; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }

            /* 评价按钮样式 */
            .rating-container {
                display: flex;
                gap: 8px;
                margin: 10px 0;
                justify-content: flex-start;
                align-items: center;
            }
            .rating-button {
                background: none;
                border: 1px solid #ddd;
                border-radius: 20px;
                padding: 4px 12px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s;
                min-height: 30px !important;
                height: 30px !important;
            }
            .rating-button:hover {
                background-color: #f0f0f0;
                border-color: #007BFF;
            }
            .rating-button.selected {
                background-color: #007BFF;
                color: white;
                border-color: #007BFF;
            }
            .rating-label {
                font-size: 12px;
                color: #666;
                margin-right: 8px;
            }
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def initialize_client():
    """Initializes and returns the ZhipuAI client."""
    try:
        api_key = st.secrets.get("ZHIPUAI_API_KEY")
        if not api_key:
            st.error("请在Streamlit Secrets中设置 ZHIPUAI_API_KEY。")
            st.stop()
        return ZhipuAI(api_key=api_key)
    except Exception as e:
        st.error(f"初始化客户端失败: {str(e)}")
        st.stop()


@st.cache_data
def load_knowledge():
    """Loads knowledge base from the Excel file and pre-processes it."""
    try:
        df = pd.read_excel("huai-an_kb.xlsx")
        df.fillna('', inplace=True)
        if '类别' in df.columns:
            df['类别'] = df['类别'].str.strip()
        return df
    except FileNotFoundError:
        st.sidebar.error("❌ 未找到知识库 'huai-an_kb.xlsx'！")
        return pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"❌ 加载知识库失败: {str(e)}")
        return pd.DataFrame()


def search_knowledge_context(question, df):
    """Searches the knowledge base for an exact question match to get context."""
    if df.empty or not question:
        return ""
    result = df.loc[df['问题'].str.lower() == question.lower(), '核心知识点']
    if not result.empty:
        return result.iloc[0]
    # Fallback for questions from chat_input that might not be in the KB
    return "在本地知识库中未找到与您输入完全匹配的问题。请根据我的已有知识进行回答。"


def get_ai_response_stream(question, context, persona, client, personas_config):
    """Gets a streaming AI response from the ZhipuAI client."""
    prompt = personas_config[persona]["prompt"].format(context=context, question=question)
    try:
        response_stream = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.75, max_tokens=2048, stream=True
        )
        return (chunk.choices[0].delta.content for chunk in response_stream if chunk.choices[0].delta.content)
    except Exception as e:
        st.error(f"AI服务调用失败: {str(e)}")

        def error_generator():
            yield "抱歉，AI服务暂时不可用。"

        return error_generator()


# ==============================================================================
# --- SECTION 1.5: FEEDBACK SYSTEM ---
# ==============================================================================

def save_feedback(question, answer, persona, rating, feedback_text=""):
    """保存用户反馈到JSON文件"""
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "persona": persona,
        "rating": rating,
        "feedback_text": feedback_text
    }

    feedback_file = "feedback.json"

    # 读取现有反馈数据
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedbacks = json.load(f)
        except:
            feedbacks = []
    else:
        feedbacks = []

    # 添加新反馈
    feedbacks.append(feedback_data)

    # 保存到文件
    try:
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存反馈失败: {str(e)}")
        return False


def rate_response(message_index, rating):
    """处理用户评价"""
    if message_index < len(st.session_state.messages):
        # 找到对应的问答对
        if message_index > 0:
            question_msg = st.session_state.messages[message_index - 1]
            answer_msg = st.session_state.messages[message_index]

            if question_msg["role"] == "user" and answer_msg["role"] == "assistant":
                # 保存反馈 - 这是关键调用
                success = save_feedback(
                    question=question_msg["content"],
                    answer=answer_msg["content"],
                    persona=answer_msg["persona"],
                    rating=rating
                )

                if success:
                    # 更新消息的评价状态
                    st.session_state.messages[message_index]["rating"] = rating
                    st.success(f"感谢您的{'好评' if rating == 'good' else '反馈'}！我们会继续改进服务。")
                    st.rerun()


def display_rating_buttons(message_index):
    """显示评价按钮"""
    message = st.session_state.messages[message_index]

    # 检查是否已经评价过
    current_rating = message.get("rating", None)

    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        good_style = "selected" if current_rating == "good" else ""
        if st.button("👍 有用", key=f"good_{message_index}",
                     help="这个回答对我很有帮助",
                     disabled=current_rating is not None):
            rate_response(message_index, "good")

    with col2:
        bad_style = "selected" if current_rating == "bad" else ""
        if st.button("👎 改进", key=f"bad_{message_index}",
                     help="这个回答需要改进",
                     disabled=current_rating is not None):
            rate_response(message_index, "bad")

    # 显示评价状态
    if current_rating:
        with col3:
            if current_rating == "good":
                st.markdown("✅ 已标记为有用")
            else:
                st.markdown("📝 已反馈需要改进")


# ==============================================================================
# --- SECTION 2: STATE MANAGEMENT & UI CALLBACKS ---
# ==============================================================================

def set_category(category):
    """Callback function to set the selected category and reset chat."""
    st.session_state.selected_category = category
    st.session_state.messages = []
    st.session_state.asked_questions = set()


def ask_question(question):
    """Callback function to set the question that needs to be processed."""
    st.session_state.question_to_ask = question


# ==============================================================================
# --- SECTION 3: APP LAYOUT & MAIN SCRIPT ---
# ==============================================================================

st.set_page_config(page_title="淮安AI导游天团", page_icon="🏮", layout="centered")
inject_custom_css()

PERSONAS = {
    "运小安": {"icon": "📜", "desc": "风趣幽默的历史系学长",
               "prompt": "作为'运小安'，一位风趣幽默的历史系学长，请基于以下知识：【{context}】，生动有趣地回答问题：【{question}】"},
    "淮博士": {"icon": "🤖", "desc": "高效精准的知识官",
               "prompt": "作为'淮博士'，一位高效精准的知识官，请基于以下知识：【{context}】，结构化、清晰地回答问题：【{question}】"},
    "阿淮": {"icon": "🍻", "desc": "热情接地气的本地咖",
             "prompt": "作为'阿淮'，一位接地气的淮安本地朋友，请基于以下知识：【{context}】，用充满生活气息的口吻回答问题：【{question}】"}
}

client = initialize_client()
knowledge_df = load_knowledge()

# --- Initialize Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "asked_questions" not in st.session_state: st.session_state.asked_questions = set()
if "selected_category" not in st.session_state: st.session_state.selected_category = None
if "question_to_ask" not in st.session_state: st.session_state.question_to_ask = None

# --- Sidebar UI ---
with st.sidebar:
    st.image("huaianliyunhe_47.png", caption="淮安里运河")
    st.header("📢 选择你的AI导游")
    st.radio(
        "选择导游风格:", list(PERSONAS.keys()),
        format_func=lambda x: f"{PERSONAS[x]['icon']} {x} - {PERSONAS[x]['desc']}",
        key="persona_selector"
    )
    if st.button("🗑️ 清空所有记录", help="清空聊天记录并返回主题选择界面"):
        st.session_state.clear()
        st.rerun()
    if not knowledge_df.empty:
        st.sidebar.success(f"✅ 已加载 {len(knowledge_df)} 条知识")

# --- Main Page UI ---
st.title("🏮 淮安AI导游天团")
st.caption("我是您的专属淮安导游，选择一个主题，开启一场“填鸭式”淮安文化之旅吧！")
st.write("---")

# --- Display Chat History from session_state ---
for i, msg in enumerate(st.session_state.messages):
    icon = PERSONAS[msg["persona"]]["icon"] if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])

        # 为AI回答添加评价按钮
        if msg["role"] == "assistant":
            display_rating_buttons(i)

# --- "LIVE" Q&A Handling ---
# Check if a new question has been submitted via a button click or chat input
if st.session_state.question_to_ask:
    question = st.session_state.question_to_ask
    st.session_state.question_to_ask = None  # Reset the state to prevent re-triggering

    # 1. Add user message to history and display it instantly
    st.session_state.asked_questions.add(question)
    st.session_state.messages.append(
        {"role": "user", "content": question, "persona": st.session_state.persona_selector})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # 2. Get context and stream the AI response LIVE
    context = search_knowledge_context(question, knowledge_df)
    response_stream = get_ai_response_stream(
        question, context, st.session_state.persona_selector, client, PERSONAS
    )

    with st.chat_message("assistant", avatar=PERSONAS[st.session_state.persona_selector]["icon"]):
        # Use write_stream to render the response live and get the full text
        full_response = st.write_stream(response_stream)

    # 3. Add the complete assistant response to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "persona": st.session_state.persona_selector})

    # 4. Force a final rerun to clear the "live" part and show the new recommendations
    st.rerun()

# --- Recommendation & Interaction UI ---
if st.session_state.selected_category is None:
    st.subheader("💡 请选择您感兴趣的主题")
    if not knowledge_df.empty:
        all_categories = sorted(knowledge_df['类别'].dropna().unique().tolist())
        num_cols = 4
        cat_cols = st.columns(num_cols)
        for i, category in enumerate(all_categories):
            col_index = i % num_cols
            cat_cols[col_index].button(category, key=f"cat_{category}", on_click=set_category, args=(category,))
    else:
        st.warning("知识库为空，无法显示主题。")

else:
    st.subheader(f"热门问题 · {st.session_state.selected_category}")
    category_df = knowledge_df[knowledge_df['类别'] == st.session_state.selected_category]
    available_qs = [q for q in category_df['问题'].dropna().tolist() if q not in st.session_state.asked_questions]

    if available_qs:
        questions_to_show = random.sample(available_qs, min(3, len(available_qs)))
        q_cols = st.columns(len(questions_to_show))
        for i, q in enumerate(questions_to_show):
            # Use on_click to trigger the question asking logic
            q_cols[i].button(q, key=f"q_{q}", on_click=ask_question, args=(q,))
    else:
        st.success("✨ 该主题下的问题已全部探索完毕！")

    if st.button("« 返回所有主题"):
        st.session_state.selected_category = None
        st.rerun()

# --- Chat Input for direct questions ---
# 修复语法错误
question_from_input = st.chat_input("您也可以直接向我提问...")
if question_from_input:
    ask_question(question_from_input)
    st.rerun()
