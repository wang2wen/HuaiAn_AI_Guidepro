import streamlit as st
import pandas as pd
from zhipuai import ZhipuAI
import random
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import re
import hashlib


# ==============================================================================
# --- SECTION 1: CORE CONFIG & FUNCTIONS ---
# ==============================================================================

# 用户认证配置
def load_users():
    """从配置文件加载用户信息"""
    try:
        config_file = "user_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("users", {})
        else:
            # 如果配置文件不存在，创建默认配置
            default_users = {
                "admin": {
                    "password": hashlib.sha256("admin123".encode()).hexdigest(),
                    "role": "admin",
                    "name": "管理员"
                },
                "user": {
                    "password": hashlib.sha256("user123".encode()).hexdigest(),
                    "role": "user",
                    "name": "普通用户"
                },
                "demo": {
                    "password": hashlib.sha256("demo123".encode()).hexdigest(),
                    "role": "user",
                    "name": "演示用户"
                }
            }

            # 创建默认配置文件
            default_config = {"users": default_users}
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)

            return default_users
    except Exception as e:
        st.error(f"加载用户配置失败: {e}")
        return {}


# 加载用户配置
USERS = load_users()


def authenticate_user(username, password):
    """验证用户登录"""
    if username in USERS:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if USERS[username]["password"] == hashed_password:
            return USERS[username]
    return None


def register_user(username, password, name, role="user"):
    """注册新用户"""
    if not username or not password or not name:
        return False, "请填写所有必填字段"

    if username in USERS:
        return False, "用户名已存在"

    # 密码强度检查
    if len(password) < 6:
        return False, "密码长度至少6位"

    # 用户名格式检查
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        return False, "用户名只能包含字母、数字、下划线，长度3-20位"

    try:
        # 创建新用户
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        new_user = {
            "password": hashed_password,
            "role": role,
            "name": name
        }

        # 添加到用户字典
        USERS[username] = new_user

        # 保存到配置文件
        config_file = "user_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {"users": {}}

        config["users"][username] = new_user

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        return True, "注册成功"

    except Exception as e:
        # 如果保存失败，从内存中移除
        if username in USERS:
            del USERS[username]
        return False, f"注册失败: {str(e)}"


def register_form():
    """显示注册表单"""
    st.subheader("📝 用户注册")

    with st.form("register_form"):
        username = st.text_input("用户名", placeholder="请输入用户名（3-20位字母、数字、下划线）",
                                 help="用户名只能包含字母、数字、下划线")
        name = st.text_input("显示名称", placeholder="请输入您的显示名称")
        password = st.text_input("密码", type="password", placeholder="请输入密码（至少6位）", help="密码长度至少6位")
        confirm_password = st.text_input("确认密码", type="password", placeholder="请再次输入密码")

        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("注册", type="primary")
        with col2:
            cancel_button = st.form_submit_button("取消")

        if cancel_button:
            st.session_state.show_register = False
            st.rerun()

        if submit_button:
            # 验证输入
            if not username or not name or not password or not confirm_password:
                st.error("请填写所有字段")
                return

            if password != confirm_password:
                st.error("两次输入的密码不一致")
                return

            # 执行注册
            success, message = register_user(username, password, name)

            if success:
                st.success(message)
                st.info("注册成功！请使用新账号登录")
                st.session_state.show_register = False
                # 延迟一下让用户看到成功消息
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)

    # 显示注册提示
    st.markdown("---")
    st.markdown("### 💡 注册说明")
    st.markdown("""
    - 用户名：3-20位字母、数字、下划线
    - 密码：至少6位字符
    - 显示名称：将作为您的昵称显示
    - 注册后即可使用所有功能
    - 如需管理员权限，请联系系统管理员
    """)


def login_form():
    """显示登录表单"""
    st.title("🔐 淮安AI导游系统")
    st.subheader("用户登录")

    with st.form("login_form"):
        username = st.text_input("用户名", placeholder="请输入用户名")
        password = st.text_input("密码", type="password", placeholder="请输入密码")
        submit_button = st.form_submit_button("登录")

        if submit_button:
            if not username or not password:
                st.error("请输入用户名和密码")
                return None

            user_info = authenticate_user(username, password)
            if user_info:
                # 先保存当前聊天记录（如果有用户登录）
                if st.session_state.logged_in:
                    save_chat_history()

                # 设置新的登录状态
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_role = user_info["role"]
                st.session_state.user_name = user_info["name"]

                # 加载该用户的历史聊天记录
                user_history = load_user_chat_history(username)
                if user_history:
                    st.session_state.messages = user_history["messages"]
                    st.session_state.asked_questions = set(user_history["asked_questions"])
                    st.session_state.selected_category = user_history["selected_category"]
                    st.success(f"登录成功！欢迎 {user_info['name']}，已恢复您的聊天记录")
                else:
                    # 新用户，初始化空的聊天状态
                    st.session_state.messages = []
                    st.session_state.asked_questions = set()
                    st.session_state.selected_category = None
                    st.success(f"登录成功！欢迎 {user_info['name']}")

                st.rerun()
            else:
                st.error("用户名或密码错误")

    # 显示使用提示
    st.markdown("---")
    st.markdown("### 💡 使用提示")
    st.markdown("""
    - 请输入您的用户名和密码进行登录
    - 登录后可以保存您的聊天记录
    - 管理员账号可以查看数据统计和管理功能
    - 如需账号帮助，请联系系统管理员
    """)


def logout_button():
    """显示登出按钮"""
    if st.button("🚪 退出登录", key="logout"):
        # 保存当前用户的聊天记录
        if st.session_state.logged_in:
            save_chat_history()

        # 清除所有session_state但保留基本的初始化状态
        st.session_state.clear()
        # 重新初始化基本的session_state
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.user_name = None
        st.session_state.messages = []
        st.session_state.asked_questions = set()
        st.session_state.selected_category = None
        st.session_state.question_to_ask = None
        st.session_state.show_stats = False
        st.session_state.show_user_history = False
        st.session_state.show_admin_panel = False
        st.rerun()


def check_permission(required_role="admin"):
    """检查用户权限"""
    return st.session_state.get("user_role") == required_role


def save_user_feedback(message_index, rating, feedback_text=""):
    """保存用户反馈到文件"""
    if not st.session_state.logged_in:
        return False

    try:
        # 获取消息信息
        if message_index < len(st.session_state.messages):
            msg = st.session_state.messages[message_index]
            if msg["role"] == "assistant":
                feedback_data = {
                    "timestamp": datetime.now().isoformat(),
                    "username": st.session_state.username,
                    "user_role": st.session_state.user_role,
                    "question": st.session_state.messages[message_index - 1]["content"] if message_index > 0 else "",
                    "answer": msg["content"],
                    "persona": msg["persona"],
                    "rating": rating,
                    "feedback_text": feedback_text
                }

                # 加载现有反馈
                feedback_file = "user_feedbacks.json"
                if os.path.exists(feedback_file):
                    with open(feedback_file, 'r', encoding='utf-8') as f:
                        feedbacks = json.load(f)
                else:
                    feedbacks = []

                # 添加新反馈
                feedbacks.append(feedback_data)

                # 保存反馈
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    json.dump(feedbacks, f, ensure_ascii=False, indent=2)

                return True
    except Exception as e:
        st.error(f"保存反馈失败: {e}")
        return False


def save_chat_history():
    """保存用户聊天记录到文件"""
    if not st.session_state.logged_in or not st.session_state.messages:
        return False

    try:
        chat_data = {
            "username": st.session_state.username,
            "user_role": st.session_state.user_role,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "asked_questions": list(st.session_state.asked_questions),
            "selected_category": st.session_state.selected_category
        }

        # 加载现有聊天记录
        chat_file = "user_chat_history.json"
        if os.path.exists(chat_file):
            with open(chat_file, 'r', encoding='utf-8') as f:
                all_chats = json.load(f)
        else:
            all_chats = []

        # 查找并更新该用户的最新聊天记录
        user_chat_index = None
        for i, chat in enumerate(all_chats):
            if chat["username"] == st.session_state.username:
                user_chat_index = i
                break

        if user_chat_index is not None:
            # 更新现有聊天记录
            all_chats[user_chat_index] = chat_data
        else:
            # 添加新聊天记录
            all_chats.append(chat_data)

        # 保存聊天记录
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(all_chats, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        st.error(f"保存聊天记录失败: {e}")
        return False


def load_user_chat_history(username):
    """加载指定用户的聊天记录"""
    try:
        chat_file = "user_chat_history.json"
        if os.path.exists(chat_file):
            with open(chat_file, 'r', encoding='utf-8') as f:
                all_chats = json.load(f)

            for chat in all_chats:
                if chat["username"] == username:
                    return chat
        return None
    except Exception as e:
        st.error(f"加载聊天记录失败: {e}")
        return None


def load_user_feedbacks():
    """加载所有用户反馈"""
    try:
        feedback_file = "user_feedbacks.json"
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"加载反馈失败: {e}")
        return []


def display_user_history():
    """显示用户历史记录"""
    if not st.session_state.logged_in:
        st.warning("请先登录查看历史记录")
        return

    st.title("📋 我的历史记录")
    st.subheader(f"👤 {st.session_state.user_name} 的聊天历史")

    # 显示当前会话的信息
    if st.session_state.messages:
        st.write(f"### 📱 当前会话")
        st.write(f"当前会话共有 {len(st.session_state.messages)} 条消息")

        # 显示最近几条消息
        recent_messages = st.session_state.messages[-6:]  # 显示最近6条消息
        for i, msg in enumerate(recent_messages):
            icon = PERSONAS[msg["persona"]]["icon"] if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=icon):
                st.markdown(msg["content"])

        if len(st.session_state.messages) > 6:
            st.info("只显示最近6条消息，完整的聊天记录会自动保存")
    else:
        st.info("当前会话暂无聊天记录")

    st.write("---")

    # 显示保存的历史记录信息
    st.write("### 💾 历史记录说明")
    st.markdown("""
    - ✅ 您的聊天记录会自动保存到本地文件
    - ✅ 重新登录时会自动恢复之前的对话
    - ✅ 切换用户账号时会分别保存各自的记录
    - ✅ 退出登录时会保存当前会话
    """)

    # 显示历史记录文件信息
    try:
        chat_file = "user_chat_history.json"
        if os.path.exists(chat_file):
            with open(chat_file, 'r', encoding='utf-8') as f:
                all_chats = json.load(f)

            user_chat = None
            for chat in all_chats:
                if chat["username"] == st.session_state.username:
                    user_chat = chat
                    break

            if user_chat:
                st.write(f"### 📊 您的统计信息")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("总消息数", len(user_chat["messages"]))
                with col2:
                    st.metric("已问问题", len(user_chat["asked_questions"]))

                st.write(f"**最后保存时间:** {user_chat['timestamp']}")
                if user_chat["selected_category"]:
                    st.write(f"**最后选择的主题:** {user_chat['selected_category']}")
    except Exception as e:
        st.error(f"读取历史记录信息失败: {e}")


def display_admin_panel():
    """显示管理员面板"""
    if not check_permission("admin"):
        st.error("权限不足")
        return

    st.title("🛡️ 管理员面板")
    st.subheader("用户管理和反馈详情")

    # 创建标签页
    tab1, tab2 = st.tabs(["📊 用户反馈", "👥 用户管理"])

    with tab1:
        display_feedback_details()

    with tab2:
        display_user_management()


def display_feedback_details():
    """显示用户反馈详情"""
    st.subheader("📊 用户反馈详情")

    feedbacks = load_user_feedbacks()

    if not feedbacks:
        st.info("暂无用户反馈")
        return

    # 反馈统计
    total_feedbacks = len(feedbacks)
    good_feedbacks = len([f for f in feedbacks if f.get("rating") == "good"])
    bad_feedbacks = len([f for f in feedbacks if f.get("rating") == "bad"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总反馈数", total_feedbacks)
    with col2:
        st.metric("好评数", good_feedbacks)
    with col3:
        st.metric("差评数", bad_feedbacks)

    st.write("---")

    # 反馈列表
    st.subheader("📋 反馈列表")

    # 筛选选项
    rating_filter = st.selectbox("筛选评价", ["全部", "good", "bad"])

    filtered_feedbacks = feedbacks
    if rating_filter != "全部":
        filtered_feedbacks = [f for f in feedbacks if f.get("rating") == rating_filter]

    # 显示反馈
    for i, feedback in enumerate(filtered_feedbacks):
        with st.expander(f"📅 {feedback['timestamp']} - {feedback['username']} ({feedback['user_role']})"):
            st.markdown(f"**问题:** {feedback['question']}")
            st.markdown(f"**回答:** {feedback['answer']}")
            st.markdown(f"**AI导游:** {feedback['persona']}")
            st.markdown(f"**评价:** {'👍 好评' if feedback['rating'] == 'good' else '👎 差评'}")
            if feedback.get('feedback_text'):
                st.markdown(f"**反馈意见:** {feedback['feedback_text']}")


def display_user_management():
    """显示用户管理"""
    st.subheader("👥 用户管理")

    # 显示系统用户
    st.markdown("### 📋 系统用户")
    user_data = []
    for username, info in USERS.items():
        user_data.append({
            "用户名": username,
            "角色": info["role"],
            "显示名称": info["name"]
        })

    if user_data:
        st.dataframe(pd.DataFrame(user_data))

    # 配置文件信息
    st.markdown("### ⚙️ 配置文件信息")
    try:
        config_file = "user_config.json"
        if os.path.exists(config_file):
            file_size = os.path.getsize(config_file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(config_file))
            st.markdown(f"**配置文件:** `user_config.json`")
            st.markdown(f"**文件大小:** `{file_size}` 字节")
            st.markdown(f"**修改时间:** `{mod_time.strftime('%Y-%m-%d %H:%M:%S')}`")
            st.markdown(f"**用户总数:** `{len(USERS)}`")
        else:
            st.warning("用户配置文件不存在")
    except Exception as e:
        st.error(f"读取配置文件信息失败: {e}")

    # 在线用户（简单模拟）
    st.markdown("### 🔄 在线用户")
    st.info("注意：当前显示的是登录会话信息")
    if st.session_state.logged_in:
        st.markdown(f"👤 **{st.session_state.user_name}** ({st.session_state.user_role}) - 在线")
    else:
        st.markdown("暂无登录用户")

    # 管理员操作说明
    st.markdown("---")
    st.markdown("### 🔧 管理员操作")
    st.markdown("""
    **要添加/修改用户，请按以下步骤：**
    1. 编辑 `user_config.json` 文件
    2. 添加新用户或修改现有用户信息
    3. 密码需要是SHA256哈希值
    4. 重启应用使更改生效

    **密码生成示例：**
    ```python
    import hashlib
    password_hash = hashlib.sha256("你的密码".encode()).hexdigest()
    ```
    """)


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

            /* 主题卡片样式 */
            .category-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .category-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px;
                padding: 25px 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                border: 2px solid transparent;
                position: relative;
                overflow: hidden;
                min-height: 140px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }
            .category-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
                border-color: #007BFF;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .category-card:hover .category-icon {
                transform: scale(1.2);
            }
            .category-card:hover .category-count {
                background: rgba(255,255,255,0.2);
                color: white;
            }
            .category-icon {
                font-size: 48px;
                margin-bottom: 15px;
                transition: transform 0.3s ease;
            }
            .category-name {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 8px;
            }
            .category-count {
                background: rgba(0,0,0,0.1);
                color: #666;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                transition: all 0.3s ease;
            }
            .category-card.selected {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-color: #007BFF;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .category-card.selected .category-count {
                background: rgba(255,255,255,0.2);
                color: white;
            }

            /* 普通按钮样式 */
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

            /* 智能推荐样式 */
            .recommendation-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                padding: 20px;
                margin: 15px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .recommendation-title {
                color: white;
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }
            .recommendation-title .ai-badge {
                background: rgba(255,255,255,0.2);
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 10px;
                margin-left: 10px;
            }
            .recommendation-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 10px;
            }
            .recommendation-item {
                background: rgba(255,255,255,0.9);
                border: none;
                border-radius: 10px;
                padding: 12px 16px;
                text-align: left;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 14px;
                line-height: 1.4;
                position: relative;
                overflow: hidden;
            }
            .recommendation-item:hover {
                background: white;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .recommendation-item::before {
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                height: 100%;
                width: 3px;
                background: #667eea;
            }
            .similarity-score {
                font-size: 11px;
                color: #666;
                margin-top: 5px;
            }

            /* AI生成内容标注 */
            .ai-generated-badge {
                background: #f0f2f5;
                color: #666;
                font-size: 11px;
                padding: 2px 6px;
                border-radius: 10px;
                margin-left: 8px;
                font-weight: normal;
            }

            /* 移动端适配 */
            @media (max-width: 768px) {
                /* 调整网格布局 */
                .category-grid {
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin: 15px 0;
                }
                .category-card {
                    padding: 20px 15px;
                    min-height: 120px;
                }
                .category-icon {
                    font-size: 36px;
                    margin-bottom: 10px;
                }
                .category-name {
                    font-size: 16px;
                }

                /* 调整按钮大小 */
                .stButton>button {
                    min-height: 50px;
                    padding: 10px 15px;
                    font-size: 14px;
                }

                /* 调整侧边栏 */
                .css-1d391kg {
                    padding: 1rem;
                }

                /* 调整聊天消息 */
                .stChatMessage {
                    padding: 10px !important;
                }

                /* 调整标题大小 */
                h1 {
                    font-size: 24px !important;
                }
                h2 {
                    font-size: 20px !important;
                }
                h3 {
                    font-size: 18px !important;
                }

                /* 智能推荐移动端适配 */
                .recommendation-grid {
                    grid-template-columns: 1fr;
                }

                /* 调整评价按钮 */
                .rating-container {
                    flex-wrap: wrap;
                }
            }

            /* 超小屏幕适配 */
            @media (max-width: 480px) {
                .category-grid {
                    grid-template-columns: 1fr;
                    gap: 10px;
                }
                .category-card {
                    padding: 15px;
                    min-height: 100px;
                }
                .category-icon {
                    font-size: 30px;
                }
                .category-name {
                    font-size: 15px;
                }

                /* 更小的按钮 */
                .stButton>button {
                    min-height: 45px;
                    font-size: 13px;
                }
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


def get_smart_recommendations(current_question, knowledge_df, user_history, top_n=3):
    """基于TF-IDF相似度计算智能推荐问题"""
    if knowledge_df.empty or not current_question:
        return []

    # 获取用户历史问题
    history_questions = [msg["content"] for msg in user_history if msg["role"] == "user"]

    # 准备所有候选问题
    all_questions = knowledge_df['问题'].dropna().tolist()

    # 过滤掉已经问过的问题
    available_questions = [q for q in all_questions if q not in history_questions]

    if not available_questions:
        return []

    # 中文文本预处理
    def preprocess_text(text):
        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 使用jieba分词
        words = jieba.lcut(text)
        return ' '.join(words)

    # 准备语料库
    corpus = [preprocess_text(current_question)] + [preprocess_text(q) for q in available_questions]

    try:
        # 计算TF-IDF矩阵
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # 计算相似度
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # 获取最相似的问题
        top_indices = similarity_scores.argsort()[-top_n:][::-1]

        recommendations = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.1:  # 相似度阈值
                recommendations.append({
                    'question': available_questions[idx],
                    'similarity': similarity_scores[idx]
                })

        return recommendations

    except Exception as e:
        # 如果推荐系统出错，返回随机推荐
        return random.sample(available_questions, min(top_n, len(available_questions)))


def get_simple_stats():
    """获取简单的统计数据"""
    stats = {
        'total_questions': 0,
        'total_answers': 0,
        'total_feedbacks': 0,
        'good_feedbacks': 0,
        'popular_personas': {},
        'recent_activity': []
    }

    # 统计消息数据
    if 'messages' in st.session_state:
        messages = st.session_state.messages
        stats['total_questions'] = len([m for m in messages if m['role'] == 'user'])
        stats['total_answers'] = len([m for m in messages if m['role'] == 'assistant'])

        # 统计人格使用频率
        for msg in messages:
            if msg['role'] == 'assistant':
                persona = msg.get('persona', '未知')
                stats['popular_personas'][persona] = stats['popular_personas'].get(persona, 0) + 1

        # 最近活动
        recent_messages = messages[-10:]  # 最近10条消息
        stats['recent_activity'] = [
            {
                'time': '最近',
                'type': msg['role'],
                'content': msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content']
            }
            for msg in recent_messages
        ]

    # 统计反馈数据
    feedback_file = "feedback.json"
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedbacks = json.load(f)
                stats['total_feedbacks'] = len(feedbacks)
                stats['good_feedbacks'] = len([f for f in feedbacks if f.get('rating') == 'good'])
        except:
            pass

    return stats


def display_simple_dashboard():
    """显示简单的数据仪表板"""
    st.title("📊 使用数据统计")
    st.caption("查看AI导游的使用情况")

    stats = get_simple_stats()

    # 关键指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💬 总问题", stats['total_questions'])

    with col2:
        st.metric("🤖 总回答", stats['total_answers'])

    with col3:
        st.metric("⭐ 用户反馈", stats['total_feedbacks'])

    with col4:
        satisfaction = f"{(stats['good_feedbacks'] / stats['total_feedbacks'] * 100):.1f}%" if stats[
                                                                                                   'total_feedbacks'] > 0 else "0%"
        st.metric("👍 满意度", satisfaction)

    st.write("---")

    # 人格使用分布
    if stats['popular_personas']:
        st.subheader("🎭 AI导游使用分布")

        # 准备数据
        personas = list(stats['popular_personas'].keys())
        counts = list(stats['popular_personas'].values())

        # 使用streamlit的bar_chart
        chart_data = pd.DataFrame({
            'AI导游': personas,
            '使用次数': counts
        })
        st.bar_chart(chart_data.set_index('AI导游'))

    # 最近活动
    if stats['recent_activity']:
        st.subheader("🕐 最近活动")

        for activity in stats['recent_activity']:
            icon = "👤" if activity['type'] == 'user' else "🤖"
            st.markdown(f"{icon} **{activity['type']}**: {activity['content']}")

    st.write("---")
    st.markdown("<span class='ai-generated-badge'>内容由AI生成</span>", unsafe_allow_html=True)


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
                # 如果是差评，询问详细反馈
                feedback_text = ""
                if rating == "bad" and st.session_state.logged_in:
                    feedback_text = st.text_area(
                        "请告诉我们需要改进的地方：",
                        key=f"feedback_{message_index}",
                        help="您的详细反馈会帮助我们改进服务质量",
                        placeholder="比如：回答不够准确、信息不够详细、语气不合适等..."
                    )

                # 保存反馈
                if st.session_state.logged_in:
                    success = save_user_feedback(message_index, rating, feedback_text)
                else:
                    # 未登录用户使用原来的反馈系统
                    success = save_feedback(
                        question=question_msg["content"],
                        answer=answer_msg["content"],
                        persona=answer_msg["persona"],
                        rating=rating
                    )

                if success:
                    # 更新消息的评价状态
                    st.session_state.messages[message_index]["rating"] = rating
                    if st.session_state.logged_in:
                        st.success(f"感谢您的{'好评' if rating == 'good' else '详细反馈'}！我们会继续改进服务。")
                    else:
                        st.success(f"感谢您的{'好评' if rating == 'good' else '反馈'}！登录后可提供更详细的反馈。")
                    st.rerun()


def display_rating_buttons(message_index):
    """显示评价按钮"""
    message = st.session_state.messages[message_index]

    # 检查是否已经评价过
    current_rating = message.get("rating", None)

    # 如果是差评且已登录，显示详细反馈区域
    show_feedback_detail = (current_rating == "bad" and
                            st.session_state.logged_in and
                            st.session_state.get(f"show_feedback_{message_index}", False))

    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if st.button("👍 有用", key=f"good_{message_index}",
                     help="这个回答对我很有帮助",
                     disabled=current_rating is not None):
            rate_response(message_index, "good")

    with col2:
        if st.button("👎 改进", key=f"bad_{message_index}",
                     help="这个回答需要改进",
                     disabled=current_rating is not None):
            if st.session_state.logged_in:
                st.session_state[f"show_feedback_{message_index}"] = True
            rate_response(message_index, "bad")

    # 显示评价状态
    if current_rating:
        with col3:
            if current_rating == "good":
                st.markdown("✅ 已标记为有用")
            else:
                if st.session_state.logged_in and show_feedback_detail:
                    st.markdown("📝 已反馈需要改进 - 点击查看详情")
                else:
                    st.markdown("📝 已反馈需要改进")

    # 显示详细反馈信息（仅管理员或在特定条件下）
    if current_rating == "bad" and check_permission("admin"):
        if st.button("📋 查看反馈详情", key=f"view_feedback_{message_index}"):
            show_feedback_detail_modal(message_index)


def show_feedback_detail_modal(message_index):
    """显示反馈详情弹窗"""
    if message_index < len(st.session_state.messages):
        msg = st.session_state.messages[message_index]
        if msg["role"] == "assistant" and message_index > 0:
            question_msg = st.session_state.messages[message_index - 1]

            # 查找对应的详细反馈
            feedbacks = load_user_feedbacks()
            matching_feedback = None

            for feedback in feedbacks:
                if (feedback["question"] == question_msg["content"] and
                        feedback["answer"] == msg["content"] and
                        feedback["persona"] == msg["persona"]):
                    matching_feedback = feedback
                    break

            if matching_feedback:
                st.markdown("### 📋 反馈详情")
                st.markdown(f"**用户:** {matching_feedback['username']} ({matching_feedback['user_role']})")
                st.markdown(f"**时间:** {matching_feedback['timestamp']}")
                st.markdown(f"**AI导游:** {matching_feedback['persona']}")
                st.markdown("---")
                st.markdown("**问题:**")
                st.markdown(matching_feedback['question'])
                st.markdown("**回答:**")
                st.markdown(matching_feedback['answer'])
                st.markdown("**评价:** 👎 需要改进")
                if matching_feedback.get('feedback_text'):
                    st.markdown("**详细反馈:**")
                    st.info(matching_feedback['feedback_text'])
                else:
                    st.markdown("*用户未提供详细反馈*")
            else:
                st.warning("未找到对应的详细反馈信息")


def display_smart_recommendations(current_question, knowledge_df, user_history):
    """显示智能推荐问题"""
    recommendations = get_smart_recommendations(current_question, knowledge_df, user_history)

    if recommendations:
        st.markdown("""
        <div class="recommendation-container">
            <div class="recommendation-title">
                🤖 智能推荐
                <span class="ai-badge">内容由AI生成</span>
            </div>
            <div class="recommendation-grid">
        """, unsafe_allow_html=True)

        # 创建推荐按钮网格
        cols = st.columns(len(recommendations))

        for i, (col, rec) in enumerate(zip(cols, recommendations)):
            with col:
                button_content = f"""
                <div class="recommendation-item">
                    <div>{rec['question']}</div>
                    <div class="similarity-score">相关度: {rec['similarity']:.2f}</div>
                </div>
                """

                if st.button(button_content, key=f"rec_{i}", help=f"相关度: {rec['similarity']:.2f}"):
                    ask_question(rec['question'])
                    st.rerun()

        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)


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
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = None
if "user_role" not in st.session_state: st.session_state.user_role = None
if "user_name" not in st.session_state: st.session_state.user_name = None
if "messages" not in st.session_state: st.session_state.messages = []
if "asked_questions" not in st.session_state: st.session_state.asked_questions = set()
if "selected_category" not in st.session_state: st.session_state.selected_category = None
if "question_to_ask" not in st.session_state: st.session_state.question_to_ask = None
if "show_stats" not in st.session_state: st.session_state.show_stats = False
if "show_user_history" not in st.session_state: st.session_state.show_user_history = False
if "show_admin_panel" not in st.session_state: st.session_state.show_admin_panel = False
if "show_register" not in st.session_state: st.session_state.show_register = False

# --- Optional Login Check ---
# 不强制登录，所有用户都可以使用基本功能

# --- Sidebar UI ---
with st.sidebar:
    st.image("huaianliyunhe_47.png", caption="淮安里运河")
    st.header("📢 选择你的AI导游")
    st.radio(
        "选择导游风格:", list(PERSONAS.keys()),
        format_func=lambda x: f"{PERSONAS[x]['icon']} {x} - {PERSONAS[x]['desc']}",
        key="persona_selector"
    )

    st.write("---")
    # 用户登录区域
    if st.session_state.logged_in:
        # 已登录状态
        st.markdown(f"👤 **{st.session_state.user_name}**")
        st.markdown(f"🔑 角色: `{st.session_state.user_role}`")
        logout_button()

        # 如果登录了，显示历史记录按钮
        if st.button("📋 我的历史记录", help="查看您的聊天历史"):
            # 关闭其他页面，打开历史记录页面
            st.session_state.show_user_history = True
            st.session_state.show_stats = False
            st.session_state.show_admin_panel = False
            st.rerun()
    else:
        # 未登录状态
        st.subheader("🔐 用户登录")

        # 显示注册/登录选项卡
        tab1, tab2 = st.tabs(["登录", "注册"])

        with tab1:
            with st.expander("点击登录", expanded=True):
                login_form()

        with tab2:
            if st.button("新用户注册", help="创建新账号"):
                st.session_state.show_register = True
                st.rerun()

            st.markdown("""
            ### 💡 注册说明
            - 免费注册，即可使用所有功能
            - 支持保存聊天记录和个人偏好
            - 注册后可享受个性化服务
            """)

    st.write("---")

    # 权限控制：只有管理员才能看到数据统计入口
    if check_permission("admin"):
        st.write("---")
        if st.button("📊 使用统计", help="查看简单的使用数据统计"):
            # 关闭其他页面，打开统计页面
            st.session_state.show_stats = True
            st.session_state.show_admin_panel = False
            st.session_state.show_user_history = False
            st.rerun()

        if st.button("🛡️ 管理员面板", help="查看用户管理和反馈详情"):
            # 关闭其他页面，打开管理员面板
            st.session_state.show_stats = False
            st.session_state.show_admin_panel = True
            st.session_state.show_user_history = False
            st.rerun()

    st.write("---")
    if st.button("🗑️ 清空当前会话", help="清空聊天记录并返回主题选择界面"):
        # 保存当前聊天记录（如果用户已登录）
        if st.session_state.logged_in:
            save_chat_history()

        st.session_state.messages = []
        st.session_state.asked_questions = set()
        st.session_state.selected_category = None
        st.session_state.question_to_ask = None
        st.rerun()
    if not knowledge_df.empty:
        st.sidebar.success(f"✅ 已加载 {len(knowledge_df)} 条知识")

# --- Main Page UI ---
# 统计页面
if st.session_state.show_stats:
    display_simple_dashboard()

    # 添加返回按钮
    if st.button("← 返回对话界面"):
        st.session_state.show_stats = False
        st.rerun()

# 用户历史记录页面
elif st.session_state.show_user_history:
    display_user_history()

    # 添加返回按钮
    if st.button("← 返回对话界面"):
        st.session_state.show_user_history = False
        st.rerun()

# 管理员面板页面
elif st.session_state.show_admin_panel:
    display_admin_panel()

    # 添加返回按钮
    if st.button("← 返回对话界面"):
        st.session_state.show_admin_panel = False
        st.rerun()

# 主对话界面
else:
    st.title("🏮 淮安AI导游天团")
    caption_text = "我是您的专属淮安导游，选择一个主题，开启一场'填鸭式'淮安文化之旅吧！"
    if st.session_state.logged_in:
        caption_text += f" 👤 欢迎，{st.session_state.user_name}！"
    st.caption(caption_text)
    st.write("---")

    # --- Display Chat History from session_state ---
    for i, msg in enumerate(st.session_state.messages):
        icon = PERSONAS[msg["persona"]]["icon"] if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=icon):
            if msg["role"] == "assistant":
                # 为AI回答添加标注
                content_with_badge = f"{msg['content']} <span class='ai-generated-badge'>内容由AI生成</span>"
                st.markdown(content_with_badge, unsafe_allow_html=True)
            else:
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

        # 为实时AI回答添加标注
        st.markdown("<span class='ai-generated-badge'>内容由AI生成</span>", unsafe_allow_html=True)

    # 3. Add the complete assistant response to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "persona": st.session_state.persona_selector})

    # 4. 自动保存聊天记录（如果用户已登录）
    if st.session_state.logged_in:
        save_chat_history()

    # 5. 显示智能推荐
    display_smart_recommendations(question, knowledge_df, st.session_state.messages)

    # 6. Force a final rerun to clear the "live" part and show the new recommendations
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
