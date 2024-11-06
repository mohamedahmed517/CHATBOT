import subprocess

def install_packages():
    # List of installation commands
    commands = [
        'pip install pip3-autoremove',
        'pip-autoremove torch torchvision torchaudio -y',
        'pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121',
        'pip install unsloth',
        'pip install azure-ai-textanalytics'
    ]

    for command in commands:
        print(f"Running command: {command}")
        subprocess.check_call(command, shell=True)

install_packages()

import re
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from unsloth.chat_templates import get_chat_template # type: ignore
from unsloth import FastLanguageModel # type: ignore

def classify_text(query):
    try:

        ai_endpoint = 'https://sentimentanalysis10.cognitiveservices.azure.com/'
        ai_key = '4kyIh8KGdZYB9j9Yj71gT09yOE3x46rXQpfXilONXKm8CFL7ydK6JQQJ99AJACYeBjFXJ3w3AAAaACOGjCS5'
        project_name = 'MentalHealth10'
        deployment_name = 'MentalHealth'

        # Create client using endpoint and key
        credential = AzureKeyCredential(ai_key)
        ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

        # Prepare the query for classification
        batchedDocuments = [query]

        # Get Classification
        operation = ai_client.begin_single_label_classify(
            batchedDocuments,
            project_name=project_name,
            deployment_name=deployment_name
        )

        document_results = operation.result()

        # Extract classification result
        for classification_result in document_results:
            if classification_result.kind == "CustomDocumentClassification":
                classification = classification_result.classifications[0]
                return classification.category, classification.confidence_score
            elif classification_result.is_error:
                return None, classification_result.error.message

    except Exception as ex:
        return None, str(ex)

def generation(question,model,tokenizer):

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    category, confidence_score = classify_text(question)

    context = f"انت معالج بالذكاء الاصطناعي قيد التدريب ومهمتك هي تقديم دعم عاطفي مدروس ومخصص لكل مستخدم بناء علي حالته النفسيه الحاليه ستقوم بالاستماع بعنايه الي مخاوفهم ومشاعرهم مع مراعاه ان الحاله النفسيه للمريض هي {category} قم بتقديم استجابات ملاءمه للوضع الذي يمر به استخدم معرفتك بمختلف المناهج العلاجيه لتقديم تقنيات ومحادثات داعمه بلهجه ودوده ومتفهمه تذكر انك مورد للدعم العاطفي والارشاد ولست بديلا عن المعالج البشري تعامل مع المريض بصدق واحترام وكن مرنا في محادثاتك لتتناسب مع حالته النفسيه وحاول التخفيف عنه بطريقه لطيفه ومتوازنه"

    # Create the messages list with the context and user input
    messages = [
        {"from": "system", "value": context},
        {"from": "human", "value": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 300, use_cache = True)
    model_answer = tokenizer.batch_decode(outputs)

    return model_answer

def simple_output(model_answer):

    model_answer_str = model_answer[0]

    pattern = re.compile(r'<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)<\|eot_id\|>', re.DOTALL)

    sections = pattern.findall(model_answer_str)

    for role, content in sections:
        role = role.strip().lower()
        content = content.strip()
        if role == 'system':
            continue
        elif role == 'user':
            user_qus = content
        elif role == 'assistant':
            model_ans = content

    if 'assistant' not in [role.strip().lower() for role, _ in sections]:
        assistant_start_index = model_answer_str.find('<|start_header_id|>assistant<|end_header_id|>')
        if assistant_start_index != -1:
            assistant_content = model_answer_str[assistant_start_index:].split('<|eot_id|>')[0]
            model_ans = assistant_content.replace('<|start_header_id|>', '').replace('<|end_header_id|>', '').strip()

    return user_qus, model_ans

model, tokenizer = FastLanguageModel.from_pretrained("mohamed517/Arabic-fine-Tuning-LLaMA-Model")

st.title("LLAMA Streamlit Chatbot")

if 'history' not in st.session_state:
    st.session_state['history'] = []

user_input = st.text_input("You:")

if user_input:
    model_answer = generation(user_input, model, tokenizer)
    question_, answer_ = simple_output(model_answer)

    if question_:
        st.session_state.history.append((question_, answer_))

# Display chat history
for question_, answer_ in st.session_state.history:
    st.write(f"You: {question_}")
    st.write(f"Bot: {answer_}")
