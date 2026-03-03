import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Lasersan AI", page_icon="🤖", layout="centered",initial_sidebar_state="collapsed")
custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stChatInputContainer > div {
        border-radius: 24px !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        padding-left: 10px;
    }
    
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #10a37f !important;
        color: white !important;
    }
    
    .main-title {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #10a37f, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-top: 20px;
    }
    
    .sub-title {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 40px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .welcome-text {
        text-align: center;
        color: #9ca3af;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 100px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>Lasersan Akıllı Asistan</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Tüm şirket cihazları hakkında anında ve güvenilir bilgi alın.</p>", unsafe_allow_html=True)
@st.cache_resource #baştan sonra return
def sistemi_hazirla():
    loader = CSVLoader(file_path='brosur.csv', encoding='utf-8')
    belgeler = loader.load()
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(documents=belgeler, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 86})
    
    llm = OllamaLLM(model="qwen2.5", temperature=0)
    
    sistem_kurallari = (
        "Sen savunma sanayii alanında üretim yapan Lasersan şirketimizin resmi yapay zeka asistanısın. "
        "GÖREVLERİN VE KURALLARIN:\n"
        "1. ŞAHAN, NOXIS, NEBULA, TUNAY, AURA, FOCUS, ODAK, TALOS-L, KOZGU, AVCI, TOYGAR, BARBAROS, MIRACLE, "
        "ALAGÖZ, YALMAN-150PT, DELTA-180PT, DELTA-225PT, YALMAN-660PT, YALMAN-1100PT, DELTA-1100PT, BARKIN-2D, "
        "KURSAD-20A ve RAYPATH şirketimizin ürettiği teknolojik cihazlardır. Kullanıcı bu cihazlar hakkında "
        "'bilgi ver', 'nedir' veya 'anlat' derse, broşürdeki bilgileri harmanlayarak akıcı bir Türkçe ile özetle.\n"
        "2. Cevaplarını her zaman doğal ve kusursuz bir Türkçe ile ver.\n"
        "3. Sadece sana aşağıda 'Broşür Bilgileri' kısmında verilen metinleri kullanarak cevap ver. Metinde olmayan bir şeyi kesinlikle uydurma.\n"
        "4. Eğer sorunun cevabı broşürde yoksa sadece 'Maalesef bu konuda broşürde bir bilgi bulunmuyor.' de.\n\n"
        "5. Eğer kullanıcı senden tüm ürünleri/cihazları listelemeni, sıralamanı veya saymanı isterse, listende her cihazın adını SADECE BİR KERE yaz. "
        "Kesinlikle (Elektriksel Özellikler, Çevresel Özellikler vb.) gibi kategori adlarını listeye ekleme ve aynı cihazı asla iki kere yazarak tekrara düşme.\n\n"
        "Broşür Bilgileri:\n{context}"
    ) 
    prompt = ChatPromptTemplate.from_messages([
        ("system", sistem_kurallari),
        ("human", "{input}"),
    ])
    def belgeleri_birlestir(belgeler):
        return "\n".join(belge.page_content for belge in belgeler)
    
    rag_zinciri = (
        {"context": retriever | belgeleri_birlestir, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ) 
    return rag_zinciri

zincir = sistemi_hazirla()


if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

if len(st.session_state.mesajlar) == 0:
    st.markdown("<p class='welcome-text'>Bugün size nasıl yardımcı olabilirim?</p>", unsafe_allow_html=True)

for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["icerik"])

soru = st.chat_input("Broşürdeki cihazlarla ilgili bir şey sorun...")

if soru:
    with st.chat_message("user"):
        st.markdown(soru)
    st.session_state.mesajlar.append({"rol": "user", "icerik": soru})

    with st.chat_message("assistant"):
        cevap_alani = st.empty()
        cevap_alani.markdown("Broşürler taranıyor...")

        cevap = zincir.invoke(soru)
        cevap_alani.markdown(cevap)
        
    st.session_state.mesajlar.append({"rol": "assistant", "icerik": cevap})