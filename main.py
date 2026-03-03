from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("1. Broşür verileri okunup taze bir veritabanı oluşturuluyor...")
loader = CSVLoader(file_path='brosur.csv', encoding='utf-8')
belgeler = loader.load()

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_documents(documents=belgeler, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

print("2. Llama 3.2 beyni uyandırılıyor...")
llm = OllamaLLM(model="llama3.1")

sistem_kurallari = (
    "Sen şirketimizin resmi yapay zeka asistanısın. "
    "Sadece sana aşağıda 'Broşür Bilgileri' kısmında verilen metinleri kullanarak cevap ver. "
    "Eğer sorulan sorunun cevabı bu metinlerde yoksa, asla kendi bilgilerini kullanma ve uydurma. "
    "Sadece 'Maalesef bu konuda broşürde bir bilgi bulunmuyor.' de ve dur.\n\n"
    "Broşür Bilgileri:\n{context}"
    
)

prompt = ChatPromptTemplate.from_messages([
    ("system", sistem_kurallari),
    ("human", "{input}"),
])

def belgeleri_birlestir(belgeler):
    birlestirilmis_metin = "\n".join(belge.page_content for belge in belgeler)
    print("\n--- YAPAY ZEKAYA GİDEN BİLGİLER (KOPYA KAĞIDI) ---")
    print(birlestirilmis_metin)
    print("--------------------------------------------------\n")
    return birlestirilmis_metin

rag_zinciri = (
    {"context": retriever | belgeleri_birlestir, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

soru = "ŞAHAN kamerasının çalışma sıcaklığı nedir?"
print(f"Soru: {soru}")
print("Cevap düşünülüyor...\n")

cevap = rag_zinciri.invoke(soru)

print("--- CHATBOT'UN CEVABI ---")
print(cevap)
print("-------------------------")