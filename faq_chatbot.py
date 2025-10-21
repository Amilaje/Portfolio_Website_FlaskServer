import os
import re
from typing import List, Optional, Dict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from pydantic import BaseModel, Field

# --- Pydantic DTO 정의 (app.py와 공유) ---

class ChatQueryRequest(BaseModel):
    """클라이언트로부터 받은 질문 요청 DTO"""
    query: str

class ChatResponse(BaseModel):
    """클라이언트에 반환할 챗봇 응답 DTO"""
    response: str
    latencyMs: int = 0
    sourceDocuments: List[str] = Field(default_factory=list)

# 파일 및 소스 정보
FAQ_FILE_NAME = "jinho_faq.txt"
RAG_SOURCES = [FAQ_FILE_NAME] 
FALLBACK_SOURCE = ["AI 개발자 기본 정보 (Fallback)"] # 폴백 답변의 출처

# ----------------------------------------------------------------------------------
# 🔑 [CONSTANTS] 사용자 정보 (저장된 정보 활용)
# ----------------------------------------------------------------------------------

TECH_STACK = ['Python', 'Java', 'LangGraph', 'LangChain', 'YOLO', 'PyTorch', 'TensorFlow', 'Scikit-learn', 'NumPy', 'Pandas', 'Spring Boot', 'React', 'PyCharm', 'VSCode', 'IntelliJ IDEA', 'GitHub', 'AWS', 'Azure', 'Docker', 'Kubernetes', 'MSA', 'MSA EZ Modeling', 'Kafka']
CERTIFICATIONS = ['AICE Basic', 'AICE Associate', 'Computer Literacy Level 2', 'Deep Learning Application Ability Level 3', 'IT Plus Level 2', 'Word Processor', '빅데이터 분석기사 (필기 합격)']
PROJECTS = ['Meteorological Administration Data Contest  Heat Demand Prediction Model (Awarded)', 'KT Aivle School Big Project – Pharmacy Automation Platform (YeahYak)', 'MSA-based Book Subscription Platform Construction Project', 'Spring Boot + React-based Book Management Web System', 'AWS-based Cloud Infrastructure Construction', 'LLM Agent-based AI Interviewer Agent System', 'Machine Learning-based Motion Classification System', 'Image Classification (MobileNetV2-based)', 'Unity-based Game Development']
CONTACT_ME_GMAIL = "choijinho321@gmail.com"
CONTACT_ME_EMAIL = "choijinho123@naver.com"
CONTACT_Telephone = "010-7115-5860"
# ----------------------------------------------------------------------------------
# 💡 [NEW/MODIFIED] 1. Fallback 정보 및 연락처 안내 생성
# ----------------------------------------------------------------------------------
def get_fallback_info() -> str:
    """사용자의 저장된 포트폴리오 정보를 문자열로 반환하여 LLM에게 전달합니다."""
    # ... (기존 정보 유지) ...
    project_list = "\n".join([f"- {p}" for p in PROJECTS])
    tech_stack_list = ", ".join(TECH_STACK)
    cert_list = ", ".join(CERTIFICATIONS)

    return f"""
    ### 기본 포트폴리오 정보 (RAG 실패 시 대체용)
    - **소개:** AI 개발자로 전공(천문학)을 전환했으며, 현재 KT Aivle School 을 수효했습니다.
    - **목표:** 협업 및 실행 능력을 갖춘 문제 해결 중심의 실무형 AI 개발자입니다.
    - **기술 스택:** {tech_stack_list}
    - **주요 프로젝트:** {project_list}
    - **자격증:** {cert_list}
    """

def get_contact_fallback_response(latency: int = 0) -> ChatResponse:
    """RAG에서 정보를 찾지 못했을 때, 연락처를 포함한 안내 응답을 생성합니다."""
    contact_info = (
        f"**현재의 포트폴리오 문서 내에서는 요청하신 정보를 찾을 수 없습니다.** "
        f"혹시 기타 문의 사항이나 개인적인 질문이 있으시다면, "
        f"**다음 연락처를 통해 AI 개발자님께 직접 연락해 보시는 것을 추천드립니다.**\n\n"
        f"▶ **E-mail**: {CONTACT_ME_EMAIL}\n"
        f"▶ **Telephone**: {CONTACT_Telephone}\n"
    )
    return ChatResponse(
        response=contact_info,
        latencyMs=latency,
        sourceDocuments=FALLBACK_SOURCE
    )

# ----------------------------------------------------------------------------------
# 2. Fallback 체인 정의 (사용하지 않고, get_contact_fallback_response 사용)
# ----------------------------------------------------------------------------------
# def create_fallback_chain(llm: ChatOpenAI) -> Runnable:
#     """RAG 실패 시 Fallback 로직이 변경되었으므로, 이 함수는 레거시로 남겨두거나 삭제 가능합니다."""
#     # 현재 요구사항에 따라 이 복잡한 체인 대신 get_contact_fallback_response를 사용합니다.
#     return None 

# ----------------------------------------------------------------------------------

class FAQChatbot:
    """포트폴리오 FAQ에 기반한 RAG 챗봇 클래스"""

    def __init__(self):
        self.rag_chain: Optional[Runnable] = None
        # self.fallback_chain: Optional[Runnable] = None # Fallback 체인 사용 안 함
        self.initialize_rag_chain()
        self.contact_keywords = ["연락", "이메일", "전화", "번호", "email", "contact"] 


    def initialize_rag_chain(self):
        """RAG 체인을 초기화합니다."""
        print("--- RAG 체인 초기화 시작 (faq_chatbot.py) ---")

        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY 환경 변수가 설정되지 않아 RAG 체인을 초기화할 수 없습니다.")
            return

        try:
            # 1~3. 데이터 로드, 청킹, 벡터 저장소 및 리트리버 설정
            loader = TextLoader(FAQ_FILE_NAME, encoding="utf-8")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            print(f"총 {len(texts)}개의 문서 청크 생성됨.")
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(texts, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # 4. LLM 및 프롬프트 정의
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
            # RAG 시스템 프롬프트 (Fallback 감지 문구 유지)
            system_message = (
                "당신은 AI 개발자님의 포트폴리오를 분석하는 전문 챗봇입니다. "
                "사용자의 질문에 대해 제공된 포트폴리오 정보 내에서만 간결하고 정확하게 답변하세요. "
                "만약 관련 정보를 찾을 수 없다면 '제공된 포트폴리오 정보 내에서는 답변을 찾을 수 없습니다.'라고 응답하십시오. "
                "답변 시 1인칭 대명사('저', '저의')를 사용하여 자연스러운 답변을 생성해도 좋습니다. "
                "반드시 공손하고 전문적인 한국어 어조를 유지하십시오."
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_message),
                    HumanMessagePromptTemplate.from_template(
                        "질문: {question}\n\n참고 문서:\n{context}"
                    ),
                ]
            )

            # 5. RAG 체인 조합
            self.rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            print("--- RAG 체인 초기화 완료 ---")

        except Exception as e:
            print(f"ERROR: RAG 체인 초기화 실패: {e}")
            self.rag_chain = None
            
    # ----------------------------------------------------------------------------------
    # 3. 특수 쿼리 처리 로직 (연락처 정보 get_contact_fallback_response 함수 사용)
    # ----------------------------------------------------------------------------------
    def handle_special_query(self, query: str) -> Optional[ChatResponse]:
        """연락처 요청 등 특수 쿼리를 처리합니다."""
        
        query_lower = query.lower()

        # 1. 연락처 요청 처리
        if not query or any(keyword in query_lower for keyword in self.contact_keywords):
            # 특수 쿼리 응답도 get_contact_fallback_response 함수를 사용하여 통일
            return get_contact_fallback_response(latency=0)
        
        # 2. 단순 인사말 처리 (기존과 동일)
        greeting_keywords = ["안녕", "반가워", "소개", "헬로", "hi", "인사"]
        if any(keyword in query_lower for keyword in greeting_keywords):
            return ChatResponse(
                response="반갑습니다! 저는 AI 개발자님의 포트폴리오 전문 챗봇입니다. 어떤 프로젝트나 기술 스택에 대해 궁금하신가요?",
                latencyMs=0,
                sourceDocuments=[]
            )
        
        return None

    # ----------------------------------------------------------------------------------
    # 4. 최종 응답 생성 로직 (RAG 실패 시 연락처 Fallback)
    # ----------------------------------------------------------------------------------
    def get_rag_response(self, query: str) -> ChatResponse:
        """RAG 체인을 실행하고, 실패 시 연락처 안내 응답을 반환합니다."""
        
        # 1. RAG 체인 실행
        rag_response = self.rag_chain.invoke(query)
        
       # 2. RAG 실패 감지 로직 개선:
        # "정보 내에서는 답변을 찾을 수 없습니다."나 "정보 내에서는 ~~ 정보를 찾을 수 없습니다." 모두 포함하는지 확인
        if "정보 내에서는" in rag_response and ("찾을 수 없습니다" in rag_response or "없습니다" in rag_response):
            print("INFO: RAG 정보 찾기 실패. 연락처 Fallback 응답 제공.")
            
            # 3. Fallback: 연락처 정보가 포함된 ChatResponse 객체 반환
            return get_contact_fallback_response(latency=0) 
        
        # 4. RAG 성공 시, ChatResponse 객체 반환
        return ChatResponse(
            response=rag_response,
            latencyMs=0, 
            sourceDocuments=RAG_SOURCES
        )