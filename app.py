import os
import time
from flask import Flask, request, jsonify 
from flask_cors import CORS 
from pydantic import ValidationError

# 분리된 모듈에서 Chatbot 클래스와 DTO를 가져옵니다.
from faq_chatbot import FAQChatbot, ChatQueryRequest, ChatResponse, RAG_SOURCES

# --- 1. Flask 및 Chatbot 초기화 ---

app = Flask(__name__)
# CORS 설정 추가 (모든 출처 허용)
CORS(app) 

# 전역 챗봇 인스턴스 생성
CHATBOT = FAQChatbot()

# --- 2. 챗봇 API 엔드포인트 ---

@app.route("/query", methods=["POST"])
def query_chatbot():
    start_time = time.time()
    
    # 1. 요청 데이터 유효성 검사
    try:
        json_data = request.get_json() 
        
        if json_data is None:
            # Content-Type이 application/json이 아니거나, 본문이 비어있는 경우
            return jsonify({"error": "Invalid JSON format: No JSON data received or wrong Content-Type"}), 400
            
        # 파싱된 딕셔너리(json_data)를 Pydantic의 model_validate()에 전달.
        data = ChatQueryRequest.model_validate(json_data) 
        query = data.query.strip()
        
    except ValidationError as e:
        # Pydantic 모델 검증 실패
        print(f"Pydantic 유효성 검사 실패: {e.errors()}")
        return jsonify({"error": f"Invalid input format (Missing 'query' key): {e.errors()}"}), 400
    except Exception as e:
        # request.get_json()에서 발생한 예외 또는 기타 오류
        print(f"요청 처리 중 일반 오류 발생: {e}")
        return jsonify({"error": "Invalid JSON format or missing data"}), 400

    # 2. 챗봇 초기화 상태 확인
    if CHATBOT.rag_chain is None:
        return jsonify(ChatResponse(
            response="죄송합니다. 챗봇 서버가 아직 초기화되지 않았거나 오류가 발생했습니다. (API Key 확인 필요)",
            latencyMs=0,
            sourceDocuments=[]
        ).model_dump()), 500

    # 3. 특수 쿼리 우선 처리
    special_response = CHATBOT.handle_special_query(query)
    if special_response:
        end_time = time.time()
        special_response.latencyMs = int((end_time - start_time) * 1000)
        # Pydantic 객체를 딕셔너리로 변환하여 반환
        return jsonify(special_response.model_dump())

    # 4. RAG 체인 실행
    try:
        # CHATBOT.get_rag_response는 이미 ChatResponse 객체를 반환
        chat_response_obj = CHATBOT.get_rag_response(query)
        end_time = time.time()
        
        # 반환된 ChatResponse 객체의 latencyMs 필드만 업데이트.
        chat_response_obj.latencyMs = int((end_time - start_time) * 1000)
        
        # 5. 응답 반환: ChatResponse 객체를 딕셔너리로 변환하여 JSON 응답
        return jsonify(chat_response_obj.model_dump())

    except Exception as e:
        print(f"RAG Chain 실행 중 오류 발생: {e}")
        return jsonify(ChatResponse(
            response="RAG 모델 실행 중 내부 오류가 발생했습니다. 로그를 확인해 주세요.",
            latencyMs=int((time.time() - start_time) * 1000),
            sourceDocuments=[]
        ).model_dump()), 500

# Flask 앱 시작
if __name__ == "__main__":
    # debug=True는 개발 단계에서 유용하지만, 프로덕션에서는 False로 설정하는 것이 좋음.
    app.run(host="0.0.0.0", port=5000, debug=False)
