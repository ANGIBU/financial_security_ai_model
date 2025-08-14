# data_processor.py

"""
데이터 처리기
- 객관식/주관식 분류
- 텍스트 정리
- 답변 검증
"""

import re
from typing import Dict, List

class SimpleDataProcessor:
    """데이터 처리기"""
    
    def __init__(self):
        # 객관식 패턴
        self.mc_patterns = [
            r'①.*②.*③.*④.*⑤',  # 동그라미 숫자
            r'1\).*2\).*3\).*4\).*5\)',  # 번호 형식
            r'1\s+.*2\s+.*3\s+.*4\s+.*5\s+',  # 번호 공백 형식
            r'해당하지.*않는.*것',
            r'적절하지.*않는.*것', 
            r'옳지.*않는.*것',
            r'틀린.*것',
            r'맞는.*것',
            r'옳은.*것',
            r'적절한.*것',
            r'올바른.*것',
            r'가장.*적절한.*것',
            r'가장.*옳은.*것'
        ]
        
        # 주관식 패턴
        self.subj_patterns = [
            r'설명하세요',
            r'기술하세요', 
            r'서술하세요',
            r'작성하세요',
            r'무엇인가요',
            r'어떻게.*해야.*하며',
            r'방안을.*기술',
            r'대응.*방안'
        ]
        
        # 도메인 키워드
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", 
                "고유식별정보", "동의", "법정대리인", "아동"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", 
                "전자서명", "전자인증", "공인인증서", "분쟁조정"
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱",
                "스미싱", "랜섬웨어", "해킹", "딥페이크", "RAT"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", 
                "접근통제", "암호화", "방화벽", "침입탐지"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업",
                "소비자금융업", "보험중개업", "자본시장법"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용",
                "리스크", "내부통제", "컴플라이언스"
            ]
        }
    
    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석"""
        
        question = question.strip()
        
        # 1차: 명확한 선택지 기호 확인
        if re.search(r'①.*②.*③.*④.*⑤', question, re.DOTALL):
            return "multiple_choice"
        
        if re.search(r'1\).*2\).*3\).*4\).*5\)', question, re.DOTALL):
            return "multiple_choice"
        
        if re.search(r'1\s+[가-힣].*2\s+[가-힣].*3\s+[가-힣]', question, re.DOTALL):
            return "multiple_choice"
        
        # 2차: 주관식 패턴 우선 확인
        for pattern in self.subj_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"
        
        # 3차: 객관식 패턴 확인
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "multiple_choice"
        
        # 4차: "것은?" "것?" 패턴 (객관식 가능성 높음)
        if re.search(r'것은\?|것\?|것은\s*$', question):
            return "multiple_choice"
        
        # 5차: 질문 길이로 판단
        if len(question) < 100 and question.endswith('?'):
            return "multiple_choice"
        
        # 기본값: 주관식
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """도메인 추출"""
        question_lower = question.lower()
        
        # 각 도메인별 키워드 매칭 점수 계산
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return "일반"
        
        # 가장 높은 점수의 도메인 반환
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 불필요한 문자 제거
        text = re.sub(r'[\u4e00-\u9fff]', '', text)  # 중국어
        text = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', text)  # 특수 기호
        text = re.sub(r'[^\w\s가-힣.,!?()[\]-]', '', text)  # 특수문자
        
        # 반복 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def validate_answer(self, answer: str, question_type: str) -> bool:
        """답변 유효성 검증"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 객관식: 1-5 범위의 숫자
            return bool(re.match(r'^[1-5]$', answer))
        
        else:
            # 주관식: 최소 길이와 한국어 비율
            clean_answer = self.clean_text(answer)
            
            if len(clean_answer) < 10:
                return False
            
            # 한국어 문자 비율 계산
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            total_chars = len(re.sub(r'[^\w]', '', clean_answer))
            
            if total_chars == 0:
                return False
            
            korean_ratio = korean_chars / total_chars
            return korean_ratio > 0.2 and len(clean_answer) >= 15
    
    def extract_choices(self, question: str) -> List[str]:
        """객관식 선택지 추출"""
        choices = []
        
        # 번호 형식 선택지 찾기
        patterns = [
            r'(\d+)\s+([^0-9\n]+?)(?=\d+\s+|$)',
            r'(\d+)\)\s*([^0-9\n]+?)(?=\d+\)|$)',
            r'[①②③④⑤]\s*([^①②③④⑤\n]+?)(?=[①②③④⑤]|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.MULTILINE)
            if matches and len(matches) >= 3:
                if isinstance(matches[0], tuple):
                    choices = [match[1].strip() for match in matches]
                else:
                    choices = [match.strip() for match in matches]
                break
        
        return choices[:5]  # 최대 5개 선택지
    
    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()
        
        # 전문 용어 개수
        technical_terms = [
            "isms", "pims", "sbom", "rat", "apt", "dlp", "siem",
            "트로이", "멀웨어", "랜섬웨어", "딥페이크", "피싱"
        ]
        
        term_count = sum(1 for term in technical_terms if term in question_lower)
        
        # 문장 길이
        length = len(question)
        
        # 난이도 계산
        if term_count >= 2 or length > 200:
            return "고급"
        elif term_count >= 1 or length > 100:
            return "중급"
        else:
            return "초급"
    
    def normalize_answer(self, answer: str, question_type: str) -> str:
        """답변 정규화"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 숫자만 추출
            numbers = re.findall(r'[1-5]', answer)
            return numbers[0] if numbers else ""
        
        else:
            # 주관식 답변 정리
            answer = self.clean_text(answer)
            
            # 길이 제한
            if len(answer) > 500:
                answer = answer[:500] + "."
            
            # 마침표 확인
            if answer and not answer.endswith(('.', '다', '요')):
                answer += "."
            
            return answer
    
    def cleanup(self):
        """정리"""
        pass