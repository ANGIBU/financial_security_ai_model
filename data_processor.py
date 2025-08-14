# data_processor.py

"""
간단하고 정확한 데이터 처리기
- 복잡성 제거
- 객관식/주관식 정확한 분류
- 핵심 기능에만 집중
"""

import re
from typing import Dict, List

class SimpleDataProcessor:
    """단순하고 정확한 데이터 처리기"""
    
    def __init__(self):
        # 객관식 패턴 (명확한 것들만)
        self.mc_patterns = [
            r'①.*②.*③.*④.*⑤',  # 동그라미 숫자
            r'1\).*2\).*3\).*4\).*5\)',  # 번호 형식
            r'해당하지.*않는.*것',
            r'적절하지.*않는.*것', 
            r'옳지.*않는.*것',
            r'틀린.*것',
            r'맞는.*것',
            r'옳은.*것',
            r'적절한.*것'
        ]
    
    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석 - 단순하고 정확하게"""
        
        question = question.strip()
        
        # 1차: 선택지 기호가 있으면 무조건 객관식
        if re.search(r'①.*②.*③.*④.*⑤', question, re.DOTALL):
            return "multiple_choice"
        
        if re.search(r'1\).*2\).*3\).*4\).*5\)', question, re.DOTALL):
            return "multiple_choice"
        
        # 2차: 객관식 패턴 확인
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "multiple_choice"
        
        # 3차: "것은?" "것은" "것?" 패턴
        if re.search(r'것은\?|것은\s*$|것\?|것\s*$', question):
            return "multiple_choice"
        
        # 기본값: 주관식
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """도메인 추출"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["트로이", "악성코드", "멀웨어", "바이러스"]):
            return "사이버보안"
        elif any(word in question_lower for word in ["개인정보", "정보주체", "개인정보보호법"]):
            return "개인정보보호"
        elif any(word in question_lower for word in ["전자금융", "전자적", "접근매체"]):
            return "전자금융"
        elif any(word in question_lower for word in ["정보보안", "보안관리", "ISMS"]):
            return "정보보안"
        else:
            return "일반"
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 문제가 되는 문자들 제거
        text = re.sub(r'[\u4e00-\u9fff]', '', text)  # 중국어
        text = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', text)  # 특수 기호
        
        return text
    
    def validate_answer(self, answer: str, question_type: str) -> bool:
        """답변 유효성 검증"""
        if not answer:
            return False
        
        if question_type == "multiple_choice":
            return bool(re.match(r'^[1-5]$', answer.strip()))
        else:
            # 주관식: 최소 길이와 한국어 포함
            clean_answer = self.clean_text(answer)
            if len(clean_answer) < 15:
                return False
            
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            total_chars = len(re.sub(r'[^\w]', '', clean_answer))
            
            if total_chars == 0:
                return False
            
            korean_ratio = korean_chars / total_chars
            return korean_ratio > 0.3
    
    def cleanup(self):
        """정리"""
        pass