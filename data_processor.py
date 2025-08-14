# data_processor.py

"""
실제 데이터 처리 시스템
- 질문 구조 분석
- 텍스트 전처리 및 정리
- 도메인 분류
- 한국어 품질 관리
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd

class RealDataProcessor:
    """실제 데이터 처리기 - 대회 규칙 준수"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.domain_keywords = {
            "개인정보": ["개인정보", "정보주체", "개인정보보호법", "민감정보", "고유식별정보"],
            "전자금융": ["전자금융", "전자적", "접근매체", "전자금융거래법", "전자서명"],
            "사이버보안": ["트로이", "악성코드", "해킹", "멀웨어", "피싱", "스미싱", "랜섬웨어"],
            "정보보안": ["정보보안", "보안관리", "ISMS", "보안정책", "접근통제", "암호화"]
        }
        
    def analyze_question_structure(self, question: str) -> Dict:
        """질문 구조 분석"""
        question = question.strip()
        
        # 객관식 패턴 검사
        mc_patterns = [
            r'①.*②.*③.*④.*⑤',
            r'1\).*2\).*3\).*4\).*5\)',
            r'가\).*나\).*다\).*라\).*마\)',
            r'다음.*중.*옳은.*것은',
            r'다음.*중.*맞는.*것은',
            r'다음.*중.*적절한.*것은',
            r'다음.*중.*해당하는.*것은',
            r'다음.*설명.*중.*올바른.*것은'
        ]
        
        is_multiple_choice = any(re.search(pattern, question, re.DOTALL | re.IGNORECASE) for pattern in mc_patterns)
        
        # 복잡도 점수 계산
        complexity_score = self._calculate_complexity(question)
        
        return {
            "question_type": "multiple_choice" if is_multiple_choice else "subjective",
            "complexity_score": complexity_score,
            "length": len(question),
            "has_legal_terms": self._has_legal_terms(question),
            "domain_hints": self._extract_domain_hints(question),
            "negative_question": self._is_negative_question(question),
            "technical_level": self._assess_technical_level(question)
        }
    
    def _calculate_complexity(self, question: str) -> float:
        """질문 복잡도 계산"""
        complexity_score = 0
        
        # 길이 기반
        complexity_score += len(question) * 0.001
        
        # 전문 용어 개수
        complexity_score += len(re.findall(r'[가-힣]{4,}', question)) * 0.01
        
        # 법률 용어
        legal_terms = ['법', '규정', '조치', '의무', '책임', '처벌', '위반']
        complexity_score += sum(question.count(term) for term in legal_terms) * 0.05
        
        # 기술 용어
        tech_terms = ['시스템', '프로그램', '네트워크', '데이터베이스', '서버']
        complexity_score += sum(question.count(term) for term in tech_terms) * 0.03
        
        # 문장 구조 복잡도
        sentence_count = question.count('.') + question.count('?') + 1
        if sentence_count > 3:
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    def _has_legal_terms(self, question: str) -> bool:
        """법률 용어 포함 여부"""
        legal_terms = ['법', '규정', '조치', '의무', '책임', '처벌', '위반', '준수', '시행', '적용']
        return any(term in question for term in legal_terms)
    
    def _extract_domain_hints(self, question: str) -> List[str]:
        """도메인 힌트 추출"""
        hints = []
        question_lower = question.lower()
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                hints.append(domain)
        
        return hints if hints else ["일반"]
    
    def _is_negative_question(self, question: str) -> bool:
        """부정형 질문 여부"""
        negative_patterns = [
            r'해당하지.*않는',
            r'적절하지.*않는',
            r'옳지.*않는',
            r'틀린.*것',
            r'잘못.*설명',
            r'부적절한.*것'
        ]
        
        return any(re.search(pattern, question, re.IGNORECASE) for pattern in negative_patterns)
    
    def _assess_technical_level(self, question: str) -> str:
        """기술적 난이도 평가"""
        technical_keywords = {
            "고급": ["암호화", "해시", "디지털서명", "PKI", "SSL/TLS"],
            "중급": ["방화벽", "IDS", "VPN", "접근제어", "인증"],
            "초급": ["비밀번호", "백업", "업데이트", "바이러스", "보안"]
        }
        
        for level, keywords in technical_keywords.items():
            if any(keyword in question for keyword in keywords):
                return level
        
        return "일반"
    
    def _clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리"""
        if not text:
            return ""
        
        # 불필요한 문자 제거
        text = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', text)
        text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', '', text)  # 자음/모음 제거
        text = re.sub(r'\s+', ' ', text)  # 연속 공백 정리
        text = text.strip()
        
        # 영어 단어 정리 (전문용어 제외)
        preserve_words = ['IT', 'AI', 'API', 'DB', 'OS', 'IP', 'DNS', 'HTTP', 'HTTPS', 'SSL', 'TLS', 'VPN', 'PKI', 'ISMS']
        
        # 보존할 단어들을 임시로 치환
        temp_replacements = {}
        for i, word in enumerate(preserve_words):
            if word in text:
                temp_key = f"__PRESERVE_{i}__"
                temp_replacements[temp_key] = word
                text = text.replace(word, temp_key)
        
        # 긴 영어 단어 제거 (3글자 이상)
        text = re.sub(r'\b[A-Za-z]{3,}\b', '', text)
        
        # 보존된 단어들 복원
        for temp_key, original_word in temp_replacements.items():
            text = text.replace(temp_key, original_word)
        
        # 최종 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 최소 길이 확인
        if len(text) < 10:
            return ""
        
        return text
    
    def extract_multiple_choice_options(self, question: str) -> List[str]:
        """객관식 선택지 추출"""
        options = []
        
        # 다양한 선택지 패턴
        patterns = [
            r'①\s*([^②③④⑤]+)',
            r'②\s*([^①③④⑤]+)',
            r'③\s*([^①②④⑤]+)', 
            r'④\s*([^①②③⑤]+)',
            r'⑤\s*([^①②③④]+)',
            r'1\)\s*([^2)3)4)5)]+)',
            r'2\)\s*([^1)3)4)5)]+)',
            r'3\)\s*([^1)2)4)5)]+)',
            r'4\)\s*([^1)2)3)5)]+)',
            r'5\)\s*([^1)2)3)4)]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.DOTALL)
            if match:
                option_text = match.group(1).strip()
                if option_text:
                    options.append(option_text)
        
        return options
    
    def validate_answer_format(self, answer: str, question_type: str) -> Tuple[bool, str]:
        """답변 형식 검증"""
        if question_type == "multiple_choice":
            # 객관식: 1-5 숫자여야 함
            if re.match(r'^[1-5]$', answer.strip()):
                return True, answer.strip()
            
            # 숫자 추출 시도
            numbers = re.findall(r'[1-5]', answer)
            if numbers:
                return True, numbers[0]
            
            return False, ""
        
        else:  # subjective
            # 주관식: 한국어 포함, 최소 길이
            cleaned = self._clean_korean_text(answer)
            
            if len(cleaned) < 20:
                return False, ""
            
            # 한국어 비율 확인
            korean_chars = len(re.findall(r'[가-힣]', cleaned))
            total_chars = len(re.sub(r'[^\w]', '', cleaned))
            
            if total_chars == 0 or korean_chars / total_chars < 0.6:
                return False, ""
            
            return True, cleaned
    
    def extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 전문 용어 패턴
        professional_terms = []
        
        # 법률 용어
        legal_pattern = r'(개인정보보호법|전자금융거래법|정보통신망법|[가-힣]{2,}법)'
        legal_terms = re.findall(legal_pattern, text)
        professional_terms.extend(legal_terms)
        
        # 기술 용어  
        tech_pattern = r'(시스템|프로그램|네트워크|데이터베이스|서버|보안|암호화|인증|접근제어)'
        tech_terms = re.findall(tech_pattern, text)
        professional_terms.extend(tech_terms)
        
        # 중요 개념
        concept_pattern = r'([가-힣]{3,}조치|[가-힣]{3,}관리|[가-힣]{3,}정책)'
        concept_terms = re.findall(concept_pattern, text)
        professional_terms.extend(concept_terms)
        
        return list(set(professional_terms))
    
    def classify_question_difficulty(self, question: str) -> str:
        """질문 난이도 분류"""
        structure = self.analyze_question_structure(question)
        complexity = structure["complexity_score"]
        technical_level = structure["technical_level"]
        
        if complexity > 0.7 or technical_level == "고급":
            return "고급"
        elif complexity > 0.4 or technical_level == "중급":
            return "중급"
        else:
            return "초급"
    
    def preprocess_for_model(self, question: str) -> Dict:
        """모델 입력용 전처리"""
        structure = self.analyze_question_structure(question)
        keywords = self.extract_keywords(question)
        difficulty = self.classify_question_difficulty(question)
        
        # 질문 정리
        cleaned_question = re.sub(r'\s+', ' ', question.strip())
        
        return {
            "cleaned_question": cleaned_question,
            "structure": structure,
            "keywords": keywords,
            "difficulty": difficulty,
            "char_count": len(cleaned_question),
            "word_count": len(cleaned_question.split()),
            "domain": structure["domain_hints"][0] if structure["domain_hints"] else "일반"
        }
    
    def cleanup(self) -> None:
        """리소스 정리"""
        if self.debug_mode:
            print("데이터 처리기 정리 완료")


class DataProcessor:
    """기존 데이터 처리기 (하위 호환성)"""
    
    def __init__(self):
        self.real_processor = RealDataProcessor()
    
    def analyze_question_structure(self, question: str) -> Dict:
        return self.real_processor.analyze_question_structure(question)
    
    def _clean_korean_text(self, text: str) -> str:
        return self.real_processor._clean_korean_text(text)
    
    def cleanup(self) -> None:
        self.real_processor.cleanup()