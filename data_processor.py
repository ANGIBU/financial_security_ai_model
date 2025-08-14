# data_processor.py

"""
데이터 처리기
- 객관식/주관식 분류
- 텍스트 정리
- 답변 검증
- 한국어 전용 처리
"""

import re
import pickle
import os
from typing import Dict, List
from datetime import datetime

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
                "스미싱", "랜섬웨어", "해킹", "딥페이크", "원격제어"
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
        
        # 한국어 전용 검증 기준
        self.korean_requirements = {
            "min_korean_ratio": 0.8,  # 최소 한국어 비율 80%
            "max_english_ratio": 0.1,  # 최대 영어 비율 10%
            "min_length": 30,  # 최소 길이
            "max_length": 500  # 최대 길이
        }
        
        # 처리 통계
        self.processing_stats = {
            "total_processed": 0,
            "korean_compliance": 0,
            "validation_failures": 0,
            "domain_distribution": {}
        }
        
        # 이전 처리 기록 로드
        self._load_processing_history()
    
    def _load_processing_history(self):
        """이전 처리 기록 로드"""
        history_file = "./processing_history.pkl"
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'rb') as f:
                    saved_stats = pickle.load(f)
                    self.processing_stats.update(saved_stats)
            except Exception:
                pass
    
    def _save_processing_history(self):
        """처리 기록 저장"""
        history_file = "./processing_history.pkl"
        
        try:
            save_data = {
                **self.processing_stats,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(history_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
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
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # 통계 업데이트
        if detected_domain not in self.processing_stats["domain_distribution"]:
            self.processing_stats["domain_distribution"][detected_domain] = 0
        self.processing_stats["domain_distribution"][detected_domain] += 1
        
        return detected_domain
    
    def clean_korean_text(self, text: str) -> str:
        """한국어 전용 텍스트 정리"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 영어 문자 제거 (한국어 답변을 위해)
        text = re.sub(r'[a-zA-Z]+', '', text)
        
        # 중국어 제거
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        
        # 특수 기호 제거
        text = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', text)
        
        # 허용된 문자만 유지 (한국어, 숫자, 기본 문장부호)
        text = re.sub(r'[^\w\s가-힣.,!?()[\]-]', ' ', text)
        
        # 반복 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def calculate_english_ratio(self, text: str) -> float:
        """영어 비율 계산"""
        if not text:
            return 0.0
        
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return english_chars / total_chars
    
    def validate_korean_answer(self, answer: str, question_type: str) -> bool:
        """한국어 답변 유효성 검증"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        self.processing_stats["total_processed"] += 1
        
        if question_type == "multiple_choice":
            # 객관식: 1-5 범위의 숫자
            is_valid = bool(re.match(r'^[1-5]$', answer))
            if is_valid:
                self.processing_stats["korean_compliance"] += 1
            else:
                self.processing_stats["validation_failures"] += 1
            return is_valid
        
        else:
            # 주관식: 한국어 전용 검증
            clean_answer = self.clean_korean_text(answer)
            
            # 길이 검증
            if not (self.korean_requirements["min_length"] <= len(clean_answer) <= self.korean_requirements["max_length"]):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 한국어 비율 검증
            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < self.korean_requirements["min_korean_ratio"]:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 영어 비율 검증
            english_ratio = self.calculate_english_ratio(answer)  # 원본 텍스트에서 검증
            if english_ratio > self.korean_requirements["max_english_ratio"]:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 최소 한국어 문자 수 검증
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            if korean_chars < 20:
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
    
    def validate_answer(self, answer: str, question_type: str) -> bool:
        """답변 유효성 검증 (한국어 전용)"""
        return self.validate_korean_answer(answer, question_type)
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리 (한국어 전용)"""
        return self.clean_korean_text(text)
    
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
            "isms", "pims", "sbom", "원격제어", "침입탐지", 
            "트로이", "멀웨어", "랜섬웨어", "딥페이크", "피싱",
            "접근매체", "전자서명", "개인정보보호법", "자본시장법"
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
    
    def normalize_korean_answer(self, answer: str, question_type: str) -> str:
        """한국어 답변 정규화"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 숫자만 추출
            numbers = re.findall(r'[1-5]', answer)
            return numbers[0] if numbers else ""
        
        else:
            # 주관식 답변 한국어 정리
            answer = self.clean_korean_text(answer)
            
            # 길이 제한
            if len(answer) > self.korean_requirements["max_length"]:
                sentences = answer.split('. ')
                answer = '. '.join(sentences[:3])
                if len(answer) > self.korean_requirements["max_length"]:
                    answer = answer[:self.korean_requirements["max_length"]]
            
            # 마침표 확인
            if answer and not answer.endswith(('.', '다', '요', '함')):
                answer += "."
            
            return answer
    
    def normalize_answer(self, answer: str, question_type: str) -> str:
        """답변 정규화 (한국어 전용)"""
        return self.normalize_korean_answer(answer, question_type)
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환"""
        total = max(self.processing_stats["total_processed"], 1)
        
        return {
            "total_processed": self.processing_stats["total_processed"],
            "korean_compliance_rate": (self.processing_stats["korean_compliance"] / total) * 100,
            "validation_failure_rate": (self.processing_stats["validation_failures"] / total) * 100,
            "domain_distribution": dict(self.processing_stats["domain_distribution"])
        }
    
    def get_korean_requirements(self) -> Dict:
        """한국어 요구사항 반환"""
        return dict(self.korean_requirements)
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()