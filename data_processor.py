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
from pathlib import Path

class SimpleDataProcessor:
    """데이터 처리기"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 객관식 패턴 (개선된 버전)
        self.mc_patterns = [
            r'①.*②.*③.*④.*⑤',  # 동그라미 숫자
            r'1\s+[가-힣].*2\s+[가-힣].*3\s+[가-힣].*4\s+[가-힣].*5\s+[가-힣]',  # 번호 + 한글
            r'1\s+.*2\s+.*3\s+.*4\s+.*5\s+',  # 번호 공백 형식
            r'1\.\s*.*2\.\s*.*3\.\s*.*4\.\s*.*5\.',  # 1. 2. 3. 형식
            r'1\)\s*.*2\)\s*.*3\)\s*.*4\)\s*.*5\)',  # 1) 2) 3) 형식
        ]
        
        # 객관식 키워드 패턴
        self.mc_keywords = [
            r'해당하지.*않는.*것',
            r'적절하지.*않는.*것', 
            r'옳지.*않는.*것',
            r'틀린.*것',
            r'맞는.*것',
            r'옳은.*것',
            r'적절한.*것',
            r'올바른.*것',
            r'가장.*적절한.*것',
            r'가장.*옳은.*것',
            r'구분.*해당하지.*않는.*것',
            r'다음.*중.*것은',
            r'다음.*중.*것',
            r'다음.*보기.*중'
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
            r'대응.*방안',
            r'특징.*다음과.*같',
            r'탐지.*지표',
            r'행동.*패턴',
            r'분석하여.*제시',
            r'조치.*사항',
            r'제시하시오'
        ]
        
        # 도메인 키워드 (확장)
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", 
                "고유식별정보", "동의", "법정대리인", "아동",
                "개인정보처리자", "열람권", "정정삭제권", "처리정지권",
                "개인정보보호위원회", "손해배상", "처리방침"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", 
                "전자서명", "전자인증", "공인인증서", "분쟁조정",
                "전자지급수단", "전자화폐", "금융감독원", "한국은행",
                "전자금융업", "전자금융분쟁조정위원회"
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱",
                "스미싱", "랜섬웨어", "해킹", "딥페이크", "원격제어",
                "RAT", "원격접근", "봇넷", "백도어", "루트킷",
                "취약점", "제로데이", "사회공학", "APT", "DDoS"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", 
                "접근통제", "암호화", "방화벽", "침입탐지",
                "침입방지시스템", "IDS", "IPS", "보안관제",
                "로그관리", "백업", "복구", "재해복구", "BCP"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업",
                "소비자금융업", "보험중개업", "자본시장법",
                "집합투자업", "신탁업", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용",
                "리스크", "내부통제", "컴플라이언스", "위험식별",
                "위험분석", "위험모니터링", "위험회피", "위험전가",
                "위험감소", "잔여위험", "위험성향"
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
            "domain_distribution": {},
            "question_type_accuracy": {"correct": 0, "total": 0}
        }
        
        # 이전 처리 기록 로드
        self._load_processing_history()
    
    def _load_processing_history(self):
        """이전 처리 기록 로드"""
        history_file = self.pkl_dir / "processing_history.pkl"
        
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    saved_stats = pickle.load(f)
                    self.processing_stats.update(saved_stats)
            except Exception:
                pass
    
    def _save_processing_history(self):
        """처리 기록 저장"""
        history_file = self.pkl_dir / "processing_history.pkl"
        
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
        """질문 유형 분석 (개선된 버전)"""
        
        question = question.strip()
        self.processing_stats["question_type_accuracy"]["total"] += 1
        
        # 1차: 명확한 선택지 패턴 확인
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                self.processing_stats["question_type_accuracy"]["correct"] += 1
                return "multiple_choice"
        
        # 2차: 선택지 개수 확인 (1부터 5까지 모두 있는지)
        numbers_found = []
        for i in range(1, 6):
            if re.search(rf'\b{i}[\s\.\)]\s*[가-힣]', question):
                numbers_found.append(i)
        
        if len(numbers_found) == 5:
            self.processing_stats["question_type_accuracy"]["correct"] += 1
            return "multiple_choice"
        
        # 3차: 객관식 키워드 확인
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                # 선택지가 있는지 추가 확인
                if any(str(i) in question for i in range(1, 6)):
                    self.processing_stats["question_type_accuracy"]["correct"] += 1
                    return "multiple_choice"
        
        # 4차: 주관식 패턴 확인
        for pattern in self.subj_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"
        
        # 5차: 질문 구조 분석
        # 선택지 형태가 있는지 확인
        lines = question.split('\n')
        choice_lines = 0
        for line in lines:
            if re.match(r'^\s*[1-5][\s\.\)]\s*', line):
                choice_lines += 1
        
        if choice_lines >= 3:  # 3개 이상의 선택지가 있으면 객관식
            return "multiple_choice"
        
        # 6차: "것은?" "것?" 패턴과 길이로 추가 판단
        if re.search(r'것은\?|것\?|것은\s*$', question):
            if len(question) < 300 and any(str(i) in question for i in range(1, 6)):
                return "multiple_choice"
        
        # 7차: 질문 길이와 내용으로 최종 판단
        if len(question) < 200 and any(word in question for word in ["구분", "해당", "적절", "옳은", "올바른"]):
            if any(str(i) in question for i in range(1, 6)):
                return "multiple_choice"
        
        # 기본값: 주관식
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """도메인 추출 (개선)"""
        question_lower = question.lower()
        
        # 각 도메인별 키워드 매칭 점수 계산
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드는 가중치 부여
                    if keyword in ["개인정보보호법", "전자금융거래법", "자본시장법", "ISMS"]:
                        score += 3
                    else:
                        score += 1
            
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
        """한국어 전용 텍스트 정리 (강화)"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 깨진 문자 및 인코딩 오류 처리
        text = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', text)
        
        # 영어 문자 제거 (한국어 답변을 위해)
        text = re.sub(r'[a-zA-Z]+', '', text)
        
        # 중국어 제거
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        
        # 특수 기호 제거
        text = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', text)
        
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
        """한국어 답변 유효성 검증 (강화)"""
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
            # 주관식: 한국어 전용 검증 (더 엄격하게)
            clean_answer = self.clean_korean_text(answer)
            
            # 길이 검증
            if not (self.korean_requirements["min_length"] <= len(clean_answer) <= self.korean_requirements["max_length"]):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 한국어 비율 검증 (더 엄격하게)
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
            
            # 의미 있는 내용인지 확인
            meaningful_keywords = ["법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", "정책", "체계", "시스템", "통제"]
            if not any(word in clean_answer for word in meaningful_keywords):
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
        
        # 다양한 형식의 선택지 패턴
        patterns = [
            r'(\d+)\s+([^0-9\n]+?)(?=\d+\s+|$)',
            r'(\d+)\)\s*([^0-9\n]+?)(?=\d+\)|$)',
            r'(\d+)\.\s*([^0-9\n]+?)(?=\d+\.|$)',
            r'[①②③④⑤]\s*([^①②③④⑤\n]+?)(?=[①②③④⑤]|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
            if matches:
                if isinstance(matches[0], tuple):
                    choices = [match[1].strip() for match in matches]
                else:
                    choices = [match.strip() for match in matches]
                
                # 5개 선택지가 모두 있는지 확인
                if len(choices) >= 5:
                    break
        
        return choices[:5]  # 최대 5개 선택지
    
    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()
        
        # 전문 용어 개수
        technical_terms = [
            "isms", "pims", "sbom", "원격제어", "침입탐지", 
            "트로이", "멀웨어", "랜섬웨어", "딥페이크", "피싱",
            "접근매체", "전자서명", "개인정보보호법", "자본시장법",
            "rat", "원격접근", "탐지지표", "apt", "ddos",
            "ids", "ips", "bcp", "drp", "isms-p"
        ]
        
        term_count = sum(1 for term in technical_terms if term in question_lower)
        
        # 문장 길이
        length = len(question)
        
        # 난이도 계산
        if term_count >= 3 or length > 300:
            return "고급"
        elif term_count >= 1 or length > 150:
            return "중급"
        else:
            return "초급"
    
    def normalize_korean_answer(self, answer: str, question_type: str) -> str:
        """한국어 답변 정규화 (강화)"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 숫자만 추출
            numbers = re.findall(r'[1-5]', answer)
            return numbers[0] if numbers else ""
        
        else:
            # 주관식 답변 한국어 정리 (더 강화)
            answer = self.clean_korean_text(answer)
            
            # 의미 없는 짧은 문장 제거
            if len(answer) < 20:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
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
            "domain_distribution": dict(self.processing_stats["domain_distribution"]),
            "question_type_accuracy": self.processing_stats["question_type_accuracy"]
        }
    
    def get_korean_requirements(self) -> Dict:
        """한국어 요구사항 반환"""
        return dict(self.korean_requirements)
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()