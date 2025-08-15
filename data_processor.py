# data_processor.py

"""
데이터 처리기
- 질문 의도 분석 시스템
- 형태소 기반 질문 분류
- 한국어 전용 검증
- 답변 품질 평가
"""

import re
import pickle
import os
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

class SimpleDataProcessor:
    """데이터 처리기"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = Path("./pkl")
        self.pkl_dir.mkdir(exist_ok=True)
        
        # 질문 의도 분석 패턴 (강화)
        self.intent_patterns = {
            "기관_요청": [
                r'기관.*(?:기술하세요|설명하세요|서술하세요)',
                r'(?:어떤|어느).*기관',
                r'기관.*무엇',
                r'조정.*신청.*기관',
                r'분쟁.*조정.*기관',
                r'신청.*수.*있는.*기관',
                r'담당.*기관',
                r'관할.*기관'
            ],
            "특징_분석": [
                r'특징.*(?:설명하세요|기술하세요|서술하세요)',
                r'특성.*(?:설명하세요|기술하세요)',
                r'어떤.*특징',
                r'주요.*특징',
                r'특징.*무엇'
            ],
            "지표_나열": [
                r'지표.*(?:설명하세요|나열하세요)',
                r'탐지.*지표',
                r'주요.*지표',
                r'징후.*설명',
                r'패턴.*설명'
            ],
            "절차_설명": [
                r'절차.*(?:설명하세요|기술하세요)',
                r'과정.*설명',
                r'단계.*설명',
                r'방법.*설명'
            ],
            "법령_해석": [
                r'법.*따라',
                r'규정.*따라',
                r'법령.*근거',
                r'조항.*해석'
            ]
        }
        
        # 객관식 식별 패턴 (개선)
        self.mc_patterns = [
            r'①.*②.*③.*④.*⑤',
            r'1\s+[가-힣].*2\s+[가-힣].*3\s+[가-힣].*4\s+[가-힣].*5\s+[가-힣]',
            r'1\s+.*2\s+.*3\s+.*4\s+.*5\s+',
            r'1\.\s*.*2\.\s*.*3\.\s*.*4\.\s*.*5\.',
            r'1\)\s*.*2\)\s*.*3\)\s*.*4\)\s*.*5\)'
        ]
        
        # 객관식 키워드
        self.mc_keywords = [
            r'해당하지.*않는.*것',
            r'적절하지.*않는.*것',
            r'옳지.*않는.*것',
            r'틀린.*것',
            r'맞는.*것',
            r'옳은.*것',
            r'적절한.*것',
            r'올바른.*것',
            r'가장.*적절한.*것'
        ]
        
        # 주관식 패턴
        self.subj_patterns = [
            r'설명하세요',
            r'기술하세요',
            r'서술하세요',
            r'작성하세요',
            r'무엇인가요',
            r'방안.*기술',
            r'대응.*방안',
            r'조치.*사항'
        ]
        
        # 도메인별 키워드 (확장)
        self.domain_keywords = {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보",
                "고유식별정보", "동의", "법정대리인", "아동",
                "개인정보처리자", "열람권", "정정삭제권", "처리정지권",
                "개인정보보호위원회", "손해배상", "처리방침",
                "개인정보영향평가", "개인정보관리체계"
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법",
                "전자서명", "전자인증", "공인인증서", "분쟁조정",
                "전자지급수단", "전자화폐", "금융감독원", "한국은행",
                "전자금융업", "전자금융분쟁조정위원회",
                "전자금융거래기록", "전자금융업무"
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱",
                "스미싱", "랜섬웨어", "해킹", "딥페이크", "원격제어",
                "RAT", "원격접근", "봇넷", "백도어", "루트킷",
                "취약점", "제로데이", "사회공학", "APT", "DDoS",
                "침입탐지", "침입방지", "보안관제"
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책",
                "접근통제", "암호화", "방화벽", "침입탐지시스템",
                "침입방지시스템", "IDS", "IPS", "보안관제",
                "로그관리", "백업", "복구", "재해복구", "BCP"
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업",
                "소비자금융업", "보험중개업", "자본시장법",
                "집합투자업", "신탁업", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무"
            ]
        }
        
        # 한국어 검증 기준
        self.korean_requirements = {
            "min_korean_ratio": 0.8,
            "max_english_ratio": 0.1,
            "min_length": 30,
            "max_length": 500
        }
        
        # 처리 통계
        self.processing_stats = {
            "total_processed": 0,
            "korean_compliance": 0,
            "validation_failures": 0,
            "intent_accuracy": 0,
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
    
    def analyze_question_intent(self, question: str) -> Dict:
        """질문 의도 분석 (강화)"""
        question_lower = question.lower()
        
        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "specific_requirement": None
        }
        
        # 각 의도별 매칭 점수 계산
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    score += 1
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns
                }
        
        # 가장 높은 점수의 의도 선택
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = best_intent[1]["score"] / len(self.intent_patterns[best_intent[0]])
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]
            
            # 답변 유형 결정
            if "기관" in best_intent[0]:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["specific_requirement"] = "구체적_기관명_필수"
            elif "특징" in best_intent[0]:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["specific_requirement"] = "특성_나열_필수"
            elif "지표" in best_intent[0]:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["specific_requirement"] = "탐지지표_나열_필수"
            elif "절차" in best_intent[0]:
                intent_analysis["answer_type_required"] = "절차설명"
                intent_analysis["specific_requirement"] = "단계별_절차_필수"
            elif "법령" in best_intent[0]:
                intent_analysis["answer_type_required"] = "법령해석"
                intent_analysis["specific_requirement"] = "법적근거_필수"
        
        # 통계 업데이트
        self.processing_stats["intent_accuracy"] += 1
        
        return intent_analysis
    
    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        if not question:
            return "초급"
        
        complexity_score = 0
        
        # 길이 기반 점수
        length = len(question)
        if length > 200:
            complexity_score += 2
        elif length > 100:
            complexity_score += 1
        
        # 전문 용어 개수
        technical_terms = [
            "개인정보보호법", "전자금융거래법", "자본시장법", "ISMS",
            "트로이목마", "원격접근도구", "침입탐지시스템", "침입방지시스템",
            "개인정보영향평가", "개인정보관리체계", "분쟁조정위원회"
        ]
        
        term_count = sum(1 for term in technical_terms if term in question)
        complexity_score += term_count
        
        # 도메인 복잡도
        domain_count = 0
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in question for keyword in keywords):
                domain_count += 1
        
        if domain_count >= 2:
            complexity_score += 1
        
        # 난이도 결정
        if complexity_score >= 4:
            return "고급"
        elif complexity_score >= 2:
            return "중급"
        else:
            return "초급"
    
    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """선택지 범위 추출 (개선)"""
        question_type = self.analyze_question_type(question)
        
        if question_type != "multiple_choice":
            return "subjective", 0
        
        # 줄별 선택지 분석
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            match = re.match(r'^(\d+)\s+', line.strip())
            if match:
                choice_numbers.append(int(match.group(1)))
        
        # 연속성 검증
        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)
            
            expected_count = max_choice - min_choice + 1
            if len(choice_numbers) == expected_count and min_choice == 1:
                return "multiple_choice", max_choice
        
        # 폴백 패턴 확인
        for i in range(5, 2, -1):
            pattern = r'1\s.*' + '.*'.join([f'{j}\s' for j in range(2, i+1)])
            if re.search(pattern, question, re.DOTALL):
                return "multiple_choice", i
        
        # 객관식 키워드 확인 후 기본값
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                return "multiple_choice", 5
        
        return "subjective", 0
    
    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석 (개선)"""
        question = question.strip()
        self.processing_stats["question_type_accuracy"]["total"] += 1
        
        # 1차: 명확한 선택지 패턴 확인
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                self.processing_stats["question_type_accuracy"]["correct"] += 1
                return "multiple_choice"
        
        # 2차: 줄별 선택지 분석
        lines = question.split('\n')
        choice_lines = 0
        for line in lines:
            if re.match(r'^\s*[1-5][\s\.]\s*', line):
                choice_lines += 1
        
        if choice_lines >= 3:
            self.processing_stats["question_type_accuracy"]["correct"] += 1
            return "multiple_choice"
        
        # 3차: 객관식 키워드 확인
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                if any(str(i) in question for i in range(1, 6)):
                    self.processing_stats["question_type_accuracy"]["correct"] += 1
                    return "multiple_choice"
        
        # 4차: 주관식 패턴 확인
        for pattern in self.subj_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"
        
        # 기본값: 주관식
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """도메인 추출 (개선)"""
        question_lower = question.lower()
        
        # 각 도메인별 키워드 매칭 점수
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드 가중치
                    if keyword in ["개인정보보호법", "전자금융거래법", "자본시장법", "ISMS"]:
                        score += 3
                    else:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return "일반"
        
        # 가장 높은 점수의 도메인
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # 통계 업데이트
        if detected_domain not in self.processing_stats["domain_distribution"]:
            self.processing_stats["domain_distribution"][detected_domain] = 0
        self.processing_stats["domain_distribution"][detected_domain] += 1
        
        return detected_domain
    
    def clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 깨진 문자 처리
        text = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', text)
        
        # 영어 문자 제거
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
    
    def validate_mc_answer_range(self, answer: str, max_choice: int) -> bool:
        """객관식 답변 범위 검증"""
        if not answer or not answer.isdigit():
            return False
        
        answer_num = int(answer)
        return 1 <= answer_num <= max_choice
    
    def validate_answer_intent_match(self, answer: str, question: str, intent_analysis: Dict) -> bool:
        """답변과 질문 의도 일치성 검증"""
        if not answer or not intent_analysis:
            return False
        
        required_type = intent_analysis.get("answer_type_required", "설명형")
        answer_lower = answer.lower()
        
        # 기관명이 필요한 경우
        if required_type == "기관명":
            institution_keywords = [
                "위원회", "감독원", "은행", "기관", "센터", "청", "부", "원",
                "전자금융분쟁조정위원회", "금융감독원", "개인정보보호위원회",
                "한국은행", "금융위원회", "과학기술정보통신부"
            ]
            return any(keyword in answer_lower for keyword in institution_keywords)
        
        # 특징 설명이 필요한 경우
        elif required_type == "특징설명":
            feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할", "원리"]
            return any(keyword in answer_lower for keyword in feature_keywords)
        
        # 지표 나열이 필요한 경우
        elif required_type == "지표나열":
            indicator_keywords = ["지표", "신호", "징후", "패턴", "행동", "활동", "모니터링", "탐지"]
            return any(keyword in answer_lower for keyword in indicator_keywords)
        
        # 기본적으로 통과
        return True
    
    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """한국어 답변 유효성 검증"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        self.processing_stats["total_processed"] += 1
        
        if question_type == "multiple_choice":
            # 객관식: 지정된 범위의 숫자
            if not self.validate_mc_answer_range(answer, max_choice):
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
        
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
            english_ratio = self.calculate_english_ratio(answer)
            if english_ratio > self.korean_requirements["max_english_ratio"]:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 최소 한국어 문자 수 검증
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            if korean_chars < 20:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 의미 있는 내용 확인
            meaningful_keywords = ["법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", "정책", "체계", "시스템", "통제"]
            if not any(word in clean_answer for word in meaningful_keywords):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 질문 의도 일치성 검증
            if question:
                intent_analysis = self.analyze_question_intent(question)
                if not self.validate_answer_intent_match(answer, question, intent_analysis):
                    self.processing_stats["validation_failures"] += 1
                    return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
    
    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """한국어 답변 정규화"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 숫자만 추출하고 범위 검증
            numbers = re.findall(r'[1-9]', answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num
            
            return ""
        
        else:
            # 주관식 답변 정리
            answer = self.clean_korean_text(answer)
            
            # 최소 길이 확인
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
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환"""
        total = max(self.processing_stats["total_processed"], 1)
        
        return {
            "total_processed": self.processing_stats["total_processed"],
            "korean_compliance_rate": (self.processing_stats["korean_compliance"] / total) * 100,
            "validation_failure_rate": (self.processing_stats["validation_failures"] / total) * 100,
            "intent_accuracy_rate": (self.processing_stats["intent_accuracy"] / max(self.processing_stats["intent_accuracy"], 1)) * 100,
            "domain_distribution": dict(self.processing_stats["domain_distribution"]),
            "question_type_accuracy": self.processing_stats["question_type_accuracy"]
        }
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()